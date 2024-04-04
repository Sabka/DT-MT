from checkpoint_fv import fv_from_checkpoint
from time import time as tm
import matplotlib.pyplot as plt
from torch import isnan
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import json
from parameters import get_parameters
import os


class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    def __init__(self, m, n, dim, niter, args, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.args = args
        self.m = m
        self.n = n
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m * n, dim).to(args.device)
        self.locations = torch.LongTensor(
            np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)
        self.quant_err = 0
        self.num_err = 0
        self.winner_occurences = [0 for i in range(self.m*self.n)]
        self.d = {}
        self.dist_sum = 0

        self.all_quant_err = []
        self.all_winner = []
        self.all_entr = []
        self.all_dists = []

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def get_som_stats(self):
        quant_err = self.quant_err/self.num_err
        winner_discrimination = sum(
            tmp > 0 for tmp in self.winner_occurences) / (self.m * self.n)

        px = torch.FloatTensor(
            [x/self.num_err for x in self.winner_occurences])
        logpx = torch.log2(px)
        product = px * logpx
        entropy = - torch.nansum(product)

        dists = self.dist_sum / self.num_err

        if isnan(entropy):
            print("self.winner_occurences {} \npx {} \nlogpx {} \nproduct {}".format(
                self.winner_occurences, px, logpx, product))

        self.quant_err = 0
        self.num_err = 0
        self.winner_occurences = [0 for i in range(self.m*self.n)]
        self.dist_sum = 0

        return quant_err, winner_discrimination, entropy, dists

    def save_som_stats(self):
        quant_err, winner_discrimination, entropy, dists = self.get_som_stats()

        self.all_quant_err.append(quant_err)
        self.all_winner.append(winner_discrimination)
        self.all_entr.append(entropy)
        self.all_dists.append(dists.item())

        return quant_err, winner_discrimination, entropy, dists

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: torch.norm(vect - self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def bmu_loc(self, x):

        x_matrix = torch.stack([x.squeeze()
                               for i in range(self.m * self.n)]).to(self.args.device)
        dists = self.pdist(x_matrix, self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()
        return bmu_loc, bmu_loc[0] * self.m + bmu_loc[1]

    def forward(self, x, y, it):
        """ Take one input x and find location of its BMU in 2D net,
            then calculate distances of all neurons to this BMU and
            update their positions. """

        x = x.to(self.args.device)

        bmu_loc, bmu_loc_1D = self.bmu_loc(x)

        # calculate distance of bmu position in ND space and x
        # winner position in 3d space
        winner = self.weights[bmu_loc_1D]

        if it == -1:                                              # test, no som weight adjustment
            return winner

        self.quant_err += torch.sum(torch.pow(x - winner, 2))
        self.num_err += 1
        self.dist_sum += torch.norm(x.squeeze()-winner)
        # print(x, winner, torch.norm(x-winner), sep="\n")

        # calculate winner occurences for stats
        self.winner_occurences[bmu_loc_1D] += 1

        if bmu_loc_1D.item() not in self.d:
            self.d[bmu_loc_1D.item()] = {}
        if y.item() not in self.d[bmu_loc_1D.item()]:
            self.d[bmu_loc_1D.item()][y.item()] = 0
        self.d[bmu_loc_1D.item()][y.item()] += 1

        learning_rate_op = 1.0 - it / self.niter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        tmp = torch.stack([bmu_loc for i in range(self.m * self.n)])
        tmp = self.locations.float() - tmp.float()
        tmp = torch.pow(tmp, 2)
        bmu_distance_squares = torch.sum(tmp, 1)

        neighbourhood_func = torch.exp(
            torch.neg(torch.div(bmu_distance_squares, sigma_op ** 2)))

        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack(
            [learning_rate_op[i:i + 1].repeat(self.dim) for i in range(self.m * self.n)])
        delta = torch.mul(learning_rate_multiplier.to(self.args.device), (torch.stack(
            [x.squeeze() for i in range(self.m * self.n)]).to(self.args.device) - self.weights))

        # print("self.weights: ", self.weights.shape, "delta: ", delta.shape)
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights
        # print("self.weights2: ", self.weights.shape)

        return winner

    def predict(self, x):

        bmu_loc, bmu_loc_1D = self.bmu_loc(x)
        winner = self.weights[bmu_loc_1D]
        return winner, bmu_loc


def show_umatrix(n, m, d, offset=0, name="tmp"):

    classes = ('airplanes', 'cars', 'birds', 'cats',
               'deers', 'dogs', 'frogs', 'horses', 'ships', 'trucks')

    dct = {}
    for i in range(len(classes)):
        dct[i] = classes[i]

    neuron_classes = []
    percentage = []
    x = []
    y = []
    for r in range(n):
        for c in range(m):

            act_neuron = r * n + c
            if act_neuron not in d:
                continue

            unlabeled_num = 0
            if -1.0 in d[act_neuron]:
                unlabeled_num = d[act_neuron][-1.0]
                d[act_neuron][-1.0] = 0
                if max(d[act_neuron], key=d[act_neuron].get) == -1.0:
                    continue

            tmp = max(d[act_neuron], key=d[act_neuron].get) + offset
            neuron_classes.append(tmp)
            percentage.append(d[act_neuron][tmp-offset] /
                              sum(d[act_neuron].values()) * 100*4.5)
            d[-1.0] = unlabeled_num
            x.append(c)
            y.append(r)

    plt.figure()
    c = neuron_classes
    s = percentage
    plt.rc('axes', axisbelow=True)
    plt.xticks(np.arange(0, n, 1))
    plt.xlim(-1, n)
    plt.yticks(np.arange(0, m, 1))
    plt.ylim(-1, m)
    plt.grid(linestyle='dashed')
    scatter = plt.scatter(x, y, c=c, s=s, cmap='turbo')

    handles, labels = scatter.legend_elements()
    new_labels = [dct[class_val] for class_val in range(len(dct))]

    plt.legend(handles, new_labels, loc='upper left',
               bbox_to_anchor=(1, 1), title="Classes")

    # plt.legend(*scatter.legend_elements(),
    #           loc='upper left', bbox_to_anchor=(1, 1), title="Classes")
    # plt.imshow(neuron_classes, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    plt.title(f'Neuron activation classes {n}x{m}')
    # plt.show()
    plt.savefig(name, bbox_inches='tight')


def show_som_stats(all_quant, all_winner, all_entr, all_dist=[], name="tmp"):

    plt.figure(figsize=(12, 4))
    plt.subplot(141)  # 1 row, 3 columns, 1st subplot
    plt.plot(torch.tensor(all_quant).cpu())
    plt.title('quant_err')

    plt.subplot(142)  # 1 row, 3 columns, 2nd subplot
    plt.plot(all_winner)
    plt.title('winner_discrimination')

    plt.subplot(143)  # 1 row, 3 columns, 3rd subplot
    plt.plot(all_entr)
    plt.title('entropy')

    plt.subplot(144)
    plt.plot(all_dist)
    plt.title('sample - prototype distance')

    plt.tight_layout()

    # plt.show()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def train_som(args, EPS=100, szs=[10], in_dim=128):

    fv, lab = fv_from_checkpoint(
        args, "ckpts/cifar10/04-02-14:49/convlarge,Adam,200epochs,b256,lr0.2/checkpoint.10.ckpt")

    # training SOM

    for n in szs:

        som = SOM(n, n, in_dim, EPS, args).to(args.device)
        som.train()

        all_quant = []
        all_winner = []
        all_entr = []

        ds = []

        t = tm()
        for ep in range(EPS):
            som.d = {}
            print(f"Epoch: {ep+1}", end=" : ")
            with torch.no_grad():
                for i, (input, label) in enumerate(zip(fv, lab)):
                    if i % 5000 == 0:
                        print(f"{i} / {len(fv)}, time: {tm() - t}")
                        with open("som.log", "a") as a:
                            a.write(
                                f"{i} / {len(fv)}, n = {n}, time: {tm() - t}\n")
                    som(input, label, ep)
                cur_quant_err, cur_winner_discrimination, cur_entropy, dists = som.save_som_stats()

            if ep % 5 == 4:
                show_umatrix(n, n, som.d, 0, f"figs/fv-{n}n-{ep}ep.png")
                if 50 < ep < 101:
                    torch.save(som, f"pretrained_som-{ep}ep.pt")
                show_som_stats(som.all_quant_err, som.all_winner, som.all_entr,
                               som.all_dists, f"figs/fv-{n}n-{ep}ep-stat.png")
                # print(som.d)
                plt.close()
        with open("som.log", "a") as a:
            a.write(
                f"n={n}\n\t{som.all_quant_err}\n\t{som.all_winner}\n\t{som.all_entr}\n\t{som.all_dists}\n\t{som.d}\n")

        training = {"quant_err": [t.tolist() for t in som.all_quant_err], "wd": som.all_winner, "entropy": [
            t.tolist() for t in som.all_entr], "d": som.d}
        json_object = json.dumps(training, indent=4)
        if not os.path.isdir("fv_som_results"):
            os.makedirs("fv_som_results")
        with open(f"fv_som_results/{n}n.json", "w") as outfile:
            outfile.write(json_object)


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")

    train_som(args)
