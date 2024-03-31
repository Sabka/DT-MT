import time
from math import *
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch
import numpy as np
from datasets import prepare_zoo_datasets
import json


# UTILS

train_dataloader, test_dataloader, som_dataloader, device = prepare_zoo_datasets(
    batch_size=30)


def show_umatrix(n, m, d, offset=0, fig_file="", class_names=[]):
    neuron_classes = []
    percentage = []
    x = []
    y = []
    for r in range(n):
        for c in range(m):

            act_neuron = r * n + c
            if act_neuron not in d:
                continue

            neuron_classes.append(
                max(d[act_neuron], key=d[act_neuron].get) + offset)
            percentage.append(d[act_neuron][neuron_classes[-1] -
                              offset] / sum(d[act_neuron].values()) * 100 * 2.5)
            x.append(c)
            y.append(r)

    d = {0: "Mammal", 1: "Bird", 2: "Reptile", 3: "Fish",
         4: "Amphibian", 5: "Bug", 6: "Invertebrate"}
    plt.figure()
    c = neuron_classes
    s = percentage
    plt.rc('axes', axisbelow=True)
    plt.xticks(np.arange(-4, n, 1))
    plt.xlim(-4, n)
    plt.yticks(np.arange(-1, m, 1))
    plt.ylim(-1, m)
    plt.grid(linestyle='dashed')
    scatter = plt.scatter(x, y, c=c, s=s, cmap='turbo')
    legend_handles = scatter.legend_elements()[0]
    custom_labels = ['Mammal', 'Bird', 'Reptile',
                     'Fish', 'Amphibian', 'Bug', 'Invertebrate']
    plt.legend(legend_handles, custom_labels,
               loc="lower left", title="Classes")
    # plt.imshow(neuron_classes, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    plt.title(f'Neuron activation classes {n}x{m}')
    plt.show()

    if fig_file != "":
        plt.savefig(fig_file)


def show_som_stats(all_quant, all_winner, all_entr, all_dist=[]):
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

    plt.show()


def show_3_som_stats(all_quant, all_winner, all_entr, fig_file=""):

    eps = len(torch.tensor(all_quant).cpu())+1
    plt.style.use('bmh')
    plt.figure(figsize=(12, 4))
    plt.subplot(131)  # 1 row, 3 columns, 1st subplot
    plt.plot([i for i in range(1, eps)],
             torch.tensor(all_quant).cpu())
    plt.title('Quantizaiton error')

    plt.subplot(132)  # 1 row, 3 columns, 2nd subplot
    plt.plot([i for i in range(1, eps)], 100 * np.array(all_winner))
    plt.title('Winner discrimination (%)')

    plt.subplot(133)  # 1 row, 3 columns, 3rd subplot
    plt.plot([i for i in range(1, len(torch.tensor(all_quant).cpu())+1)], all_entr)
    plt.title('Entropy')

    plt.tight_layout()

    plt.show()

    if fig_file != "":
        plt.savefig(fig_file)


def show_conf_matrix(confusion, class_labels):
    plt.figure(figsize=(4, 4))
    sns.set(font_scale=1.2)  # Adjust the font size for clarity

    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


class Losses4:
    def __init__(self, total_loss, sup_loss, som_loss_same, som_loss_dif) -> None:
        self.total_loss = total_loss
        self.sup_loss = sup_loss
        self.som_loss_same = som_loss_same
        self.som_loss_dif = som_loss_dif


def show_3losses(losses):
    plt.figure(figsize=(16, 4))

    plt.subplot(141)  # 1 row, 3 columns, 1st subplot
    plt.plot(losses.total_loss)
    plt.title('total loss')

    plt.subplot(142)  # 1 row, 3 columns, 2nd subplot
    plt.plot(losses.sup_loss)
    plt.title('supervised loss')

    plt.subplot(143)  # 1 row, 3 columns, 3rd subplot
    plt.plot(losses.som_loss_same)
    plt.title('som congruent dist')

    plt.subplot(144)  # 1 row, 3 columns, 3rd subplot
    plt.plot(losses.som_loss_dif)
    plt.title('som incongruent dist')

    plt.tight_layout()

    plt.show()


# SOM

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

        self.weights = torch.randn(m * n, dim).to(device)
        self.locations = torch.LongTensor(
            np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)
        self.quant_err = 0
        self.num_err = 0
        self.winner_occurences = [0 for i in range(self.m * self.n)]
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
        quant_err = self.quant_err / self.num_err
        winner_discrimination = sum(
            tmp > 0 for tmp in self.winner_occurences) / (self.m * self.n)

        px = torch.FloatTensor(
            [x / self.num_err for x in self.winner_occurences])
        logpx = torch.log2(px)
        product = px * logpx
        entropy = - torch.nansum(product)

        dists = self.dist_sum / self.num_err

        if isnan(entropy):
            print("self.winner_occurences {} \npx {} \nlogpx {} \nproduct {}".format(self.winner_occurences, px, logpx,
                                                                                     product))

        self.quant_err = 0
        self.num_err = 0
        self.winner_occurences = [0 for i in range(self.m * self.n)]
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
                               for i in range(self.m * self.n)]).to(device)
        dists = self.pdist(x_matrix, self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()

        return bmu_loc, bmu_loc[0] * self.m + bmu_loc[1]

    def forward(self, x, y, it):
        """ Take one input x and find location of its BMU in 2D net,
            then calculate distances of all neurons to this BMU and
            update their positions. """

        bmu_loc, bmu_loc_1D = self.bmu_loc(x)

        # calculate distance of bmu position in ND space and x
        winner = self.weights[bmu_loc_1D]  # winner position in 3d space

        if it == -1:  # test, no som weight adjustment
            return winner

        self.quant_err += torch.sum(torch.pow(x - winner, 2))
        self.num_err += 1
        self.dist_sum += torch.norm(x.squeeze() - winner)
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
        delta = torch.mul(learning_rate_multiplier.to(device),
                          (torch.stack([x.squeeze() for i in range(self.m * self.n)]).to(device) - self.weights))

        # print("self.weights: ", self.weights.shape, "delta: ", delta.shape)
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights
        # print("self.weights2: ", self.weights.shape)

        return winner

    def predict(self, x):

        bmu_loc, bmu_loc_1D = self.bmu_loc(x)
        # winner = self.weights[bmu_loc_1D]
        winner = bmu_loc
        return winner


def pretrain_som():

    repeat = True
    EPS = 50
    som_size = 8
    exp_run_time = time.time()
    sample_size = 16

    while repeat:
        som = SOM(som_size, som_size, sample_size, EPS, {}).to(device)
        som.train()

        all_quant = []
        all_winner = []
        all_entr = []

        ds = []

        for ep in range(EPS):
            som.d = {}
            print(f"Epoch: {ep + 1}", end=" : ")
            with torch.no_grad():
                for batch, sample1 in enumerate(som_dataloader):
                    Xs1, ys1 = sample1[:, :-1].type(torch.float32).to(device), sample1[:, -1:].type(torch.float32).to(
                        device)

                    som(Xs1, ys1, ep)

            cur_quant_err, cur_winner_discrimination, cur_entropy, dists = som.save_som_stats()

            print(
                f"SOM trained on new x_convs, quant_err: {cur_quant_err}, winner_discrimination: {cur_winner_discrimination}, entropy: {cur_entropy}, SP dist: {dists}",
                sep="\t")

            # if ep % 30 == 0:
            # show_umatrix(5, 5,som.d)
            # show_som_stats(som.all_quant_err, som.all_winner, som.all_entr, som.all_dists)

            ds.append(som.d)

        show_umatrix(som_size, som_size, som.d, 0, f"som{exp_run_time}.png")
        torch.save(som, f"pretrained_som{exp_run_time}.pt")

        if cur_winner_discrimination >= 0.5:
            print("Are you okay with SOM ? Y/N")
            if input().strip() == "Y":
                repeat = False

    # show_som_stats(som.all_quant_err, som.all_winner, som.all_entr, som.all_dists)

    som_stats = {"qe": [round(tensor.item(), 5) for tensor in som.all_quant_err],
                 "wd": [round(i, 5) for i in som.all_winner], "e": [round(tensor.item(), 5) for tensor in som.all_entr],
                 "dist": [round(i, 5) for i in som.all_dists]}
    json_object = json.dumps(som_stats, indent=4)
    with open(f"som-stats-{exp_run_time}.json", "w") as outfile:
        outfile.write(json_object)


def load_show_stats():
    f = open('som-stats-1710884692.939712.json')
    data = json.load(f)

    show_3_som_stats(data['qe'], data['wd'], data['e'], "som-stats.png")


# pretrain_som()
load_show_stats()
