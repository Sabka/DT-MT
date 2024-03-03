import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("ucimlrepo")  # install wine dataset repo
install("seaborn")
from ucimlrepo import fetch_ucirepo

import numpy as np
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F

import sklearn.model_selection
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
from math import *

import time

device = "cuda" if torch.cuda.is_available() else "cpu"

### DOWNLOAD DATA

# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features.to_numpy()  # (178, 13)
y = wine.data.targets.to_numpy()  # 1, 2 or 3

print(X.shape, y.shape)
dataset_arr = np.hstack((X, y))
batch_size = 30

### PREPARE DATA

# divide into classes
class1 = dataset_arr[:59]  # 59x14
class2 = dataset_arr[59:130]
class3 = dataset_arr[130:]

# TODO nevadi ze to je nahodne a nie pseudonahodne ??
np.random.shuffle(class1)  # 59x14
np.random.shuffle(class2)
np.random.shuffle(class3)

size = 48
class1 = class1[:48]  # 48x14
class2 = class2[:48]
class3 = class3[:48]

print(class1.shape, class1[0])

dataset = []
paired_labels = []

for i in range(size):
    for j in range(size):
        if i != j:
            dataset.append((class1[i], class1[j]))
            paired_labels.append(1)
            dataset.append((class2[i], class2[j]))
            paired_labels.append(2)
            dataset.append((class3[i], class3[j]))
            paired_labels.append(3)

for i in range(size):
    for j in range(size):
        dataset.append((class1[i], class2[j]))
        paired_labels.append(4)
        dataset.append((class1[i], class3[j]))
        paired_labels.append(5)
        dataset.append((class2[i], class3[j]))
        paired_labels.append(6)

print(len(dataset))  # 13680 list of tuples of ndarrays of data and label

paired_train_dataset, paired_test_dataset, labels1, labels2 = sklearn.model_selection.train_test_split(dataset,
                                                                                                       paired_labels,
                                                                                                       test_size=0.25,
                                                                                                       random_state=42,
                                                                                                       shuffle=True,
                                                                                                       stratify=paired_labels)
print(len(paired_train_dataset), len(paired_test_dataset))

print("xx", np.array(paired_train_dataset).shape)
train_dataloader = DataLoader(torch.tensor(np.array(paired_train_dataset)).to(device), batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(torch.tensor(np.array(paired_test_dataset)).to(device), batch_size=batch_size,
                             shuffle=True)

som_dataloader = DataLoader(torch.tensor(np.concatenate((class1, class2, class3))).to(device), shuffle=True)
print(len(som_dataloader))


def generate_triplet(x, class1, class2):
    cong = class1[np.random.randint(0, class1.shape[0])]
    incong1 = class2[np.random.randint(0, class2.shape[0])]
    return np.array((x, cong, incong1))


# divide into classes
class1 = dataset_arr[:59]
class2 = dataset_arr[59:130]
class3 = dataset_arr[130:]

# nahodne preusporiadanie so seedom
np.random.seed(4742)
np.random.shuffle(class1)
np.random.seed(4742)
np.random.shuffle(class2)
np.random.seed(4742)
np.random.shuffle(class3)

# test dataset
class_test_size = 15
test_dataset = np.vstack((class1[:class_test_size], class2[:class_test_size], class3[:class_test_size]))
print(test_dataset.shape)
test_dataloader = DataLoader(torch.tensor(test_dataset).to(device), batch_size=batch_size, shuffle=True)

# train dataset
use_of_x_times = 25
dataset_size = 6650
dataset_triplets = np.empty((dataset_size, 3, 14))
class1, class2, class3 = class1[class_test_size:], class2[class_test_size:], class3[class_test_size:]
position = 0

for i in range(use_of_x_times):

    for x in class1:
        dataset_triplets[position] = generate_triplet(x, class1, class2)
        position += 1
        dataset_triplets[position] = generate_triplet(x, class1, class3)
        position += 1

    for x in class2:
        dataset_triplets[position] = generate_triplet(x, class2, class1)
        position += 1
        dataset_triplets[position] = generate_triplet(x, class2, class3)
        position += 1

    for x in class3:
        dataset_triplets[position] = generate_triplet(x, class3, class1)
        position += 1
        dataset_triplets[position] = generate_triplet(x, class3, class2)
        position += 1

print(dataset_triplets.shape)
train_dataloader = DataLoader(torch.tensor(dataset_triplets).to(device), batch_size=batch_size, shuffle=True)


### UTILS + VISUALIZATION

def one_hot(labels, classes, offset):
    labels = labels - offset
    labels = labels.squeeze().to(torch.int)
    res = torch.eye(classes, device=device)[labels.long()]
    return res.to(device)


# print(one_hot(to)rch.tensor([1, 2, 3, 1, 3]), 3, 1))

def show_umatrix(n, m, d, offset=0, fig_file=""):
    neuron_classes = []
    percentage = []
    x = []
    y = []
    for r in range(n):
        for c in range(m):

            act_neuron = r * n + c
            if act_neuron not in d:
                continue

            neuron_classes.append(max(d[act_neuron], key=d[act_neuron].get) + offset)
            percentage.append(d[act_neuron][neuron_classes[-1] - offset] / sum(d[act_neuron].values()) * 100 * 2.5)
            x.append(c)
            y.append(r)

    plt.figure()
    c = neuron_classes
    s = percentage
    plt.rc('axes', axisbelow=True)
    plt.xticks(np.arange(-2, n, 1))
    plt.xlim(-2, n)
    plt.yticks(np.arange(-1, m, 1))
    plt.ylim(-1, m)
    plt.grid(linestyle='dashed')
    scatter = plt.scatter(x, y, c=c, s=s, cmap='turbo')
    plt.legend(*scatter.legend_elements(),
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


### SOM

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
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
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
        winner_discrimination = sum(tmp > 0 for tmp in self.winner_occurences) / (self.m * self.n)

        px = torch.FloatTensor([x / self.num_err for x in self.winner_occurences])
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

        x_matrix = torch.stack([x.squeeze() for i in range(self.m * self.n)]).to(device)
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

        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op ** 2)))

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


repeat = True
EPS = 20
som_size = 6

while repeat:
    som = SOM(som_size, som_size, 13, EPS, {}).to(device)
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

    show_umatrix(som_size, som_size, som.d, 0, "som.png")

    print("Are you okay with SOM ? Y/N")
    if input().strip() == "Y":
        repeat = False

# show_som_stats(som.all_quant_err, som.all_winner, som.all_entr, som.all_dists)

som_stats = {"qe": [round(tensor.item(), 5) for tensor in som.all_quant_err],
             "wd": [round(i, 5) for i in som.all_winner], "e": [round(tensor.item(), 5) for tensor in som.all_entr],
             "dist": [round(i, 5) for i in som.all_dists]}


### MODEL

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # FIXED
        self.layers = nn.Sequential(
            nn.Linear(13, 15),
            nn.Sigmoid(),
            nn.Linear(15, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


class SomSupLoss(nn.Module):
    def __init__(self):
        super(SomSupLoss, self).__init__()

    def forward(self, pred_x, som_pred_x, pred_c, som_pred_c, pred_i, som_pred_i, targets_x, targets_c, targets_i,
                kappa, want_som):
        loss_fn = nn.MSELoss(reduction='mean')

        sup_loss = F.mse_loss(pred_x, one_hot(targets_x, 3, 1), reduction='none').mean(dim=1)

        if want_som:
            dist_c = torch.sqrt(torch.sum(torch.pow(som_pred_x - som_pred_c, 2), dim=1))
            dist_i = torch.sqrt(torch.sum(torch.pow(som_pred_x - som_pred_i, 2), dim=1))
            som_loss = kappa * (0.5 - 0.5 * (dist_i - dist_c) / (dist_i + dist_c))
            loss = (sup_loss + som_loss).nanmean()  ### NOT FIXING AT ALL, NANS SHOULDNT BE HERE !!!
            # print(f"sup:{sup_loss.nanmean()}, som:{som_loss.nanmean()}")
        else:
            loss = sup_loss.mean()

        # print("computed loss: ", loss)
        return loss, sup_loss.mean(), dist_c.mean() if want_som else 0, dist_i.mean() if want_som else 0


def train(dataloader, model, som_model, loss_fn, optimizer, kappa, ep, total_eps):
    losses = []
    sup_losses = []
    som_losses_same_cat, som_losses_dif_cat = [], []

    size = len(dataloader.dataset)
    model.train()
    for batch, paired_sample in enumerate(dataloader):
        shape1 = min(batch_size, paired_sample[:, 0:1, :].shape[0])
        sample1, sample2, sample3 = paired_sample[:, 0:1, :].reshape(shape1, 14), paired_sample[:, 1:2, :].reshape(
            shape1, 14), paired_sample[:, 2:3, :].reshape(shape1, 14)
        Xs1, ys1 = sample1[:, :-1].type(torch.float32).to(device), sample1[:, -1:].type(torch.float32).to(device)
        Xs2, ys2 = sample2[:, :-1].type(torch.float32).to(device), sample2[:, -1:].type(torch.float32).to(device)
        Xs3, ys3 = sample3[:, :-1].type(torch.float32).to(device), sample3[:, -1:].type(torch.float32).to(device)

        optimizer.zero_grad()

        # Compute prediction error + train som on Xs1, Xs2
        pred1 = model(Xs1)
        pred2 = model(Xs2)
        pred3 = model(Xs3)

        som_pred1 = torch.empty(shape1, 2).to(device)
        som_pred2 = torch.empty(shape1, 2).to(device)
        som_pred3 = torch.empty(shape1, 2).to(device)

        for i in range(len(Xs1)):
            som_pred1[i] = som_model.predict(Xs1[i])
            som_pred2[i] = som_model.predict(Xs2[i])
            som_pred3[i] = som_model.predict(Xs3[i])

        cur_kappa = kappa * (1 - (ep / total_eps))  # linear rampup of kappa
        # print("kappa", cur_kappa)
        loss, sup_loss, som_loss_same_cat, som_loss_dif_cat = loss_fn(pred1, som_pred1, pred2, som_pred2, pred3,
                                                                      som_pred3, ys1, ys2, ys3, cur_kappa, True)

        # Backpropagation
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        sup_losses.append(sup_loss.item())
        som_losses_same_cat.append(som_loss_same_cat)
        som_losses_dif_cat.append(som_loss_dif_cat)

        # if True in torch.isnan(loss):
        #	print(loss.item())
        #	print(sup_loss.item())
        #	print(som_loss_same_cat)
        #	print(som_loss_dif_cat)
        #	input()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader):>5d}]")

    return torch.tensor(losses).mean(), torch.tensor(sup_losses).mean(), torch.tensor(
        som_losses_same_cat).float().mean(), torch.tensor(som_losses_dif_cat).float().mean()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    predicted_values, real_values = [], []

    with torch.no_grad():
        for sample in dataloader:
            Xs1, ys1 = sample[:, :-1].type(torch.float32).to(device), sample[:, -1:].type(torch.float32).to(device)
            pred1 = model(Xs1).to(device)

            predicted_values += list(pred1.argmax(1))
            real_values += list((ys1 - 1).squeeze())

            test_loss += loss_fn(pred1, one_hot(ys1, 3, 1)).item()
            correct += (pred1.argmax(1) == (ys1 - 1).squeeze()).type(torch.float).sum().item()

    predicted_values = torch.tensor(predicted_values)
    real_values = torch.tensor(real_values)
    confusion = confusion_matrix(real_values, predicted_values)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return confusion, 100 * correct


mlp = NeuralNetwork().to(device)
loss_fn = SomSupLoss()
loss_fn2 = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)

EPS = 50
MODS = 10
final_losses, accs, confs = {}, {}, {}
tm = time.time()
for kappa in [0.5, 0.2, 0.8, 0.1, 0.01, 0.001]:
	for mod_i in range(10):
		print(f"{mod_i + 1}. model starts in {tm - time.time()} sec")
		mlp = NeuralNetwork().to(device)
		loss_fn = SomSupLoss()
		loss_fn2 = nn.MSELoss()
		optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)
		model_losses, model_accs, model_confs = {}, {}, {}
		for ep in range(EPS):
		    print(f"Epoch: {ep + 1}")
		    losses = train(train_dataloader, mlp, som, loss_fn, optimizer, kappa, ep, EPS)
		    confusion, acc = test(test_dataloader, mlp, loss_fn2)
		    model_losses[ep] = round(float(losses[0].numpy()), 5)
		    model_accs[ep] = round(acc, 5)
		    model_confs[ep] = confusion.tolist()

		final_losses[mod_i] = model_losses
		accs[mod_i] = model_accs
		confs[mod_i] = model_confs

	training = {"som": som_stats, "train_loss": final_losses, "test_acc": accs, "conf_mats": confs}
	json_object = json.dumps(training, indent=4)
	with open(f"model_results/{MODS}models{EPS}eps{kappa}k.json", "w") as outfile:
		outfile.write(json_object)
