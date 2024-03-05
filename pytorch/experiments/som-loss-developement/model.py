import subprocess
import sys
import os

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
from som import SOM

device = "cuda" if torch.cuda.is_available() else "cpu"

### DOWNLOAD DATA

# fetch dataset
wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = wine.data.features.to_numpy()  # (178, 13)
y = wine.data.targets.to_numpy()  # 1, 2 or 3

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


paired_train_dataset, paired_test_dataset, labels1, labels2 = sklearn.model_selection.train_test_split(dataset,
                                                                                                       paired_labels,
                                                                                                       test_size=0.25,
                                                                                                       random_state=42,
                                                                                                       shuffle=True,
                                                                                                       stratify=paired_labels)

train_dataloader = DataLoader(torch.tensor(np.array(paired_train_dataset)).to(device), batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(torch.tensor(np.array(paired_test_dataset)).to(device), batch_size=batch_size,
                             shuffle=True)

som_dataloader = DataLoader(torch.tensor(np.concatenate((class1, class2, class3))).to(device), shuffle=True)


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


### UTILS
def one_hot(labels, classes, offset):
    labels = labels - offset
    labels = labels.squeeze().to(torch.int)
    res = torch.eye(classes, device=device)[labels.long()]
    return res.to(device)


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

        cur_kappa = kappa * (1 - (ep / total_eps))  # linear rampdown of kappa
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader):>5d}]")

    return torch.tensor(losses).mean(), torch.tensor(sup_losses).mean(), torch.tensor(
        som_losses_same_cat).float().mean(), torch.tensor(som_losses_dif_cat).float().mean()


def test(dataloader, model, loss_fn, is_test_dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    predicted_values, real_values = [], []

    with torch.no_grad():
        for sample in dataloader:
            if is_test_dataloader:
                Xs1, ys1 = sample[:, :-1].type(torch.float32).to(device), sample[:, -1:].type(torch.float32).to(device)
                
            else:
                shape1 = min(batch_size, sample[:, 0:1, :].shape[0])
                sample1, sample2, sample3 = sample[:, 0:1, :].reshape(shape1, 14), sample[:, 1:2, :].reshape(shape1, 14), sample[:, 2:3, :].reshape(shape1, 14)
                Xs1, ys1 = sample1[:, :-1].type(torch.float32).to(device), sample1[:, -1:].type(torch.float32).to(device)
                Xs2, ys2 = sample2[:, :-1].type(torch.float32).to(device), sample2[:, -1:].type(torch.float32).to(device)
                Xs3, ys3 = sample3[:, :-1].type(torch.float32).to(device), sample3[:, -1:].type(torch.float32).to(device)
		    
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

som = torch.load("pretrained_som1709630916.2606385.pt")
som.eval()

EPS = 1
MODS = 1
final_losses, accs, train_accs, confs = {}, {}, {}, {}
tm = time.time()
for kappa in [0, 0.7, 0.8]:
	for mod_i in range(MODS):
		print(f"{mod_i + 1}. model starts in {tm - time.time()} sec")
		mlp = NeuralNetwork().to(device)
		loss_fn = SomSupLoss()
		loss_fn2 = nn.MSELoss()
		optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)
		model_losses, model_accs, model_confs, model_train_accs = {}, {}, {}, {}
		for ep in range(EPS):
		    print(f"Epoch: {ep + 1}")
		    losses = train(train_dataloader, mlp, som, loss_fn, optimizer, kappa, ep, EPS)
		    confusion, acc = test(test_dataloader, mlp, loss_fn2, True)
		    confision2, train_acc = test(train_dataloader, mlp, loss_fn2, False)
		    model_losses[ep] = round(float(losses[0].numpy()), 5)
		    model_accs[ep] = round(acc, 5)
		    model_train_accs[ep] = round(train_acc, 5)
		    model_confs[ep] = confusion.tolist()

		final_losses[mod_i] = model_losses
		accs[mod_i] = model_accs
		train_accs[mod_i] = model_train_accs
		confs[mod_i] = model_confs

	training = {"train_loss": final_losses, "train_acc": model_train_accs, "test_acc": accs, "conf_mats": confs}
	json_object = json.dumps(training, indent=4)
	if not os.path.isdir("model_results"): 
		os.makedirs("model_results") 
	with open(f"model_results/{MODS}models{EPS}eps{kappa}k.json", "w") as outfile:
		outfile.write(json_object)
