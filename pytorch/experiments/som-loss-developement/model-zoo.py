import time
from math import *
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
from torch import nn
import torch
import json
import numpy as np
from ucimlrepo import fetch_ucirepo
import subprocess
import sys
import os
from datasets import prepare_zoo_datasets
from som import SOM


train_dataloader, test_dataloader, som_dataloader, device = prepare_zoo_datasets(
    batch_size=30)
batch_size = 30

# UTILS


def one_hot(labels, classes, offset):
    labels = labels - offset
    labels = labels.squeeze().to(torch.int)
    res = torch.eye(classes, device=device)[labels.long()]
    return res.to(device)


# MODEL

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # FIXED
        self.input_dim = 16

        self.input_layer = nn.Linear(self.input_dim, 10)
        self.hid_layer = nn.Linear(10, 7)
        with torch.no_grad():
            nn.init.normal_(self.input_layer.weight, mean=0, std=0.2)
            nn.init.normal_(self.hid_layer.weight, mean=0, std=0.2)

        self.layers = nn.Sequential(
            self.input_layer,
            nn.Sigmoid(),
            self.hid_layer,
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
        num_classes = 7

        sup_loss = F.mse_loss(pred_x, one_hot(
            targets_x, num_classes, 1), reduction='none').mean(dim=1)

        if want_som:
            dist_c = torch.sqrt(
                torch.sum(torch.pow(som_pred_x - som_pred_c, 2), dim=1))
            dist_i = torch.sqrt(
                torch.sum(torch.pow(som_pred_x - som_pred_i, 2), dim=1))
            som_loss = kappa * \
                (0.5 - 0.5 * (dist_i - dist_c) / (dist_i + dist_c))
            # NOT FIXING AT ALL, NANS SHOULDNT BE HERE !!!
            loss = (sup_loss + som_loss).nanmean()
            # print(f"sup:{sup_loss.nanmean()}, som:{som_loss.nanmean()}")
        else:
            loss = sup_loss.mean()

        # print("computed loss: ", loss)
        return loss, sup_loss.mean(), dist_c.mean() if want_som else 0, dist_i.mean() if want_som else 0


def train(dataloader, model, som_model, loss_fn, optimizer, kappa, ep, total_eps):
    losses = []
    sup_losses = []
    som_losses_same_cat, som_losses_dif_cat = [], []
    input_dim = 16 + 1

    size = len(dataloader.dataset)
    model.train()
    for batch, paired_sample in enumerate(dataloader):
        shape1 = min(batch_size, paired_sample[:, 0:1, :].shape[0])
        sample1, sample2, sample3 = paired_sample[:, 0:1, :].reshape(shape1, input_dim), paired_sample[:, 1:2, :].reshape(
            shape1, input_dim), paired_sample[:, 2:3, :].reshape(shape1, input_dim)
        Xs1, ys1 = sample1[:, :-1].type(torch.float32).to(
            device), sample1[:, -1:].type(torch.float32).to(device)
        Xs2, ys2 = sample2[:, :-1].type(torch.float32).to(
            device), sample2[:, -1:].type(torch.float32).to(device)
        Xs3, ys3 = sample3[:, :-1].type(torch.float32).to(
            device), sample3[:, -1:].type(torch.float32).to(device)

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
    num_classes = 7
    input_dim = 17
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    predicted_values, real_values = [], []

    with torch.no_grad():
        for sample in dataloader:
            if is_test_dataloader:
                Xs1, ys1 = sample[:, :-1].type(torch.float32).to(
                    device), sample[:, -1:].type(torch.float32).to(device)

            else:
                shape1 = min(batch_size, sample[:, 0:1, :].shape[0])
                sample1, sample2, sample3 = sample[:, 0:1, :].reshape(
                    shape1, input_dim), sample[:, 1:2, :].reshape(shape1, input_dim), sample[:, 2:3, :].reshape(shape1, input_dim)
                Xs1, ys1 = sample1[:, :-1].type(torch.float32).to(
                    device), sample1[:, -1:].type(torch.float32).to(device)
                Xs2, ys2 = sample2[:, :-1].type(torch.float32).to(
                    device), sample2[:, -1:].type(torch.float32).to(device)
                Xs3, ys3 = sample3[:, :-1].type(torch.float32).to(
                    device), sample3[:, -1:].type(torch.float32).to(device)

            pred1 = model(Xs1).to(device)
            predicted_values += list(pred1.argmax(1))
            real_values += list((ys1 - 1).squeeze())

            test_loss += loss_fn(pred1, one_hot(ys1, num_classes, 1)).item()
            correct += (pred1.argmax(1) == (ys1 - 1).squeeze()
                        ).type(torch.float).sum().item()

    predicted_values = torch.tensor(predicted_values)
    real_values = torch.tensor(real_values)
    confusion = confusion_matrix(real_values, predicted_values)

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return confusion, 100 * correct


mlp = NeuralNetwork().to(device)
loss_fn = SomSupLoss()
loss_fn2 = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)

som = torch.load("pretrained_som1710884692.939712.pt")
som.eval()

EPS = 200
MODS = 10
final_losses, accs, train_accs, confs = {}, {}, {}, {}
tm = time.time()
for kappa in [0.6, 0.7, 0.8, 0.9, 1, 1.5]:
    for mod_i in range(MODS):
        print(f"{mod_i + 1}. model starts in {tm - time.time()} sec")
        mlp = NeuralNetwork().to(device)
        loss_fn = SomSupLoss()
        loss_fn2 = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)
        model_losses, model_accs, model_confs, model_train_accs = {}, {}, {}, {}
        for ep in range(EPS):
            print(f"Epoch: {ep + 1}")
            losses = train(train_dataloader, mlp, som,
                           loss_fn, optimizer, kappa, ep, EPS)
            confusion, acc = test(test_dataloader, mlp, loss_fn2, True)
            confision2, train_acc = test(
                train_dataloader, mlp, loss_fn2, False)
            model_losses[ep] = round(float(losses[0].numpy()), 5)
            model_accs[ep] = round(acc, 5)
            model_train_accs[ep] = round(train_acc, 5)
            model_confs[ep] = confusion.tolist()

        final_losses[mod_i] = model_losses
        accs[mod_i] = model_accs
        train_accs[mod_i] = model_train_accs
        confs[mod_i] = model_confs

    training = {"train_loss": final_losses,
                "train_acc": model_train_accs, "test_acc": accs, "conf_mats": confs}
    json_object = json.dumps(training, indent=4)
    if not os.path.isdir("model_results"):
        os.makedirs("model_results")
    with open(f"model_results/{MODS}models{EPS}eps{kappa}k.json", "w") as outfile:
        outfile.write(json_object)
