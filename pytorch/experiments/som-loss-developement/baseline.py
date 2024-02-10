import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("ucimlrepo") # install wine dataset repo
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
X = wine.data.features.to_numpy() # (178, 13)
y = wine.data.targets.to_numpy() # 1, 2 or 3

print(X.shape, y.shape)
dataset_arr = np.hstack((X, y))
batch_size = 30

### PREPARE DATA

# divide into classes
class1 = dataset_arr[:59] # 59x14
class2 = dataset_arr[59:130]
class3 = dataset_arr[130:]


# TODO nevadi ze to je nahodne a nie pseudonahodne ??
np.random.shuffle(class1) # 59x14
np.random.shuffle(class2)
np.random.shuffle(class3)


size = 48
class1 = class1[:48] # 48x14
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


print(len(dataset)) # 13680 list of tuples of ndarrays of data and label

paired_train_dataset, paired_test_dataset, labels1, labels2 = sklearn.model_selection.train_test_split(dataset, paired_labels, test_size=0.25, random_state=42, shuffle=True, stratify=paired_labels)
print(len(paired_train_dataset), len(paired_test_dataset))


print("xx", np.array(paired_train_dataset).shape)
train_dataloader = DataLoader(torch.tensor(np.array(paired_train_dataset)).to(device), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(torch.tensor(np.array(paired_test_dataset)).to(device), batch_size=batch_size, shuffle=True)

som_dataloader = DataLoader(torch.tensor(np.concatenate((class1, class2, class3))).to(device) , shuffle=True)
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

def show_umatrix(n, m, d, offset = 0):

    neuron_classes = []
    percentage = []
    x = []
    y = []
    for r in range(n):
        for c in range(m):

            act_neuron = r * n + c
            if act_neuron not in d:
                continue

            neuron_classes.append(max(d[act_neuron], key=d[act_neuron].get)+offset)
            percentage.append(d[act_neuron][neuron_classes[-1]-offset] / sum(d[act_neuron].values()) * 100*2.5)
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
    
def show_som_stats(all_quant, all_winner, all_entr, all_dist = []):

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

### BASELINE


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()   # FIXED
        self.layers = nn.Sequential(
            nn.Linear(13, 100),
            nn.Sigmoid(),
            nn.Linear(100, 150),
            nn.Sigmoid(),
            nn.Linear(150, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
        
# only supervised model training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    sum_loss = 0
    for batch, sample in enumerate(dataloader):

        shape1 = min(batch_size, sample[:, 0:1, :].shape[0])
        orig_x, cong_x, incong_x = sample[:, 0:1, :].reshape(shape1, 14), sample[:, 1:2, :].reshape(shape1, 14), sample[:, 2:3, :].reshape(shape1, 14)

        Xs1, ys1 = orig_x[:, :-1].type(torch.float32).to(device), orig_x[:, -1:].type(torch.float32).to(device)

        # Compute prediction error
        pred1 = model(Xs1).to(device)

        # Loss
        loss = loss_fn(pred1, one_hot(ys1, torch.tensor(3).to(device), torch.tensor(1).to(device)))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloader):>5d}]")

    return [sum_loss / len(dataloader)]


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
            real_values += list((ys1-1).squeeze())

            test_loss += loss_fn(pred1, one_hot(ys1, 3, 1)).item()
            correct += (pred1.argmax(1) == (ys1-1).squeeze()).type(torch.float).sum().item()

    predicted_values = torch.tensor(predicted_values)
    real_values = torch.tensor(real_values)
    confusion = confusion_matrix(real_values, predicted_values)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return confusion, 100*correct

def baseline_experiment():
  mlp = NeuralNetwork().to(device)
  initial_weights = {name: param.clone() for name, param in mlp.named_parameters()}
  loss_fn = nn.MSELoss(reduction='mean')
  optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)

  EPS = 50
  all_losses = {}
  all_accs = {}
  all_confusions = {}
  for ep in range(EPS):
      print(f"Epoch: {ep+1}")
      losses = train(train_dataloader, mlp, loss_fn, optimizer)
      confusion, acc = test(test_dataloader, mlp, loss_fn)



      all_losses[ep] = round(losses[0], 5)
      all_accs[ep] = round(acc, 5)
      all_confusions[ep] = confusion.tolist()
      
      #if ep % 5 == 0:
      #  class_labels = ["1", "2", "3"]

      #  show_conf_matrix(confusion, class_labels)

  # class_labels = ["1", "2", "3"]
  # show_conf_matrix(confusion, class_labels)
  # plt.plot(all_losses)
  # print(losses)
  # plt.title("Train loss")
  # plt.show()
  return all_losses, all_accs, all_confusions

losses, accs, confs = {}, {}, {}
for i in range(20):
  l, a, c = baseline_experiment()
  losses[i] = l
  accs[i] = a
  confs[i] = c
  # accs.append(acc)
# print(losses)
# print(accs)
# print(confs)

training = {"train_loss": losses, "test_acc": accs, "conf_mats": confs}
json_object = json.dumps(training, indent=4)
with open("baseline_results/20models50eps.json", "w") as outfile:
	outfile.write(json_object)


#plt.plot(accs)
#print(accs)
#plt.title("Acuracies")
#plt.show()
