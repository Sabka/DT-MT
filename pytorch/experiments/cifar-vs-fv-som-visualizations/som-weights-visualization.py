from torch import isnan
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import pickle
from time import time as tm
device = "cuda" if torch.cuda.is_available() else "cpu"

t = tm()
ep = 99
neurons = 10
print("WEIGHT LOADING")
params = torch.load(f'models/cifar10-{neurons}n-{ep}ep-new.pt')
print(f"WEIGHT LOADED in {tm() - t}s")

num_row = 10
num_col = 10
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))


for i, nw in enumerate(params):
	neuron_weights = torch.clone(nw)
	neuron_weights *= 1/2 # normalization from -1, 1 to -0.5, 0.5
	neuron_weights += 1/2 # normalization from -0.5, 0.5 to 0, 1
	neuron_weights = neuron_weights.reshape([3, 32, 32]).permute(1, 2, 0).detach().cpu().numpy()
	ax = axes[num_col - i // num_col - 1, i % num_col]
	ax.imshow(neuron_weights)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	
plt.tight_layout()
plt.savefig(f"from-som-weights-{neurons}n-{ep}ep.svg", format="svg")
plt.show()


