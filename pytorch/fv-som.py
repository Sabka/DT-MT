from torch import isnan
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import main
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from time import time as tm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

classes = ('airplanes', 'cars', 'birds', 'cats',
           'deers', 'dogs', 'frogs', 'horses', 'ships', 'trucks')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

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
        winner_discrimination = sum(tmp > 0 for tmp in self.winner_occurences) / (self.m * self.n)

        px = torch.FloatTensor([x/self.num_err for x in self.winner_occurences])
        logpx = torch.log2(px)
        product = px * logpx
        entropy = - torch.nansum(product)


        dists = self.dist_sum / self.num_err

        if isnan(entropy):
          print("self.winner_occurences {} \npx {} \nlogpx {} \nproduct {}".format(self.winner_occurences, px, logpx, product))

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

        x_matrix = torch.stack([x.squeeze() for i  in range(self.m * self.n)]).to(device)
        dists = self.pdist(x_matrix, self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index, :]
        bmu_loc = bmu_loc.squeeze()

        return bmu_loc, bmu_loc[0] * self.m + bmu_loc[1]

    def forward(self, x, y, it):

        """ Take one input x and find location of its BMU in 2D net,
            then calculate distances of all neurons to this BMU and
            update their positions. """
            
        x = x.to(device)

        bmu_loc, bmu_loc_1D = self.bmu_loc(x)

        # calculate distance of bmu position in ND space and x
        winner = self.weights[bmu_loc_1D]                    # winner position in 3d space

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

        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op ** 2)))

        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack(
            [learning_rate_op[i:i + 1].repeat(self.dim) for i in range(self.m * self.n)])
        delta = torch.mul(learning_rate_multiplier.to(device), (torch.stack([x.squeeze() for i in range(self.m * self.n)]).to(device) - self.weights))

        # print("self.weights: ", self.weights.shape, "delta: ", delta.shape)
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights
        # print("self.weights2: ", self.weights.shape)

        return winner


    def predict(self, x):


        bmu_loc, bmu_loc_1D = self.bmu_loc(x)
        winner = self.weights[bmu_loc_1D]
        return winner

def show_umatrix(n, m, d, offset = 0, name="tmp"):

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

            tmp = max(d[act_neuron], key=d[act_neuron].get) + offset
            neuron_classes.append(tmp)
            percentage.append(d[act_neuron][tmp-offset] / sum(d[act_neuron].values()) * 100*4.5)
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

    plt.legend(handles, new_labels, loc='upper left', bbox_to_anchor=(1, 1), title="Classes")

    #plt.legend(*scatter.legend_elements(),
    #           loc='upper left', bbox_to_anchor=(1, 1), title="Classes")
    # plt.imshow(neuron_classes, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    plt.title(f'Neuron activation classes {n}x{m}')
    # plt.show()
    plt.savefig(name, bbox_inches='tight')

def show_som_stats(all_quant, all_winner, all_entr, all_dist = [], name="tmp"):

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
  
# loading data
class model_params:
  def __init__(self):
    self.arch_name = 'cifar_sarmad'
    self.lr = 0.1
    self.momentum = 0.9
    self.nesterov = False
    self.weight_decay = 1e-4


class data_params:
  def __init__(self):
    self.train_subdir = "train"
    self.eval_subdir = "val"
    self.exclude_unlabeled = True
    self.labeled_batch_size = 62
    self.labels = "data-local/labels/cifar10/1000_balanced_labels/00.txt"
    self.batch_size = 512
    self.workers = 2


ds_name = 'cifar10'
checkpoint_path = "results/best.ckpt"
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"
print("Using device", device)

# dataset_config = datasets.__dict__[ds_name]()
# num_classes = dataset_config.pop('num_classes')
# data_params = data_params()
# train_loader, eval_loader = main.create_data_loaders(**dataset_config, args=data_params)

hyparams = model_params()
hyparams.device = device
model_factory = architectures.__dict__[hyparams.arch_name]
model_params = dict(pretrained=False, num_classes=10)
model = model_factory(**model_params)
model = nn.DataParallel(model).to(device)
# optimizer = torch.optim.SGD(model.parameters(), hyparams.lr,
#                     momentum=hyparams.momentum,
#                     weight_decay=hyparams.weight_decay,
#                    nesterov=hyparams.nesterov)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
# ema_model.load_state_dict(checkpoint['ema_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# class_criterion = nn.CrossEntropyLoss(size_average=False).to(device)
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('airplanes', 'cars', 'birds', 'cats',
           'deers', 'dogs', 'frogs', 'horses', 'ships', 'trucks')

t = tm()
loaded_data = []
with torch.no_grad():
  for i, (input, target) in enumerate(trainloader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.to(device))
        minibatch_size = len(target_var)
        output1, output2, x_conv = model(input_var)
        for i, x_c in enumerate(x_conv):
            loaded_data.append((x_c, target_var[i]))
  
  for i, (input, target) in enumerate(testloader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.to(device))
        minibatch_size = len(target_var)
        output1, output2, x_conv = model(input_var)
        for i, x_c in enumerate(x_conv):
            loaded_data.append((x_c, target_var[i]))
   
            
print(f"FV created in {tm() - t} seconds")

# training SOM

EPS = 100

som = SOM(8, 8, 128, EPS, {}).to(device)
som.train()

all_quant = []
all_winner = []
all_entr = []

ds = []

t = tm()
for ep in range(EPS):
    som.d = {}
    print(f"Epoch: {ep+1}", end = " : ")
    with torch.no_grad():
      for i, x_c in enumerate(loaded_data, 0):
        if i % 1000 == 0:
          print(f"{i} / {len(loaded_data)}, time: {tm() - t}")
        
        input, label = x_c
        input = input.view(-1)
        som(input, label, ep)


    cur_quant_err, cur_winner_discrimination, cur_entropy, dists = som.save_som_stats()


    print(f"SOM trained on feature vectors, quant_err: {cur_quant_err}, winner_discrimination: {cur_winner_discrimination}, entropy: {cur_entropy}, SP dist: {dists}", sep = "\t")

    if ep % 1 == 0:
      show_umatrix(8, 8, som.d, 0, f"figs/fv-{ep}ep.png")
      show_som_stats(som.all_quant_err, som.all_winner, som.all_entr, som.all_dists, f"figs/fv-{ep}ep-stat.png")
      print(som.d)

    # ds.append(som.d)


# show_som_stats(som.all_quant_err, som.all_winner, som.all_entr, som.all_dists)
