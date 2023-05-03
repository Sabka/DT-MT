import torch
from torch import nn
import torch.nn.functional as F
import main
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from my_som import SOM


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
dataset_config = datasets.__dict__[ds_name]()
num_classes = dataset_config.pop('num_classes')
data_params = data_params()
train_loader, eval_loader = main.create_data_loaders(**dataset_config, args=data_params)

hyparams = model_params()
hyparams.device = device
model_factory = architectures.__dict__[hyparams.arch_name]
model_params = dict(pretrained=False, num_classes=10)
model = model_factory(**model_params)
model = nn.DataParallel(model).to(device)
optimizer = torch.optim.SGD(model.parameters(), hyparams.lr,
                            momentum=hyparams.momentum,
                            weight_decay=hyparams.weight_decay,
                            nesterov=hyparams.nesterov)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
# ema_model.load_state_dict(checkpoint['ema_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
class_criterion = nn.CrossEntropyLoss(size_average=False).to(device)
model.eval()

loaded_data = []
som = SOM(5, 5, 128, 10, hyparams).to(device)
som.train()


# SOM training
for epoch in range(3):
    for i, (input, target) in enumerate(eval_loader):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.to(device))
        minibatch_size = len(target_var)
        output1, output2, x_conv = model(input_var)
        for x_c in x_conv:
            with torch.no_grad():
                som(x_c, epoch)
            #print('train', torch.max(som.get_weights()))
    
    quant_err = 0
    for x_c in loaded_data:
        _, bmu_loc_1D = som.bmu_loc(x_c)
        winner = som.weights[bmu_loc_1D]
        quant_err += torch.min(torch.abs(x_c - winner))
    quant_err /= 1
    print(f"Epoch {epoch+1}, quant err {quant_err}")
   
   
# PREDICTION 
d = {}
for i, (input, target) in enumerate(eval_loader):
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target.to(device))
    minibatch_size = len(target_var)
    output1, output2, x_conv = model(input_var)
    for i,  x_c in enumerate(x_conv):
        _, bmu_loc_1D = som.bmu_loc(x_c)
        winner = som.weights[bmu_loc_1D]
        

        if bmu_loc_1D.item() not in d:
            d[bmu_loc_1D.item()] = {}
        if target_var[i].item() not in d[bmu_loc_1D.item()]:
            d[bmu_loc_1D.item()][target_var[i].item()] = 0
        #print(d)
        #print(d[bmu_loc_1D])
        #print(target_var)
        #print(target_var[i])
        #print(d[bmu_loc_1D][target_var[i]])
        d[bmu_loc_1D.item()][target_var[i].item()] += 1

print(d)
