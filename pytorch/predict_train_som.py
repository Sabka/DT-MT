import torch
from torch import nn
import torch.nn.functional as F
import main
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from my_som import SOM
import random

def find_som_colors(n, m, epochs):
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
    with torch.no_grad():
        for i, (input, target) in enumerate(eval_loader):
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target.to(device))
                minibatch_size = len(target_var)
                output1, output2, x_conv = model(input_var)
                for x_c in x_conv:
                    loaded_data.append(x_c)



    som = SOM(n, m, 128, epochs, hyparams).to(device)


    # SOM training
    for epoch in range(epochs):
        som.train()
        random.shuffle(loaded_data)
        for x_c in loaded_data:
            with torch.no_grad():
                som(x_c, epoch)
                #print('train', torch.max(som.get_weights()))
    
        som.eval()
        quant_err = 0
        for x_c in loaded_data:
            _, bmu_loc_1D = som.bmu_loc(x_c)
            winner = som.weights[bmu_loc_1D]
            quant_err += torch.min(torch.abs(x_c - winner))
        quant_err /= 5000
        print(f"Epoch {epoch+1}, quant err {quant_err}")
   
   
    # PREDICTION 
    som.eval()
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
            d[bmu_loc_1D.item()][target_var[i].item()] += 1

    return d

def umatrix(n, m, d):

    neuron_classes = []
    percentage = []
    x = []
    y = []
    for r in range(n):
        for c in range(m):

            act_neuron = r * n + c
            if act_neuron not in d:
                continue

            neuron_classes.append(max(d[act_neuron], key=d[act_neuron].get))
            percentage.append(neuron_classes[-1] / sum(d[act_neuron].values()) * 100 * 30)
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
    plt.title('Neuron activation classes')
    plt.show()



# umatrix(8, 8, {13: {0: 26, 1: 1, 2: 20, 3: 8, 4: 6, 5: 5, 6: 5, 7: 1, 8: 21, 9: 2}, 29: {0: 20, 1: 16, 2: 2, 3: 1, 4: 1, 5: 1, 6: 1, 8: 32, 9: 2}, 30: {0: 35, 1: 5, 2: 2, 4: 1, 8: 32}, 6: {0: 45, 1: 4, 2: 4, 4: 6, 6: 1, 7: 3, 8: 2, 9: 1}, 23: {0: 4, 8: 55}, 36: {0: 16, 1: 3, 2: 13, 3: 14, 4: 2, 5: 19, 6: 2, 7: 7, 8: 21, 9: 6}, 15: {0: 49, 2: 3, 3: 1, 8: 35}, 7: {0: 58, 2: 14, 4: 2, 6: 1, 8: 5, 9: 1}, 54: {0: 3, 1: 11, 2: 2, 6: 1, 7: 1, 8: 1, 9: 68}, 38: {0: 23, 1: 46, 8: 3, 9: 15}, 20: {0: 12, 1: 1, 2: 16, 3: 3, 4: 13, 5: 3, 6: 7, 7: 5, 8: 8, 9: 2}, 21: {0: 43, 1: 3, 2: 5, 3: 1, 4: 6, 5: 1, 6: 3, 7: 5, 8: 19, 9: 5}, 61: {0: 4, 1: 17, 4: 1, 6: 5, 7: 1, 9: 20}, 4: {0: 3, 1: 1, 2: 9, 3: 19, 4: 4, 5: 20, 6: 24, 7: 3, 9: 1}, 25: {0: 3, 1: 5, 2: 6, 3: 5, 4: 9, 5: 2, 6: 21, 7: 4, 8: 4, 9: 8}, 43: {0: 11, 1: 1, 2: 11, 3: 10, 4: 8, 5: 12, 6: 2, 7: 27, 9: 2}, 0: {0: 1, 2: 30, 3: 2, 4: 43, 5: 4, 6: 8, 7: 6}, 12: {0: 11, 2: 28, 3: 21, 4: 10, 5: 11, 6: 3, 7: 5, 8: 5}, 45: {0: 14, 1: 13, 2: 1, 3: 1, 6: 2, 7: 2, 8: 21, 9: 48}, 5: {0: 13, 2: 18, 3: 6, 4: 3, 5: 3, 7: 5, 8: 3}, 19: {0: 2, 2: 14, 3: 13, 4: 17, 5: 10, 6: 5, 7: 7, 9: 1}, 14: {0: 22, 1: 1, 2: 1, 3: 2, 5: 2, 8: 62, 9: 11}, 22: {0: 7, 1: 1, 2: 1, 8: 79, 9: 4}, 56: {0: 1, 2: 4, 3: 13, 4: 3, 5: 61, 6: 1, 7: 8}, 53: {0: 4, 1: 12, 2: 2, 3: 9, 4: 3, 5: 3, 6: 5, 7: 3, 8: 1, 9: 19}, 35: {0: 1, 1: 2, 2: 3, 3: 22, 4: 8, 5: 28, 6: 4, 7: 7, 8: 2, 9: 2}, 60: {0: 8, 1: 3, 2: 17, 3: 1, 4: 26, 5: 2, 6: 3, 7: 16, 8: 2, 9: 3}, 37: {0: 15, 1: 12, 2: 2, 4: 1, 7: 2, 8: 2, 9: 35}, 27: {0: 6, 1: 2, 2: 13, 3: 34, 4: 8, 5: 11, 6: 6, 7: 14, 8: 3, 9: 6}, 17: {0: 2, 1: 1, 2: 12, 3: 8, 4: 13, 5: 3, 6: 69, 7: 2}, 31: {0: 5, 1: 9, 8: 35, 9: 4}, 34: {0: 1, 1: 2, 2: 4, 3: 14, 4: 4, 5: 25, 6: 4, 7: 7, 9: 6}, 44: {0: 18, 1: 7, 2: 5, 3: 6, 4: 10, 6: 2, 7: 5, 8: 3, 9: 22}, 39: {0: 5, 1: 42, 8: 4, 9: 8}, 11: {0: 2, 2: 17, 3: 22, 4: 10, 5: 21, 6: 4, 7: 5}, 46: {0: 8, 1: 36, 7: 1, 8: 7, 9: 31}, 26: {0: 2, 1: 5, 2: 20, 3: 11, 4: 8, 5: 2, 6: 19, 7: 5, 8: 11, 9: 4}, 42: {0: 4, 2: 5, 3: 14, 4: 7, 5: 3, 6: 10, 7: 17, 8: 2, 9: 8}, 28: {0: 1, 1: 40, 2: 5, 3: 4, 4: 2, 5: 4, 6: 2, 7: 1, 8: 15, 9: 23}, 2: {0: 1, 2: 20, 3: 10, 4: 26, 5: 19, 6: 1, 7: 21}, 10: {0: 2, 2: 28, 3: 2, 4: 29, 5: 2, 6: 14, 7: 1}, 51: {0: 3, 1: 4, 2: 1, 4: 6, 5: 2, 7: 25, 8: 1, 9: 4}, 40: {0: 1, 1: 1, 2: 10, 3: 33, 4: 6, 5: 22, 6: 8, 7: 17, 8: 2, 9: 1}, 47: {0: 1, 1: 81, 8: 4, 9: 12}, 55: {1: 13, 7: 1, 9: 48}, 62: {1: 25, 2: 1, 3: 2, 4: 1, 5: 4, 6: 3, 7: 2, 8: 1, 9: 7}, 63: {1: 23, 9: 18}, 52: {1: 5, 3: 4, 5: 2, 6: 1, 7: 38, 9: 26}, 33: {1: 3, 2: 8, 3: 10, 4: 12, 5: 12, 6: 22, 7: 2, 9: 2}, 16: {1: 2, 2: 5, 3: 1, 6: 51, 7: 1}, 50: {1: 2, 2: 7, 3: 4, 4: 12, 5: 8, 6: 1, 7: 55, 9: 1}, 3: {2: 8, 3: 25, 4: 14, 5: 31, 6: 13, 7: 5}, 1: {2: 15, 4: 27, 5: 1, 6: 2, 7: 26}, 9: {2: 18, 3: 5, 4: 42, 5: 5, 6: 19, 7: 7}, 8: {2: 18, 3: 4, 4: 18, 5: 1, 6: 45}, 48: {2: 11, 3: 32, 4: 3, 5: 53, 6: 2, 7: 2}, 32: {2: 5, 3: 12, 4: 16, 5: 3, 6: 40, 7: 6}, 24: {2: 3, 3: 4, 4: 6, 5: 1, 6: 52}, 18: {2: 15, 3: 26, 4: 30, 5: 8, 6: 18, 7: 3}, 41: {2: 4, 3: 22, 4: 6, 5: 24, 6: 8, 7: 13, 9: 1}, 59: {2: 1, 4: 1, 5: 1, 7: 49}, 57: {2: 1, 3: 11, 5: 32, 6: 1, 7: 11}, 49: {3: 15, 4: 2, 5: 39, 6: 1, 7: 5}, 58: {4: 3, 5: 6, 7: 43}})

for eps in [10, 30, 100]:
    for neurons in [5, 8, 10]:
        print(find_som_colors(neurons, neurons, eps))
