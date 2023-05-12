import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import main
from mean_teacher import architectures, datasets, data, losses, ramps, cli
from sklearn.manifold import TSNE

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
    loaded_data_classes = []
    with torch.no_grad():
        for i, (input, target) in enumerate(eval_loader):
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target.to(device))
                minibatch_size = len(target_var)
                output1, output2, x_conv = model(input_var)
                for x_c in x_conv:
                    loaded_data.append(x_c.numpy())
                loaded_data_classes.append(target.numpy())

    plt.figure()
    tsne_results = TSNE(n_components=2).fit_transform(loaded_data)
    loaded_data['tsne-2d-one'] = tsne_results[:, 0]
    loaded_data['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    plt.scatter(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        cmap='turbo',
        data=loaded_data,
        legend="full",
        alpha=0.3
    )