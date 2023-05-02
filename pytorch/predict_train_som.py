import torch
from torch import nn
import torch.nn.functional as F
import main
from mean_teacher import architectures, datasets, data, losses, ramps, cli


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
checkpoint_path = "results/main/2023-05-02_17:54:18/0/transient/best.ckpt"
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
for i, (input, target) in enumerate(eval_loader):
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target.to(device))
    minibatch_size = len(target_var)
    output1, output2, x_conv = model(input_var)
    # loaded_data.append(x_conv)
    # not working too much memory for the tensors
    softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
    class_loss = class_criterion(output1, target_var) / minibatch_size
    # measure accuracy and record loss
    prec1, prec5 = main.accuracy(output1.data, target_var.data, topk=(1, 5))
    print("Class {} Acc topk 1 {:.4f} 5 {:.4f}".format(target[0],prec1[0],prec5[0]))
    print(x_conv.shape)
    print(x_conv[0].shape)
# print(loaded_data)
# som = SOM(5, 5, 128, 10, hyparams).to(device)