import os

import numpy as np
import torch

from Datasets.data import NO_LABEL
from misc.utils import *

from parameters import get_parameters
import models

from Datasets import data

import torchvision.transforms as transforms

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets

np.random.seed(5)
torch.manual_seed(5)

args = None

best_prec1 = 0
global_step = 0


def main(args):
    global global_step
    global best_prec1

    train_transform = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    traindir = os.path.join(args.datadir, args.train_subdir)
    evaldir = os.path.join(args.datadir, args.eval_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transform)

    # TODO refactor, just trying
    total_labels = 50000
    num_classes = 10
    labeled_portion = 1000  # max 35 000
    per_class = labeled_portion / num_classes  # max 3 500

    anim = {'bird', 'frog', 'cat', 'horse', 'dog', 'deer'}
    inanim = {'ship', 'truck', 'automobile', 'airplane'}

    labels = {}

    for group in [anim, inanim]:
        for cls in group:
            with open('data-local/labels/custom/' + cls + ".txt", "r") as f:
                labels_tmp = {}
                for line in f:
                    img, lab = line.strip().split(' ')
                    labels_tmp[img] = lab
                    if len(labels_tmp) == per_class: break
                labels.update(labels_tmp)

    labeled_idxs, unlabeled_idxs, label_frequencies = data.relabel_dataset(dataset, labels)

    print(
        f'==> Labeled: {len(labeled_idxs)}, ratio [A/INA]: {label_frequencies[dataset.class_to_idx["animate"]]}/{label_frequencies[dataset.class_to_idx["inanimate"]]}')

    #print(    f'==> Labeled: {len(labeled_idxs)}, ratio [A/INA]: {label_frequencies[dataset.class_to_idx["animate"]]}/{label_frequencies[dataset.class_to_idx["inanimate"]]}, unlabeled: {len(unlabeled_idxs)}, total: {len(labeled_idxs) + len(unlabeled_idxs)}, using labels {args.labels}')

    if args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    # vytvorenie iteratorov cez data
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    # Intializing the models
    # student
    student_model = models.__dict__[args.model](args, data=None).to(args.device)  # .cuda()
    # teacher
    #teacher_ema_model = models.__dict__[args.model](args, nograd=True, data=None).to(args.device)  # .cuda()

    optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # trenovanie mean teachera
    for epoch in range(args.start_epoch, args.epochs):

        # 1 trenovanie
        train(train_loader, student_model, optimizer, epoch)

        # evaluovanie po niekolkych trenovaniach
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            prec1 = validate(eval_loader, student_model)

            print('==> Accuracy of the Student network on the 10000 test images: %d %%' % (
                prec1))

def train(train_loader, student_model, optimizer, epoch):
    global global_step
    lossess = AverageMeter()
    running_loss = 0.0

    # Binary MT change
    class_supervised_criterion = nn.BCELoss(reduction='mean').to(args.device)  # .cuda()

    # trenovaci mod, nastavujeme kvoli spravaniu niektorych vrstiev
    student_model.train()

    # pocitanie lossy
    for i, ((input, ema_input), target) in enumerate(train_loader):  # iterujeme cez treningove batche


        if (input.size(0) != args.batch_size):
            continue

        # prekonvertovanie na tenzory
        input_var = torch.autograd.Variable(input).to(args.device)  # .cuda()
        target_var = torch.autograd.Variable(target).to(args.device)  # .cuda()) #async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        # trenovanie
        student_model_out, student_model_h = student_model(input_var)

        # cross entrophy loss - average supervised loss S
        # updated to BCELoss for mean teacher
        student_model_out = student_model_out.view(256).to(torch.float32)
        class_loss = class_supervised_criterion(student_model_out[194:],
                                                target_var.to(torch.float32)[194:]) / minibatch_size

        loss = class_loss
        # print(loss, class_loss, consistency_loss)

        # uprava vah studenta
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor s to zero.
        loss.backward()
        optimizer.step()
        global_step += 1

        running_loss += loss.item()

        pr_freq = 20
        if i % pr_freq == pr_freq - 1:  # print every <pr_freq> mini-batches
            print(
                f'Epoch: {epoch + 1}/{args.epochs}, Iteration: {i + 1}/{len(train_loader)}, Train loss: {round(running_loss / pr_freq, 5)}')  # , Acc: {None}, Time: {None}')
            running_loss = 0.0

        lossess.update(loss.item(), input.size(0))

    return lossess, running_loss


def validate(eval_loader, model):
    print("===> Validating")

    model.eval()
    total = 0
    correct = 0
    for i, (input, target) in enumerate(eval_loader):
        with torch.no_grad():
            input_var = input.to(args.device)  # .cuda()
            target_var = target.to(args.device)  # .cuda() # async=True)

            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0

            # compute output
            output1, output_h = model(input_var)

            output1 = (output1.view(target_var.size(0)).to(torch.float32) > torch.tensor([0.5]).to(
                args.device)).float() * 1

            # _, predicted = torch.max(output1.data, 1)
            total += target_var.size(0)
            correct += (output1 == target_var).sum().item()

    return 100 * correct / total


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"==> Using device {args.device}")

    main(args)
