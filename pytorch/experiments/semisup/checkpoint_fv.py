import os
import shutil

import numpy as np
import torch

from Datasets.data import NO_LABEL
from misc.utils import *
from tensorboardX import SummaryWriter
import datetime
from parameters import get_parameters
import models

from misc import ramps
from Datasets import data
from models import losses

import torchvision.transforms as transforms


import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets

np.random.seed(5)
torch.manual_seed(5)

args = None


def fv_from_checkpoint(args, checkpoint_path):

    checkpoint = torch.load(checkpoint_path)

    model = models.__dict__[args.model](args, data=None).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    train_transform = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))]))

    traindir = os.path.join(args.datadir, args.train_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transform)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)                 # deterministic

    # train(train_loader, model, ema_model, optimizer, epoch)
    feature_vectors, labels = get_fv_from_model(args, train_loader, model)
    return feature_vectors, labels


def get_fv_from_model(args, train_loader, model):

    shape_h = 128

    model.eval()
    feature_vectors = None
    labels = None
    free_position = 0

    for i, ((input, ema_input), target) in enumerate(train_loader):
        print(f"Processed batch {i+1}/{len(train_loader)}")

        if (input.size(0) != args.batch_size):
            continue

        input_var = torch.autograd.Variable(input).cuda()

        target_var = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        args.sntg = True
        with torch.no_grad():
            model_out, model_h = model(input_var)
            model_h = model_h.reshape(len(target_var), shape_h)

        if feature_vectors is None and labels is None:
            feature_vectors = torch.zeros(
                (len(train_loader)*len(target_var), shape_h))
            labels = torch.zeros(len(train_loader)*len(target_var))

        for h, tar in list(zip(model_h, target)):
            feature_vectors[free_position] = h
            labels[free_position] = tar
            free_position += 1
    return feature_vectors, labels


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")

    # fv, lab = fv_from_checkpoint(
    #    args, "ckpts/cifar10/04-02-14:49/convlarge,Adam,200epochs,b256,lr0.2/checkpoint.10.ckpt")
    # print(fv.shape, lab.shape)
