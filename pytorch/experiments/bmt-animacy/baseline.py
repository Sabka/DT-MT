import os
import numpy as np
from parameters import get_parameters
import models
from Datasets import data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets

from Datasets.data import NO_LABEL
from misc.utils import *
import torch.nn as nn
import torch
from training import train_baseline, validate
from main import prep_data, prepare_models

np.random.seed(5)
torch.manual_seed(5)

args = None

best_prec1 = 0
global_step = 0


def main(args):
    global global_step
    global best_prec1

    # SET CONSTANTS FOR PORTION OF LABELED DATA
    total_labels = 50000    # do not change
    num_classes = 10        # do not change
    labeled_portion = 1000  # change to max 35 000
    per_class = labeled_portion / num_classes  # max 3 500

    train_loader, eval_loader = prep_data(per_class, args)
    student_model, _, optimizer = prepare_models(args, both=False)

    # trenovanie baseline
    for epoch in range(args.start_epoch, args.epochs):

        train_baseline(train_loader, student_model, optimizer, epoch, args)

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            prec1 = validate(eval_loader, student_model, args)
            print('==> Accuracy of the Student network on the 10000 test images: %d %%' % (
                prec1))


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"==> Using device {args.device}")

    main(args)
