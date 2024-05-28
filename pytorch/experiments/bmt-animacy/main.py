import os
import numpy as np
from parameters import get_parameters
import models
from Datasets import data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.datasets

from training import *

np.random.seed(5)
torch.manual_seed(5)

args = None

best_prec1 = 0


def main(args):

    global best_prec1

    # SET CONSTANTS FOR PORTION OF LABELED DATA
    total_labels = 50000    # do not change
    num_classes = 10        # do not change
    labeled_portion = 1000  # change to max 35 000
    per_class = labeled_portion / num_classes  # max 3 500

    train_loader, eval_loader = prep_data(per_class, args)

    student_model, teacher_ema_model, optimizer = prepare_models(args)

    # trenovanie binary mean teachera
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, student_model,
              teacher_ema_model, optimizer, epoch, args)

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            evaluate_models(eval_loader, student_model, teacher_ema_model)


def prepare_models(args, both=True):

    student_model = models.__dict__[args.model](
        args, data=None).to(args.device)
    teacher_ema_model = None

    if both:
        teacher_ema_model = models.__dict__[args.model](
            args, nograd=True, data=None).to(args.device)
    optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    return student_model, teacher_ema_model, optimizer


def evaluate_models(eval_loader, student_model, teacher_ema_model):
    global best_prec1

    prec1 = validate(eval_loader, student_model, args)
    print('==> Accuracy of the Student network on the 10000 test images: %d %%' % (
        prec1))

    ema_prec1 = validate(eval_loader, teacher_ema_model, args)
    print('==> Accuracy of the Teacher network on the 10000 test images: %d %%' % (
        ema_prec1))

    is_best = ema_prec1 > best_prec1
    best_prec1 = max(ema_prec1, best_prec1)


def create_loaders(dataset, batch_sampler, evaldir, eval_transform, args):
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

    return train_loader, eval_loader


def create_transformations():

    train_transform = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    return train_transform, eval_transform


def prep_data(per_class, args):

    train_transform, eval_transform = create_transformations()

    traindir = os.path.join(args.datadir, args.train_subdir)
    evaldir = os.path.join(args.datadir, args.eval_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transform)

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
                    if len(labels_tmp) == per_class:
                        break
                labels.update(labels_tmp)

    labeled_idxs, unlabeled_idxs, label_frequencies = data.relabel_dataset(
        dataset, labels)

    print(f'==> Labeled: {len(labeled_idxs)}, ' +
          f'ratio [A/INA]: {label_frequencies[dataset.class_to_idx["animate"]]}/' +
          f'{label_frequencies[dataset.class_to_idx["inanimate"]]}, ' +
          f'unlabeled: {len(unlabeled_idxs)}, total: {len(labeled_idxs) + len(unlabeled_idxs)}')

    if not args.labeled_batch_size:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    batch_sampler = data.TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)

    train_loader, eval_loader = create_loaders(
        dataset, batch_sampler, evaldir, eval_transform, args)

    return train_loader, eval_loader


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"==> Using device {args.device}")

    main(args)
