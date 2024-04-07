import os
import json
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

from som_from_fv import SOM
from time import *

np.random.seed(5)
torch.manual_seed(5)

args = None


best_prec1 = 0
global_step = 0


class SomLoss(nn.Module):
    def __init__(self):
        super(SomLoss, self).__init__()

    def forward(self, som_pred1, som_pred2, kappa):

        dist = torch.sqrt(
            torch.sum(torch.pow(som_pred1 - som_pred2, 2), dim=1))
        som_loss = (kappa * dist).nanmean()
        return som_loss


def load_models_from_pts(args):
    som = torch.load("fv_som_results/pretrained_som-79ep.pt")
    som.eval()

    checkpoint_path = "ckpts/cifar10/04-02-14:49/convlarge,Adam,200epochs,b256,lr0.2/checkpoint.10.ckpt"
    checkpoint = torch.load(checkpoint_path)
    model = models.__dict__[args.model](args, data=None).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.train()

    return som, model


def main(args):
    global global_step
    global best_prec1

    som, model = load_models_from_pts(args)
    kappa = 100

    train_transform = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470,  0.2435,  0.2616))]))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470,  0.2435,  0.2616))
    ])

    traindir = os.path.join(args.datadir, args.train_subdir)
    evaldir = os.path.join(args.datadir, args.eval_subdir)

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

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)                                                        # deterministic

#   Intializing the model
    ema_model = models.__dict__[args.model](
        args, nograd=True, data=None).to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if args.evaluate:
        print('Evaluating the primary model')
        acc1 = validate(eval_loader, model)
        print('Accuracy of the Student network on the 10000 test images: %d %%' % (
            acc1))
        print('Evaluating the Teacher model')
        acc2 = validate(eval_loader, ema_model)
        print('Accuracy of the Teacher network on the 10000 test images: %d %%' % (
            acc2))
        return

    if args.saveX == True:
        save_path = '{},{},{}epochs,b{},lr{}'.format(
            args.model,
            args.optim,
            args.epochs,
            args.batch_size,
            args.lr)
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(time_stamp, save_path)
        save_path = os.path.join(args.dataName, save_path)
        save_path = os.path.join(args.save_path, save_path)
        print('==> Will save Everything to {}', save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    if args.saveX == True:
        test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    prec1 = validate(eval_loader, model)

    ema_prec1 = validate(eval_loader, ema_model)

    print('Initial accuracy of the Student network on the 10000 test images: %d %%' % (
        prec1))
    print('Initial accuracy of the Teacher network on the 10000 test images: %d %%' % (
        ema_prec1))

    train_losses = {"total": [], "sup": [], "som": []}
    eval_accs = {"student": [], "teacher": []}

    for epoch in range(args.start_epoch, args.epochs):

        lss, rl, epoch_loss = train(train_loader, model, ema_model, som,
                                    optimizer, epoch, args.epochs, kappa)
        print(
            f'EPOCH LOSSES tot:{round(epoch_loss["total"]/10,2)}\tsup:{round(epoch_loss["sup"]/10, 2)}\tsom:{round(epoch_loss["som"]/10, 2)}')

        train_losses["total"].append((epoch_loss["total"]/10))
        train_losses["sup"].append((epoch_loss["sup"]/10))
        train_losses["som"].append((epoch_loss["som"]/10))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            prec1 = validate(eval_loader, model)

            ema_prec1 = validate(eval_loader, ema_model)

            eval_accs["student"].append(prec1)
            eval_accs["teacher"].append(ema_prec1)

            print('Accuracy of the Student network on the 10000 test images: %d %%' % (
                prec1))
            print('Accuracy of the Teacher network on the 10000 test images: %d %%' % (
                ema_prec1))
            if args.saveX == True:
                test_writer.add_scalar('Accuracy Student', prec1, epoch)
                test_writer.add_scalar('Accuracy Teacher', ema_prec1, epoch)

            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        training = {"train_loss": train_losses, "test_acc": eval_accs}
        json_object = json.dumps(training, indent=4)
        if not os.path.isdir("mt_som_results"):
            os.makedirs("mt_som_results")
        with open(f"mt_som_results/{args.epochs}eps{kappa}k.json", "w") as outfile:
            outfile.write(json_object)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + (1 - alpha) * param.data


def train(train_loader, model, ema_model, som, optimizer, epoch, total_eps, kappa):

    global global_step
    lossess = AverageMeter()
    running_loss = 0.0
    epoch_loss = {'total': 0.0, 'som': 0.0, 'sup': 0.0}
    init_time = time()

    class_criterion = nn.CrossEntropyLoss(
        reduction='sum', ignore_index=NO_LABEL).to(args.device)

    model.train()
    ema_model.train()

    for i, ((input, ema_input), target) in enumerate(train_loader):

        if (input.size(0) != args.batch_size):
            continue

        input_var = torch.autograd.Variable(input).to(args.device)

        target_var = torch.autograd.Variable(target.to(args.device))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0

        if args.sntg == True:
            model_out, model_h = model(input_var)
        else:
            model_out = model(input_var)

        class_loss = class_criterion(model_out, target_var) / minibatch_size
        epoch_loss['sup'] += class_loss.item()

        if not args.supervised_mode:
            with torch.no_grad():
                ema_input_var = torch.autograd.Variable(ema_input)
                ema_input_var = ema_input_var.to(args.device)

            if args.sntg == True:
                ema_model_out, ema_h = ema_model(ema_input_var)

            else:
                ema_model_out = ema_model(ema_input_var)

            ema_logit = ema_model_out

            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

            if args.consistency:

                consistency_weight = get_current_consistency_weight(epoch)
                shape1 = min(minibatch_size, input[:, 0:1, :].shape[0])
                som_pred1 = torch.empty(shape1, 2).to(args.device)
                som_pred2 = torch.empty(shape1, 2).to(args.device)

                for j in range(len(ema_h)):
                    _, som_pred1[j] = som.predict(model_h[j])
                    _, som_pred2[j] = som.predict(ema_h[j])

                # linear rampdown of kappa
                cur_kappa = kappa * (1 - (epoch / total_eps))

                som_loss_fn = SomLoss()
                som_loss = som_loss_fn(som_pred1, som_pred2, cur_kappa)
                consistency_loss = consistency_weight * som_loss / minibatch_size
                epoch_loss['som'] += consistency_loss.item()
            else:
                consistency_loss = 0

            loss = class_loss + consistency_loss
            epoch_loss['total'] += loss.item()
        else:
            loss = class_loss
        assert not (np.isnan(loss.item()) or loss.item() >
                    1e5), 'Loss explosion: {}'.format(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # print statistics
        running_loss += loss.item()

        if i == 0 or i % 20 == 20-1:    # print every 20 mini-batches
            print('[Epoch: %d/%d, Iteration: %5d] loss: %.5f epoch time %.1f' %
                  (epoch + 1, args.epochs, i + 1, running_loss / 20, time() - init_time))
            running_loss = 0.0

        lossess.update(loss.item(), input.size(0))

    return lossess, running_loss, epoch_loss


def validate(eval_loader, model):

    model.eval()
    total = 0
    correct = 0
    for i, (input, target) in enumerate(eval_loader):

        with torch.no_grad():
            input_var = input.to(args.device)
            target_var = target.to(args.device)

            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            if args.sntg == True:
                output1, _ = model(input_var)

            else:
                # compute output
                output1 = model(input_var)

            _, predicted = torch.max(output1.data, 1)
            total += target_var.size(0)
            correct += (predicted == target_var).sum().item()

    return 100 * correct / total


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print('Best Model Saved: ')
        print(best_path)


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == '__main__':
    args = get_parameters()

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    args.saveX = False

    main(args)
