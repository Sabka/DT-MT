import numpy as np
import torch.nn as nn
import torch

from Datasets.data import NO_LABEL
from misc import ramps
from misc.utils import AverageMeter
# from tsne import visualize

global_step = 0

def train(train_loader, student_model, teacher_ema_model, optimizer, epoch, args):

    global global_step

    lossess, class_supervised_criterion, consistency_criterion = init_training(student_model, teacher_ema_model,args)
    st_correct, st_total, te_correct, te_total = 0, 0, 0, 0
    running_loss = 0.0

    # iterate through batches
    for i, ((input, ema_input), target) in enumerate(train_loader):

        # TODO find out how to identify data

        if (input.size(0) != args.batch_size):
            print("skipping batch", i)
            continue

        # prekonvertovanie na tenzory
        input_var = torch.autograd.Variable(input).to(args.device)
        target_var = torch.autograd.Variable(target).to(args.device)

        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input)
            ema_input_var = ema_input_var.to(args.device).to(args.device)

        # forward props
        student_model_out, student_model_h, teacher_ema_model_out, teacher_ema_h \
            = forward_props(student_model, input_var, teacher_ema_model, ema_input_var)

        # compute loss
        loss = compute_loss(student_model_out, class_supervised_criterion, target_var, ema_input, epoch, consistency_criterion, student_model_h, teacher_ema_h, args)

        # train accuracy calculation
        st_total, st_correct, te_total, te_correct \
            = train_acc(student_model_out, target_var, st_total, st_correct, teacher_ema_model_out, te_total, te_correct, args)

        # update student and teacher weights
        update_weights(optimizer, loss, student_model, teacher_ema_model, args)

        # stat printing
        running_loss += loss.item()
        pr_freq = 20
        if i % pr_freq == pr_freq - 1:  # print every <pr_freq> mini-batches
            print_stats(st_correct, st_total, te_correct, te_total, epoch, i, len(train_loader), running_loss, pr_freq, args)
            # print(st_correct, te_correct, st_total, te_total)
            st_correct, st_total, te_correct, te_total, running_loss  = 0, 0, 0, 0, 0.0
            lossess.update(loss.item(), input.size(0))

    return lossess, running_loss

def init_training(student_model, teacher_ema_model, args):
    lossess = AverageMeter()

    # cost definitions
    class_supervised_criterion = nn.BCELoss(reduction='mean').to(args.device)  # .cuda()
    consistency_criterion = nn.MSELoss(reduction='sum').to(args.device)  # .cuda()

    # trenovaci mod, nastavujeme kvoli spravaniu niektorych vrstiev
    student_model.train()
    teacher_ema_model.train()

    return lossess, class_supervised_criterion, consistency_criterion

def forward_props(student_model, input_var, teacher_ema_model, ema_input_var):
    student_model_out, student_model_h = student_model(input_var)
    teacher_ema_model_out, teacher_ema_h = teacher_ema_model(ema_input_var)

    return student_model_out, student_model_h, teacher_ema_model_out, teacher_ema_h

def train_acc(student_model_out, target_var, st_total, st_correct, teacher_ema_model_out, te_total, te_correct, args):
    # student train accuracy
    output1 = (student_model_out.view(target_var.size(0)).to(torch.float32).to(args.device) > torch.tensor(
        [0.5]).to(args.device)).float() * 1
    st_total += target_var.size(0) - (target_var == -1).sum().item()
    st_correct += (output1 == target_var).sum().item()

    # teacher train accuracy
    output1 = (teacher_ema_model_out.view(target_var.size(0)).to(torch.float32).to(args.device) > torch.tensor(
        [0.5]).to(args.device)).float() * 1
    te_total += target_var.size(0) - (target_var == -1).sum().item()
    te_correct += (output1 == target_var).sum().item()

    return st_total, st_correct, te_total, te_correct

def compute_loss(student_model_out, class_supervised_criterion, target_var, ema_input, epoch, consistency_criterion,
                 student_model_h, teacher_ema_h, args):
    # supervised loss calculation
    student_model_out = student_model_out.view(256).to(torch.float32)
    class_loss = class_supervised_criterion(student_model_out[194:],
                                            target_var.to(torch.float32)[194:]) / len(target_var)


    consistency_weight = get_current_consistency_weight(epoch, args)
    consistency_loss = consistency_weight * consistency_criterion(student_model_h, teacher_ema_h) / len(target_var)

    loss = class_loss + consistency_loss

    return loss

def print_stats(st_correct, st_total, te_correct, te_total, epoch, act_iter, total_iters, running_loss, pr_freq, args):
    st_train_acc = st_correct / st_total
    t_train_acc = te_correct / te_total

    print(f'Epoch: {epoch + 1}/{args.epochs}, '
          f'Iteration: {" " if act_iter + 1 < 100 else ""}{act_iter + 1}/{total_iters}, '
          f'Train loss: { "{0:.5f}".format(round(running_loss / pr_freq, 5)) } '
          f'Train accuracy (S, T): { "{0:.3f}".format(round(st_train_acc * 100, 3))}%, { "{0:.3f}".format(round(t_train_acc * 100, 2))}%')  # , Acc: {None}, Time: {None}')

def update_weights(optimizer, loss, student_model, teacher_ema_model, args):

    global global_step

    # uprava vah studenta
    optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor s to zero.
    loss.backward()
    optimizer.step()
    global_step += 1

    # uprava vah teachera
    update_ema_variables(student_model, teacher_ema_model, args.ema_decay, global_step)

def validate(eval_loader, model, args):
    print("===> Validating")

    viz_data = []
    labels = []

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

            # prepare for visualization
            for i in range(len(output_h)):
                viz_data.append(output_h[i].tolist())
            for i in range(len(target)):
                labels.append(target[i].tolist())

            output1 = (output1.view(target_var.size(0)).to(torch.float32) > torch.tensor([0.5]).to(
                args.device)).float() * 1

            # _, predicted = torch.max(output1.data, 1)
            total += target_var.size(0)
            correct += (output1 == target_var).sum().item()

    # visualize(np.array(viz_data), labels, f"{global_step}-tsne.png")

    return 100 * correct / total

def get_current_consistency_weight(epoch, args):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha)
        ema_param.data = torch.add(ema_param.data, param.data, alpha=(1 - alpha))
