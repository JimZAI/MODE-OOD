from __future__ import print_function

import os
import sys
import argparse
import time
import math
import torch.nn as nn

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss, ContrastiveLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # contrastive learning
    parser.add_argument('--feat_dim', default=80, type=int, help='the dimension of the projection features')
    parser.add_argument('--temperature_s', type=float, default=10.0, help='temperature for spatial contrastive loss')
    parser.add_argument('--aggregation', type=str, default="mean", choices=["mean", "max", "sum", "logsum"], help='the aggregation function used to compute the total similarity')
    parser.add_argument('--spatial_cont_loss', action='store_true', help="Use spatial_cont_loss")
    parser.add_argument('--lam', type=float, default=1.0, help='balance')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--alpa_train', action='store_true',
                        help='finetune pretrained models')
    parser.add_argument('--alpa_finetune', action='store_true',
                        help='finetune pretrained models')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '/mnt/hdd4/zhangji/OOD-datasets/cifar10/'
    opt.model_path = '/home/zhangji/OOD-project/knn-ood-master/pretrained/SupCon/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.alpa_finetune:
        opt.model_name = 'Finetune_{}_{}_{}_lr_{}_bsz_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                   opt.batch_size,  opt.trial, opt.lam)
    elif opt.alpa_train:
        opt.model_name = 'Joint_{}_{}_{}_lr_{}_bsz_{}_trial_{}_lam_{}'.\
            format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                   opt.batch_size,  opt.trial, opt.lam)
    else:
        opt.model_name = 'Base_{}_{}_{}_lr_{}_bsz_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                   opt.batch_size,  opt.trial, opt.lam)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
            
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt, criterion_contrast_spatial, attention):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_scl = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        spatial_f, features = model(images)

        # finetune
        if opt.finetune:
            loss = 0
            opt.lam = 1
            loss_contrast_spatial = criterion_alpa(spatial_f, labels=labels, attention=attention)
            Mode = "Finetune"
        else:
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)
            if opt.alpa_train:
                loss_contrast_spatial = criterion_alpa(spatial_f, labels=labels, attention=attention)
                Mode = "Joint"
            else:
                loss_contrast_spatial = loss-loss
                Mode = "Base"

        losses.update(loss, bsz)
        losses.update(loss_contrast_spatial.item(), bsz)
        loss = loss + opt.lam * loss_contrast_spatial

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train ('+ Mode +'): [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss_contrast_spatial {loss_contrast_spatial.val:.3f} ({loss_contrast_spatial.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_contrast_spatial=losses_scl))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    criterion_alpa = ContrastiveLoss(temperature=opt.temperature_s)
    train_loader = set_loader(opt)
    model, criterion = set_model(opt)

    if opt.finetune:
        if opt.model == "resnet18":
            checkpoint = torch.load("./ckp/ce/checkpoint_500.pth")
        else:
            checkpoint = torch.load("./ckp/ce/checkpoint_500.pth")
        model.load_state_dict(checkpoint, strict=False)

    # build optimizer
    optimizer, attention = set_optimizer(opt, model)

    if torch.cuda.is_available():
        model = model.cuda()
        if opt.alpa_train or opt.alpa_finetune:
            attention = attention.cuda()

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, criterion_alpa, attention)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # if epoch > 200  and epoch % opt.save_freq == 0:
        if epoch % 100 == 0:
            if opt.finetune:
                save_file = os.path.join(opt.save_folder, 'Finetune_epoch_{epoch}.pth'.format(bsz=opt.batch_size, epoch=epoch))
            elif opt.spatial_cont_loss:
                save_file = os.path.join(opt.save_folder,  'Train_epoch_{epoch}.pth'.format(bsz=opt.batch_size, epoch=epoch))
            else:
                save_file = os.path.join(opt.save_folder,  'Base_epoch_{epoch}.pth'.format(bsz=opt.batch_size, epoch=epoch))
            save_model(model, attention, optimizer, opt, epoch, save_file, opt.spatial_cont_loss)

if __name__ == '__main__':
    main()
