'''
Training script for CIFAR-10/100
Inspired on
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

# Basic things
import os
import pdb
import sys
import time
import shutil
import random
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Torch-related stuff
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.alexnet import AlexNet
from models.vgg import VGG16
from models.resnet import ResNet18
from models.madrynet import MadryNet
from models.lenet import LeNet
from models.wide_resnet import Wide_ResNet

# Import the models

# Import attacks
from utils.adversarial_attacks import attack_cw, attack_pgd

# Utils for bars and stuff
from utils import (Bar, Logger, AverageMeter, accuracy, mkdir_p, 
    savefig)

# Ensure we have reproducibility
cudnn.deterministic = True
cudnn.benchmark = False

def main_attack(args):
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    # Random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Data
    print(f'==> Preparing dataset {args.dataset}')
    if args.dataset in ['cifar10', 'cifar100']:
        detph = 28
        widen_factor = 10
        dropout = 0.3
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    elif args.dataset == 'tiny-imagenet':
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    elif args.dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif args.dataset == 'mnist':
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    elif args.dataset == 'SVHN':
        detph = 16
        widen_factor = 4
        dropout = 0.4
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    print(f'Running on dataset {args.dataset}')
    if args.dataset in ['cifar10', 'cifar100', 'mnist']:
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
        elif args.dataset == 'mnist':
            dataloader = datasets.MNIST
            num_classes = 10

        testset = dataloader(root='.data', train=False, download=False, transform=transform_test)

    elif args.dataset == 'tiny-imagenet':
        testset = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
        num_classes = 200

    elif args.dataset == 'imagenet':
        testset = datasets.ImageFolder('imagenet/val', transform=transform_test)
        num_classes = 1000

    elif args.dataset == 'SVHN':
        testset = datasets.SVHN('data', split='test', transform=transform_test, download=True)
        num_classes = 10
    
    testloader = data.DataLoader(testset, batch_size=args.test_batch, 
        shuffle=False, num_workers=args.workers)
    # Model
    if args.arch == 'vgg16':
        model = VGG16(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'resnet18':
        model = ResNet18(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'madry':
        model = MadryNet(
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'lenet':
        model = LeNet(
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'alexnet':
        model = AlexNet(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'wide-resnet':
        model = Wide_ResNet(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta,
            finetune=False,
            depth=detph,
            widen_factor=widen_factor,
            dropout_rate=dropout,
            use_7x7=args.use_7x7
        )

    
    print('Model:')
    print(model)
    
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    # Compute number of parameters and print them
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_txt = f'    Total trainable params: {param_num:d}'
    print(param_txt)

    criterion = nn.CrossEntropyLoss()
    # Resume
    # Load checkpoint.
    print('==> Resuming from checkpoint...')
    checkpoint_filename = osp.join(args.checkpoint, 'model_best.pth.tar')
    assert osp.isfile(checkpoint_filename), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    print('\nEvaluation only')
    test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    print(f'Running {args.attack} attack!')

    if args.attack == 'cw':
        c_vals = torch.logspace(start=-2, end=2, steps=9)
        for c in c_vals:
            print(f'Running attack with c = {c:5.3f}')
            attack_cw(model, testloader, device=device, c=c)
            print('\n')
    else:
        if args.dataset == 'mnist':
            epsilons = [.1, .2, .3, .4]
        else:
            epsilons = [2/255, 8/255, 16/255, .1]
        print(f'Epsilons are: {epsilons}')
        minimum = 0.
        maximum = 1.
        print(f'Images maxima: {maximum} -- minima: {minimum}')
        df = {
            'epsilons' : [0., ],
            'test_set_accs' : [test_acc, ],
            'flip_rates' : [0., ],
        }
        for eps in epsilons:
            print(f'Running attack with epsilon = {eps:5.3f}')
            acc_test_set, flip_rate = attack_pgd(model, testloader, device=device, 
                minimum=minimum, maximum=maximum, eps=eps)
            df['epsilons'].append(eps)
            df['test_set_accs'].append(acc_test_set)
            df['flip_rates'].append(flip_rate)
            print('\n')
        df = pd.DataFrame.from_dict(df)
        print('Overall results: \n', df)
        filename = osp.join(args.checkpoint, 'attack_results.csv')
        df.to_csv(filename, index=False)



def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        (prec1, ) = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    num_sets = ['A', 'B', 'all']
    attack_names = ['pgd', 'cw']
    dataset_names = ['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet',
                     'mnist', 'SVHN']
    model_names = ['vgg16', 'resnet18', 'madry', 'lenet', 'alexnet',
                   'wide-resnet']

    parser = argparse.ArgumentParser(
        description='PyTorch Gabor-DNN attack')
    # Datasets
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
        help='number of data loading workers (default: 4)')
    parser.add_argument('-d', '--dataset', default='cifar100', type=str,
        choices=dataset_names)

    # Optimization options
    parser.add_argument('--test-batch', default=64, type=int, metavar='N',
        help='test batchsize')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', required=True, type=str, 
        metavar='PATH', help='path to save checkpoint')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
        choices=model_names,
        help='model architecture: ' +
            ' | '.join(model_names) +
            ' (default: vgg16)')
    parser.add_argument('--orientations', type=int, default=8, 
        help='Number of orientations between 0 and 2*pi.')
    parser.add_argument('--kernels1', type=int, default=None, 
        help='Number of inner kernels in Gabor layer 1.')
    parser.add_argument('--kernels2', type=int, default=None, 
        help='Number of inner kernels in Gabor layer 2.')
    parser.add_argument('--kernels3', type=int, default=None, 
        help='Number of inner kernels in Gabor layer 3.')
    parser.add_argument('--learn-theta', action='store_true',
        help='Use Gabor layer with learnable theta')
    parser.add_argument('--use-7x7', action='store_true',
        help='Use a kernel of 7x7 as the first conv (only available for wide-resnet)')

    # Miscs
    parser.add_argument('--seed', type=int, help='manual seed', default=111)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        help='evaluate model on validation set')
    # Device options
    parser.add_argument('--gpu-id', default='0', type=str,
        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--attack', default='pgd', type=str, required=True,
        help='type of adversarial attack', choices=attack_names)

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    best_acc = 0  # best test accuracy
    txt_file_path = osp.join('.', args.checkpoint + '-attack', 'params.txt')

    main_attack(args)