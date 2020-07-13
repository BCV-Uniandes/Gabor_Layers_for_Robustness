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

# Import the models
from models.alexnet import AlexNet
from models.vgg import VGG16
from models.resnet import ResNet18
from models.lenet import LeNet
from models.madrynet import MadryNet
from models.wide_resnet import Wide_ResNet

# Utils for bars and stuff
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.adversarial_attacks import pgd_attack

# Import attack script
from main_attack import main_attack

# Import SVD computation script 
from compute_conv_svs import main_svs_computation

# Ensure we have reproducibility
cudnn.deterministic = True
cudnn.benchmark = False

model_names = ['vgg16', 'resnet18', 'madry', 'lenet', 'alexnet', 'wide-resnet']
dataset_names = ['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet', 'mnist',
                 'SVHN']

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/CIFAR100/MNIST Gabor-DNN training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str,
    choices=dataset_names)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
    help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
    help='train batchsize')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200],
    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, 
    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--optim', default='SGD', type=str,
    help='optimizer (SGD or Adam)')
parser.add_argument('--lambd', default=0., type=float,
    help='lambda for regularization of the sigma parameters of the Gabor layer')

# Checkpoints
parser.add_argument('-c', '--checkpoint', required=True, type=str, 
    metavar='PATH', help='path to save checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
    help='path to latest checkpoint (default: none)')

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
parser.add_argument('--attack', default='pgd', type=str,
    help='type of adversarial attack', choices=['pgd'])
parser.add_argument('--train-adv', action='store_true',
    help='perform adversarial training')
parser.add_argument('--finetune', action='store_true',
    help='finetune from imagenet pre-trained weights')
parser.add_argument('--use-7x7', action='store_true',
    help='Use a kernel of 7x7 as the first conv (only available for wide-resnet)')

# Miscs
parser.add_argument('--seed', type=int, help='manual seed', default=111)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

if args.finetune:
    assert 'imagenet' in args.dataset, "Pre-trained weights are from IN, so fine-tuning should be on (tiny) ImageNet"

if args.dataset == 'mnist':
    assert args.arch in ['madry', 'lenet'], 'Running on MNIST should be done with Madrys/Lenet architecture'

# Boolean to check if there are any Gabor kernels
any_kernels = any(
    [
        k is not None 
        for k in [args.kernels1, args.kernels2, args.kernels3]
    ]
)
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

best_acc = 0  # best test accuracy
txt_file_path = osp.join('.', args.checkpoint, 'params.txt')

def print_to_log(text, txt_file_path):
	with open(txt_file_path, 'a') as text_file:
		print(text, file=text_file)

def print_training_params(args, txt_file_path):
	d = vars(args)
	text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
	# Print to log and console
	print_to_log(text, txt_file_path)

def get_sigmas(args, model):
    sigmas = []
    if args.arch == 'vgg16':
        if args.kernels1 is not None:
            k1_sigmas = model.module.net.features[0].sigmas
            sigmas.append(k1_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels2 is not None:
            k2_sigmas = model.module.net.features[2].sigmas
            sigmas.append(k2_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels3 is not None:
            k3_sigmas = model.module.net.features[5].sigmas
            sigmas.append(k3_sigmas.squeeze(3).squeeze(2).squeeze(1))
    elif args.arch == 'resnet18':
        if args.kernels1 is not None:
            k1_sigmas = model.module.net.conv1.sigmas
            sigmas.append(k1_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels2 is not None:
            k2_sigmas = model.module.net.layer1[0].conv1.sigmas
            sigmas.append(k2_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels3 is not None:
            k3_sigmas = model.module.net.layer1[0].conv2.sigmas
            sigmas.append(k3_sigmas.squeeze(3).squeeze(2).squeeze(1))
    elif args.arch in ['madry', 'lenet']:
        if args.kernels1 is not None:
            k1_sigmas = model.module.conv1.sigmas
            sigmas.append(k1_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels2 is not None:
            k2_sigmas = model.module.conv2.sigmas
            sigmas.append(k2_sigmas.squeeze(3).squeeze(2).squeeze(1))
    elif args.arch == 'alexnet':
        if args.kernels1 is not None:
            k1_sigmas = model.module.conv1[0].sigmas
            sigmas.append(k1_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels2 is not None:
            k2_sigmas = model.module.conv2[0].conv1.sigmas
            sigmas.append(k2_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels3 is not None:
            k3_sigmas = model.module.conv3[0].conv2.sigmas
            sigmas.append(k3_sigmas.squeeze(3).squeeze(2).squeeze(1))
    elif args.arch == 'wide-resnet':
        if args.kernels1 is not None:
            k1_sigmas = model.module.conv1.sigmas
            sigmas.append(k1_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels2 is not None:
            k2_sigmas = model.module.layer1[0].conv1.sigmas
            sigmas.append(k2_sigmas.squeeze(3).squeeze(2).squeeze(1))
        if args.kernels3 is not None:
            k3_sigmas = model.module.layer1[0].conv2.sigmas
            sigmas.append(k3_sigmas.squeeze(3).squeeze(2).squeeze(1))

    return torch.cat(sigmas, dim=0)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print_training_params(args=args, txt_file_path=txt_file_path)

    # Data
    print(f'==> Preparing dataset {args.dataset}')
    if args.dataset in ['cifar10', 'cifar100']:
        detph = 28
        widen_factor = 10
        dropout = 0.3
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    elif args.dataset == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

    elif args.dataset == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    elif args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
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

        trainset = dataloader(root='.data', train=True, download=True, transform=transform_train)
        testset = dataloader(root='.data', train=False, download=False, transform=transform_test)

    elif args.dataset == 'tiny-imagenet':
        trainset = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
        testset = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
        num_classes = 200

    elif args.dataset == 'imagenet':
        trainset = datasets.ImageFolder('imagenet/train', transform=transform_train)
        testset = datasets.ImageFolder('imagenet/val', transform=transform_test)
        num_classes = 1000

    elif args.dataset == 'SVHN':
        trainset = datasets.SVHN('data', split='train', transform=transform_train, download=True)
        testset = datasets.SVHN('data', split='test', transform=transform_test, download=True)
        num_classes = 10
    
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch == 'vgg16':
        model = VGG16(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta,
            finetune=args.finetune
        )

    elif args.arch == 'resnet18':
        model = ResNet18(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta,
            finetune=args.finetune
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
            finetune=args.finetune,
            depth=detph,
            widen_factor=widen_factor,
            dropout_rate=dropout,
            use_7x7=args.use_7x7
        )

    print('Model:')
    print(model)
    print_to_log(text=repr(model), txt_file_path=txt_file_path)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    
    # Compute number of parameters and print them
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_txt = f'    Total trainable params: {param_num:d}'
    print_to_log(text=param_txt, txt_file_path=txt_file_path)
    print(param_txt)

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        print('Using SGD optimizer')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
       optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Resume
    title = f'{args.dataset}-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint...')
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = osp.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, device)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(
            trainloader, model, criterion, optimizer, epoch, device, train_adv=args.train_adv, args=args)
        test_loss, test_acc = test(
            testloader, model, criterion, epoch, device)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

        if args.kernels1 is not None:
            plot_kernels(model, args.checkpoint, epoch, device)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

    print('Training finished. Running attack')
    main_attack(args)

    print('Running SVD computation')
    main_svs_computation(args)


def plot_kernels(model, checkpoint, epoch, device):
    # Generate kernels for plotting
    if args.arch == 'vgg16':
        kernels = model.module.net.features[0].generate_gabor_kernels()
    elif args.arch == 'resnet18':
        kernels = model.module.net.conv1.generate_gabor_kernels()
    elif args.arch in ['madry', 'lenet']:
        kernels = model.module.conv1.generate_gabor_kernels()
    elif args.arch == 'alexnet':
        kernels = model.module.conv1[0].generate_gabor_kernels()
    elif args.arch == 'wide-resnet':
        kernels = model.module.conv1.generate_gabor_kernels()
    kernels = kernels.squeeze()
    n_kernels = kernels.size(0)
    fig = plt.figure()
    # For tight layout
    extra_kernels = 0
    rows = args.kernels1 if extra_kernels == 0 else args.kernels1+1
    orientations = 8
    gs1 = gridspec.GridSpec(
        rows, orientations,
        width_ratios=[1 for _ in range(orientations)], 
        wspace=0.0, hspace=0.0, 
        top=0.95, bottom=0.05, left=0.17, right=0.845)
    curr_row, curr_col = 0, 0
    # Families of filters
    for k in range(n_kernels):
        if device == 'cuda':
            x_filter = kernels[k].detach().cpu().numpy()
        else:
            x_filter = kernels[k].detach().numpy()
        axes = fig.add_subplot(gs1[curr_row, curr_col])
        # axes[curr_row, curr_col].matshow(x_filter)
        axes.matshow(x_filter)
        # axes[curr_row, curr_col].axis('off')
        axes.axis('off')
        curr_col += 1
        if curr_col == orientations:
            curr_col = 0
            curr_row += 1
    # Save figure to file
    filename = os.path.join(checkpoint, f'kernels_epoch_{epoch}.pdf')
    fig.savefig(filename)
    plt.close()

def train(trainloader, model, criterion, optimizer, epoch, device, train_adv, args):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)
        if train_adv:
            orig_outputs = model(inputs)
            _, orig_preds = torch.max(orig_outputs.data, 1)
            iters, eps = 20, 8/255
            inputs = pgd_attack(
                model, inputs, targets, 
                orig_preds=orig_preds,
                alpha=eps/iters, eps=eps, 
                minimum=0, maximum=1, 
                iterations=iters, training=True
            )
            model.train()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # check if there's regularization on the sigmas of Gabor layers
        if args.lambd > 0.:
            sigmas = get_sigmas(args, model)
            reg_loss = (sigmas**2).mean()
            reg_losses.update(reg_loss.item(), inputs.size(0))
            # subtract this loss from the other loss
            loss = loss - args.lambd * reg_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Reg. loss: {reg_loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    reg_loss=reg_losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, device):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    reg_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # check if there's regularization on the sigmas of Gabor layers
        if args.lambd > 0.:
            sigmas = get_sigmas(args, model)
            reg_loss = (sigmas**2).mean()
            reg_losses.update(reg_loss.item(), inputs.size(0))
            # subtract this loss from the other loss
            loss = loss - args.lambd * reg_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Reg. loss: {reg_loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    reg_loss=reg_losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
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
    main()
