import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from .gabor_layers import GaborLayer, GaborLayerLearnable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
                               bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):

    def __init__(self, dataset='cifar100',
        num_classes=100, input_channels=3,
        kernels1=None, kernels2=None, kernels3=None,
        orientations=8, learn_theta=False, finetune=False,
        depth=16, widen_factor=4, dropout_rate=0.0, use_7x7=False):

        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', 'SVHN']
        if dataset in ['cifar10', 'cifar100']:
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.2, 0.2, 0.2]
            pretrained = False
        elif dataset == 'imagenet':
            data_mean = [0.485, 0.456, 0.406]
            data_std = [0.229, 0.224, 0.225]
            pretrained = finetune
        elif dataset == 'tiny-imagenet':
            data_mean = [0.4802, 0.4481, 0.3975]
            data_std = [0.2302, 0.2265, 0.2262]
            pretrained = False
        elif dataset == 'SVHN':
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.5, 0.5, 0.5]
            pretrained = False

        self.mean = nn.Parameter(torch.tensor(data_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3), 
            requires_grad=False)
        self.std = nn.Parameter(torch.tensor(data_std).unsqueeze(0).unsqueeze(2).unsqueeze(3), 
            requires_grad=False)

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        if use_7x7:
            self.conv1 = conv3x3(input_channels, nStages[0])
        else:
            self.conv1 = conv3x3(input_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        # Modify layers to add Gabor Layers

        if learn_theta:
            gabor_layer = GaborLayerLearnable
        else:
            gabor_layer = GaborLayer

        if kernels1 is not None:
            print('Modified first conv layer')
            # First convolutional layer
            self.conv1 = gabor_layer(
                in_channels=input_channels, out_channels=nStages[0], 
                kernel_size=3, stride=1, padding=1, kernels=kernels1)

        if kernels2 is not None:
            print('Modified second conv layer')
            # Second convolutional layer
            self.layer1[0].conv1 = gabor_layer(
                in_channels=nStages[0], out_channels=nStages[1], 
                kernel_size=3, stride=1, padding=1, kernels=kernels2)

        if kernels3 is not None:
            print('Modified third conv layer')
            # Second convolutional layer
            self.layer1[0].conv2 = gabor_layer(
                in_channels=nStages[1], out_channels=nStages[1], 
                kernel_size=3, stride=1, padding=1, kernels=kernels3)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        # import pdb; pdb.set_trace()
        strides = [stride] + [1]*int(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
