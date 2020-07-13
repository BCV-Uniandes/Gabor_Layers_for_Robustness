import torch
import torch.nn as nn
from torchvision.models import resnet18 
from .gabor_layers import GaborLayer, GaborLayerLearnable

import pdb, traceback, sys

# mean and std were taken from 
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

class ResNet18(nn.Module):
    def __init__(self, dataset='cifar100', 
            num_classes=10, input_channels=3, 
            kernels1=None, kernels2=None, kernels3=None, 
            orientations=8, learn_theta=False, finetune=False):
        super(ResNet18, self).__init__()
        assert dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet',
                           'SVHN']
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
            pretrained = finetune
        elif dataset == 'SVHN':
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.5, 0.5, 0.5]
            pretrained = False

        self.mean = nn.Parameter(torch.tensor(data_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3), 
            requires_grad=False)
        self.std = nn.Parameter(torch.tensor(data_std).unsqueeze(0).unsqueeze(2).unsqueeze(3), 
            requires_grad=False)
        # Which Gabor layer to use
        if learn_theta:
            gabor_layer = GaborLayerLearnable
        else:
            gabor_layer = GaborLayer
        # Net from the model zoo
        self.net = resnet18(pretrained=pretrained)
        # Freeze things if we want to finetune
        if finetune:
            # Freeze the whole net
            for param in self.net.parameters():
                param.requires_grad = False
        # Modify last layer to match the number of classes
        if dataset != 'imagenet':
            # This comes with .requires_grad = True, by default
            self.net.fc = nn.Linear(
                in_features=512, out_features=num_classes, bias=True)

        # Modify first convolution
        self.net.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, padding=1, bias=False)
        if kernels1 is not None:
            print('Modified first conv layer')
            # First convolutional layer
            self.net.conv1 = gabor_layer(
                in_channels=input_channels, out_channels=64, 
                kernel_size=3, stride=1, padding=1, kernels=kernels1)
        if kernels2 is not None:
            print('Modified second conv layer')
            # Second convolutional layer
            self.net.layer1[0].conv1 = gabor_layer(
                in_channels=64, out_channels=64, 
                kernel_size=3, stride=1, padding=1, kernels=kernels2)
        if kernels3 is not None:
            print('Modified third conv layer')
            # Second convolutional layer
            self.net.layer1[0].conv2 = gabor_layer(
                in_channels=64, out_channels=64, 
                kernel_size=3, stride=1, padding=1, kernels=kernels3)
        # Remove average pooling
        try:
            del self.net.avgpool
            self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        except:
            print(f'This version of torchvision does not have avgpooling in ResNet18')

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)
