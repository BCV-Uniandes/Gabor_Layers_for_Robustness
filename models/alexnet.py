# Everything taken from 
# https://github.com/bearpaw/pytorch-classification/blob/master
'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gabor_layers import GaborLayer, GaborLayerLearnable

class AlexNet(nn.Module):
    def __init__(self, dataset='cifar100', 
            num_classes=10, input_channels=3,
            kernels1=None, kernels2=None, kernels3=None, 
            orientations=8, learn_theta=False):
        super(AlexNet, self).__init__()
        assert dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', 'SVHN']
        if dataset in ['cifar10', 'cifar100']:
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.2, 0.2, 0.2]
        elif dataset == 'imagenet':
            data_mean = [0.485, 0.456, 0.406]
            data_std = [0.229, 0.224, 0.225]
        elif dataset == 'tiny-imagenet':
            data_mean = [0.4802, 0.4481, 0.3975]
            data_std = [0.2302, 0.2265, 0.2262]
        elif dataset == 'SVHN':
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.5, 0.5, 0.5]

        self.mean = nn.Parameter(torch.tensor(data_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3), 
            requires_grad=False)
        self.std = nn.Parameter(torch.tensor(data_std).unsqueeze(0).unsqueeze(2).unsqueeze(3), 
            requires_grad=False)

        # Which Gabor layer to use
        if learn_theta:
            gabor_layer = GaborLayerLearnable
        else:
            gabor_layer = GaborLayer

        if kernels1 is not None:
            # First convolutional layer
            conv_layer_1 = gabor_layer(
                in_channels=3, out_channels=96, 
                kernel_size=11, stride=4, padding=5, kernels=kernels1)
        else:
            conv_layer_1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5)

        if kernels2 is not None:
            # Second convolutional layer
            conv_layer_2 = gabor_layer(
                in_channels=96, out_channels=256, 
                kernel_size=5, stride=1, padding=2, kernels=kernels2)
        else:
            conv_layer_2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        
        if kernels3 is not None:
            # Third convolutional layer
            conv_layer_3 = gabor_layer(
                in_channels=256, out_channels=384, 
                kernel_size=3, stride=1, padding=1, kernels=kernels3)
        else:
            conv_layer_3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)

        self.conv1 = nn.Sequential(
            conv_layer_1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            conv_layer_2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            conv_layer_3,
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        # # Convolutions
        # First conv
        x = self.conv1(x)
        # Second conv
        x = self.conv2(x)
        # Third conv
        x = self.conv3(x)

        # # Classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

