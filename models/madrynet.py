'''LeNet in PyTorch.'''
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gabor_layers import GaborLayer, GaborLayerLearnable

__all__ = ['MadryNet']

# Avgs and stds taken from 
# https://github.com/pytorch/examples/blob/master/mnist/main.py
class MadryNet(nn.Module):
    def __init__(self, kernels1=None, kernels2=None, kernels3=None, 
            orientations=8, learn_theta=False):
        super(MadryNet, self).__init__()

        data_mean = [0.1307]
        data_std = [0.3081]
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
            self.conv1 = gabor_layer(
                in_channels=1, out_channels=32, 
                kernel_size=5, stride=1, padding=2, kernels=kernels1)
        else:
            self.conv1 = nn.Conv2d(1, 32, 5, padding=2, bias=True)

        if kernels2 is not None:
            # Second convolutional layer
            self.conv2 = gabor_layer(
                in_channels=32, out_channels=64, 
                kernel_size=5, stride=1, padding=2, kernels=kernels2)
        else:
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=True)
            
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = F.max_pool2d(torch.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(torch.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
