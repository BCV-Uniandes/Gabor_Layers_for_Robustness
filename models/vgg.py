import torch
import torch.nn as nn
from torchvision.models import vgg16 
from torchvision.transforms.functional import normalize
from .gabor_layers import GaborLayer, GaborLayerLearnable

import pdb

class VGG16(nn.Module):
    def __init__(self, dataset='cifar100', 
            num_classes=10, input_channels=3,
            kernels1=None, kernels2=None, kernels3=None, 
            orientations=8, learn_theta=False, finetune=False):
        super(VGG16, self).__init__()
        assert dataset in ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', 'svhn']
        if dataset in ['cifar10', 'cifar100']:
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.2, 0.2, 0.2]
            pretrained = False
        elif dataset == 'svhn':
            data_mean = [0.5, 0.5, 0.5]
            data_std = [0.5, 0.5, 0.5]
            pretrained = False
        elif dataset == 'imagenet':
            data_mean = [0.485, 0.456, 0.406]
            data_std = [0.229, 0.224, 0.225]
            pretrained = finetune
        elif dataset == 'tiny-imagenet':
            data_mean = [0.4802, 0.4481, 0.3975]
            data_std = [0.2302, 0.2265, 0.2262]
            pretrained = finetune
        if pretrained:
            print('The model being created IS pretrained on ImageNet')

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
        self.net = vgg16(pretrained=pretrained)
        # Freeze things if we want to finetune
        if finetune:
            # Freeze the whole net
            for param in self.net.parameters():
                param.requires_grad = False
                
        if kernels1 is not None:
            # First convolutional layer
            self.net.features[0] = gabor_layer(
                in_channels=input_channels, out_channels=64, 
                kernel_size=3, stride=1, padding=1, kernels=kernels1)
            if kernels2 is not None:
                # Second convolutional layer
                self.net.features[2] = gabor_layer(
                    in_channels=64, out_channels=64, 
                    kernel_size=3, stride=1, padding=1, kernels=kernels2)
                if kernels3 is not None:
                    # Second convolutional layer
                    self.net.features[5] = gabor_layer(
                        in_channels=64, out_channels=128, 
                        kernel_size=3, stride=1, padding=1, kernels=kernels3)
        # Remove average pooling
        if dataset != 'imagenet':
            try:
                del self.net.avgpool
                self.net.avgpool = lambda x: x
            except:
                print(f'This version of torchvision does not have avgpooling in VGG16')
            # Change last linear layers
            lin1_inp_size = 512
            lin1_out_size = 512
            lin2_inp_size = 512
            lin2_out_size = 512
            lin3_inp_size = 512
            self.net.classifier._modules['0'] = nn.Linear(lin1_inp_size, lin1_out_size)
            self.net.classifier._modules['3'] = nn.Linear(lin2_inp_size, lin2_out_size)
            self.net.classifier._modules['6'] = nn.Linear(lin3_inp_size, num_classes)


    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)
