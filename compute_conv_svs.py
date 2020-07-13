import os
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp

from models.vgg import VGG16
from models.resnet import ResNet18

# Taken _explicitly_ from
# The Singular Values of Convolutional Layers
# https://openreview.net/pdf?id=rJevYoA9Fm
def SingularValues(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)

def main_svs_computation(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Input from dataset
    inp_s = 32
    INPUT_SHAPE = (inp_s, inp_s)

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    # Load state dictionary
    path = osp.join(args.checkpoint, 'model_best.pth.tar')
    state_dict = torch.load(path, map_location=device)['state_dict'] 

    # Model
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100

    if args.arch == 'vgg16':
        model = VGG16(
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'resnet18':
        model = ResNet18(
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)

    arguments = [args.kernels1, args.kernels2, args.kernels3]
    indices = [0, 2, 5]
    kernels = []
    for iteration, (ks, idx) in enumerate(zip(arguments, indices)):
        # If the network is a vgg16
        if args.arch == 'vgg16':
            layer = model.module.net.features[idx]
        # If the network is a resnet18
        elif args.arch == 'resnet18':
            if iteration == 0:
                layer = model.module.net.conv1
            elif iteration == 1:
                layer = model.module.net.layer1[0].conv1
            elif iteration == 2:
                layer = model.module.net.layer1[0].conv2
        if ks is None:
            this_kernel = layer.weight
        else:
            this_kernel = layer.generate_gabor_kernels()
        kernels.append(this_kernel)

    # kernel is of size output-channels x input-channels x k x k
    # kernel should be a (k × k × m × m) tensor, according to:
    # The Singular Values of Convolutional Layers
    # https://openreview.net/pdf?id=rJevYoA9Fm
    # k x k x output-channels x input-channels
    # so we need to transpose things
    kernels = [k.permute(2, 3, 0, 1) for k in kernels]

    # Convert to numpy
    kernels = [k.cpu().detach().numpy() for k in kernels]

    # Compute singular values
    for idx, kernel in enumerate(kernels):
        singular_values = SingularValues(
            kernel=kernel, input_shape=INPUT_SHAPE)

        # Compute descriptive statistics of values
        df = {
            'max' : singular_values.max(),
            'min' : singular_values.min(),
            'avg' : singular_values.mean(),
            'std' : singular_values.std()
        }
        df = pd.DataFrame.from_dict(df, 
            orient='index',
            columns=['value']
        )
        print('Overall results: \n', df)
        name = f'svd_results_conv{idx+1}.csv'
        filename = osp.join(args.checkpoint, name)
        df.to_csv(filename, index=False)

if __name__ == "__main__":
    dataset_names = ['cifar10', 'cifar100', 'mnist', 'fashion-mnist']
    model_names = ['vgg16', 'resnet18']
    parser = argparse.ArgumentParser(
        description='Singular Values computation for convolutional layer')
    parser.add_argument('-c', '--checkpoint', required=True, type=str, 
        metavar='PATH', help='path to save checkpoint')
    parser.add_argument('-d', '--dataset', default='cifar100', type=str,
        choices=dataset_names)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
        choices=model_names,
        help='model architecture: ' +
            ' | '.join(model_names) +
            ' (default: vgg16)')
    parser.add_argument('--gpu-id', default='0', type=str,
        help='id(s) for CUDA_VISIBLE_DEVICES')
    # If the network has Gabor layer
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
    args = parser.parse_args()
    main_svs_computation(args)