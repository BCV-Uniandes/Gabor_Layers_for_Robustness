import os
import torch
import argparse
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from models.vgg import VGG16
from models.resnet import ResNet18
from models.lenet import LeNet
from models.alexnet import AlexNet
import pdb

# This function only works for non-separable conv layers
def SingularValuesConv(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def SingularValuesSepConv(kernel, input_shape):
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    # Here, for example, we have
    # transforms.shape = (32, 32, 24, 1)
    all_values = []
    for idx in range(transforms.shape[2]):
        # current.shape == (32, 32)
        current = transforms[:, :, idx, 0]
        # expand
        current = np.expand_dims(current, axis=2)
        current = np.expand_dims(current, axis=3)
        # current.shape == (32, 32, 1, 1)
        curr_svd = np.linalg.svd(current, compute_uv=False)
        # curr_svd.shape == (32, 32, 1)
        all_values.append(curr_svd)

    all_values = np.concatenate(all_values, axis=2)
    return all_values



def main_svs_plotting(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    PERCENTILE = 95
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    # Model
    if args.dataset == 'cifar10':
        # Input from dataset
        inp_s = 32
        num_classes = 10
    elif args.dataset == 'cifar100':
        # Input from dataset
        inp_s = 32
        num_classes = 100
    elif args.dataset == 'imagenet':
        # Input from dataset
        inp_s = 224
        num_classes = 1000
    elif args.dataset == 'mnist':
        # Input from dataset
        inp_s = 28
        num_classes = 10
    elif args.dataset == 'svhn':
        # Input from dataset
        inp_s = 28
        num_classes = 10

    INPUT_SHAPE = (inp_s, inp_s)

    if args.arch == 'vgg16':
        finetune = False 
        if args.dataset == 'imagenet':
            finetune = True
        std_model = VGG16(
            dataset=args.dataset,
            num_classes=num_classes,
            finetune=finetune
        )

        gabor_model = VGG16(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )

    elif args.arch == 'resnet18':
        std_model = ResNet18(
            dataset=args.dataset,
            num_classes=num_classes
        )

        gabor_model = ResNet18(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )

    elif args.arch == 'lenet':
        std_model = LeNet()

        gabor_model = LeNet(
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )
    elif args.arch == 'alexnet':
        std_model = AlexNet(
            dataset=args.dataset,
            num_classes=num_classes,
        )

        gabor_model = AlexNet(
            dataset=args.dataset,
            num_classes=num_classes,
            kernels1=args.kernels1,
            kernels2=args.kernels2,
            kernels3=args.kernels3,
            orientations=args.orientations,
            learn_theta=args.learn_theta
        )

    std_model = torch.nn.DataParallel(std_model).cuda()
    gabor_model = torch.nn.DataParallel(gabor_model).cuda()
    # Load standard state dictionary
    if args.standard_ckpt is not None:
        std_path = osp.join(args.standard_ckpt, 'model_best.pth.tar')
        std_state_dict = torch.load(std_path, map_location=device)['state_dict'] 
        std_model.load_state_dict(std_state_dict)
    

    # Load Gabor state dictionary
    gabor_path = osp.join(args.gabor_ckpt, 'model_best.pth.tar')
    gabor_state_dict = torch.load(gabor_path, map_location=device)['state_dict'] 
    gabor_model.load_state_dict(gabor_state_dict)

    # Variables for the loop
    kernels = [args.kernels1, args.kernels2, args.kernels3]
    indices = [0, 2, 5]
    # Boolean to check if there are any Gabor kernels
    any_kernels = any([k is not None for k in kernels])
    assert any_kernels, "There should be at least one layer with Gabor kernel to compare to!"
    how_many_kernels = sum([k is not None for k in kernels])

    plt.figure(figsize=(12, 5), dpi=110, facecolor='w') 
    for iteration, (idx, kernel) in enumerate(zip(indices, kernels)):
        if kernel is None:
            continue
        # # Get standard kernels
        # If the network is a vgg16
        if args.arch == 'vgg16':
            std_layer = std_model.module.net.features[idx]
        # If the network is a resnet18
        elif args.arch == 'resnet18':
            if iteration == 0:
                std_layer = std_model.module.net.conv1
            elif iteration == 1:
                std_layer = std_model.module.net.layer1[0].conv1
            elif iteration == 2:
                std_layer = std_model.module.net.layer1[0].conv2
        # If the network is a lenet
        elif args.arch == 'lenet':
            if iteration == 0:
                std_layer = std_model.module.conv1
            elif iteration == 1:
                std_layer = std_model.module.conv2
        # If the network is an alexnet
        elif args.arch == 'alexnet':
            if iteration == 0:
                std_layer = std_model.module.conv1[0]
            elif iteration == 1:
                std_layer = std_model.module.conv2[0]
            elif iteration == 3:
                std_layer = std_model.module.conv3[0]
        std_kernels = std_layer.weight

        # # Get Gabor kernels
        # If the network is a vgg16
        if args.arch == 'vgg16':
            gabor_layer = gabor_model.module.net.features[idx]
        # If the network is a resnet18
        elif args.arch == 'resnet18':
            if iteration == 0:
                gabor_layer = gabor_model.module.net.conv1
            elif iteration == 1:
                gabor_layer = gabor_model.module.net.layer1[0].conv1
            elif iteration == 2:
                gabor_layer = gabor_model.module.net.layer1[0].conv2
        # If the network is a lenet
        elif args.arch == 'lenet':
            if iteration == 0:
                gabor_layer = gabor_model.module.conv1
            elif iteration == 1:
                gabor_layer = gabor_model.module.conv2
        # If the network is an alexnet
        elif args.arch == 'alexnet':
            if iteration == 0:
                gabor_layer = gabor_model.module.conv1[0]
            elif iteration == 1:
                gabor_layer = gabor_model.module.conv2[0]
            elif iteration == 3:
                gabor_layer = gabor_model.module.conv3[0]
        gabor_kernels = gabor_layer.generate_gabor_kernels()

        # kernel is of size output-channels x input-channels x k x k
        # kernel should be a (k × k × m × m) tensor, according to:
        # The Singular Values of Convolutional Layers
        # https://openreview.net/pdf?id=rJevYoA9Fm
        # k x k x output-channels x input-channels
        # so we need to transpose things
        std_kernels = std_kernels.permute(2, 3, 0, 1)
        gabor_kernels = gabor_kernels.permute(2, 3, 0, 1)

        # Convert to numpy
        std_kernels = std_kernels.cpu().detach().numpy()
        gabor_kernels = gabor_kernels.cpu().detach().numpy()

        # Compute singular values
        std_singular_values = SingularValuesConv(
                    kernel=std_kernels, input_shape=INPUT_SHAPE).flatten()
        gabor_singular_values = SingularValuesSepConv(
                    kernel=gabor_kernels, input_shape=INPUT_SHAPE).flatten()

        std_max = std_singular_values.max()
        gabor_max = gabor_singular_values.max()
        if std_max > gabor_max:
            top_value = np.percentile(std_singular_values, q=PERCENTILE)
        else:
            top_value = np.percentile(gabor_singular_values, q=PERCENTILE)
        step = 1e-1
        top_value += step
        bins = np.arange(0, top_value, step)
        # Weights to have normalized histograms
        std_w = np.ones_like(std_singular_values)/float(len(std_singular_values))
        gabor_w = np.ones_like(gabor_singular_values)/float(len(gabor_singular_values))
        # Plot histogram of singular values
        plt.subplot(1, how_many_kernels, iteration+1)
        plt.hist(
            [std_singular_values, gabor_singular_values], 
            bins,
            label=[f'Standard - max={std_max:3.2f}', f'Gabor - max={gabor_max:3.2f}'], 
            weights=[std_w, gabor_w]
        )
        plt.xlim((0, top_value))
        plt.grid(True)
        plt.xlabel('Singular values', fontsize=18)
        if iteration == 0:
            plt.ylabel('Frequency', fontsize=18)
        plt.legend(loc='upper right', fontsize=14)
        conv_layer = iteration+1
        plt.title(f'Conv. layer {conv_layer}', fontsize=18)
        basename = args.gabor_ckpt.split('/')[-1]
        if basename == '':
            basename = args.gabor_ckpt.split('/')[-2]
        filename = osp.join(args.gabor_ckpt, f'{basename}_std_sing_vals_{args.arch}_{args.dataset}_conv{conv_layer}.npy')
        np.save(filename, std_singular_values)
        filename = osp.join(args.gabor_ckpt, f'{basename}_gabor_sing_vals_{args.arch}_{args.dataset}_conv{conv_layer}.npy')
        np.save(filename, gabor_singular_values)
    
    figname = osp.join(args.gabor_ckpt, 'singular_value_comp.eps')
    plt.tight_layout()
    plt.savefig(figname)




if __name__ == "__main__":
    dataset_names = ['cifar10', 'cifar100', 'mnist', 'fashion-mnist', 'imagenet', 'svhn']
    model_names = ['vgg16', 'resnet18', 'lenet', 'alexnet']
    parser = argparse.ArgumentParser(
        description='Singular Values computation for convolutional layer')
    parser.add_argument('--standard-ckpt', default=None, type=str, 
        metavar='PATH', help='path to standard checkpoint')
    parser.add_argument('--gabor-ckpt', required=True, type=str, 
        metavar='PATH', help='path to gabor checkpoint')
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
    main_svs_plotting(args)