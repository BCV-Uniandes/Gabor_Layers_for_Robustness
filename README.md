# Gabor Layers Enhance Network Robustness
This repository provides the implementation for the paper "Gabor Layers Enhance Network Robustness", to be published at the European Conference on Computer Vision (ECCV) 2020. The arXiv version of the paper is available [here](https://arxiv.org/abs/1912.05661).

In this work, we revisit the benefits of merging classical vision concepts with deep learning models. In particular, we explore the effect on robustness against adversarial attacks of replacing the first layers of various deep architectures with Gabor layers, _i.e._ convolutional layers with filters that are based on *learnable* Gabor parameters. We observe that architectures enhanced with Gabor layers gain a consistent boost in robustness over regular models and preserve high generalizing test performance. We then exploit the closed form expression of Gabor filters to derive an expression for a Lipschitz constant of such filters, and harness this theoretical result to develop a regularizer we use during training to further enhance network robustness. We conduct extensive experiments with various architectures (LeNet, AlexNet, VGG16 and WideResNet) on several datasets (MNIST, SVHN, CIFAR10 and CIFAR100) and demonstrate large empirical robustness gains. Furthermore, we experimentally show how our regularizer provides consistent robustness improvements.

In summary, here we provide the implementation for Gabor layers: convolutional layers whose filters are constrained to be the result of evaluating Gabor functions on a 2-dimensional grid. Gabor functions are parameterized functions of the form: <img src="https://render.githubusercontent.com/render/math?math=G_{\theta}(x',y';\sigma,\gamma,\lambda,\psi):=e^{-\:(x'^2+\gamma^2\:y'^2)/\sigma^2}\:\cos(\lambda\:x'+\psi)">, where <img src="https://render.githubusercontent.com/render/math?math=x'=x\cos\theta-y\sin\theta"> and <img src="https://render.githubusercontent.com/render/math?math=y'=x\sin\theta+y\cos\theta">. In a Gabor layer, the parameters <img src="https://render.githubusercontent.com/render/math?math=\sigma">, <img src="https://render.githubusercontent.com/render/math?math=\gamma">, <img src="https://render.githubusercontent.com/render/math?math=\lambda">, and <img src="https://render.githubusercontent.com/render/math?math=\psi"> are *learnable*, _i.e._ they are adjusted during training according to backpropagation.

Since all Gabor filters are single-channeled (contrary to multiple-channeled filters of standard convolutional layers), the Gabor layer learns a set of Gabor filters and performs separable convolution on its input. Each of the resulting responses of this operation is then passed through ReLU, and then a <img src="https://render.githubusercontent.com/render/math?math=1\times1"> convolution is applied to obtain the desired amount of channels. Please refer to Figure 2 in the [paper](https://arxiv.org/abs/1912.05661) for a graphical guide on this operation. Note that the purpose of the <img src="https://render.githubusercontent.com/render/math?math=1\times1"> convolution is to allow for straightforward substitution of arbitrary convolutional layers with Gabor layers.

# Disclaimer regarding code
This repository was built on top of [this public repository](https://github.com/bearpaw/pytorch-classification), which implements standard training of Convolutional Neural Networks on CIFAR10/100. We thank [Wei Yang](https://github.com/bearpaw) for making this code publicly available.

# Dependencies
This implementation requires [PyTorch](https://pytorch.org/).

# Code structure
The main script of this repository is `main.py`. Running this script trains a Convolutional Neural Network on a given dataset, with the option of substituting traditional convolutional layers with Gabor layers.

# Citation
```
@article{perez2020gabor,
    title={Gabor Layers Enhance Network Robustness},
    author={Juan C. {P{\'e}rez} and Motasem Alfarra and Guillaume Jeanneret and Adel Bibi and Ali Thabet and Bernard Ghanem and Pablo {Arbel{\'a}ez}},
    year={2020},
    journal={arXiv:1912.05661},
}
```
