# Gabor Layers Enhance Network Robustness
This repository provides the implementation for the paper "Gabor Layers Enhance Network Robustness", to be published at the European Conference on Computer Vision (ECCV) 2020. The arXiv version of the paper is available [here](https://arxiv.org/abs/1912.05661).

*Abstract:* In this work, we revisit the benefits of merging classical vision concepts with deep learning models. In particular, we explore the effect on robustness against adversarial attacks of replacing the first layers of various deep architectures with Gabor layers, _i.e._ convolutional layers with filters that are based on *learnable* Gabor parameters. We observe that architectures enhanced with Gabor layers gain a consistent boost in robustness over regular models and preserve high generalizing test performance. We then exploit the closed form expression of Gabor filters to derive an expression for a Lipschitz constant of such filters, and harness this theoretical result to develop a regularizer we use during training to further enhance network robustness. We conduct extensive experiments with various architectures (LeNet, AlexNet, VGG16 and WideResNet) on several datasets (MNIST, SVHN, CIFAR10 and CIFAR100) and demonstrate large empirical robustness gains. Furthermore, we experimentally show how our regularizer provides consistent robustness improvements.

In summary, here we provide the implementation for Gabor layers: convolutional layers whose filters are constrained to be the result of evaluating Gabor functions on a 2-dimensional grid. Gabor functions are parameterized functions of the form: <img src="https://render.githubusercontent.com/render/math?math=G_{\theta}(x',y';\sigma,\gamma,\lambda,\psi):=e^{-\:(x'^2+\gamma^2\:y'^2)/\sigma^2}\:\cos(\lambda\:x'+\psi)">, where <img src="https://render.githubusercontent.com/render/math?math=x'=x\cos\theta-y\sin\theta"> and <img src="https://render.githubusercontent.com/render/math?math=y'=x\sin\theta+y\cos\theta">. In a Gabor layer, the parameters <img src="https://render.githubusercontent.com/render/math?math=\sigma">, <img src="https://render.githubusercontent.com/render/math?math=\gamma">, <img src="https://render.githubusercontent.com/render/math?math=\lambda">, and <img src="https://render.githubusercontent.com/render/math?math=\psi"> are *learnable*, _i.e._ they are adjusted during training according to backpropagation.

Since all Gabor filters are single-channeled (contrary to multiple-channeled filters of standard convolutional layers), the Gabor layer learns a set of Gabor filters and performs separable convolution on its input. Each of the resulting responses of this operation is then passed through ReLU, and then a <img src="https://render.githubusercontent.com/render/math?math=1\times1"> convolution is applied to obtain the desired amount of channels. Please refer to Figure 2 in the [paper](https://arxiv.org/abs/1912.05661) for a graphical guide on this operation. Note that the purpose of the <img src="https://render.githubusercontent.com/render/math?math=1\times1"> convolution is to allow for straightforward substitution of arbitrary convolutional layers with Gabor layers.

# Disclaimer regarding code
This repository was built on top of [this public repository](https://github.com/bearpaw/pytorch-classification), which implements standard training of Convolutional Neural Networks on CIFAR10/100. We thank [Wei Yang](https://github.com/bearpaw) for making this code publicly available.

# Dependencies
This implementation requires [PyTorch](https://pytorch.org/).

# Code structure
The main script of this repository is `main.py`. Running this script trains a Convolutional Neural Network on a given dataset, with the option of substituting traditional convolutional layers with Gabor layers. This script provides several arguments for controlling parameters related to the training procedure and the architecture of the ConvNet. Run `python main.py --help` for a detailed explanation of all the available parameters. The script `main_attack.py` is concerned with conducting (PGD) adversarial attacks on a given neural network. The directory `models` contains the code for several Convolutional Neural Network architectures (AlexNet, LeNet, ResNet, VGG, and WideResNet). The directory `utils` contains the utilities used when training models.

# Gabor layers
The script `models/gabor_layers.py` contains the `GaborLayer` class. This class implements the Gabor layers proposed in the paper. Next, we provide a short example of how this class can be used for replacing standard convolutional layers with Gabor layers.

Say you have a convolutional layer defined as
```python
from torch.nn import Conv2d
conv = Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=1)
```
You can replace this layer by a Gabor layer like this:
```python
from models.gabor_layers import GaborLayer
conv = GaborLayer(in_channels=1, out_channels=6, kernel_size=5, padding=1, kernels=1)
```
The `kernels` argument controls the number of kernels in that Gabor Layer, that is, the number of distinct Gabor filters that are learnt. You can also change the number of orientations (uniformly-spaced rotations between between 0 and <img src="https://render.githubusercontent.com/render/math?math=2\pi"> for each filter in the Gabor Layer) by setting the `orientations` argument.

# Training Gabor-layered architectures
Next, we describe the arguments related to training Gabor-layered architectures:
* `--kernelsX`, where `X` is either 1, 2 or 3. These arguments control the number of different Gabor filters that are learnt (<img src="https://render.githubusercontent.com/render/math?math=\sigma">, <img src="https://render.githubusercontent.com/render/math?math=\gamma">, <img src="https://render.githubusercontent.com/render/math?math=\lambda">, and <img src="https://render.githubusercontent.com/render/math?math=\psi">) at each layer, _i.e._ these arguments are equivalent to <img src="https://render.githubusercontent.com/render/math?math=p"> in the notation of the paper. For instance, running `python main.py --kernels1 Y --kernels3 Z --checkpoint expA`, where `Y` and `Z` are numbers, will run training on a ConvNet in which the first convolutional layer (hence the `--kernels1` argument) was replaced with a Gabor layer with `Y` sets of parameters, and the third convolutional layer was replaced with a Gabor layer with `Z` sets of parameters. The results of the experiment will be saved at directory `expA`.

* `--orientations` controls the number of orientations (uniformly spaced between 0 and <img src="https://render.githubusercontent.com/render/math?math=2\pi">) for which each Gabor filter is rotated.

* `--lambd` controls the degree of the penalty for regularizing the Gabor layers' Lipschitz constant, according to Theorem 1 in the paper. This argument corresponds to <img src="https://render.githubusercontent.com/render/math?math=\beta"> in the paper's notation.

# Output files
When running a training procedure (_e.g._, running `python main.py --checkpoint expA --kernels1 3`), several outputs are generated under the directory `expA`. Next, we describe each of these outputs:
* `checkpoint.pth.tar`. File containing the state dictionary of both the *last* model and the corresponding optimizer, the epoch in which the model was saved, the validation accuracy of the model, and the best validation accuracy that the training procedure had obtained at the moment this file was created.

* `model_best.pth.tar`. File with the same information as `checkpoint.pth.tar`, but the model is the on that had the best validation accuracy of the training procedure.

* `params.txt`. Text file stating the values for all the arguments passed to `main.py`.

* `log.txt`. Text file in which the logging of the training was saved. It contains information regarding Learning Rate, Train Loss, Validation Loss, Train Accuracy, and Validation Accuracy for each epoch.

* Files of the form `kernels_epoch_X.pdf` where `X` is a number from 0 to the number of epochs for which training was run. This file is a PDF depicting the filters that were learnt in the *first* Gabor layer of the model. It shows all the rotation of each filter in such layer. The filters shown are those that the Gabor layer has by the end of epoch `X`.


# Citation
```
@article{perez2020gabor,
    title={Gabor Layers Enhance Network Robustness},
    author={Juan C. {P{\'e}rez} and Motasem Alfarra and Guillaume Jeanneret and Adel Bibi and Ali Thabet and Bernard Ghanem and Pablo {Arbel{\'a}ez}},
    year={2020},
    journal={arXiv:1912.05661},
}
```
