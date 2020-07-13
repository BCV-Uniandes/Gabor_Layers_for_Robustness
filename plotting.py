import os
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt

import pdb


def main_plotting(args):

    epsilons = [0, 2/255, 8/255, 16/255, .1]

    standard_path = osp.join(args.standard_ckpt, 'model_best.pth.tar')
    standard_attack_results = osp.join(args.standard_ckpt, 'attack_results.csv')    
    standard_df = pd.read_csv(standard_attack_results, 
        usecols=['test_set_accs', 'flip_rates'])

    
    gabor_path = osp.join(args.gabor_ckpt, 'model_best.pth.tar')
    gabor_attack_results = osp.join(args.gabor_ckpt, 'attack_results.csv')    
    gabor_df = pd.read_csv(gabor_attack_results, 
        usecols=['test_set_accs', 'flip_rates'])

    # Subplot for accuracies
    plt.subplot(1, 2, 1)
    plt.plot(epsilons, standard_df['test_set_accs'].values, label='standard')
    plt.plot(epsilons, gabor_df['test_set_accs'].values, label='gabor')
    plt.xlabel('Attack strength')
    plt.ylabel('Accuracy (larger is better)')
    plt.grid(True)

    # Subplot for flip rates
    plt.subplot(1, 2, 2)
    plt.plot(epsilons, standard_df['flip_rates'].values, label='standard')
    plt.plot(epsilons, gabor_df['flip_rates'].values, label='gabor')
    plt.xlabel('Attack strength')
    plt.ylabel('Flip rate (lower is better)')
    plt.grid(True)

    plt.legend()

    plt.suptitle(args.gabor_ckpt)
    plt.show()



if __name__ == "__main__":
    dataset_names = ['cifar10', 'cifar100', 'mnist', 'fashion-mnist']
    model_names = ['vgg16', 'resnet18']
    parser = argparse.ArgumentParser(
        description='Singular Values computation for convolutional layer')
    parser.add_argument('--standard-ckpt', required=True, type=str, 
        metavar='PATH', help='path to standard checkpoint')
    parser.add_argument('--gabor-ckpt', required=True, type=str, 
        metavar='PATH', help='path to gabor checkpoint')
    args = parser.parse_args()
    main_plotting(args)


""" import argparse
import numpy as np
import matplotlib.pyplot as plt

DATASET = 'mnist'

if DATASET == 'cifar100':
    epsilons = np.array([0, 2/255, 8/255, 16/255, 0.1])
    log_scale = False
    # Accuracies
    std_acc = np.array([67.54, 27.22, 18.46, 10.49, 6.51])
    gab_acc = np.array([64.49, 31.12, 25.82, 15.4, 8.7])
    gab_reg_acc = np.array([64.52, 31.68, 26.64, 16.06, 9.09])

    # Flip rates
    std_fr = np.array([0.0, 57.05, 77.94, 90.48, 95.98])
    gab_fr = np.array([0.0, 50.94, 68.95, 85.75, 93.51])
    gab_reg_fr = np.array([0.0, 51.05, 68.45, 85.16, 93.36])

elif DATASET == ' cifar10':
    epsilons = np.array([0, 2/255, 8/255, 16/255, 0.1])
    log_scale = False
    # Accuracies 
    std_acc = np.array([92.03, 34.22, 23.63, 13.88, 6.8])
    gab_acc = np.array([91.35, 37.6, 30.11, 19.5, 8.91])
    # gab_reg_acc = np.array([64.52, 31.68, 26.64, 16.06, 9.09])

    # Flip rates 
    std_fr = np.array([0.0, 60.03, 74.51, 86.47, 94.13]) 
    gab_fr = np.array([0.0, 56.42, 67.83, 80.84, 92.34])
    # gab_reg_fr = np.array([0.0, 51.05, 68.45, 85.16, 93.36])

elif DATASET == 'mnist':
    epsilons = np.array([0, .1, .2, .3, .4])
    log_scale = False
    # Accuracies
    std_acc = np.array([99.36, 83.04, 4.38, 0.34, 0.36])
    gab_acc = np.array([99.03, 80.58, 7.94, 0.78, 0.56])
    gab_reg_acc = np.array([98.75, 88.47, 21.9, 0.69, 0.75])

    # Flip rates
    std_fr = np.array([0.0, 16.62, 95.48, 99.75, 99.82])
    gab_fr = np.array([0.0, 18.88, 91.85, 99.24, 99.75])
    gab_reg_fr = np.array([0.0, 11.03, 78.12, 99.6, 99.77])

# Robustness measure
std_rob = 100 - std_fr
gab_rob = 100 - gab_fr
gab_reg_rob = 100 - gab_reg_fr

plt.figure(figsize=(12, 5), dpi=110, facecolor='w') 
plt.subplot(1, 2, 1)
plt.plot(epsilons, std_acc, label='Std.')
plt.plot(epsilons, gab_acc, label='Gab.')
plt.plot(epsilons, gab_reg_acc, label='Gab.+reg.')
if log_scale:
    plt.yscale('log')
plt.grid(True)
plt.xlabel(r'$\epsilon$')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(std_rob[1:], std_acc[1:], label='Std.')
plt.plot(gab_rob[1:], gab_acc[1:], label='Gab.')
plt.plot(gab_reg_rob[1:], gab_reg_acc[1:], label='Gab.+reg.')
if log_scale:
    plt.yscale('log')
plt.grid(True)
plt.xlabel('Robustness')
plt.legend()

plt.tight_layout()
plt.savefig(f'robust_plot_{DATASET}_normal.png')
 """
