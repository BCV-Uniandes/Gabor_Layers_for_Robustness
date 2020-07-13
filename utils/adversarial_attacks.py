import torch
import numpy as np
from torch import nn
from tqdm import tqdm

import pdb

def cw_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
    # Define f-function
    def f(x):
        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(outputs.device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())       
        # If targeted, optimize for making the other class most likely 
        if targeted:
            return torch.clamp(i-j, min=-kappa)       
        # If untargeted, optimize for making the other class most likely 
        else:
            return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True)
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    prev = 1e10
    mse_loss = nn.MSELoss(reduction='sum')
    for step in range(max_iter):
        a = 1/2*(torch.tanh(w) + 1)
        loss1 = mse_loss(a, images)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0:
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        #print(f'Learning Progress : {(step+1)/max_iter*100}')
    attack_images = 1/2*(torch.tanh(w) + 1)
    return attack_images


def attack_cw(model, testloader, device, c=0.1):
    print('Attack Image & Predicted Label')
    model.eval()
    correct, total = 0, 0
    flipped_decisions, cum_l2_distort = 0, 0
    mse_loss = nn.MSELoss(reduction='sum')
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)
        adv_images = cw_l2_attack(model, images, labels, targeted=False, c=c)
        # Original predictions
        orig_outputs = model(images)
        _, orig_pre = torch.max(orig_outputs.data, 1)

        # Predictions of adversarial images
        outputs = model(adv_images)
        _, pre = torch.max(outputs.data, 1)

        # Check flipped decisions
        flipped_idxs = orig_pre != pre
        flipped_decisions += flipped_idxs.sum()
        flipped_images = images[flipped_idxs]
        flipped_adv_images = adv_images[flipped_idxs]

        # Compute L2 distortion of adversarial images and accumulate
        l2_distort = mse_loss(flipped_images, flipped_adv_images)
        cum_l2_distort += l2_distort

        total += labels.size(0) # batch_size
        correct += (pre == labels).sum()
        
    print('Accuracy of test set: %f %%' % (100 * float(correct) / total))
    print(f'Average L2 distortion on adversarial images: {l2_distort / flipped_decisions}')
    print(f'{flipped_decisions} examples out of {total} were flipped: ' +
        f'{flipped_decisions}/{total} = {100 * float(flipped_decisions)/total}%')




def pgd_attack(model, images, labels, orig_preds, alpha=8/255, eps=0.1, 
        minimum=0, maximum=1, iterations=200, rand=True, training=False):
    loss = nn.CrossEntropyLoss()
    # Assume the image is a tensor
    if rand:
        pert = 2*eps*torch.rand_like(images) - eps
        perturbed_images = images.data + pert
        perturbed_images = torch.clamp(perturbed_images, min=minimum, max=maximum).data.clone()
    else:
        perturbed_images = images.data.clone()

    # Check for which images was the attack already succesful
    inds_ims_left = torch.arange(perturbed_images.size(0))

    for _ in range(iterations): 
        if training:
            inds_ims_left = torch.arange(perturbed_images.size(0))
        # Gradient for the image
        perturbed_images.requires_grad = True
        # Compute forward
        outputs = model(perturbed_images[inds_ims_left])
        # Gradient to zero
        model.zero_grad()
        # Compute label predictions
        _, perturbed_preds = torch.max(outputs.data, 1)
        # Check where the attack was not successful
        where_not_success = orig_preds[inds_ims_left] == perturbed_preds
        # Remove image indices for which there's already an adversarial example
        inds_ims_left = inds_ims_left[where_not_success]
        if inds_ims_left.size(0) == 0:
            break
        # Backward
        cost = loss(outputs[where_not_success], labels[inds_ims_left])
        cost.backward()
        with torch.no_grad():
            # Sign of gradient times 'learning rate'
            eta = perturbed_images.grad[inds_ims_left].sign()
            perturbed_images[inds_ims_left] += alpha*eta
            # Project to noise within epsilon ball aroung original images
            noise = torch.clamp(
                perturbed_images[inds_ims_left] - images[inds_ims_left], 
                min=-eps, max=eps
            )
            # Project to images within space of possible images
            perturbed_images[inds_ims_left] = torch.clamp(
                images[inds_ims_left] + noise, 
                min=minimum, max=maximum
            )

    return perturbed_images


def attack_pgd(model, testloader, device, minimum, maximum, eps=0.1, n_rand_restarts=10):
    model.eval()
    correct, total, flipped_decisions = 0, 0, 0
    # # Attack parameters
    # Upper_Bound on the infinity norm of the perturbation. \|\delta\|_{infty} < eps 
    # eps = 0.1    
    # Lower bound of the input value, i.e. 0 in the case if we want the image between [0,1]
    # minimum = 0      
    # Upper bound of the input value, i.e  1 in the case if we want the image between [0,1]
    # maximum = 1      
    # You can put train_loader as well
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)
        total += images.size(0) # batch_size
        # Original predictions
        orig_outputs = model(images)
        _, orig_preds = torch.max(orig_outputs.data, 1)
        # Compute adversarial examples
        final_adv_preds = orig_preds.clone()
        inds_ims_left = torch.arange(final_adv_preds.size(0))
        for _ in range(n_rand_restarts):
            curr_ims = images[inds_ims_left]
            curr_labels = labels[inds_ims_left]
            curr_orig_preds = orig_preds[inds_ims_left]
            # Compute adversarial examples for images that are left
            adv_images = pgd_attack(
                model, curr_ims, curr_labels, orig_preds=curr_orig_preds,
                eps=eps, minimum=minimum, maximum=maximum
            )
            # Compute prediction on adversarial examples
            outputs = model(adv_images)
            _, adv_preds = torch.max(outputs.data, 1)
            # Check instances in which attack was succesful
            where_success = curr_orig_preds != adv_preds
            num_inds_where_success = inds_ims_left[where_success]
            # Replace predictions where attack was succesful
            final_adv_preds[num_inds_where_success] = adv_preds[where_success]
            # Remove image indices for which an adversarial example was found
            inds_ims_left = inds_ims_left[~where_success]
            # Check if there are no images left
            if inds_ims_left.size(0) == 0:
                break

        # Compare original predictions with predictions from adv examples
        flipped_idxs = orig_preds != final_adv_preds
        flipped_decisions += flipped_idxs.sum()

        correct += (final_adv_preds == labels).sum()

    acc_test_set = 100 * float(correct) / total
    flip_rate = 100 * float(flipped_decisions) / total
    print(f'Accuracy of test set: {acc_test_set:5.4f}')
    print(f'{flipped_decisions} examples out of {total} were flipped: ' +
        f'{flipped_decisions}/{total} = {flip_rate:5.4f}%')
    return acc_test_set, flip_rate

