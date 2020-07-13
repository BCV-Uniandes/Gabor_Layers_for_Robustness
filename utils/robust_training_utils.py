import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time

import pdb

def update_and_clamp(img, grad, delta, alpha, epsilon, llimit, ulimit):
    for i,_ in enumerate(epsilon): #updating the adversary
        delta.data[:,i,:,:] = delta.data[:,i,:,:] + alpha[i]*torch.sign(grad)[:,i,:,:]
        delta.data[:,i,:,:].clamp_(-epsilon[i].item(), epsilon[i].item())
        delta.data[:,i,:,:] = torch.min(torch.max(delta.data[:,i,:,:],llimit[i] - img[:,i,:,:]), ulimit[i] - img[:,i,:,:])
    return delta 

def train(model, trainloader, criterion, optimizer,
        device = 'cpu', robust = False):
    model.to(device)        
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    # bar = Bar('Processing', max = len(trainloader))
    model.train()
    for img,label in tqdm(trainloader):
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = criterion(output, label)

        #Logging Data:
        acc = accuracy(output.data, label.data)
        train_loss.update(loss.item(), img.size(0))
        train_acc.update(acc.item(), img.size(0))
        
        #SGD Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, train_loss.avg, train_acc.avg


def train_fast(model, trainloader, criterion, optimizer,scheduler,
        device = 'cpu', delta = 0, alpha = 0, init = 'random',
         epsilon = 0, llimit = 0, ulimit = 0):
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    model.train()
    for img,label in tqdm(trainloader):
        img, label = img.to(device), label.to(device)
        delta.requires_grad = True
        if init == 'random': #initializing the adversary
            for i,_ in enumerate(epsilon):
                delta[:, i, :, :].data.uniform_(-epsilon[i].item(), epsilon[i].item())
                delta[:,i,:,:].data = torch.max(torch.min(delta[:,i,:,:], ulimit[i] - img[:,i,:,:]), llimit[i] - img[:,i,:,:])
        #one time for updating the adversary.
        output = model(img + delta)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        grad = delta.grad.detach()
        delta = update_and_clamp(img, grad, delta, alpha, epsilon, llimit, ulimit)

        #another time for updating the model weights.
        delta = delta.detach()
        output = model(img + delta)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        acc = accuracy(output.data, label.data)
        train_loss.update(loss.item(), img.size(0))
        train_acc.update(acc.item(), img.size(0))
        
    return model, train_loss.avg, train_acc.avg, delta

def evaluate(model, testloader, criterion, device = 'cpu'):
    #model.to(device)
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    for img, label in tqdm(testloader):
        
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            output = model(img)
        loss = criterion(output, label)
        acc = accuracy(output.data, label.data)
        test_acc.update(acc)
        test_loss.update(loss.item())

    return test_loss.avg, test_acc.avg

def attack_pgd(model, testloader, device, minimum, maximum, eps=0.1, n_rand_restarts=10, alpha = 2/255):
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
                eps=eps, minimum=minimum, maximum=maximum, alpha = alpha
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

def get_rand_perturb(images, eps):
    # Between 0 and 1
    pert = torch.rand_like(images)
    # Now between -eps and +eps
    pert = 2*eps*pert - eps
    return pert

def channelwise_clamp(images, minima, maxima):
    # torch.clamp does not allow the limits to be 
    # tensors, but torch.min and max DOES!
    images = torch.max(torch.min(images, maxima), minima)
    return images

def pgd_attack(model, images, labels, orig_preds, eps, minimum, maximum,
    alpha, iterations=50, rand=True, training=False):
    loss = nn.CrossEntropyLoss()
    # Unsqueeze alpha, eps, minima and maxima so that we don't need an extra for-loop
    # (and move it to appropriate device)
    alphas = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(images.device)
    minima = minimum.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(images.device)
    maxima = maximum.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(images.device)
    epss = eps.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(images.device)
    # Assume the image is a tensor
    if rand:
        perturbed_images = images.data + get_rand_perturb(images, epss)
        perturbed_images = channelwise_clamp(perturbed_images, 
            minima=minima, maxima=maxima).data.clone()
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
            # perturbed_images[inds_ims_left] += alpha*eta
            perturbed_images[inds_ims_left] += alphas*eta
            # Project to noise within epsilon ball around original images
            noise = channelwise_clamp(
                perturbed_images[inds_ims_left] - images[inds_ims_left], 
                minima=-epss, maxima=epss
            )
            # Project to images within space of possible images
            perturbed_images[inds_ims_left] = channelwise_clamp(
                images[inds_ims_left] + noise, 
                minima=minima, maxima=maxima
            )

    return perturbed_images

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = torch.sum(correct).type(torch.DoubleTensor)/batch_size

    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cluster_test(model, dataloader, activation_number, device = 'cuda'):
    model.eval()
    model.to(device)
    activation_values = []
    list_of_labels = []
    for img,label in tqdm(dataloader):
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            output = model.modified_forward(img, activation_number)
        activation_values += np.array(output.to('cpu').detach()).tolist()
        list_of_labels += np.array(label.to('cpu').detach()).tolist()
    return activation_values, list_of_labels
