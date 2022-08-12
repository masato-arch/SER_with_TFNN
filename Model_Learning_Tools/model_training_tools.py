#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:28:37 2022

@author: Ark_001
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np

# =============================================================================
# User Interfaces:
#   train_epoch(): Method for 1 epoch training
#   valid(): Method for validation during training
#   valid_multi_dataloaders() validation during training for multiple valid dataloaders
#   display_log(): Method to display the log during training
#   get_pred_true(): Method to get predicted and true labels to get confusion matrix and accuracy
#   get_confusion_matrix(): Method to calculate confusion matrix
#   accuracy_score(): from sklearn.metrics
#   balanced_accuracy_score(): from sklearn.metrics
# =============================================================================

def train_epoch(model, trainloader, criterion, optimizer, device='cpu', 
        epoch=None, measure_gpu_time=False):
    
    """Method to train the model for 1 epoch"""
    
    # initialize the model
    model.train() # training mode
    train_loss = 0.0 # total train loss of the epoch
    running_loss = 0.0 # temporal loss during training

    # if GPU time measurement is enabled, start measuring here
    if measure_gpu_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
    # forward and backward processing
    for count, item in enumerate(trainloader):
        inputs, labels = item
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        running_loss += loss.item()
        # output monitor for debugging
        
        if count % 100 == 0:
            print(f'#{epoch}, data:{count*4}, running_loss:{running_loss / 100:1.3f}')
            running_loss = 0.0
    
    # time measuring end
    if measure_gpu_time:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000
    else:
        elapsed_time = None
    
    # return train loss and elapsed time
    train_loss /= len(trainloader)
    return train_loss, elapsed_time
        
def valid(model, validloader, criterion, device='cpu'):
    
    """Method to validate the model"""
    # initilaize the model
    model.eval() # evaluation (validation) mode
    
    with torch.no_grad():
        total = 0 # total number of datas
        correct = 0 # total number of correct answers
        valid_loss = 0 # total valid loss
            
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward processing
            outputs = model(inputs)
            
            # calculate the valid loss
            valid_loss += criterion(outputs, labels).item()
            
            # calculate the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
        
        valid_loss /= len(validloader)
        valid_accuracy = correct / total

    return valid_loss, valid_accuracy

def valid_multi_dataloaders(model, validloaders, criterion, device='cpu'):
    
    """Method for validation with multiple dataloaders"""
    # =============================================================================
    # In validation dataset, the number of speech segments is different by files. 
    # Therefore we need multiple dataloaders.
    # =============================================================================
    with torch.no_grad():
        # total dataloader length for calculating mean loss
        dataloader_len = sum([len(loader) for loader in validloaders])
        model.eval()
        
        corrects = 0
        total = 0
        valid_loss = 0.0
        
        # calculate total loss, number of correct answers and number of datas over all dataloaders
        for loader in validloaders:
            v_loss, corrects_, total_ = _half_valid(model, loader, criterion, device=device)
            valid_loss += v_loss
            corrects += corrects_
            total += total_
        
        # calculate means of them
        valid_loss /= dataloader_len
        valid_accuracy = corrects / total
    
    return valid_loss, valid_accuracy

def display_curves(train_loss_log, valid_loss_log, train_accuracy_log, valid_accuracy_log):
    
    """Method to display learning and accuracy curves"""
    
    fig = plt.figure(figsize=(12, 4))
    
    # add subplots
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_acc = fig.add_subplot(1, 2, 2)
    
    # set the axis range
    x = np.arange(1, len(train_loss_log) + 1)
    
    # set titles
    ax_loss.set_title('Loss Curve')
    ax_acc.set_title('Accuracy Curve')
    
    # set axis labels
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    
    # plot loss and accuracy logs
    ax_loss.plot(x, train_loss_log, label='Train Loss')
    ax_loss.plot(x, valid_loss_log, label='Validation Loss')
    ax_loss.legend()
    ax_acc.plot(x, train_accuracy_log, label='Train Accuracy')
    ax_acc.plot(x, valid_accuracy_log, label='Validation Accuracy')
    ax_acc.legend()
    
    plt.tight_layout()
    plt.show()

def get_pred_true(model, testloader, device='cpu', class_labels=None):
    
    """Method to get predicted labels and true labels"""
    
    with torch.no_grad():
        # variables to get accuracy
        total = 0
        correct = 0
        
        # variables to get confusion matrix
        predictions = []
        true_labels = []
        
        for data in testloader:
            inputs, labels = data
            
            # record the true labels to calculate the confusion matrix later
            true_labels.extend(list(labels.numpy())) 
            
            # forward processings
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            
            # record the number of correct answers
            total += len(outputs)
            correct += (prediction==labels).sum().item()
            
            # record the predicted labels
            predictions.extend(list(prediction.to('cpu').numpy()))
        
    return predictions, true_labels

def get_pred_true_multi_dataloaders(model, testloaders, device='cpu', class_labels=None):
    with torch.no_grad():
        # temporal variables
        total = 0
        correct = 0
        
        # variables to return
        predictions = []
        true_labels = []
        
        for loader in testloaders:
            
            for data in loader:
                inputs, labels = data
                
                # record the true labels to calculate the confusion matrix later
                true_labels.extend(list(labels.numpy())) 
                
                # forward processings
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, prediction = torch.max(outputs, 1)
                
                # record the number of correct answers
                total += len(outputs)
                correct += (prediction==labels).sum().item()
                
                # record the predicted labels
                predictions.extend(list(prediction.to('cpu').numpy()))
        
    return predictions, true_labels

def get_confusion_matrix(predictions, true_labels, class_labels=None, normalize=False):
    cm = pd.DataFrame(confusion_matrix(true_labels, predictions), index=class_labels, columns=class_labels)
    if normalize:
        cm = cm.apply(lambda x: x / sum(x), axis=1)
    return cm

def show_confusion_matrix(cm):
    ax = sns.heatmap(cm, square=True, cbar=True, annot=True, fmt='.3f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True labels')
    plt.show()

"""
Following codes are for internal processings. You don't have to read.
"""
    
def _half_valid(model, validloader, criterion, device='cpu'):
    # =============================================================================
    # Valid method for valid_multi_dataloaders() method.
    # 'half' means that the loss and accuracy aren't be calculated completely.
    # Returns the total loss, the total number of data, 
    #   and the number of correct answers in a single dataloader.
    # =============================================================================
    with torch.no_grad():
        total = 0
        correct = 0
        valid_loss = 0
            
        # forward processing
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # calculating the valid loss
            valid_loss += criterion(outputs, labels).item()
            # calculating the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()

    return valid_loss, correct, total