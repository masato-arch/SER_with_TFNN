#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:28:37 2022

@author: Ark_001
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# =============================================================================
# User Interfaces:
#   train_epoch(): Method for 1 epoch training
#   valid(): Method for validation during training
#   display_log(): Method to display the log during training
#   classification_test(): Method for calculating confusion matrix and accuracy after training
# =============================================================================

def train_epoch(model, trainloader, criterion, optimizer, device='cpu', 
        epoch=None, measure_time=False):
    
    """Method to train the model for 1 epoch"""
    
    train_loss = 0.0
    running_loss = 0.0

    # if you want to measure time, start measuring here
    if measure_time:
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
        
        if count % 1000 == 0:
            print(f'#{epoch}, data:{count*4}, running_loss:{running_loss / 100:1.3f}')
            running_loss = 0.0
    
    # time measuring end
    if measure_time:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) / 1000
    else:
        elapsed_time = None
    
    # return train loss and elapsed time
    train_loss /= len(trainloader)
    return train_loss, elapsed_time

def valid(model, testloader, criterion, device='cpu', measure_time=False):
    
    """Method to test the model"""
    
    with torch.no_grad():
        total = 0
        correct = 0
        test_loss = 0

        # if time measurement is enabled, start here
        if measure_time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
        # forward processing
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # calculating the test loss
            test_loss += criterion(outputs, labels).item()
            # calculating the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
        
        # if time measurement is enabled, end here
        if measure_time:
            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000
        else:
            elapsed_time = None
        
        test_loss /= len(testloader)
        test_accuracy = correct / total

    return test_loss, test_accuracy, elapsed_time

def display_log(train_losses, test_losses, train_accuracy, test_accuracy, epoch=None):
    
    """Method to display the learning process"""
    
    print(f'plot_log: #{epoch}, train_accuracy:{train_accuracy[-1]}, test_accuracy:{test_accuracy[-1]}')
    plt.figure(figsize=(12, 4))
    
    # plot the loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend()
    
    # plot the accuracy curve１
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='train_accuracy')
    plt.plot(test_accuracy, label='test_accuracy')
    plt.legend()
    
    plt.show()

def classification_test(model, testloader, device='cpu', class_labels=None):
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
        
        accuracy = correct / total
        print(f'accuracy: {correct}/{total} = {accuracy}')
        cm = _get_confusion_matrix(true_labels, predictions, class_labels=class_labels)
        _show_confusion_matrix(cm)
        
    return accuracy, cm

"""
Following codes are for internal processings. You don't have to read.
"""

def _get_confusion_matrix(true_labels, predictions, class_labels=None):
    cm = confusion_matrix(true_labels, predictions)
    cm = pd.DataFrame(data=cm, index=class_labels, columns=class_labels)
    return cm

def _show_confusion_matrix(cm):
    ax = sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True labels')
    plt.show()