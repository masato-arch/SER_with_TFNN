#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:33:22 2022

@author: Ark_001
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

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

def get_pred_true_md(model, testloaders, device='cpu', class_labels=None):
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

def get_cm(y_pred, y_true, class_names=None, normalize=False):
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=class_names, columns=class_names)
    if normalize:
        cm = cm.apply(lambda x: x / sum(x), axis=1)
    return cm

def show_confusion_matrix(cm, fmt='.3f'):
    ax = sns.heatmap(cm, square=True, annot=True, fmt=fmt, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True labels')
    plt.show()