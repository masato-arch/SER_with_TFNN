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
#   train_epoch_tm(): Method for 1 epoch training with GPU time measurement
#   valid(): Method for validation during training
#   valid_multi_dataloaders() validation during training for multiple valid dataloaders
#   display_log(): Method to display the log during training
#   get_pred_true(): Method to get predicted and true labels to get confusion matrix and accuracy
#   get_confusion_matrix(): Method to calculate confusion matrix
#   show_confusion_matrix(): Method to show confusion matrix
#   accuracy_score(): from sklearn.metrics
#   balanced_accuracy_score(): from sklearn.metrics
# =============================================================================

def train_epoch(model, trainloader, criterion, optimizer, device='cpu', epoch=None):
    
    """Method to train the model for 1 epoch"""
    
    # initialize the model
    model.train() # training mode
    train_loss = 0.0 # total train loss of the epoch
    running_loss = 0.0 # temporal loss during training
    
    # forward and backward processing
    for count, item in enumerate(trainloader):
        inputs, labels = item
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # forward processing
        outputs = model(inputs)
        
        # backward processing
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        running_loss += loss.item()
        
        if count % 100 == 0:
            print(f'#{epoch}, data:{count*4}, running_loss:{running_loss / 100:1.3f}')
            running_loss = 0.0
    
    # return train loss and elapsed time
    train_loss /= len(trainloader)
    return train_loss

def train_epoch_tm(model, trainloader, criterion, optimizer, device='cpu', 
        epoch=None):
    
    """Method to measure GPU time to train 1 epoch"""
    
    # initialize the model
    model.train() # training mode
    train_loss = 0.0 # total train loss of the epoch
    running_loss = 0.0 # temporal loss during training
    fw_elapsed_time = []
    bw_elapsed_time = []

    # Event object to measure GPU time
    
    # for forward processing
    fw_start = torch.cuda.Event(enable_timing=True)
    fw_end = torch.cuda.Event(enable_timing=True)
    
    # for backward processing
    bw_start = torch.cuda.Event(enable_timing=True)
    bw_end = torch.cuda.Event(enable_timing=True)
    
    # forward and backward processing
    for count, item in enumerate(trainloader):
        inputs, labels = item
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        fw_start.record() # start measurement of forward here
        outputs = model(inputs)
        fw_end.record() # end measurement 
        torch.cuda.synchronize() # wait until the processing on GPU ends
        fw_elapsed_time.append(fw_start.elapsed_time(fw_end) / 1000)
        
        bw_start.record() # start measurement of backward here
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        bw_end.record() # end measurement
        torch.cuda.synchronize()
        bw_elapsed_time.append(bw_start.elapsed_time(bw_end) / 1000)
        
        train_loss += loss.item()
        running_loss += loss.item()
        
        if count % 100 == 0:
            print(f'#{epoch}, data:{count * trainloader.batch_size}, running_loss:{running_loss / 100:1.3f}')
            running_loss = 0.0
    
    # return train loss and elapsed time
    train_loss /= len(trainloader)
    
    print(f'forward elapsed time len {len(fw_elapsed_time)}')
    print(f'backward elapsed time len {len(bw_elapsed_time)}')
    print(fw_elapsed_time)
    mean_fw_time = np.mean(np.array(fw_elapsed_time))
    mean_bw_time = np.mean(np.array(bw_elapsed_time))
    return train_loss, mean_fw_time, mean_bw_time

def train_epoch_layerwise_tm(model, trainloader, criterion, optimizer, device='cpu', 
        epoch=None):
    
    """Method to measure layerwise time consumption"""
    
    # NOTE: the model must return layerwise time consumption
    
    # initialize the model
    model.train() # training mode
    train_loss = 0.0 # total train loss of the epoch
    running_loss = 0.0 # temporal loss during training
    times = []
    
    # forward and backward processing
    for count, item in enumerate(trainloader):
        inputs, labels = item
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # forward processing
        outputs, batch_times = model(inputs)
        
        # backward processing
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        running_loss += loss.item()
        times.append(batch_times)
        
        if count % 100 == 0:
            print(f'#{epoch}, data:{count * trainloader.batch_size}, batch_times:{batch_times}, sum:{sum(batch_times):.6f}')
            running_loss = 0.0
    
    # return train loss and elapsed time
    train_loss /= len(trainloader)
    times = np.array(times)
    mean_times = np.mean(times, axis=0)
    return train_loss, mean_times
        
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

def valid_tm(model, validloader, criterion, device='cpu'):
    
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
            outputs, times = model(inputs)
            
            # calculate the valid loss
            valid_loss += criterion(outputs, labels).item()
            
            # calculate the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
        
        valid_loss /= len(validloader)
        valid_accuracy = correct / total

    return valid_loss, valid_accuracy

def valid_vislayer(model, validloader, criterion, device='cpu'):
    
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
            outputs, vis_layer = model(inputs)
            
            # calculate the valid loss
            valid_loss += criterion(outputs, labels).item()
            
            # calculate the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
        
        valid_loss /= len(validloader)
        valid_accuracy = correct / total

    return valid_loss, valid_accuracy

def valid_vislayer(model, validloader, criterion, device='cpu'):
    
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
            outputs, _ = model(inputs)
            
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

def valid_multi_dataloaders_tm(model, validloaders, criterion, device='cpu'):
    
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
        times = []
        
        # calculate total loss, number of correct answers and number of datas over all dataloaders
        for loader in validloaders:
            v_loss, corrects_, total_, times_ = _half_valid_tm(model, loader, criterion, device=device)
            valid_loss += v_loss
            corrects += corrects_
            total += total_
            times.append(times_)
        
        # calculate means of them
        valid_loss /= dataloader_len
        valid_accuracy = corrects / total
    
    return valid_loss, valid_accuracy, times

def valid_multi_dataloaders_vislayer(model, validloaders, criterion, device='cpu'):
    
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
        vis_layers = []
        
        # calculate total loss, number of correct answers and number of datas over all dataloaders
        for loader in validloaders:
            v_loss, corrects_, total_, vis = _half_valid_vislayer(model, loader, criterion, device=device)
            valid_loss += v_loss
            corrects += corrects_
            total += total_
        
        # calculate means of them
        valid_loss /= dataloader_len
        valid_accuracy = corrects / total
        
        vis_layers.append(vis)
    
    return valid_loss, valid_accuracy, vis_layers

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

def get_pred_true_multi_dataloaders_tm(model, testloaders, device='cpu', class_labels=None):
    with torch.no_grad():
        # temporal variables
        total = 0
        correct = 0
        
        # variables to return
        predictions = []
        true_labels = []
        times = []
        
        for loader in testloaders:
            times_ = []
            
            for data in loader:
                inputs, labels = data
                
                # record the true labels to calculate the confusion matrix later
                true_labels.extend(list(labels.numpy())) 
                
                # forward processings
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, t = model(inputs)
                _, prediction = torch.max(outputs, 1)
                
                # record the number of correct answers
                total += len(outputs)
                correct += (prediction==labels).sum().item()
                times_.extend(t)
                
                # record the predicted labels
                predictions.extend(list(prediction.to('cpu').numpy()))
            
            times.append(times_)
        
    return predictions, true_labels, times

def get_pred_true_multi_dataloaders_vislayer(model, testloaders, device='cpu', class_labels=None):
    with torch.no_grad():
        # temporal variables
        total = 0
        correct = 0
        
        # variables to return
        predictions = []
        true_labels = []
        vis_layers = []
        
        for loader in testloaders:
            vis_layers_ = []
            
            for data in loader:
                inputs, labels = data
                
                # record the true labels to calculate the confusion matrix later
                true_labels.extend(list(labels.numpy())) 
                
                # forward processings
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, vis = model(inputs)
                _, prediction = torch.max(outputs, 1)
                
                # record the number of correct answers
                total += len(outputs)
                correct += (prediction==labels).sum().item()
                vis_layers_.extend(vis)
                
                # record the predicted labels
                predictions.extend(list(prediction.to('cpu').numpy()))
            
            vis_layers.append(vis_layers_)
        
    return predictions, true_labels, vis_layers

def get_confusion_matrix(predictions, true_labels, class_labels=None, normalize=False):
    cm = pd.DataFrame(confusion_matrix(true_labels, predictions), index=class_labels, columns=class_labels)
    if normalize:
        cm = cm.apply(lambda x: x / sum(x), axis=1)
    return cm

def show_confusion_matrix(cm):
    ax = sns.heatmap(cm, square=True, annot=True, fmt='.3f', cmap='Blues')
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

def _half_valid_tm(model, validloader, criterion, device='cpu'):
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
        times = []
            
        # forward processing
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, time = model(inputs)

            # calculating the valid loss
            valid_loss += criterion(outputs, labels).item()
            # calculating the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
            times.extend(time)

    return valid_loss, correct, total, times

def _half_valid_vislayer(model, validloader, criterion, device='cpu'):
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
        vis = []
            
        # forward processing
        for data in validloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, x = model(inputs)

            # calculating the valid loss
            valid_loss += criterion(outputs, labels).item()
            # calculating the accuracy
            _, prediction = torch.max(outputs, 1)
            total += len(outputs)
            correct += (prediction==labels).sum().item()
            
            vis.extend(x)

    return valid_loss, correct, total, vis