#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:28:42 2022

@author: Ark_001
"""

import torch

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

def valid_md(model, validloaders, criterion, device='cpu'):
    
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