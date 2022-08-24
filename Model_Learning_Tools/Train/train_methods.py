#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:17:44 2022

@author: Ark_001
"""

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
