#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 02:14:58 2022

@author: Ark_001
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Following modules are self-made
import Datasets as datasets
import Model_Learning_Tools.Train as trainer
import Model_Learning_Tools.Validation as validator
import Model_Learning_Tools.Evaluation as evaluator
import Model_Learning_Tools.Display as display
import Models as models

# =============================================================================
# Load the datasets
# =============================================================================

"""NOTE: choose either and comment the other out"""
# Emo-DB
loader = datasets.EmoDB_loader()
# IEMOCAP
# loader = datasets.IEMOCAP_loader()

# load melsp datasets
melsp_data, labels, speakers = loader.load_melsp_dataset(requires_speakers=True)

# =============================================================================
# Create speaker-dependent or speaker-independent TensorDatasets
# =============================================================================

# tensor dataset creator
tdc = datasets.TensorDatasetCreatorForSER()

"""NOTE: choose either and comment the other out"""

# speaker-dependent (sd) datasets
tdc.set_random_seed(2222) # random seed for train_test_split
train_dataset, valid_datasets = tdc.speaker_dependent_dataset(melsp_data, labels)

# speaker-independent (si) datasets
# test_speakers = ['Ses01M', 'Ses05F'] # for IEMOCAP
# test_speakers = [9, 14] # for Emo-DB
# train_dataset, valid_datasets, valid_speakers = tdc.speaker_independent_dataset(\
#     melsp_data, labels, speakers, test_speakers=test_speakers)

# =============================================================================
# Define DataLoaders
# =============================================================================

"""NOTE: choose either and comment the other out"""
batch_size = 15 # batch size for Emo-DB
# batch_size = 25 # batch size for IEMOCAP

train_loader = DataLoader(train_dataset, batch_size=batch_size)
valid_loaders = [DataLoader(dataset, batch_size=batch_size) for dataset in valid_datasets]

# =============================================================================
# Instantiate model and learning tools
# =============================================================================

epochs = 100 # number of epochs
patience = 20 # patience of early stopping
acc_threshold = 1 # accuracy threshold of early stopping
device = 'cuda' if torch.cuda.is_available() else 'cpu' # device
lr = 0.0001
display_interval = 5 # interval of displaying curves


# instantiate earlystopping modules
# NOTE: YOU SHOULD CHANGE filename value appropriately if you have multiple models

path = './save_models'
filename = 'checkpoint_model.sav'
earlystopping = trainer.EarlyStopping(patience=patience, criterion='accuracy', verbose=True, \
                                  acc_threshold=acc_threshold, path=path, filename=filename)

# leaning logs
train_loss_log = []
valid_loss_log = []
train_accuracy_log = []
valid_accuracy_log = []

# instantiate the model
model = models.TFNN_for_SER().to(device)

# criterion and optimizer
# we have different criterions for train and validation this time
criterion_train = nn.CrossEntropyLoss()
criterion_valid = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)


# =============================================================================
# Train the model
# =============================================================================

for epoch in range(1, epochs + 1):
    
    # training mode
    model.train()
    
    # calculate train loss (and training time) of the epoch
    # NOTE: don't enable measure_gpu_time if you don't use GPU
    t_loss = trainer.train_epoch(model, train_loader, criterion_train, optimizer, device=device, epoch=epoch)
    # evaluation mode
    model.eval()
    
    # calculate train accuracy of the epoch
    _, t_acc = validator.valid(model, train_loader, criterion_valid, device=device)
    
    # calculate valid loss and valid accuracy of the epoch
    v_loss, v_acc = validator.valid_md(model, valid_loaders, criterion_valid, device=device)
    
    # record losses and accuracies
    train_loss_log.append(t_loss)
    valid_loss_log.append(v_loss)
    train_accuracy_log.append(t_acc)
    valid_accuracy_log.append(v_acc)
    
    # display current losses and accuracies
    print(f'#{epoch}: Train loss = {train_loss_log[-1]:.3f}, Validation loss = {valid_loss_log[-1]:.3f}, Train accuracy = {train_accuracy_log[-1]:.3f}, valid_accuracy = {valid_accuracy_log[-1]:.3f}')
    
    # display learning and accuracy curves at certain interval
    if epoch % display_interval == 0:
        display.display_curves(train_loss_log, valid_loss_log, train_accuracy_log, valid_accuracy_log)
    
    # early stopping
    earlystopping(v_acc, t_acc, v_acc, model)
    if earlystopping.early_stop:
        print('Early Stopping!')
        break
    
# =============================================================================
# Display the final outputs
# =============================================================================
display.display_curves(train_loss_log, valid_loss_log, train_accuracy_log, valid_accuracy_log)