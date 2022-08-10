#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 21:26:54 2022

@author: Ark_001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Following modules are self-made
import Datasets as datasets
import Model_Learning_Tools as mlt
import Models as models

# =============================================================================
# Load the datasets
# =============================================================================

# dataset loaders
emodb_loader = datasets.EmoDB_loader()
iemocap_loader = datasets.IEMOCAP_loader()
tdc = datasets.TensorDatasetCreatorForSER()

# load melsp datasets
iemocap_melsp, iemocap_labels, iemocap_speakers = iemocap_loader.load_melsp_dataset(requires_speakers=True)
emodb_melsp, emodb_labels, emodb_speakers = emodb_loader.load_melsp_dataset(requires_speakers=True)

# =============================================================================
# Create TensorDatasets
# =============================================================================

# speaker-dependent (sd) datasets
sd_iemocap_train_dataset, sd_iemocap_valid_datasets = tdc.speaker_dependent_dataset(iemocap_melsp, iemocap_labels)
sd_emodb_train_dataset, sd_emodb_valid_datasets = tdc.speaker_dependent_dataset(emodb_melsp, emodb_labels)

# speaker-independent (si) datasets
si_iemocap_train_dataset, si_iemocap_valid_datasets, iemocap_valid_speakers = tdc.speaker_independent_dataset(\
    iemocap_melsp, iemocap_labels, iemocap_speakers, test_speakers=['Ses01M', 'Ses05F'])
si_emodb_train_dataset, si_emodb_valid_datasets, emodb_valid_speakers = tdc.speaker_independent_dataset(\
    emodb_melsp, emodb_labels, emodb_speakers)

# =============================================================================
# Define DataLoaders
# =============================================================================

emodb_batch_size = 15 # batch size for Emo-DB
iemocap_batch_size = 25 # batch size for IEMOCAPS

# speaker-dependent (sd) dataloaders
sd_iemocap_train_loader = DataLoader(sd_iemocap_train_dataset, batch_size=iemocap_batch_size)
sd_iemocap_valid_loaders = [DataLoader(dataset, batch_size=iemocap_batch_size) for dataset in sd_iemocap_valid_datasets]
sd_emodb_train_loader = DataLoader(sd_iemocap_train_dataset, batch_size=emodb_batch_size)
sd_emodb_valid_loaders = [DataLoader(dataset, batch_size=iemocap_batch_size) for dataset in sd_emodb_valid_datasets]

# speaker-independent (si) dataloaders
si_iemocap_train_loader = DataLoader(si_iemocap_train_dataset, batch_size=iemocap_batch_size)
si_iemocap_valid_loaders = [DataLoader(dataset, batch_size=iemocap_batch_size) for dataset in si_iemocap_valid_datasets]
si_emodb_train_loader = DataLoader(si_emodb_train_dataset, batch_size=emodb_batch_size)
si_emodb_valid_loaders = [DataLoader(dataset, batch_size=emodb_batch_size) for dataset in si_emodb_valid_datasets]

# =============================================================================
# Instantiate model and learning tools
# =============================================================================

epochs = 100 # number of epochs
patience = 7 # patience of early stopping
device = 'cuda' if torch.cuda.is_available() else 'cpu' # device
display_interval = 5

# instantiate earlystopping modules
# NOTE: you should change path and filename values appropriately if you have multiple models
earlystopping = mlt.EarlyStopping(patience=patience, verbose=True, \
                                  path='./save_models', filename='checkpoint_model.sav')

# leaning logs
train_loss_log = []
valid_loss_log = []
train_accuracy_log = []
valid_accuracy_log = []
elapsed_time = []

# instantiate the model
model = models.TFNN_for_SER().to(device)

# criterion and optimizer
# we have different criterions for train and validation this time
criterion_train = nn.CrossEntropyLoss()
criterion_valid = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# =============================================================================
# Train the model
# =============================================================================

# speaker-dependent learning
for epoch in range(1, epochs + 1):
    
    # training mode
    model.train()
    
    # calculate train loss (and training time) of the epoch
    # NOTE: don't enable measure_gpu_time if you don't use GPU
    t_loss, e_time = mlt.train_epoch(model, \
                                     sd_iemocap_train_loader, criterion_train, optimizer, device=device, epoch=epoch)
    # evaluation mode
    model.eval()
    
    # calculate train accuracy of the epoch
    _, t_acc = mlt.valid(model, sd_iemocap_train_loader, criterion_valid, device=device)
    
    # calculate valid loss and valid accuracy of the epoch
    v_loss, v_acc = mlt.valid_multi_dataloaders(model, sd_iemocap_valid_loaders, criterion_valid, device=device)
    
    # record losses and accuracies
    train_loss_log.append(t_loss)
    valid_loss_log.append(v_loss)
    train_accuracy_log.append(t_acc)
    valid_accuracy_log.append(v_acc)
    elapsed_time.append(e_time)
    
    # display current losses and accuracies
    print(f'#{epoch}: Train loss = {train_loss_log[-1]:.3f}, Validation loss = {valid_loss_log[-1]:.3f}, Train accuracy = {train_accuracy_log[-1]:.3f}, valid_accuracy = {valid_accuracy_log[-1]:.3f}')
    
    # display learning and accuracy curves at certain interval
    if epoch % display_interval == 0:
        mlt.display_curves(train_loss_log, valid_loss_log, train_accuracy_log, valid_accuracy_log, epoch=epoch)
    
    # early stopping
    earlystopping(v_loss, t_acc, v_acc, model)
    if earlystopping.early_stop:
        print('Early Stopping!')
        break
    
# =============================================================================
# Evaluate the model
# =============================================================================

# speaker-dependent
sd_iemocap_cm, sd_iemocap_accuracy = mlt.classification_test_multi_dataloaders(model, sd_iemocap_valid_loaders, device=device)
