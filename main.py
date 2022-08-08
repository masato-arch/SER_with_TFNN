#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 21:26:54 2022

@author: Ark_001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from Datasets.Load_EmoDB import EmoDB_loader
from Datasets.Load_IEMOCAP import IEMOCAP_loader
from Datasets.TensorDatasetCreatorForSER import TensorDatasetCreatorForSER
import Model_Learning_Tools as mlt
import Models.TFNN_for_SER as TFNN_for_SER

# =============================================================================
# Loading the datasets
# =============================================================================

# dataset loaders
emodb_loader = EmoDB_loader()
iemocap_loader = IEMOCAP_loader()
tdc = TensorDatasetCreatorForSER()

# load melsp datasets
iemocap_melsp, iemocap_labels, iemocap_speakers = iemocap_loader.load_melsp_dataset(requires_speakers=True)
emodb_melsp, emodb_labels, emodb_speakers = emodb_loader.load_melsp_dataset(requires_speakers=True)

# =============================================================================
# Create TensorDatasets
# =============================================================================

# speaker-dependent datasets
sd_iemocap_train_dataset, sd_iemocap_test_datasets = tdc.speaker_dependent_dataset(iemocap_melsp, iemocap_labels)
sd_emodb_train_dataset, sd_emodb_test_datasets = tdc.speaker_dependent_dataset(emodb_melsp, emodb_labels)

# concatenate the train datasets
sd_train_dataset = ConcatDataset([sd_iemocap_train_dataset, sd_emodb_train_dataset])

# speaker-independent datasets
si_iemocap_train_dataset, si_iemocap_test_datasets, iemocap_test_speakers = tdc.speaker_independent_dataset(\
    iemocap_melsp, iemocap_labels, iemocap_speakers, test_speakers=['Ses01M', 'Ses05F'])
si_emodb_train_dataset, si_emodb_test_datasets, emodb_test_speakers = tdc.speaker_independent_dataset(\
    emodb_melsp, emodb_labels, emodb_speakers)
    
# concatenate the train datasets
si_train_dataset = ConcatDataset([si_iemocap_train_dataset, si_emodb_train_dataset])

# =============================================================================
# Define DataLoaders
# =============================================================================

batch_size = 15

# speaker-dependent dataloaders
sd_train_loader = DataLoader(sd_train_dataset, batch_size=batch_size)
sd_iemocap_valid_loaders = [DataLoader(dataset, batch_size=(batch_size)) for dataset in sd_iemocap_test_datasets]
sd_emodb_valid_loaders = [DataLoader(dataset, batch_size=(batch_size)) for dataset in sd_emodb_test_datasets]
sd_valid_loaders = sd_iemocap_valid_loaders + sd_emodb_valid_loaders

# speaker-independent dataloaders
si_train_loader = DataLoader(si_train_dataset, batch_size=batch_size)
si_iemocap_valid_loaders = [DataLoader(dataset, batch_size=(batch_size)) for dataset in si_iemocap_test_datasets]
si_emodb_valid_loaders = [DataLoader(dataset, batch_size=(batch_size)) for dataset in si_emodb_test_datasets]
si_valid_loaders = si_iemocap_valid_loaders + si_emodb_valid_loaders

# =============================================================================
# Define model and learning tools
# =============================================================================

epochs = 100
patience = 7
device = 'cuda'

sd_earlystopping = mlt.EarlyStopping(patience=patience, path='./save_models', filename='sd_checkpoint_model.sav')
si_earlystopping = mlt.EarlyStopping(patience=patience, path='./save_models', filename='si_checkpoint_model.sav')

train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []
elapsed_time = []

sd_model = TFNN_for_SER().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(sd_model.parameters())


# =============================================================================
# Train the model
# =============================================================================

# speaker-dependent
for epoch in range(epochs):
    
    sd_model.train()
    
    # calculate train loss and training time of the epoch
    t_loss, e_time = mlt.train_epoch(sd_model, \
                                     sd_train_loader, criterion, optimizer, device=device, epoch=epoch, measure_time=True)
    
    sd_model.eval()
    
    # calculate train accuracy of the epoch
    _, t_acc = mlt.valid(sd_model, sd_train_loader, criterion, device=device)
    
    # calculate valid loss and valid accuracy of the epoch
    v_loss, v_acc = mlt.valid_multi_dataloaders(sd_model, sd_valid_loaders, criterion, device=device)
    
    # record losses and accuracies
    train_loss.append(t_loss)
    valid_loss.append(v_loss)
    train_accuracy.append(t_acc)
    valid_accuracy.append(v_acc)
    elapsed_time.append(e_time)
    
    # display log
    print(f'#{epoch}: train_loss = {t_loss:.3f}, train_acc={t_acc:.3f}, valid_loss={v_loss:.3f}, valid_acc={v_acc:.3f}')
    if epoch % 10 == 0:
        mlt.display_log(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch=epoch)
    
    # early stopping
    sd_earlystopping(v_loss, sd_model)
    if sd_earlystopping.early_stop:
        print('Early Stopping!')
        break