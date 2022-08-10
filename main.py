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
from Datasets.Load_EmoDB import EmoDB_loader
from Datasets.Load_IEMOCAP import IEMOCAP_loader
from Datasets.TensorDatasetCreatorForSER import TensorDatasetCreatorForSER
import Model_Learning_Tools as mlt
import Models as models

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

# instantiate earlystopping modules
earlystopping = mlt.EarlyStopping(patience=patience, path='./save_models', filename='sd_iemocap_TFNN_checkpoint_model.sav')

# records
train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []
elapsed_time = []
outputs_list = []

# instantiate the model
model = models.TFNN_for_SER().to(device)

# criterion and optimizer
criterion_train = nn.CrossEntropyLoss()
criterion_valid = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# =============================================================================
# Train the model
# =============================================================================

# speaker-dependent learning
for epoch in range(epochs):
    
    # training mode
    model.train()
    
    # calculate train loss and training time of the epoch
    t_loss, e_time, outputs_list_ = mlt.train_epoch(model, \
                                     sd_iemocap_train_loader, criterion_train, optimizer, device=device, epoch=epoch, measure_time=True)
    # evaluation mode
    model.eval()
    
    # calculate train accuracy of the epoch
    _, t_acc = mlt.valid(model, sd_iemocap_train_loader, criterion_valid, device=device)
    
    # calculate valid loss and valid accuracy of the epoch
    v_loss, v_acc = mlt.valid_multi_dataloaders(model, sd_iemocap_valid_loaders, criterion_valid, device=device)
    
    # record losses and accuracies
    train_loss.append(t_loss)
    valid_loss.append(v_loss)
    train_accuracy.append(t_acc)
    valid_accuracy.append(v_acc)
    elapsed_time.append(e_time)
    outputs_list.append(outputs_list_)
    
    # display log
    print(f'#{epoch}: train_loss = {t_loss:.3f}, train_acc={t_acc:.3f}, valid_loss={v_loss:.3f}, valid_acc={v_acc:.3f}')
    # print(f'#{epoch}: train_loss = {t_loss:.3f}, valid_loss={v_loss:.3f}, valid_acc={v_acc:.3f}')
    if epoch % 10 == 0:
        mlt.display_log(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch=epoch)
    # early stopping
    earlystopping(v_loss, model)
    if earlystopping.early_stop:
        print('Early Stopping!')
        break
    
# =============================================================================
# Evaluate the model
# =============================================================================

# speaker-dependent
sd_iemocap_cm, sd_iemocap_accuracy = mlt.classification_test_multi_dataloaders(model, sd_iemocap_valid_loaders, device=device)
