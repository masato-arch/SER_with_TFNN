#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 00:11:25 2022

@author: user
"""
import os
import torch
from torch.utils.data import DataLoader

# Following modules are self-made
import Datasets as datasets
import Model_Learning_Tools as mlt
import Models as models

# =============================================================================
# Load the dataset
# =============================================================================

batch_size = 15

# instantiate datasets
loader = datasets.EmoDB_loader() # choose IEMOCAP_loader() or EmoDB_loader()
tdc = datasets.TensorDatasetCreatorForSER()
tdc.set_random_seed(2222)

melsp_data, labels, speakers = loader.load_melsp_dataset(requires_speakers=True)

# =============================================================================
# Create speaker-dependent or speaker-independent TensorDataset
# =============================================================================

"""NOTE: please comment out either"""

# speaker-dependent
_, test_datasets = tdc.speaker_dependent_dataset(melsp_data, labels)

# speaker-independent
# test_speakers = ['Ses01M', 'Ses05F'] # test speakers for IEMOCAP
# test_speakers = [9, 14] # test speakers for Emo-DB
# _, test_datasets, test_speakers = tdc.speaker_independent_dataset(melsp_data, labels, speakers, test_speakers=test_speakers)

# =============================================================================
# Create the Test dataloaders
# =============================================================================
test_dataloaders = [DataLoader(dataset, batch_size=batch_size) for dataset in test_datasets]

# get class labels
class_labels = list(tdc.emotion_mapping.keys()) #['Anger', 'Happiness', 'Neutral', 'Sadness']

# =============================================================================
# Load the model
# =============================================================================

"""NOTE: change filename according to the models"""

path = './save_models'
filename = 'si_emodb_bestmodel_r2222.sav'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_model = models.TFNN_for_SER()
best_model.load_state_dict(torch.load(os.path.join(path, filename)))
best_model = best_model.to(device)
best_model.eval()

# =============================================================================
# Evaluate the model
# =============================================================================

# get predicted and true_labels
predictions, true_labels = mlt.get_pred_true_multi_dataloaders(best_model, test_dataloaders, device=device)

# calculate confusion matrix
cm = mlt.get_confusion_matrix(predictions, true_labels, class_labels=class_labels, normalize=True)
cm_raw = mlt.get_confusion_matrix(predictions, true_labels, class_labels=class_labels, normalize=False)
mlt.show_confusion_matrix(cm)
mlt.show_confusion_matrix(cm_raw)

# calculate accuracy
total = len(true_labels)
corrects = sum([p == t for p, t in zip(predictions, true_labels)])
weighted_accuracy = mlt.accuracy_score(true_labels, predictions)
unweighted_accuracy = mlt.balanced_accuracy_score(true_labels, predictions)

print(f'WA: {corrects}/{total} = {weighted_accuracy}')
print(f'UA: {unweighted_accuracy}')