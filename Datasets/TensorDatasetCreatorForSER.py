# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import ConcatDataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
from Load_IEMOCAP import IEMOCAP_loader
from Load_EmoDB import EmoDB_loader

emodb_loader = EmoDB_loader()
iemocap_loader = IEMOCAP_loader()

emodb_melsp, emodb_labels, emodb_speakers = emodb_loader.load_melsp_dataset(requires_speakers=True)
iemocap_melsp, iemocap_labels, iemocap_speakers = iemocap_loader.load_melsp_dataset(requires_speakers=True)

class TensorDatasetCreatorForSER:
    
    def __init__(self):
        
        self.random_seed = 2222
        self.default_emotion_mapping = {
            'Anger' : 0,
            'Happiness' : 1,
            'Neutral' : 2,
            'Sadness' : 3
            }
        
    def speaker_dependent_dataset(self, datas, labels, emotion_mapping, train_size=0.8, shuffle=True):
        # =============================================================================
        # Method to create speaker-dependent dataset, i.e. dataset with 
        # the same set of speakers in training and testing.
        #
        # First, split the whole dataset at the prescribed train_size(=0.8),
        # and remake the train dataset to treat all the speech segments as individual files.
        # =============================================================================
        
        train_datas, test_datas, train_labels, test_labels = \
            train_test_split(datas, labels, random_state=self.random_seed, \
                             shuffle=shuffle, train_size=train_size)
        
        sd_train_dataset = self._create_train_dataset(train_datas, train_labels, emotion_mapping=emotion_mapping)
        sd_test_dataset = self._create_test_dataset(test_datas, test_labels, emotion_mapping=emotion_mapping)
        
        return sd_train_dataset, sd_test_dataset
    
    def speaker_independent_dataset(self, datas, labels, train_size=0.8):
        return
                
    
    def _create_train_dataset(self, datas, labels, emotion_mapping):
        # =============================================================================
        # Method to create train dataset.
        # Decompose a speech file (consists of several segments) into individual segments.
        # For training, we treat all the speech segments come from same files as individual files.
        # Reshape the data into 3 dimentional tensor: (n_segments=1, width, height)
        # =============================================================================
        
        train_datas, train_labels = [], []
        
        for data, label in zip(datas, labels):
            
            for segment in data:
                train_datas.append(segment.reshape((1,) + segment.shape))
                train_labels.append(emotion_mapping[label])
        
        train_datas = torch.FloatTensor(np.array(train_datas))
        train_labels = torch.LongTensor(np.array(train_labels))
        train_dataset = TensorDataset(train_datas, train_labels)
        
        return train_dataset
        
    
    def _create_test_dataset(self, datas, labels, emotion_mapping):
        # =============================================================================
        # Method to create test dataset.
        # We have different number of segments by files.
        # Reshape the data into 3 dimentional tensor: (n_segments, width, height)
        # =============================================================================
        
        n_segments = [len(data) for data in datas]
        set_of_n_segments = list(set(n_segments))
        
        test_datas = [[]*i for i in range(len(set_of_n_segments))]
        test_labels = [[]*i for i in range(len(set_of_n_segments))]
        
        for i, n in enumerate(n_segments):
            idx = set_of_n_segments.index(n)
            test_datas[idx].append(datas[i])
            test_labels[idx].append(emotion_mapping[labels[i]])
        
        test_datas = [torch.FloatTensor(np.array(td)) for td in test_datas]
        test_labels = [torch.LongTensor(np.array(tl)) for tl in test_labels]
        test_datasets = [TensorDataset(test_datas[i], test_labels[i]) for i in range(len(test_datas))]
        test_dataset = ConcatDataset(test_datasets)
        
        return test_dataset

tensor_dataset_creator = TensorDatasetCreatorForSER()

emotion_mapping = {
    'Anger' : 0,
    'Happiness' : 1,
    'Neutral' : 2,
    'Sadness' : 3
    }

iemocap_train_dataset, iemocap_test_dataset = tensor_dataset_creator.speaker_dependent_dataset(iemocap_melsp, iemocap_labels, emotion_mapping=emotion_mapping)