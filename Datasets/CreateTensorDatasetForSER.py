# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split

class CreateTensorDatasetForSER:
    
    def __init__(self):
        
        self.random_seed = 2222
        self.default_emotion_mapping = {
            'Anger' : 0,
            'Happiness' : 1,
            'Neutral' : 2,
            'Sadness' : 3
            }
        
    def speaker_dependent_dataset(self, datas, labels, train_size=0.8, shuffle=True):
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
        
        train_datas, train_labels = self._decompose_a_data(train_datas, train_labels)
        train_datas = torch.FloatTensor(np.array(train_datas))
        
        return
    
    def speaker_independent_dataset(self, datas, labels, train_size=0.8):
        return
                
    
    def _create_train_dataset(self, datas, labels, emotion_mapping=None):
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
        
        return train_datas, train_labels
        
    
    def _create_test_dataset(self, datas, labels):
        # =============================================================================
        # Method to create test dataset.
        # We have different number of segments by files.
        # Reshape the data into 3 dimentional tensor: (n_segments, width, height)
        # =============================================================================
        
        return
            