# -*- coding: utf-8 -*-
"""
Spyder Editor

@Auther: Ark_001
"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import ConcatDataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
from Load_IEMOCAP import IEMOCAP_loader
from Load_EmoDB import EmoDB_loader

class TensorDatasetCreatorForSER:
    
    def __init__(self):
        # =============================================================================
        # Initialize the module.
        # The module holds a mapping to convert string emotion labels to int numbers,
        #   and a random seed.
        # =============================================================================
        
        self.random_seed = 2222
        self.emotion_mapping = {
            'Anger' : 0,
            'Happiness' : 1,
            'Neutral' : 2,
            'Sadness' : 3
            }
    
    # =============================================================================
    # User Interfaces:
    #   speaker_dependent_dataset(): returns a speaker-dependent TensorDataset.
    #   speaker_independent_dataset(): returns a speaker-independent TensorDataset.
    #   set_random_seed(): use if you want to change random seed.
    # =============================================================================
    
    def speaker_dependent_dataset(self, datas, labels, train_size=0.8, shuffle=True):
        # =============================================================================
        # Method to create a speaker-dependent dataset, i.e. dataset with 
        # the same set of speakers in training and testing.
        #
        # First, split the whole dataset at the prescribed train_size(=0.8),
        # and remake the train dataset to treat all the speech segments as individual files.
        # =============================================================================
        
        # split the dataset into training and test datasets randomly
        train_datas, test_datas, train_labels, test_labels = \
            train_test_split(datas, labels, random_state=self.random_seed, \
                             shuffle=shuffle, train_size=train_size)
        
        # create the TensorDatasets
        # sd stands for speaker-dependent
        sd_train_dataset = self._create_train_dataset(train_datas, train_labels)
        sd_test_dataset = self._create_test_dataset(test_datas, test_labels)
        return sd_train_dataset, sd_test_dataset
    
    def speaker_independent_dataset(self, datas, labels, speakers, test_speakers=None):
        # =============================================================================
        # Method to create a speaker-independent dataset, i.e. dataset with
        # the different set of speakers in training and testing. 
        # The classification will be more difficult than speaker-dependent condition.
        # =============================================================================
        
        # if test_speakers are not specified, choose randomly
        if not test_speakers:
            speaker_set = list(set(speakers))
            _, test_speakers = train_test_split(speaker_set, test_size=0.2, random_state=self.random_seed)
        
        # split the dataset into training and test datasets
        # si stands for speaker-independent
        si_train_datas, si_test_datas, si_train_labels, si_test_labels = \
            self._speaker_independent_data_split(datas, labels, speakers, test_speakers=test_speakers)
        
        # create the TensorDatasets
        si_train_dataset = self._create_train_dataset(si_train_datas, si_train_labels)
        si_test_dataset = self._create_test_dataset(si_test_datas, si_test_labels)
        
        print(f'Speaker-independent dataset created. Test speakers:{test_speakers}')
        
        return si_train_dataset, si_test_dataset, test_speakers
    
    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
    
    """
    Following codes are for internal processings. You don't have to read.'
    """
    def _create_train_dataset(self, datas, labels):
        # =============================================================================
        # Method to create a train dataset.
        # Decompose a speech file (consists of several segments) into individual segments.
        # For training, we treat all the speech segments come from same files as individual files.
        # Reshape the data into 3 dimentional tensor: (n_segments=1, width, height)
        # =============================================================================
        
        train_datas, train_labels = [], []
        
        for data, label in zip(datas, labels):
            
            for segment in data:
                train_datas.append(segment.reshape((1,) + segment.shape))
                train_labels.append(self.emotion_mapping[label])
        
        train_datas = torch.FloatTensor(np.array(train_datas))
        train_labels = torch.LongTensor(np.array(train_labels))
        train_dataset = TensorDataset(train_datas, train_labels)
        
        return train_dataset
        
    
    def _create_test_dataset(self, datas, labels):
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
            test_labels[idx].append(self.emotion_mapping[labels[i]])
        
        test_datas = [torch.FloatTensor(np.array(td)) for td in test_datas]
        test_labels = [torch.LongTensor(np.array(tl)) for tl in test_labels]
        test_datasets = [TensorDataset(test_datas[i], test_labels[i]) for i in range(len(test_datas))]
        test_dataset = ConcatDataset(test_datasets)
        
        return test_dataset
    
    def _speaker_independent_data_split(self, datas, labels, speakers, test_speakers):
        train_datas, test_datas, train_labels, test_labels = [], [], [], []
        for data, label, speaker in zip(datas, labels, speakers):
            if speaker in test_speakers:
                test_datas.append(data)
                test_labels.append(label)
            else:
                train_datas.append(data)
                train_labels.append(label)
    
        return train_datas, test_datas, train_labels, test_labels