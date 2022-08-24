#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:06:45 2022

@author: Ark_001
"""

import os
import numpy as np
import torch

class EarlyStopping:
    """earlystopping class"""

    def __init__(self, patience=5, criterion='loss', acc_threshold=0.12, verbose=True, path='models/', filename='checkpoint_model.sav'):
        # =============================================================================
        # Module for early stopping.
        
        # [parameters]
        #   patience:  (int) stop counter limit
        #   criterion: (str: 'loss' or 'accuracy') score criterion 
        #   verbose: (bool) whether to display log
        #   path: (str) directory path to save the model
        #   filename: (str) filename to save the model
        #
        # Earlystop is triggerd when:
        #   1, the model couldn't update its best score over long epochs defined by 'patience'
        #   2, the difference between train accuracy and validation accuracy exceeded defined 'acc_threshold'
        # =============================================================================

        self.patience = patience    # Set stop counter
    
        if criterion == 'loss' or criterion == 'accuracy':
            self.criterion = criterion # Criterion
        else:
            raise ValueError(f'parameter <criterion> must be \'loss\' or \'accuracy\', but received {criterion}')
        
        self.acc_threshold = acc_threshold # Accuracy difference threshold
        self.verbose = verbose    # Whether to display or not
        self.checkpoint_msg = 'Validation loss decreased' if self.criterion == 'loss' else 'Validation accuracy increased'
        self.counter = 0    # Current Counter
        self.best_score = None     # For remembering the best score
        self.early_stop = False     # Early stopping flag
        self.last_best_score = np.Inf if self.criterion == 'loss' else 0.0 # For remembering the last best score
        self.path = os.path.join(path, filename) # Path to storage the best model
        
        if not os.path.isdir(path):
            os.mkdir(path)

    def __call__(self, score, train_accuracy, valid_accuracy, model):
        """
        The part that calculates whether or not the minimum loss was actually updated in the learning loop"""
        score = -score if self.criterion == 'loss' else score # reverse the sign if criterion is loss

        if self.best_score is None:  # Processing for the first epoch
            self.best_score = score   # In the first epoch, record it as the best score
            self.checkpoint(score, model)  # Save the model after recording and display the score
        elif score < self.best_score:  # If you couldn't update the best score
            self.counter += 1   # +1 the stop counter
            if self.verbose:  # Show progress when display is enabled
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # Change the stop flag to True if the set count is exceeded
                self.early_stop = True
        else:  # If you update the best score
            self.best_score = score  # Overwrite the best score
            self.checkpoint(score, model)  # Save the model and display the score
            self.counter = 0  # Reset the stop counter
        
        # If the difference between train accuracy and validation accuracy exceeds the acc_threshold,
        # change the stop flag to True
        if np.abs(train_accuracy - valid_accuracy) >= self.acc_threshold:
            if self.verbose:
                print(f'Difference between Train accuracy and Validation accuracy exceeded the threshold. EarlyStopping!')
            self.early_stop = True

    def checkpoint(self, score, model):
        '''Checkpoint function executed when the best score is updated'''
        
        if self.verbose:  # If display is enabled, display the difference from the previous best score
            score_todisplay = -score if self.criterion == 'loss' else score
            print(f'{self.checkpoint_msg} ({self.last_best_score:.6f} --> {score_todisplay:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  # Save the best model to the set path
        self.last_best_score = score  # Record the loss at the time
        