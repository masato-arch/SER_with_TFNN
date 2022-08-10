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

    def __init__(self, patience=5, acc_threshold=0.12, verbose=True, path='models/', filename='checkpoint_model.sav'):
        # =============================================================================
        # Module for early stopping.
        # parameters: patience(set stop counter), verbose(whether to display or not), path, filename
        #
        # Earlystop is triggerd when:
        #   1, the model couldn't update its best score over long epochs defined by 'patience'
        #   2, the difference between train accuracy and validation accuracy exceeded defined 'acc_threshold'
        # =============================================================================

        self.patience = patience    # Set stop counter
        self.acc_threshold = acc_threshold # Accuracy difference threshold
        self.verbose = verbose    # Whether to display or not
        self.counter = 0    # Current Counter
        self.best_score = None     # For remembering the best score
        self.early_stop = False     # Early stopping flag
        self.valid_loss_min = np.Inf   # For remembering the last best score
        self.path = os.path.join(path, filename) # Path to storage the best model
        
        if not os.path.isdir(path):
            os.mkdir(path)

    def __call__(self, valid_loss, train_accuracy, valid_accuracy, model):
        """
        The part that calculates whether or not the minimum loss was actually updated in the learning loop"""
        score = -valid_loss

        if self.best_score is None:  # Processing for the first epoch
            self.best_score = score   # In the first epoch, record it as the best score
            self.checkpoint(valid_loss, model)  # Save the model after recording and display the score
        elif score < self.best_score:  # If you couldn't update the best score
            self.counter += 1   # +1 the stop counter
            if self.verbose:  # Show progress when display is enabled
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # Change the stop flag to True if the set count is exceeded
                self.early_stop = True
        else:  # If you update the best score
            self.best_score = score  # Overwrite the best score
            self.checkpoint(valid_loss, model)  # Save the model and display the score
            self.counter = 0  # Reset the stop counter
        
        # If the difference between train accuracy and validation accuracy exceeds the acc_threshold,
        # change the stop flag to True
        if np.abs(train_accuracy - valid_accuracy) >= self.acc_threshold:
            self.early_stop = True

    def checkpoint(self, valid_loss, model):
        '''Checkpoint function executed when the best score is updated'''
        
        if self.verbose:  # If display is enabled, display the difference from the previous best score
            print(f'Validation loss decreased ({self.valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  # Save the best model to the set path
        self.valid_loss_min = valid_loss  # Record the loss at the time

