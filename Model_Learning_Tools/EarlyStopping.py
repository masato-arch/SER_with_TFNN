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

    def __init__(self, patience=5, verbose=False, path='models/', filename='checkpoint_model.pth'):
        # =============================================================================
        # Module for early stopping.
        # parameters: patience(set stop counter), verbose(whether to display or not), path, filename
        # =============================================================================

        self.patience = patience    # Set stop counter
        self.verbose = verbose    # Whether to display or not
        self.counter = 0    # Current Counter
        self.best_score = None     # For remembering the best score
        self.early_stop = False     # Early stopping flag
        self.val_loss_min = np.Inf   # For remembering the last best score
        self.path = os.path.join(path, filename) # Path to storage the best model
        
        if not os.path.isdir(path):
            os.mkdir(path)

    def __call__(self, val_loss, model):
        """
        The part that calculates whether or not the minimum loss was actually updated in the learning loop"""
        score = -val_loss

        if self.best_score is None:  # Processing for the first epoch
            self.best_score = score   # In the first epoch, record it as the best score
            self.checkpoint(val_loss, model)  # Save the model after recording and display the score
        elif score < self.best_score:  # If you couldn't update the best score
            self.counter += 1   # +1 the stop counter
            if self.verbose:  # Show progress when display is enabled
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # Change the stop flag to True if the set count is exceeded
                self.early_stop = True
        else:  # If you update the best score
            self.best_score = score  # Overwrite the best score
            self.checkpoint(val_loss, model)  # Save the model and display the score
            self.counter = 0  # Reset the stop counter

    def checkpoint(self, val_loss, model):
        '''Checkpoint function executed when the best score is updated'''
        
        if self.verbose:  # If display is enabled, display the difference from the previous best score
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  # Save the best model to the set path
        self.val_loss_min = val_loss  # Record the loss at the time

