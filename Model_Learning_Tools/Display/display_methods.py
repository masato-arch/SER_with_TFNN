#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:55:14 2022

@author: Ark_001
"""

import numpy as np
import matplotlib.pyplot as plt

def display_curves(train_loss_log, valid_loss_log, train_accuracy_log, valid_accuracy_log):
    
    """Method to display learning and accuracy curves"""
    
    fig = plt.figure(figsize=(12, 4))
    
    # add subplots
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_acc = fig.add_subplot(1, 2, 2)
    
    # set the axis range
    x = np.arange(1, len(train_loss_log) + 1)
    
    # set titles
    ax_loss.set_title('Loss Curve')
    ax_acc.set_title('Accuracy Curve')
    
    # set axis labels
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    
    # plot loss and accuracy logs
    ax_loss.plot(x, train_loss_log, label='Train Loss')
    ax_loss.plot(x, valid_loss_log, label='Validation Loss')
    ax_loss.legend()
    ax_acc.plot(x, train_accuracy_log, label='Train Accuracy')
    ax_acc.plot(x, valid_accuracy_log, label='Validation Accuracy')
    ax_acc.legend()
    
    plt.tight_layout()
    plt.show()
