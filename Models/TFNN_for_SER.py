#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 01:37:47 2022

@author: Ark_001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import tltorch
import numpy as np

class TFNN_for_SER(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.tcl1 = tltorch.factorized_layers.TCL(input_shape=(1, 128, 98), rank=(1, 120, 90))
        self.tcl2 = tltorch.factorized_layers.TCL(input_shape=(1, 120, 90), rank=(1, 110, 80))
        self.tcl3 = tltorch.factorized_layers.TCL(input_shape=(1, 110, 80), rank=(1, 100, 70))
        self.tcl4 = tltorch.factorized_layers.TCL(input_shape=(1, 100, 70), rank=(1, 90, 60))
        self.trl = tltorch.factorized_layers.TRL(input_shape=(1, 90, 60), output_shape=(4,), factorization='tucker')
        self.trl.init_from_random()
        
    def forward(self, x):
        
        # We have different forward() methods for training and testing
        
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)
        
    def _forward_train(self, x):
        # =============================================================================
        # Method for forward processing for training.
        # The number of segments of every input files is assumed to be 1
        # =============================================================================
        
        x = self.bn(F.relu(self.tcl1(x)))
        x = self.bn(F.relu(self.tcl2(x)))
        x = self.bn(F.relu(self.tcl3(x)))
        x = self.bn(F.relu(self.tcl4(x)))
        x = self.trl(x)
        
        return x
    
    def _forward_test(self, x):
        # =============================================================================
        # Method for forward processing for testing
        # The number of segments is different for each file
        # So, we give up parallel computing and process file by file
        # 
        # NOTE: This method still has some unreasonable parts, so updated in future release.
        # =============================================================================
        outputs = []
        for file in x: # for all files in x
        
            # take mean output of each segments
            mean_output = []
            for segment in file:
                # since our layer needs 4D input, forcibly reshape 128x98 segment into 1x1x128x98
                # need to be improved in the future release
                segment = torch.reshape(segment, (1, 1, ) + segment.shape)
                output = self._forward_test_(segment)
                mean_output.append(output)
            mean_output = torch.mean(torch.cat(mean_output).reshape(-1, 4), dim=0)
            outputs.append(mean_output)
        outputs = torch.cat(outputs).reshape(-1, 4)
        return outputs
    
    def _forward_test_(self, segment):
        # =============================================================================
        # Almost same method as _forward_train, but for testing
        # Method for forward processing one segment from a file
        # =============================================================================
        segment = self.bn(F.relu(self.tcl1(segment)))
        segment = self.bn(F.relu(self.tcl2(segment)))
        segment = self.bn(F.relu(self.tcl3(segment)))
        segment = self.bn(F.relu(self.tcl4(segment)))
        segment = self.trl(segment)
        
        return segment