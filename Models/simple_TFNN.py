#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:10:14 2022

@author: Ark_001
"""

from .TFNN_for_SER import TFNN_for_SER
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
import tltorch


class simple_TFNN(TFNN_for_SER):
    
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
        x = self.bn(F.relu(self.tcl1(x)))
        x = self.bn(F.relu(self.tcl2(x)))
        x = self.bn(F.relu(self.tcl3(x)))
        x = self.bn(F.relu(self.tcl4(x)))
        x = self.trl(x)
        return x
