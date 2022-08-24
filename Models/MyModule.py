#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 23:02:33 2022

@author: Ark_001
"""

import torch.nn as nn

class MyModule(nn.Module):
    
    def __init__(self, time_mesure=False):
        super().__init__()
        self.time_mesure = time_mesure
        
    def set_time_measure(self, time_mesure):
        self.time_mesure = time_mesure
    
    def forward(self, x):
        return
    
    def forward_tm(self, x):
        return
    
    def feature_map(self, x):
        return