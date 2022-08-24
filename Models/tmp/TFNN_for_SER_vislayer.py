#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 02:30:26 2022

@author: Ark_001
"""

import numpy as np
import torch
import torch
import torch.nn.functional as F
from .TFNN_for_SER import TFNN_for_SER

class TFNN_for_SER_vislayer(TFNN_for_SER):
    
    def _forward_test(self, x):
        # =============================================================================
        # Method for forward processing for validation
        # Validation set contains different number of segments by file.
        # So we give up parallel computing and process file by file.
        #
        # Calculate class probability for each segment, and take its mean 
        # as utterance-level class probability
        # =============================================================================
        
        outputs = []
        outputs_x = []
        #print('outputs defined')
        for utterance in x:
            segment_wise_p = []
            segment_wise_x = []
            #print("segment_wise_p defined")
            for segment in utterance:
                
                # calculate the probabilities of each segment
                segment = torch.reshape(segment, (1, 1,) + segment.shape)
                p_segment, x = self._forward_test_(segment)
                segment_wise_p.append(p_segment)
                segment_wise_x.append(x)
            
            # concatenate segment-wise probabilities
            segment_wise_p = torch.reshape(torch.cat(segment_wise_p), (len(segment_wise_p), -1))
            
            # take mean of segment-wise probabilities and get utterance-level probability
            utterance_level_p = torch.mean(segment_wise_p, dim=0)
            outputs.append(utterance_level_p)
            outputs_x.append(segment_wise_x)
        
        # finally reshape outputs to tensor
        outputs = torch.reshape(torch.cat(outputs, dim=0), (len(outputs), -1))
        
        return outputs, outputs_x
    def _forward_test_(self, segment):
        # =============================================================================
        # Method for calculating segment-wise class probability
        # Receives a segment, forward process that, and calculate class probability 
        # by applying softmax
        # =============================================================================
       
        # forward process the segment
        x = F.relu(self.tcl1(segment))
        x = F.relu(self.tcl2(x))
        x = F.relu(self.tcl3(x))
        x = F.relu(self.tcl4(x))
        a = self.trl(x)
        
        # apply softmax to the segment and get the class probability
        p = F.softmax(a, dim=1)
        
        return p, x