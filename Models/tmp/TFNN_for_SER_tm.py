#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 20:40:55 2022

@author: Ark_001
"""

import numpy as np
import torch
import torch
import torch.nn.functional as F
from .TFNN_for_SER import TFNN_for_SER

class TFNN_for_SER_tm(TFNN_for_SER):
    
    def __init__(self):
        super().__init__()
        
    def _forward_train(self, x):
        # =============================================================================
        # Method to measure time consumed at each layer during training
        # NOTE: The number of segments of every input files is assumed to be 1
        # =============================================================================
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        
        start.record()
        x = F.relu(self.tcl1(x))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
        
        start.record()
        x = F.relu(self.tcl2(x))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
        
        start.record()
        x = F.relu(self.tcl3(x))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
        
        start.record()
        x = F.relu(self.tcl4(x))
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
        
        start.record()
        x = self.trl(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
        
        times = np.array(times)
        
        return x, times
    
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
        times = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        for utterance in x:
            segment_wise_p = []
            start.record()
            for segment in utterance:
                
                # calculate the probabilities of each segment
                segment = torch.reshape(segment, (1, 1,) + segment.shape)
                p_segment = self._forward_test_(segment)
                segment_wise_p.append(p_segment)
                
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000
            
            # concatenate segment-wise probabilities
            segment_wise_p = torch.reshape(torch.cat(segment_wise_p), (len(segment_wise_p), -1))
            
            # take mean of segment-wise probabilities and get utterance-level probability
            utterance_level_p = torch.mean(segment_wise_p, dim=0)
            outputs.append(utterance_level_p)
            times.append(time)
        
        # finally reshape outputs to tensor
        outputs = torch.reshape(torch.cat(outputs, dim=0), (len(outputs), -1))
        
        return outputs, times
    
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
        
        return p