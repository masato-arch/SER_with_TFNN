#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:12:05 2022

@author: Ark_001
"""

import os
from os.path import exists
import audeer
import librosa
import math
import numpy as np
import pickle


class EmoDB_loader:
    
    def __init__(self):
        
        # =============================================================================
        # Initialize the module
        #
        # The module holds paths to pickle files of preprocessed wav, labels and melspectrograms.
        # 
        # If you don't have pickle, the dataset will be retrieved from ./EmoDB
        # =============================================================================
        self.wav_pickle_path = 'EmoDB_wav.pkl'
        self.melsp_pickle_path = 'EmoDB_melspectrogram.pkl'
        self.labels_pickle_path = 'EmoDB_labels.pkl'
        self.speakers_pickle_path = 'EmoDB_speakers.pkl'
        
        if not exists(self.wav_pickle_path) or not exists(self.labels_pickle_path) or not exists(self.speakers_pickle_path):
            print('pkl files not found. retrieveing datasets.')
            initial_wav_datas, initial_labels, initial_speakers = self._load_data()
            wav_datas, labels, speakers = self._constrain_dataset(initial_wav_datas, initial_labels, initial_speakers)
            self._save_pickle(wav_datas, self.wav_pickle_path)
            self._save_pickle(labels, self.labels_pickle_path)
            self._save_pickle(speakers, self.speakers_pickle_path)
            print('done.')
            
    # =============================================================================
    # User Interfaces:
    #   load_wav_dataaset(): returns dataset of raw wav audios and labels
    #   load_melsp_dataset(): returns dataset of 128x98 mel-spectrograms and labels
    # =============================================================================
    
    def load_wav_dataset(self, requires_speakers=False):
        wav_datas = self._load_pickle(self.wav_pickle_path)
        labels = self._load_pickle(self.labels_pickle_path)
        if requires_speakers:
            speakers = self._load_pickle(self.speakers_pickle_path)
        else:
            speakers = None
        return wav_datas, labels, speakers
    
    def load_melsp_dataset(self, requires_speakers=False):
        if not exists(self.melsp_pickle_path):
            print('pkl file not found. calculating mel spectorgrams.')
            print('this process may take several minutes...')
            wav_datas = self._load_pickle(self.wav_pickle_path)
            melsp_datas = self._calculate_melsp_datas(wav_datas)
            self._save_pickle(melsp_datas, self.melsp_pickle_path)
            print('done.')
        else:
            melsp_datas = self._load_pickle(self.melsp_pickle_path)
            
        if requires_speakers:
            speakers = self._load_pickle(self.speakers_pickle_path)
        else:
            speakers = None
        
        labels = self._load_pickle(self.labels_pickle_path)
        return melsp_datas, labels, speakers
    
    """
    Following codes are for internal processings.
    You don't have to read.
    """
    
    def _constrain_dataset(self, wav_datas, labels, speakers, required_labels=['Anger', 'Happiness', 'Sadness', 'Neutral']):
        # =============================================================================
        # Method to retrieve only required datas.
        # 
        # We only use utterances labeled with required classes, 
        #   which are ['Anger', 'Happiness', 'Sadness', 'Neutral']
        #   because the other classes doesn't have enough amount of data
        # =============================================================================
        
        req_datas, req_labels, req_speakers = [], [], []
        for data, label, speaker in zip(wav_datas, labels, speakers):
            if label in required_labels:
                req_datas.append(data)
                req_labels.append(label)
                req_speakers.append(speaker)
        
        return req_datas, req_labels, req_speakers
    
    def _calculate_melsp_datas(self, wav_data, len_segment=50000, fft_params=(2048, 512)):
        
        def _zero_pad(data, len_segment=len_segment):
            n_segments = int(math.ceil(len(data) / len_segment))
            zero_padded = np.zeros(n_segments * len_segment)
            zero_padded[:len(data)] = data
            return zero_padded
        
        def _segment_wav(wav_data, len_segment=len_segment):
            segmented_wav = []
            n_segments = int(len(wav_data) / len_segment)
            for i in range(n_segments):
                segmented_wav.append(wav_data[i * len_segment : (i + 1) * len_segment])
            
            return segmented_wav
        
        def _calculate_melsp(x, n_fft=fft_params[0], hop_length=fft_params[1]):
            stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
            log_stft = librosa.power_to_db(stft)
            melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
            return melsp
        
        def _calculate_melsp_from_segments(segmented_wav_datas, n_fft=fft_params[0], hop_length=fft_params[1]):
            melsp_datas = []
            for wav_data in segmented_wav_datas:
                melsp_data = []
                for segment in wav_data:
                    melsp = _calculate_melsp(segment)
                    melsp_data.append(melsp)
                melsp_datas.append(melsp_data)
            
            return melsp_datas
            
        zero_padded = [_zero_pad(data) for data in wav_data]
        segmented_wav_datas = [_segment_wav(zp) for zp in zero_padded]
        melsp_datas = _calculate_melsp_from_segments(segmented_wav_datas)
        return melsp_datas
    
    def _load_data(self):
        # =============================================================================
        # Method for loading Emo-DB dataset from ./EmoDB.
        # returns wav file datas, corresponding labels and speakers (not constrained yet)
        # 
        # In Emo-DB, infomations of labels and speakers correspond to wav file names.
        # =============================================================================
        
        """
        Retrieving paths to every wav files.
        """
        base_dir = './EmoDB/wav/'
        wav_paths = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir)])
        
        
        """
        Retrieving infomations of labels and speakers.
        """
        # Method to extract other infomations from filename
        def _parse_names(names, from_i, to_i, is_number=False, mapping=None):
            for name in names:
                key = name[from_i:to_i]
                if is_number:
                    key = int(key)
                yield mapping[key] if mapping else key
                
        # correspondence between filename and labels
        emotion_mapping = {
            'W': 'Anger',
            'L': 'Boredom',
            'E': 'Disgust',
            'A': 'Fear',
            'F': 'Happiness',
            'T': 'Sadness',
            'N': 'Neutral',
        }
        
        filenames = [audeer.basename_wo_ext(p) for p in wav_paths]
        speakers = list(_parse_names(filenames, from_i=0, to_i=2, is_number=True))
        emotions = list(_parse_names(filenames, from_i=5, to_i=6, mapping=emotion_mapping))
        
        wav_datas = []
        for p in wav_paths:
            x, fs = librosa.load(p, sr=16000)
            wav_datas.append(x)
        
        return wav_datas, emotions, speakers
    
    def _load_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as pf:
            data = pickle.load(pf)
        return data
    
    def _save_pickle(self, data, pickle_path):
        with open(pickle_path, 'wb') as pf:
            pickle.dump(data, pf)
        
loader = EmoDB_loader()
