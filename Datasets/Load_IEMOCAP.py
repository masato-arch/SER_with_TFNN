#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 21:44:48 2022

@author: Ark_001

"""

import os
from os.path import exists
import pickle
import re
import collections
import librosa
import audeer
import math
import numpy as np
import tqdm


class IEMOCAP_loader:
    
    def __init__(self, dataset_path='./Datasets'):
        
        """
        REQUIREMENS:
            1, Please put this file in the <project folder>/Datasets/
            2, Please put 'IEMOCAP_full_release' folder in abovementionded path.
        """
        # =============================================================================
        # Initialize the module
        # 
        # The module holds paths to pickle files of preprocessed wav, labels and melspectrograms
        # 
        # If we don't have pickle, the dataset will be retrieved from dataset_path/IEMOCAP_full_release.
        # =============================================================================
        self.dataset_path = dataset_path # the path in which IEMOCAP_full_release exists
        self.pickle_path = os.path.join(dataset_path, 'pickles') # the path to storage pickle files
        
        # the paths for data files
        self.wav_pickle_path = os.path.join(self.pickle_path, 'IEMOCAP_wav.pkl')
        self.melsp_pickle_path = os.path.join(self.pickle_path, 'IEMOCAP_melspectrogram.pkl')
        self.labels_pickle_path = os.path.join(self.pickle_path, 'IEMOCAP_labels.pkl')
        self.speakers_pickle_path = os.path.join(self.pickle_path, 'IEMOCAP_speakers.pkl')
        
        # if we don't have preprocessed pickle dataset, it is retrieved from <project folder>/Datasets/IEMOCAP_full_release
        if not exists(self.wav_pickle_path) or not exists(self.labels_pickle_path) or not exists(self.speakers_pickle_path):
            print('pkl files not found. retrieveing datasets.')
            
            # if we don't have dataset_path/pickles folder, create one
            if not exists(self.pickle_path):
                os.mkdir(self.pickle_path)
            
            # retrieve unconstraint (initial) wav, label and speaker datas from 'IEMOCAP_full_release'
            initial_wav_datas, initial_labels, initial_speakers, _ = self._load_data()
            
            # remove unwanted datas
            wav_datas, labels, speakers = self._constrain_dataset(initial_wav_datas, initial_labels, initial_speakers)
            
            # storage the cleansed datas
            print('saving datas as .pkl files...')
            self._save_pickle(wav_datas, self.wav_pickle_path)
            self._save_pickle(labels, self.labels_pickle_path)
            self._save_pickle(speakers, self.speakers_pickle_path)
            print('pickle files were successfully created.')
            
    # =============================================================================
    # User Interfaces:
    #   load_wav_dataaset(): returns dataset of raw wav audios, labels and speakers(if required)
    #   load_melsp_dataset(): returns dataset of 128x98 mel-spectrograms, labels and speakers(if required)
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
            
            print('saving mel spectrograms as pickle files...')
            self._save_pickle(melsp_datas, self.melsp_pickle_path)
            print('pickle file was successfully created.')
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
    def _constrain_dataset(self, wav_datas, labels, speakers, required_labels=['Anger', 'Happiness', 'Sadness', 'Neutral'], excite_happy_marge=True):
        # =============================================================================
        # Method to retrieve only required datas.
        # 
        # We only use utterances labeled with required classes, 
        #   which are ['Anger', 'Happiness', 'Sadness', 'Neutral']
        #   because the other classes doesn't have enough amount of data
        #
        # First, we marge 'Excited' with 'Happiness'
        #   since they are close in valence and activation domain.
        # =============================================================================
        
        if excite_happy_marge:
            for i in range(len(labels)):
                if labels[i] == 'Excited':
                    labels[i] = 'Happiness'
                    
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
            for wav_data in tqdm.tqdm(segmented_wav_datas):
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
        # Method for loading IEMOCAP dataset from ./IEMOCAP_full_release
        # returns wav file datas, corresponding labels and comments by evaluators     
        # =============================================================================
        
        """
        retrieving paths of wav files and label files
        """
        print('retrieving paths...')
        wav_dirs = [os.path.join(self.dataset_path, 'IEMOCAP_full_release', 'Session' + str(i), 'sentences', 'wav') for i in range(1, 6)]
        label_dirs = [os.path.join(self.dataset_path, 'IEMOCAP_full_release', 'Session' + str(i), 'dialog', 'EmoEvaluation', 'Categorical') for i in range(1, 6)]
        
        def _retrieve_paths(parent_paths, select_func=None):
            target_paths = []
            for path in parent_paths:
                a_target = sorted(os.listdir(path))
                for p in a_target:
                    if select_func(p):
                        target_paths.append(os.path.join(path, p))
    
            return target_paths
        
        def _is_wavdir(path):
            return not '.' in path

        def is_wavdir(path):
            return not '.' in path

        def _is_wavfile(path):
            _, ext = os.path.splitext(path)
            return ext == '.wav'

        def _is_txtfile(path):
            _, ext = os.path.splitext(path)
            return ext == '.txt'
        
        # paths for every utterance wav file
        wav_file_paths_p = _retrieve_paths(wav_dirs, select_func=_is_wavdir)
        wav_file_paths = _retrieve_paths(wav_file_paths_p, select_func=_is_wavfile)
        
        # wav filenames without extention
        wav_filename_wo_ext = [audeer.basename_wo_ext(p) for p in wav_file_paths]
        
        # paths for every label file 
        label_file_paths = _retrieve_paths(label_dirs, select_func=_is_txtfile)
        
        print('done.')
    
        """
        Retrieving labels for every utterance
        """
        
        print('retrieving labels for every utterance...')
        
        # name of every speech dialogs
        dialog_names = [audeer.basename_wo_ext(p) for p in wav_file_paths_p]
        
        # =============================================================================
        # Retrieving all label files for each dialog
        # Each dialog has 3 label files created by different evaluators
        # =============================================================================
        
        def _label_filepath_to_dialogname(path):
            filename = audeer.basename_wo_ext(path)
            dialogname = re.sub('_e[0-9]+_cat', '', filename)
            return dialogname
        
        dialogwise_labelfiles = []
        for dialog_name in dialog_names:
            labelfiles = [label_file_paths[i] for i in range(len(label_file_paths)) if dialog_name == _label_filepath_to_dialogname(label_file_paths[i])]
            dialogwise_labelfiles.append(labelfiles)
        
        # =============================================================================
        # wav file selection and labeling
        # 
        # We only use wav files which can be labeled by majority voting
        # In other words, we don't use files which don't have a majority label,
        # which means there is no agreement between evaluators
        # =============================================================================
        
        # =============================================================================
        # infomation about special wav files
        
        # Ses04M_script01_1_F032 (evaluator no.4 says he don't think this is a part of script)
        # Ses04M_script01_1_M032 (evaluator no.4 says he don't think this is a part of script)
        # Ses05F_impro02_F000 (evaluator no.1 leaves a comment:'anxious as in expecting something big to happen')
        # =============================================================================
        
        def _preprocess_line(line):
            comment = re.search('\(.*\)', line).group()[1:-1]
            line_wo_comment = re.sub(comment, '', line)
            line_wo_comment = re.sub('Neutral state', 'Neutral', line_wo_comment)
            line_wo_comment = re.split('[();:\n ]', line_wo_comment)
            line_cleansed = [elem for elem in line_wo_comment if elem != '']
            filename = line_cleansed.pop(0)
            if 'Other' in line_cleansed:
                idx = line_cleansed.index('Other')
                line_cleansed[idx] = 'Other:' + comment
                comment = 'Other:{0}'.format(comment)
            labels = line_cleansed
            return filename, labels, comment
        
        def _majority_voting(labels):
            counter = collections.Counter(labels)
            if len(counter) == 1:
                return labels[0]
            else:
                most_common = counter.most_common()
                if most_common[0][1] > most_common[1][1]:
                    return most_common[0][0]
                else:
                    # print(f'labels:{labels} rejected')
                    return None
        
        def _extract_speaker(filename):
            session_name = filename[:5]
            sex = filename[-4]
            return session_name + sex
                
        def _extract_values_from_lines(lines):
            filenames_tobe_same = []
            labels_for_majority_voting = []
            comments = ''
            for line in lines:
                filename, labels, comment = _preprocess_line(line)
                filenames_tobe_same.append(filename)
                labels_for_majority_voting.extend(labels)
                if comment != '':
                    comments += comment + '/'
            
            if len(set(filenames_tobe_same)) != 1:
                raise Exception(f'filename mismatch detected')
            else:
                filename = list(set(filenames_tobe_same))[0]
            
            label = _majority_voting(labels_for_majority_voting)
            speaker = _extract_speaker(filename)
            
            return filename, label, speaker, comments
        
        print('done.')
        print('removing unwanted datas...')
        
        approved_wav_paths = []
        approved_labels = []
        approved_speakers = []
        comments = {}
        bugfix_filenames = []
        for labelfiles in dialogwise_labelfiles:
            file_descriptors = [open(labelfile, 'r') for labelfile in labelfiles]
            while True:
                lines = [fd.readline() for fd in file_descriptors]
                if '' in lines:
                    break
                else:
                    filename, label, speaker, comments_ = _extract_values_from_lines(lines)
                    bugfix_filenames.append(filename)
                    if label != None:
                        # print(f'labels:{labels_}, approved')
                        wav_idx = wav_filename_wo_ext.index(filename)
                        approved_wav_paths.append(wav_file_paths[wav_idx])
                        approved_labels.append(label)
                        approved_speakers.append(speaker)
                        comments[filename] = comments_
        
        if bugfix_filenames == wav_filename_wo_ext:
            print('filenames are correctly retrieved')
        else:
            raise Exception('something is wrong in retrieving filenames')
        print('done.')
        print('loading wav files, this process may take long...')
        approved_wav_datas = []
        for path in tqdm.tqdm(approved_wav_paths):
            x, fs = librosa.load(path, sr=16000)
            approved_wav_datas.append(x)
        
        print('done.')
        return approved_wav_datas, approved_labels, approved_speakers, comments
    
    def _load_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as pf:
            data = pickle.load(pf)
        return data
    
    def _save_pickle(self, data, pickle_path):
        with open(pickle_path, 'wb') as pf:
            pickle.dump(data, pf)