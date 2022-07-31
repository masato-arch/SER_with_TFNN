#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:52:02 2022

@author: Masato Arasaki
"""
import os
from os.path import exists
import audeer
import re
import collections
import librosa
import math
import numpy as np
import pickle

"""
NOTE: This file is 
"""

"""
retrieving paths of wav files and label files
"""

wav_dirs = [os.path.join('IEMOCAP_full_release', 'Session' + str(i), 'sentences', 'wav') for i in range(1, 6)]
label_dirs = [os.path.join('IEMOCAP_full_release', 'Session' + str(i), 'dialog', 'EmoEvaluation', 'Categorical') for i in range(1, 6)]


def retrieve_paths(parent_paths, select_func=None):
    target_paths = []
    for path in parent_paths:
        a_target = sorted(os.listdir(path))
        for p in a_target:
            if select_func(p):
                target_paths.append(os.path.join(path, p))
    
    return target_paths

def is_wavdir(path):
    return not '.' in path

def is_wavfile(path):
    _, ext = os.path.splitext(path)
    return ext == '.wav'

def is_txtfile(path):
    _, ext = os.path.splitext(path)
    return ext == '.txt'

# paths for every utterance wav file
wav_file_paths_p = retrieve_paths(wav_dirs, select_func=is_wavdir)
wav_file_paths = retrieve_paths(wav_file_paths_p, select_func=is_wavfile)

# wav filenames without extention
wav_filename_wo_ext = [audeer.basename_wo_ext(p) for p in wav_file_paths]

# paths for every label file 
label_file_paths = retrieve_paths(label_dirs, select_func=is_txtfile)


"""
retrieving labels for every utterance
"""

# names of all the speech dialogs
dialog_names = [audeer.basename_wo_ext(p) for p in wav_file_paths_p]

# =============================================================================
# Retrieving all label files for each dialog
# Each dialog has 3 label files created by different evaluators
# =============================================================================
def label_filepath_to_dialogname(path):
    filename = audeer.basename_wo_ext(path)
    dialogname = re.sub('_e[0-9]+_cat', '', filename)
    return dialogname

dialogwise_labelfiles = []
for dialog_name in dialog_names:
    labelfiles = [label_file_paths[i] for i in range(len(label_file_paths)) if dialog_name == label_filepath_to_dialogname(label_file_paths[i])]
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

def preprocess_line(line):
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

def majority_voting(labels):
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

def extract_speaker(filename):
    session_name = filename[:5]
    sex = filename[-4]
    return session_name + sex
            
def extract_values_from_lines(lines):
    filenames_tobe_same = []
    labels_for_majority_voting = []
    comments = ''
    for line in lines:
        filename, labels, comment = preprocess_line(line)
        filenames_tobe_same.append(filename)
        labels_for_majority_voting.extend(labels)
        if comment != '':
            comments += comment + '/'
    
    if len(set(filenames_tobe_same)) != 1:
        raise Exception(f'filename mismatch detected')
    else:
        filename = list(set(filenames_tobe_same))[0]
    
    label = majority_voting(labels_for_majority_voting)
    speaker = extract_speaker(filename)
    
    return filename, label, speaker, comments

approved_wav_paths = []
approved_labels = []
approved_speakers = []
comments = {}
for labelfiles in dialogwise_labelfiles:
    file_descriptors = [open(labelfile, 'r') for labelfile in labelfiles]
    while True:
        lines = [fd.readline() for fd in file_descriptors]
        if '' in lines:
            break
        else:
            filename, label, speaker, comments_ = extract_values_from_lines(lines)
            if label != None:
                # print(f'labels:{labels_}, approved')
                wav_idx = wav_filename_wo_ext.index(filename)
                approved_wav_paths.append(wav_file_paths[wav_idx])
                approved_labels.append(label)
                approved_speakers.append(speaker)
                comments[filename] = comments_

wav_datas = []
for path in approved_wav_paths:
    x, fs = librosa.load(path, sr=16000)
    wav_datas.append(x)

## skip this part (bugfix log)
## =============================================================================
##
## bug_dialog_list = ['Ses04M_script01_1', 'Ses05F_impro02']
## bug_dialog_idx = [dialog_names.index(dialog_name) for dialog_name in bug_dialog_list]
## bug_dialog_labelfiles = [dialogwise_labelfiles[idx] for idx in bug_dialog_idx]
## bug_labels = []
##
## for labelfiles in bug_dialog_labelfiles:
##     file_descriptors = [open(labelfile, 'r') for labelfile in labelfiles]
##     while True:
##         lines = [re.sub('Neutral state', 'Neutral', fd.readline()).split() for fd in file_descriptors]
##         if [] in lines:
##             break
##         filenames = [line.pop(0) for line in lines]
##      
##         if len(filenames) != filenames.count(filenames[0]):
##             raise Exception(f'filename mismatch was detected.')
##            
##         label = extract_label_from_lines(lines)
##         if label != None:
##             bug_labels.append(label)
## =============================================================================

# skip this part (design updated)
# =============================================================================
# """
# saving dataset using pickle
# """
# def savefile_if_not_exists(data, pickle_path):
#     if not os.path.exists(pickle_path):
#         with open(pickle_path, 'xb') as pf:
#             pickle.dump(data, pf)
#    
# wav_pickle_path = 'IEMOCAP_approved_wav.pkl'
# label_pickle_path = 'IEMOCAP_approved_labels.pkl'
# comments_pickle_path = 'IEMOCAP_comments.pkl'
#
# savefile_if_not_exists(wav_data, wav_pickle_path)
# savefile_if_not_exists(approved_labels, label_pickle_path)
# savefile_if_not_exists(comments, comments_pickle_path)
# =============================================================================

"""
select only the datas labeled with required classes
"""
# =============================================================================
# First, marge 'Excited' with 'Happiness'
#   since they are close in valence and activation domain.
# 
# We only use utterances labeled with required classes, 
#   which are ['Anger', 'Happiness', 'Sadness', 'Neutral']
#   because the other classes doesn't have enough amount of data
# =============================================================================
def dataset_with_required_classes(datas, labels, speakers, required_labels=['Anger', 'Happiness', 'Sadness', 'Neutral']):
    req_datas, req_labels, req_speakers = [], [], []
    for data, label, speaker in zip(datas, labels, speakers):
        if label in required_labels:
            req_datas.append(data)
            req_labels.append(label)
            req_speakers.append(speaker)
    
    return req_datas, req_labels, req_speakers

for i in range(len(approved_labels)):
    if approved_labels[i] == 'Excited':
        approved_labels[i] = 'Happiness'

required_datas, required_labels, required_speakers = dataset_with_required_classes(wav_datas, approved_labels, approved_speakers)

"""
create spectrogram dataset
"""

# =============================================================================
# calculate mel spectrogram from wav data
#
# first, segment the audio into fixed-length segments (default is 50000)
# (frame rate is 16000, so 50000 frame is 3.125 sec segment)
# then calculate the mel spectrograms from each segment using librosa (n_fft=2048, hop_length=512)
# this process takes several minutes
# =============================================================================

def _calculate_melsp_datas(wav_data, len_segment=50000, fft_params=(2048, 512)):
    
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

# melsp_datas = _calculate_melsp_datas(necessary_datas)

"""
Save processed files with pickle
"""
wav_pickle_path = 'IEMOCAP_wav.pkl'
labels_pickle_path = 'IEMOCAP_labels.pkl'
speakers_pickle_path = 'IEMOCAP_speakers.pkl'

def save_pickle(datas, pickle_path):
    if not exists(pickle_path):
        with open(pickle_path, 'wb') as pf:
            pickle.dump(datas, pf)

save_pickle(required_datas, wav_pickle_path)
save_pickle(required_labels, labels_pickle_path)
save_pickle(required_speakers, speakers_pickle_path)