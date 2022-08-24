#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:06:04 2022

@author: user
"""

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model_training_tools import *
import matplotlib.pyplot as plt

model = torch.nn.Sequential(
    torch.nn.Linear(1000, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 10),
    torch.nn.Softmax()
    )

train_datas = torch.randn(100, 1000)
train_corrects = torch.randint(0, 10, (100,))
train_dataset = TensorDataset(train_datas, train_corrects)
train_loader = DataLoader(train_dataset, batch_size=10)


test_datas = torch.randn(50, 1000)
test_corrects = torch.randint(0, 10, (50,))
test_dataset = TensorDataset(test_datas, test_corrects)
test_loader = DataLoader(test_dataset, batch_size=10)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

epochs = 100
train_loss = []
test_loss = []
elapsed_time = []
train_accuracy = []
test_accuracy = []

for epoch in range(epochs):
    train_loss_, elapsed_time_ = train_epoch(model, train_loader, criterion, optimizer, measure_time=(True))
    test_loss_, test_accuracy_, _ = valid(model, test_loader, criterion)
    train_loss.append(train_loss_)
    test_loss.append(test_loss_)
    test_accuracy.append(test_accuracy_)
    elapsed_time.append(elapsed_time_)
    
    
accuracy, cm = classification_test(model, test_loader)

plt.plot(test_accuracy)

