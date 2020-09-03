#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:23:16 2019

@author: wuzhixin
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
INPUT_SIZE = 28
TIME_SIZE = 28

#train_data = torchvision.datasets.MNIST(
 #       root='./mnist/',
#        train=True,
 #       transform=torchvision.transforms.ToTensor(),
  #      download=True,
#

#train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE ,shuffle=True)



class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        
        self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=64,
                num_layers=1,
                batch_first=True,)
        self.out = nn.Linear(64,10)
        
        def forward(self,x):
            r_out, (h_n,h_c) = self.rnn(x,None)#lstm包含两个hidden_state
            out = self.out(r_out[:,-1,:])#(batch,time step,input)
rnn = RNN()
print(rnn)

net = RNN()

optimizer = torch.optim.Adam(net.parameters, lr=LR)
loss_func = torch.nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x,b_y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)#reshape x to batch,time_step,input_size
        out = rnn(x)
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        optimizer.step()                    
        
