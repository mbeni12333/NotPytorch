# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:05:46 2021

@author: Mounib Benimam
"""

from src.nottorch import nn, utils
from src.nottorch.Datasets.loader import *
from src import nottorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




class Autoencoder(nn.Module):
    
    def __init__(self, bottleneck=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            [nn.Linear(256, 128),
             nn.Tanh(),
             nn.Linear(128, 64),
             nn.Tanh(),
             nn.Linear(64, bottleneck),
             nn.Tanh()]
        )
        self.decoder = nn.Sequential(
            [nn.Linear(bottleneck, 64),
             nn.Tanh(),
             nn.Linear(64, 128),
             nn.Tanh(),
             nn.Linear(128, 256)]
        )
        
        
    def forward(self, X):
        
        X = self.encoder.forward(X)
        X = self.decoder.forward(X)
        
        return X
    
    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
    
    def backward_update_gradient(self, inputs, grads, lr=1e-5):
        
        dEncoder = self.backward_delta(inputs, grads)
        self.update_parameters(lr)
        
        return dEncoder

    def backward_delta(self, inputs, grads):


        dDecoder = self.decoder.backward_delta(inputs, grads)
        dEncoder = self.encoder.backward_delta(inputs, dDecoder)

        return dEncoder

    def update_parameters(self, gradient_step=1e-3):

        self.encoder.update_parameters(gradient_step)
        self.decoder.update_parameters(gradient_step)
    

# In[]       
        
alltrainx,alltrainy = load_usps("train")
alltestx,alltesty = load_usps("test")


trainx,trainy = get_usps(list(range(10)),alltrainx,alltrainy)
testx,testy = get_usps(list(range(10)),alltestx,alltesty)

show_usps(trainx, trainy, rows=16, cols=32)

mu = trainx.mean(axis=0)
sig = trainx.std(axis=0)

# trainx = (trainx - mu)/sig
# testx = (testx - mu)/sig
trainx = trainx/(np.ptp(trainx))
testx = testx/(np.ptp(testx))

#show_usps(trainx, trainy, rows=16, cols=32)

# In[]

epochs = 100
Criterion = nn.BCELoss()
model = Autoencoder(10)
print_every = 10
lr = 1e-3

losses = []

cpt = 0

for epoch in range(epochs):
    
    for i, (X, Y) in enumerate(utils.generateBatches(trainx, trainx)):

        Yhat = model(X)
        
        loss = Criterion(Y, Yhat)
        
        losses.append(loss)
        
        if cpt % print_every == 0:
            print(f"Epoch : {epoch}/{epochs}, batch {i}, loss = {loss}")
            show_usps(X, trainy, rows=2, cols=6)
            reconstruct = nn.F.sigmoid(model(X))
            show_usps(reconstruct, trainy, rows=2, cols=6)
        cpt += 1
        
        
        model.zero_grad()
        
        dYhat = Criterion.backward(Y, Yhat)
    
        model.backward_update_gradient(Yhat, dYhat, lr)
        
    
# In[]
plt.plot(losses)

