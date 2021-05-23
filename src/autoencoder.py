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

epochs = 1000
Criterion = nn.BCELoss()
model = Autoencoder(10)
print_every = 300
lr = 5e-4

losses = []

cpt = 0

for epoch in range(epochs):
    
    for i, (X, Y) in enumerate(utils.generateBatches(trainx, trainx, batch_size=64)):

        Yhat = model(X)
        
        loss = Criterion(Y, Yhat)
        
        
        model.zero_grad()
        
        dYhat = Criterion.backward(Y, Yhat)
    
        model.backward_update_gradient(Yhat, dYhat, lr)
        
        if cpt % print_every == 0:
            print(f"Epoch : {epoch}/{epochs}, batch {i}, loss = {loss}")
            #show_usps(X, trainy, rows=2, cols=6)
            losses.append(loss)
            reconstruct = nn.F.sigmoid(Yhat)
            show_usps(reconstruct, Y, rows=2, cols=4)
            
            enc = model.encoder(testx)
            plt.scatter(enc[:, 0], enc[:, 1], c=testy, cmap="tab10")
            plt.show()
            
        cpt += 1
        
    
# In[]
plt.plot(losses)
plt.show()

# In[]
# enc = model.encoder(trainx)


# idx2 = np.argmax(trainy == 0)

# for i in range(1, 10):
    
#     idx1 = idx2
#     idx2 = np.argmax(trainy == i)
    
#     start = enc[idx1].reshape(1, -1)
#     end = enc[idx2].reshape(1, -1)
    
#     interpolation = start + (end - start)*np.linspace(0, 1, 50).reshape(-1, 1)
    
#     reconstruct = nn.F.sigmoid(model.decoder(interpolation))


#     for i in range(50):
#         show_usps(reconstruct[i].reshape(1, -1), trainy, rows=1, cols=1)


# In[]
from sklearn.manifold import TSNE

enc = model.encoder(trainx)

embd = TSNE(n_components=2).fit_transform(trainx)
# In[]
plt.scatter(embd[:, 0], embd[:, 1], c=trainy, cmap="tab10")
plt.show()

# In[]
plt.scatter(enc[:, 0], enc[:, 1], c=trainy, cmap="tab10")
plt.show()

# In[]
from sklearn.manifold import TSNE

out = model(trainx)

embd = TSNE(n_components=2).fit_transform(out)

plt.scatter(embd[:, 0], embd[:, 1], c=trainy, cmap="tab10")
plt.show()
