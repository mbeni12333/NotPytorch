from src.nottorch import nn, utils
from src.nottorch.Datasets.loader import *
from src import nottorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# In[] model linear

# X, Y = utils.generate_linear()


# model = nn.Linear(1, 1)


# criterion = nn.MSELoss()
# #optim = nottorch.optim.Optim()

# losses = utils.train(model, X, Y, criterion, epochs=2000)

# utils.plot_losses((losses))


# utils.show2Ddata(X, Y, W=model._parameters["W"], b=model._parameters["b"])

# In[] model Linear tanh linear sigmoid

# X, Y = utils.generate_classif(1000, n_clusters_per_class=1)
# utils.show2DdataClassif(X, Y)


# epochs = 1000

# Linear1 = nn.Linear(2, 2)
# Activation1 = nn.Tanh()
# Linear2 = nn.Linear(2, 1)
# Activation2 = nn.Sigmoid()

# Criterion = nn.BCELoss()
# #optim = nottorch.optim.Optim()

# losses = []
    
# for epoch in range(epochs):
    
#     print(f"Epoch : {epoch}/{epochs})")

#     Z1 = Linear1(X)
#     A1 = Activation1(Z1)
#     Z2 = Linear2(A1)
#     #A2 = Activation2(Z2)

#     #Yhat = A2
#     Yhat = Z2
    
#     loss = Criterion(Y, Yhat)
    
#     losses.append(loss)
    
#     print(f"loss = {loss}")
    
#     Linear1.zero_grad()
#     Linear2.zero_grad()
    
#     dZ2 = Criterion.backward(Y, Yhat)
#     dA1 = Linear2.backward_update_gradient(A1, dZ2)
#     dZ1 = Activation1.backward_update_gradient(Z1, dA1)
#     dA0 = Linear1.backward_update_gradient(X, dZ1)
    
    
# utils.plot_losses((losses))

# In[]

# X, Y = utils.generate_classif(1000, n_clusters_per_class=2)
# utils.show2DdataClassif(X, Y)

# epochs = 10000

# model = nn.Sequential([nn.Linear(2, 2),
#                         nn.Tanh(),
#                         nn.Linear(2, 2),
#                         nn.Tanh(),
#                         nn.Linear(2, 1)])


# Criterion = nn.BCELoss()
# #optim = nottorch.optim.Optim()

# X_transformed = np.hstack((X, (X[:, 0]**2).reshape(-1, 1), (X[:, 1]**2).reshape(-1, 1)))

# losses = []
    
# losses = utils.train(model, X, Y, Criterion, lr=0.1, epochs=epochs, print_every=100)  
    
# utils.plot_losses((losses))

# utils.show2DdataClassif(X, Y)
# utils.plot_frontiere(X_transformed, lambda x: nn.F.sigmoid(model(x.reshape(1, *(x.shape)))) >= 0.5)


# In[]

# alltrainx,alltrainy = load_usps("train")
# alltestx,alltesty = load_usps("test")
# neg, pos = 5, 6

# trainx,trainy = get_usps([neg,pos],alltrainx,alltrainy)
# testx,testy = get_usps([neg,pos],alltestx,alltesty)

# mu = trainx.mean(axis=0)
# sig = trainx.std(axis=0)

# trainx = (trainx - mu)/sig
# testx = (testx - mu)/sig

# trainy = np.where(trainy == pos, 1, 0)
# testy = np.where(testy == pos, 1, 0)

# show_usps(datax, datay, rows=16, cols=32)



# epochs = 1000

# model = nn.Sequential([nn.Linear(256, 128),
#                         nn.ReLU(),
#                         nn.Linear(128, 64),
#                         nn.ReLU(),
#                         nn.Linear(64, 1)])


# Criterion = nn.BCELoss()


# losses = utils.train(model, trainx, trainy, Criterion, epochs=epochs, print_every=10)


# utils.plot_losses((losses))


# yhat = np.where(nn.F.sigmoid(model(testx)) >= 0.5, 1, 0)
# utils.plot_report(testy, yhat, [neg, pos])

# In[]

# alltrainx,alltrainy = load_usps("train")
# alltestx,alltesty = load_usps("test")


# trainx,trainy = get_usps(list(range(10)),alltrainx,alltrainy)
# testx,testy = get_usps(list(range(10)),alltestx,alltesty)

# show_usps(trainx, trainy, rows=16, cols=32)

# mu = trainx.mean(axis=0)
# sig = trainx.std(axis=0)

# trainx = (trainx - mu)/sig
# testx = (testx - mu)/sig

# # trainy = utils.one_hot_encode(trainy, 10)
# # testy = utils.one_hot_encode(testy, 10)

# show_usps(trainx, trainy, rows=16, cols=32)

# epochs = 1000

# model = nn.Sequential([nn.Linear(256, 64),
#                        nn.ReLU(),
#                        nn.Linear(64, 10)])


# Criterion = nn.CCELoss()


# losses = utils.train(model, trainx, trainy, Criterion, lr=0.1, epochs=epochs, print_every=100)


# utils.plot_losses((losses))


# yhat = np.argmax(nn.F.softmax(model(testx)), axis=1)
# utils.plot_report(testy, yhat, list(range(10)))


# In[]

# X2, Y2 = utils.generate_checker(1000)
# utils.show2DdataClassif(X2, Y2)

# In[]
# X, Y = utils.generate_classif(1000, n_clusters_per_class=1, n_classes=4)


# utils.show2DdataClassif(X, Y)
# plt.show()

# epochs = 1000
# print_every = 100
# lr = 0.001

# model = nn.Sequential([nn.Linear(2, 2),
#                         nn.Tanh(),
#                         nn.Linear(2, 2),
#                         nn.Tanh(),
#                         nn.Linear(2, 4)])


# Criterion = nn.CCELoss()
# #optim = nottorch.optim.Optim()

# #X_transformed = np.hstack((X, (X[:, 0]**2).reshape(-1, 1), (X[:, 1]**2).reshape(-1, 1)))

# losses = []

# for epoch in range(epochs):
    
    

#     Yhat = model(X)
    
#     loss = Criterion(Y, Yhat)
    
#     losses.append(loss)
    
        
#     model.zero_grad()
    
#     dYhat = Criterion.backward(Y, Yhat)
    
#     model.backward_update_gradient(X, dYhat, lr)
    
    
#     if epoch % print_every == 0:
#         print(f"Epoch : {epoch}/{epochs}, loss = {loss}")
        
#         utils.show2DdataClassif(X, Y)
#         utils.plot_frontiere(X, lambda x: np.argmax(nn.F.softmax(model(x.reshape(-1, x.shape[1]))), axis=1), n_classes=4)
    

    
# utils.plot_losses((losses))

# In[]

X, Y = utils.generate_classif(1000, n_clusters_per_class=1, n_classes=4)

X2, Y2 = utils.generate_circles(1000, noise=0.1, factor=0.8)
Y2 += 4

X = np.vstack((X+2, X2-2))
Y = np.vstack((Y, Y2))
 
epochs = 1000

print_every = 100
lr = 0.1
cmap = "tab10"

model = nn.Sequential([nn.Linear(7, 7),
                        nn.Tanh(),
                        nn.Linear(7, 7),
                        nn.Tanh(),
                        nn.Linear(7, 6)])

Criterion = nn.CCELoss()
#optim = nottorch.optim.Optim()


transform = lambda x: np.hstack((x,
                                 (x[:, 0]**2).reshape(-1, 1),
                                 (x[:, 1]**2).reshape(-1, 1),
                                 ((x[:, 1]*x[:, 0]).reshape(-1, 1)),
                                 np.sin(x)))

X_transformed = transform(X)

losses = []

for epoch in range(epochs):
    
    

    Yhat = model(X_transformed) 
    
    loss = Criterion(Y, Yhat)
    
    losses.append(loss)
    
        
    model.zero_grad()
    
    dYhat = Criterion.backward(Y, Yhat)
    
    model.backward_update_gradient(X, dYhat, lr)
    
    
    if epoch % print_every == 0:
        print(f"Epoch : {epoch}/{epochs}, loss = {loss}")
        
        utils.show2DdataClassif(X, Y, cmap=cmap)
        utils.plot_frontiere(X,
                             lambda x: np.argmax(nn.F.softmax(model(transform(x))), axis=1),
                             n_classes=6,
                             cmap=cmap)
    

      
utils.plot_losses((losses))
utils.plot_report(Y, Yhat, list(range(6)))