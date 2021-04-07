from src.nottorch import nn, utils
from src.nottorch.Datasets.loader import *
from src import nottorch
import matplotlib.pyplot as plt
import numpy as np

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
# In[]

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


# epochs = 10000

# model = nn.Sequential([nn.Linear(4, 2),
#                        nn.Tanh(),
#                        nn.Linear(2, 2),
#                        nn.Tanh(),
#                        nn.Linear(2, 1)])


# Criterion = nn.BCELoss()
# #optim = nottorch.optim.Optim()

# X_transformed = np.hstack((X, (X[:, 0]**2).reshape(-1, 1), (X[:, 1]**2).reshape(-1, 1)))

# losses = []
    
# for epoch in range(epochs):
    
    


#     Z2 = model(X_transformed)
    
#     Yhat = Z2
    
#     loss = Criterion(Y, Yhat)
    
#     losses.append(loss)
    
#     model.zero_grad()
    
#     dA_last = Criterion.backward(Y, Yhat)   
    
#     dX = model.backward_update_gradient(Z2, dA_last, lr=1e-3)
    
#     if epoch % 100 == 0:
        
#         print(f"Epoch : {epoch}/{epochs}, loss = {loss}")
        
#         utils.show2DdataClassif(X, Y)
#         utils.plot_frontiere(X_transformed, lambda x: nn.F.sigmoid(model(x.reshape(1, *(x.shape)))) >= 0.5)
#         plt.show()
    
    
# utils.plot_losses((losses))

# utils.show2DdataClassif(X, Y)
# utils.plot_frontiere(X_transformed, lambda x: nn.F.sigmoid(model(x.reshape(1, *(x.shape)))) >= 0.5)


# In[]

alltrainx,alltrainy = load_usps("train")
alltestx,alltesty = load_usps("test")
neg, pos = 5, 6

datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
testx,testy = get_usps([neg,pos],alltestx,alltesty)


datay = np.where(datay == pos, 1, 0)
testy = np.where(testy == pos, 1, 0)

show_usps(datax, datay, rows=16, cols=32)
