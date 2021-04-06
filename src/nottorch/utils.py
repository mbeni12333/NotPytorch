from . import nn
import numpy as np
from sklearn.datasets import make_regression, load_diabetes
import matplotlib.pyplot as plt


def plot_losses(losses):
    
    
    plt.plot(losses)
    plt.show()
    

def generate_linear(n_samples=100, n_features=1, bias=True):
    
    
    X, Y = make_regression(n_samples=n_samples, n_features=n_features, bias=bias, noise=10)
    #X, Y = load_diabetes(return_X_y=True)
    
    return X, Y.reshape(-1, 1)


def show2Ddata(X, Y, W=None, b=None):
    
    plt.scatter(X, Y)
    
    z = np.linspace(-2, 2, 5).reshape(-1, 1)
    plt.plot(z, z@W + b, c="red")
    
    plt.show()

def one_hot_encode(y):
    """
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pass

def train(model:nn.Module, X, Y, Criterion, Optim=None, epochs=1000, verbose=False):
    """
    

    Parameters
    ----------
    model : nn.Module
        DESCRIPTION.
    Criterion : TYPE
        DESCRIPTION.
    Optim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    losses = []
    
    for epoch in range(epochs):
        
        print(f"Epoch : {epoch}/{epochs})")

        Yhat = model(X)
        
        loss = Criterion(Y, Yhat)
        
        losses.append(loss)
        
        print(f"loss = {loss}")
        
        dYhat = Criterion.backward(Y, Yhat)
        
        model.zero_grad()
        
        print(dYhat.shape, X.shape)
        
        model.backward_update_gradient(X, dYhat)
        
        
    return losses
        
        
        
        
        
        
        