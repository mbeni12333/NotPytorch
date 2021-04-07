from . import nn
import numpy as np
from sklearn.datasets import make_regression, load_diabetes, make_classification
import matplotlib.pyplot as plt




def generateBatches(trainx, trainy, batch_size=32, shuffle=True):
    """
    

    Parameters
    ----------
    trainx : TYPE
        DESCRIPTION.
    trainy : TYPE
        DESCRIPTION.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 32.

    Returns
    -------
    None.

    """

    
    indicies = np.random.permutation(range(len(trainy)))
    
    
    
    data = [(trainx[indicies[i*batch_size:(i+1)*batch_size]].reshape(-1, trainx.shape[1]),
             trainy[indicies[i*batch_size:(i+1)*batch_size]].reshape(-1, 1))
                        
                        for i in range(len(indicies)//batch_size)]
    
    
    return data



def plot_losses(losses):
    
    
    plt.plot(losses)
    plt.show()
    

def generate_classif(n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=1):
    
    
    X, Y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               n_redundant=0,
                               class_sep=2)
    #X, Y = load_diabetes(return_X_y=True)
    
    return X, Y.reshape(-1, 1)

def generate_linear(n_samples=200, n_features=1, bias=True):
    
    
    X, Y = make_regression(n_samples=n_samples, n_features=n_features, bias=bias, noise=10)
    #X, Y = load_diabetes(return_X_y=True)
    
    return X, Y.reshape(-1, 1)



def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):

    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0])+1,  np.min(data[:,0])-1, np.max(data[:,1])+1, np.min(data[:,1])-1
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def plot_frontiere(data,f,step=200):

    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=('red','gray'),levels=[0,0.5,1], alpha=0.5)


def show2DdataClassif(X, Y):
    
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="Set1")    


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
        
        
        
        
        
        
        