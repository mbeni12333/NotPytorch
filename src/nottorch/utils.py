"""
@author: Mounib Benimam
"""


import numpy as np
from sklearn.datasets import make_regression, make_classification, make_circles
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd


colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

def one_hot_encode(Y:np.array, num_classes:int=3) -> np.array:
    """
    Encode raw int labels to one hot vectors

    Parameters
    ----------
    Y : np.array
        Raw labels.
    num_classes : int, optional
       number of classes to encode. The default is 3.

    Returns
    -------
    np.array
        one hot encoded labels shape [numexemples, numclasses]

    """
    I = np.eye(num_classes)
    
    return I[Y]

def plot_report(y:np.array, y_hat:np.array, labels:list) -> None:
    """
    Show a report of the perfomance of the model

    Parameters
    ----------
    y : np.array
        True labels
    y_hatnp.array : np.array
        Predicted labels
    labels : list
        DESCRIPTION.

    Returns
    -------
    None

    """
    
    cm = confusion_matrix(y, y_hat)
    cr = classification_report(y, y_hat, output_dict=True)
    
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    sns.heatmap(cm, annot=True)
    plt.subplot(122)
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True)
    plt.show()


def generateBatches(trainx:np.array, trainy:np.array, batch_size:int=32, shuffle:bool=True):
    """
    generate batches of batchsize from data

    Parameters
    ----------
    trainx : np.array
        Dataset features shape [num exemple, datashape]
    trainy : np.array
        Dataset labels
    batch_size : int, optional
        Number of exemples per batch The default is 32.
    shuffle : bool, optional
        shuffle the indicies or not The default is True.

    Returns
    -------
    data : list(Tuple(np.array, np.array))
        list of the batches 

    """

    indicies = np.random.permutation(range(len(trainy)))
    
    
    
    data = [(trainx[indicies[i*batch_size:(i+1)*batch_size]].reshape(-1, trainx.shape[1]),
             trainy[indicies[i*batch_size:(i+1)*batch_size]].reshape(-1, trainy.shape[1]))
                        
                        for i in range(len(indicies)//batch_size)]
    
    
    return data



def plot_losses(losses:list):
    """
    

    Parameters
    ----------
    losses : list
        losses over time

    Returns
    -------
    None.

    """
    
    plt.plot(losses)
    plt.show()
    

def generate_checker(n_samples=1000, n=8):
    """

    Parameters
    ----------
    n_samples : TYPE, optional
        DESCRIPTION. The default is 1000.
    n : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    x = np.random.uniform(-(n//2)*np.pi, (n//2)*np.pi, size=(n_samples, 2))
    mask = 1* np.logical_or(
        np.logical_and(np.sin(x[:,0]) > 0.0, np.sin(x[:,1]) > 0.0),
        np.logical_and(np.sin(x[:,0]) < 0.0, np.sin(x[:,1]) < 0.0)
    )
    
    
    y = mask
    
    return x, y.reshape(-1, 1)


def generate_circles(n_samples=1000, noise=2, factor=0.5):
    """
    

    Parameters
    ----------
    n_samples : TYPE, optional
        DESCRIPTION. The default is 1000.
    noise : TYPE, optional
        DESCRIPTION. The default is 2.
    factor : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    x, y = make_circles(n_samples, noise=noise, factor=factor)
    
    return x, y.reshape(-1, 1)


def generate_classif(n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=1):
    """
    

    Parameters
    ----------
    n_samples : TYPE, optional
        DESCRIPTION. The default is 300.
    n_features : TYPE, optional
        DESCRIPTION. The default is 2.
    n_classes : TYPE, optional
        DESCRIPTION. The default is 2.
    n_clusters_per_class : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    X, Y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               n_redundant=0,
                               class_sep=2)
    #X, Y = load_diabetes(return_X_y=True)
    
    return X, Y.reshape(-1, 1)

def generate_linear(n_samples=200, n_features=1, bias=True):
    """
    

    Parameters
    ----------
    n_samples : TYPE, optional
        DESCRIPTION. The default is 200.
    n_features : TYPE, optional
        DESCRIPTION. The default is 1.
    bias : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    X, Y = make_regression(n_samples=n_samples, n_features=n_features, bias=bias, noise=10)
    #X, Y = load_diabetes(return_X_y=True)
    
    return X, Y.reshape(-1, 1)



def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """
    

    Parameters
    ----------
    data : TYPE, optional
        DESCRIPTION. The default is None.
    xmin : TYPE, optional
        DESCRIPTION. The default is -5.
    xmax : TYPE, optional
        DESCRIPTION. The default is 5.
    ymin : TYPE, optional
        DESCRIPTION. The default is -5.
    ymax : TYPE, optional
        DESCRIPTION. The default is 5.
    step : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    grid : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0])+1,  np.min(data[:,0])-1, np.max(data[:,1])+1, np.min(data[:,1])-1
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def plot_frontiere(data,f,step=200, n_classes=2, cmap="Set1"):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    step : TYPE, optional
        DESCRIPTION. The default is 200.
    n_classes : TYPE, optional
        DESCRIPTION. The default is 2.
    cmap : TYPE, optional
        DESCRIPTION. The default is "Set1".

    Returns
    -------
    None.

    """
    grid,x,y=make_grid(data=data,step=step)
    out = f(grid).reshape(x.shape)
    plt.contourf(x,y,out, colors=colors, vmin=0, vmax=10, alpha=0.3)
    plt.show()

def show2DdataClassif(X, Y, cmap="Set1"):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is "Set1".

    Returns
    -------
    None.

    """
    plt.scatter(X[:, 0], X[:, 1], c=colors[Y].reshape(-1))    

def show2Ddata(X, Y, W=None, b=None):
    """
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    W : TYPE, optional
        DESCRIPTION. The default is None.
    b : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    plt.scatter(X, Y)
    
    z = np.linspace(-2, 2, 5).reshape(-1, 1)
    plt.plot(z, z@W + b, c="red")
    
    plt.show()


def train(model,
          X:np.array, Y:np.array,
          Criterion, Optim=None,
          lr:float=1e-3, epochs:int=1000,
          verbose:bool=False, print_every:int=100,
          n_classe:int=2,
          batch_size=64) -> None:
    """
    Train a model using X, Y, and some criterion and optitimzer

    Parameters
    ----------
    model : nn.Module
        nottorch model
    X : np.array
        Train set
    Y : np.array
        train labels
    Criterion : nn.Loss
        Loss used
    Optim : TYPE, optional
        DESCRIPTION. The default is None.
    lr : float, optional
        Learning rate used. The default is 1e-3.
    epochs : int, optional
        Number of full passes over the dataset. The default is 1000.
    verbose : bool, optional
        Shoz debug information. The default is False.
    print_every : int, optional
        show progress. The default is 100.
    n_classe : int, optional
        number of classes. The default is 2.

    Returns
    -------
    losses : list
        Progression of the model

    """
    
    losses = []
    cpt = 0
    
    for epoch in range(epochs):
        
        for i, (batch, labels) in enumerate(generateBatches(X, Y, batch_size=batch_size)):
        
    
            Yhat = model(batch)
            
            loss = Criterion(labels, Yhat)
            
            
            if cpt % print_every == 0:
                print(f"Epoch : {epoch}/{epochs}, batch {i}, loss = {loss}")
                losses.append(loss)
                
            model.zero_grad()
            
            dYhat = Criterion.backward(labels, Yhat)
            
            model.backward_update_gradient(batch, dYhat, lr)
            
            cpt += 1
        
        
    return losses
        
        
        
        
        
        
        