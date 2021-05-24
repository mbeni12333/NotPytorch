"""
@author: Mounib Benimam
"""



import numpy as np
import pickle as pk
from .utils import one_hot_encode

class F():
    """
    Functional static operations
    """
    
    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))
    def softmax(X, dim=1):
        tmp = np.exp(X)
        return tmp/tmp.sum(dim, keepdims=True)


class Loss(object):
    """
    Base class for defining loss
    """
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

    def __call__(self, Y, Yhat):
        return self.forward(Y, Yhat)


class MSELoss(Loss):
    """
    Mean Square Loss usefull for regression
    """

    def __init__(self):
        return

    def forward(self, y:np.array, yhat:np.array):
        """
        

        Parameters
        ----------
        y : np.array
            True Labels
        yhat : np.array
            Predicted Labels

        Returns
        -------
        float
            return the mean square of the error

        """
        return np.square(np.linalg.norm(y - yhat, axis=1)).mean()

    def backward(self, y:np.array, yhat:np.array):
        """
        Backward pass of the loss

        Parameters
        ----------
        y : np.array
            True labels
        yhat : np.array
            predicted labels

        Returns
        -------
        TYPE
           Gradient of the loss with respect to yhat

        """
        assert y.shape == yhat.shape

        return 2*(yhat - y)
    
    
class BCELoss(Loss):
    """
    Binary Cross Entropy loss usefull for binary classification
    """

    def __init__(self, eps=1e-10):
        self.eps=eps

    def forward(self, y:np.array, yhat:np.array):
        """
        Expect yhat to be raw output of linear layer

        Parameters
        ----------
        y : np.array
            True Labels
        yhat : np.array
            output of last linear layer

        Returns
        -------
        np.array
            return Binary Cross Entropy

        """
        
        assert y.shape == yhat.shape
        
        yhat = 1.0 / (1.0 + np.exp(-yhat))
        
        return -np.mean(y.T @ np.log(yhat+self.eps) +
                (1-y).T @ np.log(1.0-yhat+self.eps))

    def backward(self, y:np.array, yhat:np.array) -> np.array:
        """
        Expects yhat to be raw output of the network, 
        calculate directly the gradient with respect to the last layer
        instead of multitpling dyhat*dz which introduces numerical instability

        Parameters
        ----------
        y : np,array
             
        yhat : np.array
            output of linear layer

        Returns
        -------
        TYPE
            return dZL last linear layer derivative

        """

        assert y.shape == yhat.shape

        yhat = 1.0 / (1.0 + np.exp(-yhat))

        return yhat - y
    
    
class CCELoss(Loss):
    """
    Categorical Cross Entropy loss, usefull for multiclass classification
    """
    def __init__(self, eps=1e-10):
        self.eps=eps

    def forward(self, y:np.array, yhat:np.array) -> float:
        
        y = one_hot_encode(y.reshape(-1), yhat.shape[1]) 
        
        assert y.shape == yhat.shape
        
        yhat = F.softmax(yhat)
        
        return -(y*np.log(yhat)).sum(1).mean()
        
        #return -((y*yhat).sum(keepdims=True) + np.log(np.exp(yhat).sum(1, keepdims=True))).mean()

    def backward(self, y:np.array, yhat:np.array):
        """
        Return directly dZL instead of multiploying dl/dyhat * dyhat/dZL which introduces numerical instability 

        Parameters
        ----------
        y : np.array
            raw labels (integers)
        yhat : np.array
            last linear layers output

        Returns
        -------
        np.array
            Gradient dZL last linear layer

        """

        y = one_hot_encode(y.reshape(-1), yhat.shape[1])        

        assert y.shape == yhat.shape

        yhat = F.softmax(yhat)

        return yhat - y
    


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradients = None
        self._out = None
        self.isTraining = True

    def zero_grad(self):
        # Annule gradient
        pass

    def forward(self, X):
        # Calcule la passe forward
        pass

    def __call__(self, X):
        return self.forward(X)

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    def train(self):
        isTraining = True
        
    def inference(self):
        isTraining = False

    def update_parameters(self, gradient_step=1e-3):
        # Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta, lr=1e-3):
        # Met a jour la valeur du gradient
        pass

    def backward_delta(self, X, dZ):
        # Calcul la derivee de l'erreur
        pass


class Sequential(Module):
    
    def __init__(self, Modules):
        super().__init__()
        self.layers = Modules
        self._out = {}


    def train(self):
        for i, layer in enumerate(self.layers):
            layer.train()
        
    def inference(self):
        for i, layer in enumerate(self.layers):
            layer.inference()
        
    def forward(self, X):
        self._out = {}
        self._out[-1] = X
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(X)
            else:
                out = layer(out)
            
            self._out[i] = out

        return out

    def zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.zero_grad()

    def backward_update_gradient(self, input_in, grad_in, lr=1e-3):
        dX = self.backward_delta(input_in, grad_in)
        self.update_parameters(lr)
        
        return dX

    def backward_delta(self, input_in, grad_in):
        
        for i, layer in reversed(list(enumerate(self.layers))):
            grad_out = layer.backward_delta(self._out[i-1], grad_in)
            grad_in = grad_out
            
        return grad_out

    def update_parameters(self, lr=1e-3):
        for i, layer in enumerate(self.layers):
            layer.update_parameters(lr)


class Linear(Module):

    def __init__(self, in_dim, out_dim):

        super().__init__()

        self._parameters = {}
        self._gradients = {}

        self._parameters["W"] = np.random.randn(in_dim,out_dim) * np.sqrt(2.0 / in_dim)
        self._parameters["b"] = np.zeros((1, out_dim))

        return

    def forward(self, X):

        W = self._parameters["W"]
        b = self._parameters["b"]

        self._out = X@W + b

        return self._out

    def zero_grad(self):

        if self._gradients is not None:
            self._gradients["W"] = np.zeros(self._parameters["W"].shape)
            self._gradients["b"] = np.zeros(self._parameters["b"].shape)

    def backward_update_gradient(self, A, dZ, lr=1e-5):
        dA = self.backward_delta(A, dZ)
        self.update_parameters(lr)
        
        return dA

    def backward_delta(self, A, dZ):

        batch_size = dZ.shape[0]

        self._gradients["W"] += A.T @ dZ / batch_size
        self._gradients["b"] += dZ.mean(0, keepdims=True)

        return dZ @ self._parameters["W"].T

    def update_parameters(self, gradient_step=1e-3):

        self._parameters["W"] -= gradient_step*self._gradients["W"]
        self._parameters["b"] -= gradient_step*self._gradients["b"]
        
  
        
  
    
class Conv1D(Module):

    def __init__(self, k_size,chan_in=3, chan_out=32, stride=1):

        super().__init__()

        self._parameters = {}
        self._gradients = {}
        
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        
        self._parameters["F"] = np.random.randn(self.chan_out,  self.chan_in, *k_size) * np.sqrt(2.0 / chan_in)
        self._parameters["b"] = np.zeros(1, self.chan_out)

    def forward(self, X):

        b, c, h, w = X.shape        

        F = self._parameters["F"]
        b = self._parameters["b"]

        
    
        sliding = np.lib.stride_tricks.sliding_window_view(X, window_shape=(1, self.chan_in, *(self.k)))
        #sliding = sliding[:, :, ::self.stride[0], ::stride[1]]
        
        res = (sliding * F[None, :, None, None, None, :, :, :]).sum((4, 5, 6, 7))


        return res

    def zero_grad(self):
        self._parameters["F"] = np.zeros_like(self._parameters["F"])
        self._parameters["b"] = np.zeros_like(self._parameters["b"])
        return

    def backward_update_gradient(self, A, dZ, lr=1e-5):
        dA = self.backward_delta(A, dZ)
        self.update_parameters(lr)
        
        return dA

    def backward_delta(self, inputs, grads_in):
        
        b, c, h, w = inputs.shape 
        
        return

    def update_parameters(self, gradient_step=1e-3):

        self._parameters["F"] -= gradient_step*self._gradients["F"]
        self._parameters["b"] -= gradient_step*self._gradients["b"]
    
  



class Sigmoid(Module):
    

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))
          

    def backward_update_gradient(self, Z, dA):
        dZ = self.backward_delta(Z, dA)
        self.update_parameters()
        
        return dZ

    def backward_delta(self, Z, dA):

        A = self.forward(Z)
        dZ = (A*(1-A)) * dA
        return dZ
    
    def update_parameters(self, gradient_step=1e-3):
        return


    
class Tanh(Module):
    

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 2.0 / (1.0 + np.exp(-2.0*X)) - 1.0
            

    def backward_update_gradient(self, Z, dA):
        dZ = self.backward_delta(Z, dA)
        self.update_parameters()
        
        return dZ

    def backward_delta(self, Z, dA, lr=1e-3):
        A = self.forward(Z)
        return (1.0 - np.square(A)) * dA
    
    def update_parameters(self, gradient_step=1e-3):
        return
    
    
    
class ReLU(Module):
    

    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.maximum(0, X)
            

    def backward_update_gradient(self, Z, dA, lr=1e-3):
        dZ = self.backward_delta(Z, dA)
        self.update_parameters()
        
        return dZ

    def backward_delta(self, Z, dA):
        return dA * (Z > 0.0)
    
    def update_parameters(self, gradient_step=1e-3):
        return
    
    
class Dropout1D(Module):
    """
    Regularization layer
    """
    def __init__(self, proba=0.5):
        super().__init__()
        self.proba = proba

    def forward(self, X):
        if self.isTraining:
            self.mask = np.random.rand(1, X.shape[1]) <= self.proba
            return X*self.mask
        return X
            
    def backward_update_gradient(self, input, grad_in, lr=1e-3):
        grad_out = self.backward_delta(input, grad_in) 
        return grad_out

    def backward_delta(self, input, grad_in):
         if self.isTraining:
             return self.mask*grad_in
         return grad_in
    
    def update_parameters(self, gradient_step=1e-3):
        return
    
    
    
    
    
    
    
    
