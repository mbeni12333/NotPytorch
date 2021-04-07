import numpy as np
import pickle as pk


class F():
    
    def sigmoid(X):
        return 1.0 / (1.0 + np.exp(-X))


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

    def __call__(self, Y, Yhat):
        return self.forward(Y, Yhat)


class MSELoss(Loss):

    def __init__(self):
        return

    def forward(self, y, yhat):
        return np.square(np.linalg.norm(y - yhat, axis=1)).mean()

    def backward(self, y, yhat):

        assert y.shape == yhat.shape

        return 2*(yhat - y)
    
    
class BCELoss(Loss):

    def __init__(self, eps=1e-10):
        self.eps=eps

    def forward(self, y, yhat):
        
        assert y.shape == yhat.shape
        
        yhat = 1.0 / (1.0 + np.exp(-yhat))
        
        return -np.mean(y.T @ np.log(yhat+self.eps) +
                (1-y).T @ np.log(1.0-yhat+self.eps))

    def backward(self, y, yhat):

        assert y.shape == yhat.shape

        yhat = 1.0 / (1.0 + np.exp(-yhat))

        return yhat - y
    


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradients = None
        self._out = None

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
        pass

    def inference(self):
        pass

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

        self._parameters["W"] = np.random.randn(
            in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self._parameters["b"] = np.ones((1, out_dim))

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
        self._gradients["b"] += dZ.mean(0)

        return dZ @ self._parameters["W"].T

    def update_parameters(self, gradient_step=1e-3):

        self._parameters["W"] -= gradient_step*self._gradients["W"]
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
        return 2.0 / (1 + np.exp(-2*X)) - 1
            

    def backward_update_gradient(self, Z, dA):
        dZ = self.backward_delta(Z, dA)
        self.update_parameters()
        
        return dZ

    def backward_delta(self, Z, dA, lr=1e-3):
        A = self.forward(Z)
        return 1 - np.square(A)
    
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
        return dA * (Z <= 0.0)
    
    def update_parameters(self, gradient_step=1e-3):
        return
