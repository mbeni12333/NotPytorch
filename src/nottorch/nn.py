import numpy as np
import pickle as pk


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



class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradients = None
        self._out = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
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
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, X, dZ):
        ## Calcul la derivee de l'erreur
        pass
    
    
class Linear(Module):
    
    def __init__(self, in_dim, out_dim):
        
        super().__init__()
        
        self._parameters = {}
        self._gradients = {}
        
        self._parameters["W"] = np.random.randn(in_dim,out_dim) * np.sqrt(2 / in_dim)
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

        
    
    def backward_update_gradient(self, A, dZ):
       dA = self.backward_delta(A, dZ)
       self.update_parameters()
    
    def backward_delta(self, A, dZ):
        
        batch_size = dZ.shape[0]
        
        self._gradients["W"] += A.T @ dZ / batch_size
        self._gradients["b"] += dZ.mean(0)
        
        print(dZ.shape, self._parameters["W"].shape)
    
        return dZ @ self._parameters["W"]
    
    def update_parameters(self, gradient_step=1e-3):
        
        self._parameters["W"] -= gradient_step*self._gradients["W"]
        self._parameters["b"] -= gradient_step*self._gradients["b"]
        
    

