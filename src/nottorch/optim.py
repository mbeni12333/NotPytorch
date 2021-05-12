from src.nottorch import nn


class Optim():

    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self, batch_x, batch_y):
        
        out = self.net.forward( batch_x)
        grad_in = self.loss.backward(batch_y,out)
        self.net.backward_update_gradient(out, grad_in, self.eps)

