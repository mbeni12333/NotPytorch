from src.nottorch import nn, utils
from src import nottorch

# In[] model linear

# X, Y = utils.generate_linear()


# model = nn.Linear(1, 1)


# criterion = nn.MSELoss()
# #optim = nottorch.optim.Optim()

# losses = utils.train(model, X, Y, criterion, epochs=2000)

# utils.plot_losses((losses))


# utils.show2Ddata(X, Y, W=model._parameters["W"], b=model._parameters["b"])

# In[] model Linear tanh linear sigmoid

X, Y = utils.generate_classif()
utils.show2DdataClassif(X, Y)
# In[]

epochs = 1000

Linear1 = nn.Linear(2, 2)
Activation1 = nn.Tanh()
Linear2 = nn.Linear(2, 1)
Activation2 = nn.Sigmoid()

Criterion = nn.BCELoss()
#optim = nottorch.optim.Optim()

losses = []
    
for epoch in range(epochs):
    
    print(f"Epoch : {epoch}/{epochs})")

    Z1 = Linear1(X)
    A1 = Activation1(Z1)
    Z2 = Linear2(A1)
    A2 = Activation2(Z2)

    Yhat = A2
    
    loss = Criterion(Y, Yhat)
    
    losses.append(loss)
    
    print(f"loss = {loss}")
    
    Linear1.zero_grad()
    Linear2.zero_grad()
    
    dZ2 = Criterion.backward(Y, Yhat)
    dA1 = Linear2.backward_update_gradient(A1, dZ2)
    dZ1 = Activation1.backward_update_gradient(Z1, dA1)
    dA0 = Linear1.backward_update_gradient(X, dZ1)
    
    
utils.plot_losses((losses))


