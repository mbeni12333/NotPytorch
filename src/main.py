from nottorch import nn, utils
import nottorch



X, Y = utils.generate_linear()


model = nn.Linear(1, 1)



print(X.shape)
print(model._parameters)

criterion = nn.MSELoss()
#optim = nottorch.optim.Optim()


losses = utils.train(model, X, Y, criterion, epochs=2000)

utils.plot_losses((losses))


utils.show2Ddata(X, Y, W=model._parameters["W"], b=model._parameters["b"])