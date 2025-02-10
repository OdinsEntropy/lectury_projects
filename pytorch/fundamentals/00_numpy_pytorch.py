# Description: Pytorch lecture
# Simple linear regression model in numpy and pytorch
# inspired by series https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

import torch
import torch.nn as nn
import numpy as np

# Parameter definition
mode = 'pytorch' # 'numpy' or 'partial_pytorch' or 'pytorch'

learning_rate = 0.1
n_iters = 20

# Data generation

X = np.array([1,2,3,4])
Y = np.array([2,4,6,8])

w = 0.0

# Numpy model implementation, calc loss, derive, update weight with negative gradient
def numpy_model_implement(x,y,w):

    def forward(x):
        return w*x

    def loss(y, y_predicted):
        return ((y_predicted - y)**2).mean()

    def gradient(x, y, y_predicted):
        return np.mean((2*x) * (y_predicted - y))


    print(f'Prediction before training: f(5) = {forward(5):.3f}')

    for epoch in range(n_iters):
        y_pred = forward(X)
        l = loss(Y, y_pred)
        dw = gradient(X, Y, y_pred)

        w -= learning_rate * dw

        if epoch % 1 == 0:
            print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(5) = {forward(5):.3f}')

# Partial Pytorch model implementation, calc loss, backward, update weight with negative gradient
def pytorch_partial_model_implement(X,Y,w):

    def forward(x):
        return w*x

    def loss(y, y_predicted):
        return ((y_predicted - y)**2).mean()

    print(f'Prediction before training: f(5) = {forward(5):.3f}')

    for epoch in range(n_iters):
        y_pred = forward(X)
        l = loss(Y, y_pred)

        l.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad

        

        if epoch % 1 == 0:
            print(w.grad)
            #print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

        w.grad.zero_()

    print(f'Prediction after training: f(5) = {forward(5):.3f}')

# Full Pytorch model implementation, automatic differentiation, calc loss, backward, update weight with optimizer
def pytorch_model_implement(X,Y):

    class LinearRegressionOwn(nn.Module):
            def __init__(self):
                super(LinearRegressionOwn, self).__init__()
                self.linear = nn.Linear(1,1)

            def forward(self, x):
                return self.linear(x)
            
    # Define model, loss function and optimizer
    model = LinearRegressionOwn()
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_iters):
        y_pred = model(X)
        l = loss(Y, y_pred)

        l.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 1 == 0:
            [w,b] = model.parameters()
            print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')




if __name__ == '__main__':

    if mode == 'numpy':
        numpy_model_implement(X,Y)

    elif mode == 'partial_pytorch':
        pytorch_partial_model_implement(torch.tensor(X),torch.tensor(Y),torch.tensor(w, requires_grad=True))
    
    elif mode == 'pytorch':
        pytorch_model_implement(torch.tensor(X, dtype=torch.float32).view(-1,1),torch.tensor(Y, dtype=torch.float32).view(-1,1))
    else:
        print('Invalid mode')