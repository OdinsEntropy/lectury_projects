# Description: Pytorch lecture
# Simple linear regression model in numpy and pytorch
# inspired by series https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

import torch
import numpy as np

# Parameter definition
mode = 'pytorch' # 'numpy' or 'pytorch'

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


def pytorch_model_implement(X,Y,w):

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


if __name__ == '__main__':

    if mode == 'numpy':
        numpy_model_implement(X,Y,w)

    elif mode == 'pytorch':
        pytorch_model_implement(torch.tensor(X),torch.tensor(Y),torch.tensor(w, requires_grad=True))
    else:
        print('Invalid mode')