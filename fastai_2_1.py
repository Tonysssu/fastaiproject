import numpy as np
from fastai import *
import matplotlib as plt
import torch
# A = np.array([[-3,1],[-4,2]])
# B = np.linalg.inv(A)
# n = 500
# x = np.ones([n, 2])
# x[:,0] = np.random.uniform(size=(n))
# param = np.array([3, 2]).reshape(-1, 1)
# y = x@param + np.random.rand(n)

n = 100
x = torch.ones(n, 2)
x[:,0].uniform_(-1, 1)
a = tensor(3.0, 2.0)
y = x@a+ torch.rand(n)

# element-wise subtract and square
def mse(y_hat, y):
    return ((y_hat-y)**2).mean()

a = nn.Parameter(a);

def update():
    y_hat = x@a
    loss = mse(y_hat, y)
    loss.backward()
    with torch.no_grad(): ##turn gradient compution off when you do the SGD update
        a.sub_(lr*a.grad)
        a.grad.zero_()

lr = 1e^-1
for t in range(100):
    update()
