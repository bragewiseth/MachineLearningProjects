# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(2020)

mse = []
epochs = [1, 10, 50, 100, 500, 1000]

polydegree = 2
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x**2+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x**2] 

H = (2.0/n)* X.T @ X
EigValues, EigVectors = np.linalg.eig(H)

for n_epochs in epochs:

    beta = np.random.randn(polydegree+1,1)
    eta = 0.01/np.max(EigValues)
    M = 5   #size of each minibatch
    m = int(n/M) #number of minibatches
    t0, t1 = 5, 50

    def learning_schedule(t):
        return t0/(t+t1)
    for epoch in range(n_epochs):
    # selects a random mini-batch at every epoch. it does not garanty that all the data will be used
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            eta = learning_schedule(epoch*m+i)
            beta = beta - gradients*eta
    
    mse.append((1.0/n)*np.sum((y - X@beta)**2))

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)

print(np.min(mse))