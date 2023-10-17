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

delta  = 1e-8
rho = 0
for n_epochs in epochs:

    beta = np.random.randn(polydegree+1,1)
    eta = 0.01/np.max(EigValues)
    M = 5   #size of each minibatch
    m = int(n/M) #number of minibatches
    t0, t1 = 5, 50

    def learning_schedule(t):
        return t0/(t+t1)
    
    beta1 = 0.9
    beta2 = 0.999
    iter = 0

    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        iter += 1
    # selects a random mini-batch at every epoch. it does not garanty that all the data will be used
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            first_moment = beta1*first_moment + (1-beta1)*gradients
            second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
            first_term = first_moment/(1.0-beta1**iter)
            second_term = second_moment/(1.0-beta2**iter)
            # Scaling with rho the new and the previous results
            update = eta*first_term/(np.sqrt(second_term)+delta)
            beta -= update

    
    mse.append((1.0/n)*np.sum((y - X@beta)**2))

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)

print(np.min(mse))