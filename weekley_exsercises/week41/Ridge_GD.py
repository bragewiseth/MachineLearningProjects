# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
import pandas as pd


learning_rate = [ 0.001, 0.01, 0.1, 0.25]
momentum = [0.0, 0.1, 0.5, 0.9]
change = 0.0
mse = []
lamda_list = [0.0001, 0.001, 0.01, 0.1, 1.0]


# the number of datapoints
for i in range(len(momentum)):
    mse_tmp = []
    for j in range(len(learning_rate)):
        mse_tmp_tmp = []
        for k in range(len(lamda_list)):
            polydegree = 2
            n = 100
            x = 2*np.random.rand(n,1)
            y = 4+3*x**2+np.random.randn(n,1)

            X = np.c_[np.ones((n,1)), x, x**2] 

            beta = np.random.randn(polydegree+1,1)

            eta = learning_rate[j]#learning rate
            Niterations = 1000

            lamda = lamda_list[k]

            for iter in range(Niterations):
                gradient = (2.0/n)*X.T@(X@beta - y) + 2*lamda*beta
                change = eta*gradient + momentum[j]*change
                beta -= change

            ypredict = X.dot(beta)

            mse_tmp_tmp.append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
        
        mse_tmp.append(mse_tmp_tmp)
        
    mse.append(mse_tmp)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

for i in range(len(mse)):
    panda = pd.DataFrame(data=mse[i])
    print("momentum = ", momentum[i])  
    print(panda)

print("\nmin mse:",np.min(mse))