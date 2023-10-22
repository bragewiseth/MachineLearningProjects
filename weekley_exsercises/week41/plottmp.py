# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
seed =np.random.randint(0,10000)
np.random.seed(seed)

learning_rate = [0.0001, 0.001, 0.01, 0.1]
momentum = [0.0, 0.1, 0.5, 0.9]
change = 0.0
mse = []
plts = []
# the number of datapoints
for i in range(len(momentum)):
    plts.append([])
    mse_tmp = []
    
    for j in range(len(learning_rate)):
        np.random.seed(seed)
        plts[i].append([])
        polydegree = 2
        n = 100
        x = 2*np.random.rand(n,1)
        y = 4+3*x**2+np.random.randn(n,1)

        X = np.c_[np.ones((n,1)), x, x**2] 

        beta = np.random.randn(polydegree+1,1)

        eta = learning_rate[j]#learning rate
        Niterations = 100
        iterations = range(Niterations)
        for iter in iterations:
            gradient = (2.0/n)*X.T @ (X @ beta-y)
            change = eta*gradient + momentum[i]*change
            beta -= change
            plts[i][j].append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
        
        ypredict = X.dot(beta)

        mse_tmp.append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
        
    mse.append(mse_tmp)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)
print("\nmin mse:",np.min(mse))
plots,ax = plt.subplots(int(len(learning_rate)//2),2)
for i in range(len(momentum)):
    for j in range(len(learning_rate)):
        ax[int(j//2),j%2].plot(iterations,plts[i][j])
plt.show()
