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
it = []
# the number of datapoints
for i in range(len(momentum)):
    plts.append([])
    it.append([])
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
        Niterations = 100000
        iterations = range(Niterations)
        se = False
        for iter in iterations:
            
            gradient = (2.0/n)*X.T @ (X @ beta-y)
            change = eta*gradient + momentum[i]*change
            beta -= change
            plts[i][j].append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
            
            if iter !=0 and abs(plts[i][j][iter] -plts[i][j][iter-1])<1e-100 and not se:
                it[i].append(iter)
                se = True
            
        if not se:
            it[i].append(iter)
        ypredict = X.dot(beta)

        mse_tmp.append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
        
    mse.append(mse_tmp)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)
print(len(it))
print("\nmin mse:",np.min(mse))
plots,ax = plt.subplots(int(len(learning_rate)//2),2)
for i in range(len(learning_rate)):
    
    for j in range(len(momentum)):
        ite = np.max(np.array(it)[:,i])
        ax[int(i//2),i%2].plot(range(ite-1),plts[j][i][0:ite-1])
plt.show()
