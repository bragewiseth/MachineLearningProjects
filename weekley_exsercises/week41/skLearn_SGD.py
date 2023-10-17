# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

np.random.seed(2020)

polydegree = 2
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x**2+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x**2] 

X = np.c_[np.ones((n,1)), x, x**2]
beta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
print("beta_linreg:", beta_linreg.ravel())
print("MSE (linreg): ",(1.0/n)*np.sum((y - X@beta_linreg)**2))

#learning_ratestr = "optimal" (eta = 1.0 / (alpha * (t + t0))) alpha = 1 er det vi gjør i de andre kodene
sgdreg = SGDRegressor(max_iter = 100, fit_intercept=False, eta0=0.01)
sgdreg.fit(X,y.ravel())
print("sklearnModel beta: ", sgdreg.coef_)
print("sklearnModel MSE: ", (1.0/n)*np.sum((y.ravel() - sgdreg.predict(X))**2))

 