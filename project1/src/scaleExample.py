import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import random, seed
from utils import MSE, fit_beta, fit_beta_ridge ,   makeData



n = 100
maxdegree = 5
Lambda = 0.1
np.random.seed()





# Make data set.
X , y, _,_,_,_ ,xx,yy = makeData(n)


# preprocessing
scaler = StandardScaler()
poly = PolynomialFeatures(maxdegree,include_bias=True)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)
X_train_scaled = scaler.fit_transform(X_train[:,1:])
X_test_scaled = scaler.transform(X_test[:,1:]) # type: ignore
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean


# OLS
beta_scaled = fit_beta(X_train_scaled,y_train_scaled)
beta = fit_beta(X_train, y_train)
predict = (scaler.transform(poly.transform(X)[:,1:]) @ beta_scaled) + y_train_mean # type: ignore 
ytilde_test_scaled = X_test_scaled @ beta_scaled + y_train_mean
ytilde_train_scaled = X_train_scaled @ beta_scaled + y_train_mean
ytilde_test = X_test @ beta
ytilde_train = X_train @ beta





# # RIDGE
beta_ridge = fit_beta_ridge(X_train, y_train, Lambda)
beta_ridge_scaled = fit_beta_ridge(X_train_scaled, y_train_scaled, Lambda)
predict_ridge = (scaler.transform(poly.transform(X)[:,1:]) @ beta_ridge_scaled) + y_train_mean # type: ignore 
ytilde_test_scaled_ridge = X_test_scaled @ beta_ridge_scaled + y_train_mean
ytilde_train_scaled_ridge = X_train_scaled @ beta_ridge_scaled + y_train_mean
ytilde_test_rigde = X_test @ beta_ridge
ytilde_train_rigde = X_train @ beta_ridge







# SHOWCASE HOW SCALING AFFECTS RESULTS
np.set_printoptions(precision=5, suppress=True, threshold=np.inf) # type: ignore
polyScaleShowcase = PolynomialFeatures(5)
A = polyScaleShowcase.fit_transform(np.linspace(0,1,5).reshape(-1,1))
A_scaled = scaler.fit_transform(A)
print(A)
print(A_scaled)
print(beta)
print(beta_scaled)
print(beta_ridge_scaled)
print(beta_ridge)

print(MSE(y_test,ytilde_test ))
print(MSE(y_test,ytilde_test_scaled ))
print(MSE(y_test,ytilde_test_rigde))
print(MSE(y_test,ytilde_test_scaled_ridge))