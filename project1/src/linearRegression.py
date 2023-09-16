import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from random import random, seed
from utils import MSE, R2, FrankeFunction, fit_beta, fit_beta_ridge, MyStandardScaler, plotFrankefunction, makeFigure
import matplotlib



n = 100
degree = 5
Lambda = 0.1
np.random.seed()





# Make data set.
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xx , yy = np.meshgrid(x,y)
z = FrankeFunction(xx, yy) + 0.06 *np.random.randn(n,n)
X = np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T    # design matrix


# preprocessing
scaler = StandardScaler()   # own implementation of standard scaler
poly = PolynomialFeatures(degree,include_bias=True)

x_train, x_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2)
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)
X_train_scaled = scaler.fit_transform(X_train[:,1:])
X_test_scaled = scaler.transform(X_test[:,1:]) # type: ignore
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean


# LASSO
model = Lasso(fit_intercept=False, alpha=0)
clf = model.fit(X_train_scaled, y_train_scaled)
predict_lasso = clf.predict(scaler.transform(poly.transform(X)[:,1:])) + y_train_mean
# predict_lasso = clf.predict(poly.transform(X)) 
y_tilde_train_lasso = clf.predict(X_train_scaled) + y_train_mean
y_tilde_test_lasso = clf.predict(X_test_scaled) + y_train_mean


# OLS
beta_scaled = fit_beta(X_train_scaled,y_train_scaled)
beta = fit_beta(X_train, y_train)
predict = (scaler.transform(poly.transform(X)[:,1:]) @ beta_scaled) + y_train_mean # type: ignore 
ytilde_test_scaled = X_test_scaled @ beta_scaled + y_train_mean
ytilde_train_scaled = X_train_scaled @ beta_scaled + y_train_mean
ytilde_test = X_test @ beta
ytilde_train = X_train @ beta


# RIDGE
beta_ridge = fit_beta_ridge(X_train, y_train, Lambda)
beta_ridge_scaled = fit_beta_ridge(X_train_scaled, y_train_scaled, Lambda)
predict_ridge = (scaler.transform(poly.transform(X)[:,1:]) @ beta_ridge_scaled) + y_train_mean # type: ignore 
ytilde_test_scaled_ridge = X_test_scaled @ beta_ridge_scaled + y_train_mean
ytilde_train_scaled_ridge = X_train_scaled @ beta_ridge_scaled + y_train_mean
ytilde_test_rigde = X_test @ beta_ridge
ytilde_train_rigde = X_train @ beta_ridge







# showcase how scaling effects result
np.set_printoptions(precision=5, suppress=True, threshold=np.inf) # type: ignore
polyScaleShowcase = PolynomialFeatures(5)
A = polyScaleShowcase.fit_transform(x.reshape(-1,1))
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
print(MSE(y_test,y_tilde_test_lasso))




# plot FrankeFunction
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})
fig = makeFigure((15,10))
plotFrankefunction(xx,yy,z, fig, (1,2,1) ,"Data")
plotFrankefunction(xx,yy,predict_lasso.reshape(n,n), fig, (1,2,2), "lasso") 
fig1 = makeFigure((15,10))
plotFrankefunction(xx,yy,z, fig1, (1,2,1) ,"Data")
plotFrankefunction(xx,yy,predict.reshape(n,n), fig1, (1,2,2), "Ols") 
fig2 = makeFigure((15,10))
plotFrankefunction(xx,yy,z, fig2, (1,2,1) ,"Data")
plotFrankefunction(xx,yy,predict_ridge.reshape(n,n), fig2, (1,2,2), "Ridge") 
plt.show()
