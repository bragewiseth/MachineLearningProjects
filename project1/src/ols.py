import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from myStandardScaler import MyStandardScaler
from utils import MSE, R2, FrankeFunction, fit_beta, fit_beta_ridge




n = 100
degree = 5
Lambda = 1
np.random.seed()





x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
xx , yy = np.meshgrid(x,y)
z = FrankeFunction(xx, yy) + 0.06 *np.random.randn(n,n)
# Make data set.


fig = plt.figure(figsize=(15,8))
fig.tight_layout()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.title.set_text("Data to fit")
# Plot the surface.
surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, # type: ignore
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)






X = np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T

scaler = StandardScaler()       # sklearn scaler
myScaler = MyStandardScaler()   # own implementation of standard scaler
poly = PolynomialFeatures(degree,include_bias=True)


# fit polynomial and scale
x_train, x_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2)
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)
X_train_scaled = scaler.fit_transform(X_train[:,1:])
X_test_scaled = scaler.transform(X_test[:,1:]) # type: ignore
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean


# fit beta
beta_scaled = fit_beta(X_train_scaled,y_train_scaled)
beta = fit_beta(X_train, y_train)
beta_ridge = fit_beta_ridge(X_train, y_train, Lambda)
beta_ridge_scaled = fit_beta_ridge(X_train_scaled, y_train_scaled, Lambda)

predict = (scaler.transform(poly.transform(X)[:,1:]) @ beta_scaled) + y_train_mean # type: ignore 
predict2d = predict.reshape(n,n)
ytilde_scaled = X_test_scaled @ beta_scaled + y_train_mean
ytilde = X_test @ beta
ytilde_scaled_ridge = X_test_scaled @ beta_ridge_scaled + y_train_mean
ytilde_rigde = X_test @ beta_ridge




# ax = fig.gca(projection='3d')
ax = fig.add_subplot(1,2,2,projection='3d')
ax.title.set_text("Model")

surf = ax.plot_surface(xx, yy, predict2d, cmap=cm.coolwarm, # type: ignore
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()



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


print(MSE(y_test,ytilde ))
print(MSE(y_test,ytilde_scaled ))
print(MSE(y_test,ytilde_rigde))
print(MSE(y_test,ytilde_scaled_ridge))

