import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils import FrankeFunction, makeData, MSE, R2, OLS, makeFigure, plotFrankefunction
import matplotlib as mpl
import matplotlib.pyplot as plt










maxdegree = 5
n = 100
X, y, x_train, x_test, y_train, y_test  = makeData(n, rand=0.1)
polydegree = np.zeros(maxdegree)
trainError  = np.zeros(maxdegree)
testError  = np.zeros(maxdegree)
trainR2  = np.zeros(maxdegree)
testR2  = np.zeros(maxdegree)
betas = np.zeros((maxdegree,20))
scaler = StandardScaler()
y_train_mean = np.mean(y_train)
model = OLS()

for degree in range(maxdegree):
    poly = PolynomialFeatures(degree+1,include_bias=False)
    X_train = poly.fit_transform(x_train)
    X_test = poly.transform(x_test)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # type: ignore
    y_train_mean = np.mean(y_train)
    y_train_scaled = y_train - y_train_mean
    print(X_train.shape)
    model.fit(X_train, y_train_scaled)
    betas[degree] = np.pad(model.beta,(0,20-model.beta.size))
    polydegree[degree] = degree + 1
    testError[degree] = MSE(y_test, model.predict(X_test) + y_train_mean)
    trainError[degree] = MSE(y_train, model.predict(X_train) + y_train_mean)
    testR2[degree] = R2(y_test, model.predict(X_test) + y_train_mean)
    trainR2[degree] = R2(y_train, model.predict(X_train)+ y_train_mean)


mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '11',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})

fix1 , ax1 = plt.subplots()
ax1.set_title(r"$\mathbf{\beta}$ and model complexity")
ax1.set_xlabel("Degree")
ax1.set_ylabel(r"values of $\beta$")
ax1.set(xticks=polydegree)
ax1.plot(polydegree, betas)
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].set_xlabel("Degree")
ax[1].set_xlabel("Degree")
ax[1].set_ylabel("R2")
ax[0].set_ylabel("MSE")
ax[0].set_title(r"MSE")
ax[0].set(xticks=polydegree)
ax[1].set(xticks=polydegree)
ax[0].plot(polydegree, testError, label="test")
ax[0].plot(polydegree, trainError, label="train")
ax[0].legend()
ax[1].set_title("R2")
ax[1].plot(polydegree, testR2, label="test")
ax[1].plot(polydegree, trainR2, label="train")
ax[1].legend()
plt.show()




x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
poly = PolynomialFeatures(5,include_bias=False)
z = model.predict(scaler.transform(poly.fit_transform(np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T ))) + y_train_mean
fig = makeFigure((8,8))
plotFrankefunction(xx,yy,z.reshape(100,100), fig, (1,1,1) ,"Franke's Function")
plt.show()