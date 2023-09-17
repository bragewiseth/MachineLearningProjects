from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import MSE, R2, makeData, Ridge
















maxdegree = 5
n = 100
lamdas = np.logspace(1,0,10)
X, y, x_train, x_test, y_train, y_test, xx, yy = makeData(n, rand=0.1)
polydegree = np.zeros(maxdegree)
trainError  = np.zeros((maxdegree,10))
testError  = np.zeros((maxdegree,10))
trainR2  = np.zeros((maxdegree,10))
testR2  = np.zeros((maxdegree,10))
betas = np.zeros((maxdegree,20))
scaler = StandardScaler()
model = Ridge()

for i, l in enumerate(lamdas):
    for degree in range(maxdegree):
        poly = PolynomialFeatures(degree+1, include_bias=False)
        X_train = poly.fit_transform(x_train)
        X_test = poly.transform(x_test)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # type: ignore
        y_train_mean = np.mean(y_train)
        y_train_scaled = y_train - y_train_mean
        model.fit(X_train, y_train_scaled, alpha=l)
        betas[degree] = np.pad(model.beta, (0, 20-model.beta.size))
        polydegree[degree] = degree + 1
        testError[degree,i] = MSE(y_test, model.predict(X_test) + y_train_mean)
        trainError[degree,i] = MSE(y_train, model.predict(X_train) + y_train_mean)
        testR2[degree,i] = R2(y_test, model.predict(X_test) + y_train_mean)
        trainR2[degree,i] = R2(y_train, model.predict(X_train) + y_train_mean)


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
ax[0].set_title("MSE")
ax[0].set(xticks=polydegree)
ax[1].set(xticks=polydegree)
ax[0].plot(polydegree, testError[:,0], label="test")
ax[0].plot(polydegree, trainError[:,0], label="train")
ax[0].legend()
ax[1].set_title("R2")
ax[1].plot(polydegree, testR2[:,0], label="test")
ax[1].plot(polydegree, trainR2[:,0], label="train")
ax[1].legend()
plt.show()