from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import MSE, R2, FrankeFunction, makeData, Ridge, makeFigure, plotFrankefunction















maxdegree = 5
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
n = 100
numlamdas = 8
lamdas = np.logspace(1,-6,numlamdas)
X, y, x_train, x_test, y_train, y_test, xx, yy = makeData(n, rand=0.1)
polydegree = np.zeros(maxdegree)
trainError  = np.zeros((numlamdas,maxdegree))
testError  = np.zeros((numlamdas,maxdegree))
trainR2  = np.zeros((numlamdas, maxdegree))
testR2  = np.zeros((numlamdas, maxdegree))
betas = np.zeros((maxdegree,numlamdas,numfeatures))
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
        betas[degree,i] = np.pad(model.beta, (0, numfeatures -model.beta.size))
        polydegree[degree] = degree + 1
        testError[i,degree] = MSE(y_test, model.predict(X_test) + y_train_mean)
        trainError[i,degree] = MSE(y_train, model.predict(X_train) + y_train_mean)
        testR2[i,degree] = R2(y_test, model.predict(X_test) + y_train_mean)
        trainR2[i,degree] = R2(y_train, model.predict(X_train) + y_train_mean)

mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '16',
    'xtick.labelsize': '11',
    'ytick.labelsize': '11',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})
fig1 , ax1 = plt.subplots(5,1)
fig1.suptitle(r"$\mathbf{\beta}$ and model complexity for different $\lambda$")
fig1.text(0.5, 0.04, r'$\lambda$ for all $x$-axes', ha='center', va='center')
for i, ax in enumerate(ax1):
    ax.xticks = lamdas
    ax.set_xscale("log")
    ax.set_ylabel(r"$\beta$" + " degree " + str(i+1))
    ax.plot(lamdas, betas[i])



plt.show()


fig, ax = plt.subplots(2,4, figsize=(12,4))
fig.suptitle("R2 for different model complexity and $\lambda$")
fig.text(0.5, 0.04, r'Degree of polynomial for all $x$-axes', ha='center', va='center')
fig.text(0.06, 0.5, 'R2-score for all $y$-axes', ha='center', va='center', rotation='vertical')
for i,axi in enumerate(ax.flatten()):
    axi.set_xticks(polydegree)
    axi.plot(polydegree, trainR2[i], label="Train")
    axi.plot(polydegree, testR2[i], label="Test")
    axi.legend()
    axi.set_title(r"$\lambda$" + " = " + str(lamdas[i]))

plt.show()








poly = PolynomialFeatures(5, include_bias=False)
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # type: ignore
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean
model.fit(X_train, y_train_scaled, alpha=10)
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
z = FrankeFunction(xx,yy)

fig = makeFigure((8,8))
plotFrankefunction(xx,yy,model.predict(scaler.transform(poly.transform(X))).reshape(n,n) + y_train_mean, fig, (1,1,1) ,"Franke's Function")
plt.show()