import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils import FrankeFunction, makeData, MSE, R2, OLS, makeFigure, plotFrankefunction


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n




# maxdegree = 5
X, y  = readData("../data/syntheticData.csv")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9282)


maxdegree = 6
n_boostraps = 100
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
X, y, x_train, x_test, y_train, y_test  = makeData(n, rand=0.1)
polydegree = np.zeros(maxdegree)
trainError  = np.zeros(maxdegree)
testError  = np.zeros(maxdegree)
trainR2  = np.zeros(maxdegree)
testR2  = np.zeros(maxdegree)
scaler = StandardScaler()
y_train_mean = np.mean(y_train)
model = OLS()

for degree in range(maxdegree):
    
    poly = PolynomialFeatures(degree+1,include_bias=False)
    X_train = poly.fit_transform(x_train)
    X_train = scaler.fit_transform(X_train)
    X_test = poly.transform(x_test)
    X_test = scaler.transform(X_test)
    y_pred = np.empty((y_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)
        y_ = y_ - y_train_mean
        x_ = poly.transform(x_)
        x_ = scaler.transform(x_)
        model.fit(x_, y_)
        y_pred[:, i] = ( model.predict(X_test) + y_train_mean  )
        # print(ybeta[:, i])
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test.reshape(-1,1) - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=False))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

fig, ax = plt.subplots()
ax.plot(polydegree, error,'--', label='Error', color='black')
ax.plot(polydegree, bias, label='bias', color='purple')
ax.plot(polydegree, variance, label='Variance', color='teal')
# ax.set_yscale('log')
plt.legend()
plt.show()





mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '16',
    'xtick.labelsize': '11',
    'ytick.labelsize': '11',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})



maxdegree = 3
n_boostraps = 100
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)

x = np.linspace(0,1,100)
y = np.full(100, 0.5)
poly = PolynomialFeatures(maxdegree,include_bias=False)

z =  FrankeFunction(x,y) 
znoise = z + np.random.normal(0,0.1,100)
X = np.c_[x,y]
X_train , X_test, z_train, z_test = train_test_split(X, znoise, test_size=0.2)
y_train_mean = np.mean(z_train)
z_train = z_train - y_train_mean
X_train = poly.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
X = poly.fit_transform(X)
X = scaler.transform(X)
yys = np.zeros((100,n_boostraps))
fig, ax = plt.subplots()
for i in range(n_boostraps):
    x_, y_ = resample(X_train, z_train)
    model.fit(x_, y_)
    yys[:,i] = model.predict(X) + y_train_mean
    ax.plot(x, yys[:,i], '-' ,color='lightsteelblue', alpha=0.5, zorder=0)

ax.fill_between(x, np.mean(yys, axis=1) - np.std(yys, axis=1), np.mean(yys, axis=1) + np.std(yys, axis=1), alpha=0.2, color='blue', label=r'$\pm 1$ std. dev. of $\mathbf{\tilde{y}}$')
ax.plot(x, np.mean(yys, axis=1), '--', color='navy', label=r'$\mathbb{E}[\mathbf{\tilde{y}}]$')
ax.plot(x, z, label=r'true function $\mathbf{f}$', color='forestgreen', linewidth=3)
ax.set_ylim(0.0,0.7)
plt.legend()
plt.show()
print("variance: ", np.mean( np.var(yys, axis=1) ))
print("bias: ", np.mean( (z - np.mean(yys, axis=1))**2 ))








"""
We can see that
"""
