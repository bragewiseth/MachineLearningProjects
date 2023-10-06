from numpy.core.fromnumeric import argmax, argmin, size
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
from utils import FrankeFunction, readData , MSE, R2, OLS,  plotFrankefunction


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n



np.random.seed(9898)
X, y  = readData("../data/syntheticData.csv")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


maxdegree = 6
n_boostraps = 100
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
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
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test.reshape(-1,1) - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=False))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )



# print best MSE:
i =  np.argmin(error)
print("Best MSE = {:25} \tfor polynomial of degree = {}".format(error[i],  i+1))



mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '16',
    'xtick.labelsize': '14',
    'ytick.labelsize': '14',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})



fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel('Model complexity')
ax.set_ylabel('Error')
ax.set_title('Bias-variance tradeoff')
ax.plot(polydegree, error,'--', label='Error', color='black')
ax.plot(polydegree, bias, label='bias', color='purple')
ax.plot(polydegree, variance, label='Variance', color='teal')
# ax.set_yscale('log')
plt.legend()
plt.savefig('../runsAndAdditions/bias-variance1.png')








# THE FOLLOWING CODE VISUALIZES THE BIAS-VARIANCE TRADEOFF

model= OLS()
maxdegree = [3,8]
n_boostraps = 100
n = 100
x = np.linspace(0,1,n)
y = np.full(n, 0.5)
z =  FrankeFunction(x,y) 
scaler = StandardScaler()
znoise = z + np.random.normal(0,0.08,n)
A = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
x_train , x_test, z_train, z_test = train_test_split(A, znoise, test_size=0.2)
z_train_mean = np.mean(z_train)
z_train = z_train - y_train_mean

yys = np.zeros((2,n,n_boostraps))
fig, ax = plt.subplots(1,2,figsize=(20,10))


for j , degree in enumerate(maxdegree):
    ax = ax.flatten()
    poly = PolynomialFeatures(degree ,include_bias=False)
    X_train = poly.fit_transform(x_train)
    X_train = scaler.fit_transform(X_train)
    X = poly.fit_transform(A)
    X = scaler.transform(X)

    for i in range(n_boostraps):
        x_, y_ = resample(X_train, z_train)
        model.fit(x_, y_)
        yys[j,:,i] = model.predict(X) + z_train_mean
        ax[j].plot(x, yys[j,:,i], '-' ,color='lightsteelblue', alpha=0.5, zorder=0)
    
    ax[j].fill_between(
        x, np.mean(yys[j], axis=1) - np.std(yys[j], axis=1), 
        np.mean(yys[j], axis=1) + np.std(yys[j], axis=1), 
        alpha=0.2, color='blue', 
        label=r'$\pm 1$ std. dev. of $\mathbf{\tilde{y}}$'
    )

    ax[j].plot(x, np.mean(yys[j], axis=1), '--', color='navy', label=r'$\mathbb{E}[\mathbf{\tilde{y}}]$')
    ax[j].plot(x, z, label=r'true function $\mathbf{f}$', color='red', linewidth=3)
    ax[j].text(0.6, 0.02, r'Bias: ' + str( np.round(np.mean( (z - np.mean(yys[j], axis=1))**2 ), 6 )), size=20) 
    ax[j].text(0.6, 0.05, r'Variance: ' + str(np.round( np.mean( np.var(yys[j], axis=1) ), 6 )), size=20)
    ax[j].set_ylim(0.0,0.7)
    ax[j].set_xlabel(r'$x$',size=30)
    ax[j].set_ylabel(r'$y$',size=30)
    ax[j].set_title(r'Polynomial of degree ' + str(degree))
    ax[j].legend()

fig.suptitle(r'Bias-variance tradeoff for different model complexity', size=20)



plt.savefig('../runsAndAdditions/bias-variance2.png')



