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



np.random.seed(9282) # 9282 gives a nice plot textbook plot
n = 100
maxdegree = 6
n_boostraps = 100

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
    print(y_test.reshape(-1,1).shape)
    print(y_pred.shape)
    print((y_test.reshape(-1,1) - y_pred.shape).shape)
    polydegree[degree] = degree
    error[degree] = np.mean( np.mean((y_test.reshape(-1,1) - y_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=False))**2 )
    variance[degree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

fig, ax = plt.subplots()
ax.plot(polydegree, error,'--', label='Error', color='black')
ax.plot(polydegree, bias, label='bias', color='purple')
ax.plot(polydegree, variance, label='Variance', color='teal')
# ax.set_yscale('log')
plt.legend()
plt.show()




x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
poly = PolynomialFeatures(7,include_bias=False)
z = model.predict(scaler.transform(poly.fit_transform(np.concatenate((xx.ravel(), yy.ravel())).reshape(2,-1).T ))) + y_train_mean
fig = makeFigure((8,8))
plotFrankefunction(xx,yy,z.reshape(100,100), fig, (1,1,1) ,"Franke's Function")
# plt.show()