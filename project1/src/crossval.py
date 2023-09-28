from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import MSE, R2, makeData, Ridge, makeFigure, plotFrankefunction














np.random.seed(9282)
maxdegree = 5
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
n = 100
numlamdas = 50
lamdas = np.logspace(1,-6,numlamdas)
X, y, x_train, x_test, y_train, y_test = makeData(n, rand=0.1)
poly = PolynomialFeatures(maxdegree, include_bias=False)
scaler = StandardScaler()
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # type: ignore
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean
scaler = StandardScaler()
model = Ridge()

k = 5
kfold = KFold(n_splits = k)
for trainind, testind in kfold.split(X):
    print("TRAIN:", trainind, "TEST:", testind)
