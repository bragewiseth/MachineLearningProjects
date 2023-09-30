from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import MSE, R2, makeData, Ridge, makeFigure, plotFrankefunction, OLS







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





def CrossVal(designMatrix,y, folds =  5, regtype = "ols",l = 1,ls = 0, random = True):

    def linreg(trainx, trainy, valx, valy, regtype,l,ls):
        match regtype:
            case "ols":
                ols = OLS()
                ols.fit(trainx,trainy)
                ypred =ols.predict(valx)
                score[i] = mse(valy,ypred)
            case "ridge":
                ridge = ridge()
                for j in range( l):
                    ridge.fit(trainx,trainy,ls[j])
                    ypred =ridge.predict(valx)
                    score[i][j] = mse(valy,ypred)
            case _:
                return



    if random:
        seed = np.random.randint(0,100000)

        rng = np.random.default_rng(seed=seed)
        rng.shuffle(designmatrix)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(y)
    foldsize =y.size/folds
    at =0
    score = np.zeros((folds, l))
    for i in range(folds):
        start = int(np.floor(at))
        at += foldsize
        end = int(np.floor(at))
        trainx = np.delete(designmatrix,slice(start,end),axis=0)
        trainy = np.delete(y,slice(start,end))
        valx = designmatrix[start:end,:]
        valy = y[:,start:end]
        linreg(trainx,trainy, valx,valy,regtype,l,ls)
    
    return np.mean(score,axis=1
)


