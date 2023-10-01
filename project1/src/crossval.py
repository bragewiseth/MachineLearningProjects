from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils import shuffle
from utils import MSE, R2, Ridge, makeFigure, plotFrankefunction, OLS, readData, printGrid
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score




X, y  = readData("../data/syntheticData.csv")


maxdegree = 5
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
numlamdas = 50
lambdas = np.logspace(1,-6,numlamdas)


scaler = StandardScaler()
model = Ridge()
polydegree = np.zeros(maxdegree)
trainError  = np.zeros((maxdegree,numlamdas))
testError  = np.zeros((maxdegree,numlamdas))
trainR2  = np.zeros((maxdegree,numlamdas))
testR2  = np.zeros((maxdegree,numlamdas))



k = [5,6,7,8,9,10]
kfold = KFold(n_splits = k)
scores_KFold = np.zeros((maxdegree, numlamdas, k))


for i, lmb in enumerate(lambdas):
    ridge = Ridge()
    ols = OLS()
    las = Lasso(alpha = lmb)     
    for j , degree in enumerate(range(maxdegree)):
        poly = PolynomialFeatures(degree+1, include_bias=False)
        for trainind, testind in kfold.split(X):
            x_train, x_test = X[trainind], X[testind]
            y_train, y_test = y[trainind], y[testind]
            X_train = poly.fit_transform(x_train)
            X_train = scaler.fit_transform(X_train)
            X_test = poly.transform(x_test)
            X_test = scaler.transform(X_test)  # type: ignore
            y_train_mean = np.mean(y_train)
            y_train_scaled = y_train - y_train_mean
            model.fit(X_train, y_train_scaled, alpha=lmb)
            y_pred = model.predict(X_test) + y_train_mean 
            y_predTrain = model.predict(X_train) + y_train_mean
            scores_KFold[j,i] = MSE(y_test, y_pred )
            polydegree[degree] = degree + 1
            testError[degree,i] = MSE(y_test, y_pred )
            trainError[degree,i] = MSE(y_train, y_predTrain )
            testR2[degree,i] = R2(y_test, y_pred )
            trainR2[degree,i] = R2(y_train, y_predTrain )



estimated_mse_KFold = np.mean(scores_KFold, axis = 1)
estimated_mse_sklearn = np.zeros(nlambdas)
for i, lmb in enumerate(lambdas):
    ridge = Ridge(alpha = lmb)

    X = poly.fit_transform(x[:, np.newaxis])
    estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1




skModel = SkRidge(fit_intercept=False, alpha= lamdas[numlamdas-1])
skModel.fit(X_train,y_train_scaled)

# compare ours with sklearnModel
print("sklearnModel beta: ", skModel.coef_)
print("our beta: ", betas[-1,-1])
print("sklearnModel MSE: ", MSE(y_test, skModel.predict(X_test) + y_train_mean))
print("our MSE: ", testError[-1,-1])
print("sklearnModel R2: ", R2(y_test, skModel.predict(X_test) + y_train_mean))
print("our R2: ", testR2[-1,-1])






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


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

# kfold is an instance initialized above as:
# kfold = KFold(n_splits = k)

estimated_mse_sklearn = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)

    X = poly.fit_transform(x[:, np.newaxis])
    estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

