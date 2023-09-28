import numpy as np
from utils import Ridge, OLS 
from sklearn.preprocessing import PolynomialFeatures


def CrossVal(designMatrix,y, folds =  5, regtype = "ols",l = 1,ls = 0, random = True):

    def linreg(trainx, trainy, valx, valy, regtype,l,ls):
        match regtype:
            case "ols":
                ols = OLS()
                ols.fit(trainx,trainy)
                ypred =ols.predict(valx)
                score[i] = MSE(valy,ypred)
            case "ridge":
                ridge = Ridge()
                for j in range( l):
                    ridge.fit(trainx,trainy,ls[j])
                    ypred =ridge.predict(valx)
                    score[i][j] = MSE(valy,ypred)
            case _:
                return



    if random:
        seed = np.random.randint(0,100000)

        rng = np.random.default_rng(seed=seed)
        rng.shuffle(designMatrix)
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(y)
    foldsize =y.size/folds
    at =0
    score = np.zeros((folds, l))
    for i in range(folds):
        start = int(np.floor(at))
        at += foldsize
        end = int(np.floor(at))
        trainx = np.delete(designMatrix,slice(start,end),axis=0)
        trainy = np.delete(y,slice(start,end))
        valx = designMatrix[start:end,:]
        valy = y[:,start:end]
        linreg(trainx,trainy, valx,valy,regtype,l,ls)
    
    return np.mean(score,axis=1
)



