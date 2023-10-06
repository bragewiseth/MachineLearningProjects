from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils import shuffle
from utils import MSE, R2, Ridge , OLS, readData, printGrid
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge as SkRidge
from sklearn.model_selection import cross_val_score
from matplotlib.ticker import FuncFormatter

def findParmas(error,  R2score , polydegree, lamdas=None):
    for i in range(len(polydegree)):
        errori  = np.argmin(error[i])    # find index of minimum test error (best fit)
        R2i  = np.argmax(R2score[i])     # find index of maximum test R2 (best fit)
        print("Degree of polynomial = ", polydegree[i])
        if lamdas is None:
            print("Best error = {:25}".format(error[i,errori]))
            print("Best R2 = {:26}".format(R2score[i,R2i]))
        else:
            print("Best error = {:25} \tfor λ = {}".format(error[i,errori],  lamdas[errori]))
            print("Best R2 = {:26} \tfor λ = {}".format(R2score[i,R2i],  lamdas[R2i]))






X, y  = readData("../data/syntheticData.csv")


maxdegree = 5
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
numlamdas = 20
lambdas = np.logspace(1,-5,numlamdas)
polydegree = np.array((1,2,3,4,5))

scaler = StandardScaler()
karray = [2,5,7,10,15,20]
estimatedFoldsOLS = np.zeros((maxdegree,numlamdas, len(karray)))
estimatedFoldsR2OLS = np.zeros((maxdegree,numlamdas, len(karray)))
estimatedFoldsRidge = np.zeros((maxdegree,numlamdas, len(karray)))
estimatedFoldsR2Ridge = np.zeros((maxdegree,numlamdas, len(karray)))
estimatedFoldsLasso = np.zeros((maxdegree,numlamdas, len(karray)))
estimatedFoldsR2Lasso = np.zeros((maxdegree,numlamdas, len(karray)))

bestk = np.zeros(len(karray))

for n, k_i in enumerate(karray):
    kfold = KFold(k_i)
    scores_KFold_R2OLS = np.zeros((maxdegree, numlamdas, k_i))
    scores_KFoldOLS = np.zeros((maxdegree, numlamdas, k_i))
    scores_KFold_R2Ridge = np.zeros((maxdegree, numlamdas, k_i))
    scores_KFoldRidge = np.zeros((maxdegree, numlamdas, k_i))
    scores_KFold_R2Lasso = np.zeros((maxdegree, numlamdas, k_i))
    scores_KFoldLasso = np.zeros((maxdegree, numlamdas, k_i))

    for i, lmb in enumerate(lambdas):
        ridge = Ridge()
        ols = OLS()
        lasso = Lasso(alpha = lmb ,max_iter=10000 )     
        for j , degree in enumerate(range(maxdegree)):
            poly = PolynomialFeatures(degree+1, include_bias=False)
            k = 0
            for trainind, testind in kfold.split(X):
                x_train, x_test = X[trainind], X[testind]
                y_train, y_test = y[trainind], y[testind]
                X_train = poly.fit_transform(x_train)
                X_train = scaler.fit_transform(X_train)
                X_test = poly.transform(x_test)
                X_test = scaler.transform(X_test)  # type: ignore
                y_train_mean = np.mean(y_train)
                y_train_scaled = y_train - y_train_mean
                ridge.fit(X_train, y_train_scaled, alpha=lmb)
                ols.fit(X_train, y_train_scaled)
                lasso.fit(X_train, y_train_scaled)
                scores_KFoldOLS[j,i,k] = MSE(y_test, ols.predict(X_test) + y_train_mean)
                scores_KFold_R2OLS[j,i,k] = R2(y_test, ols.predict(X_test) + y_train_mean)
                scores_KFoldRidge[j,i,k] = MSE(y_test, ridge.predict(X_test) + y_train_mean)
                scores_KFold_R2Ridge[j,i,k] = R2(y_test, ridge.predict(X_test) + y_train_mean)
                scores_KFoldLasso[j,i,k] = MSE(y_test, lasso.predict(X_test) + y_train_mean)
                scores_KFold_R2Lasso[j,i,k] = R2(y_test, lasso.predict(X_test) + y_train_mean)
                k += 1

    bestk[n] = np.min(np.mean(scores_KFoldOLS, axis=2))
    estimatedFoldsOLS[:,:,n] = np.mean(scores_KFoldOLS, axis=2)
    estimatedFoldsR2OLS[:,:,n] = np.mean(scores_KFold_R2OLS, axis=2)
    estimatedFoldsRidge[:,:,n] = np.mean(scores_KFoldRidge, axis=2)
    estimatedFoldsR2Ridge[:,:,n] = np.mean(scores_KFold_R2Ridge, axis=2)
    estimatedFoldsLasso[:,:,n] = np.mean(scores_KFoldLasso, axis=2)
    estimatedFoldsR2Lasso[:,:,n] = np.mean(scores_KFold_R2Lasso, axis=2)
 
 











findParmas(estimatedFoldsOLS[:,:,2], estimatedFoldsR2OLS[:,:,2], polydegree )
findParmas(estimatedFoldsRidge[:,:,2], estimatedFoldsR2Ridge[:,:,2], polydegree, lambdas)
findParmas(estimatedFoldsLasso[:,:,2], estimatedFoldsR2Lasso[:,:,2], polydegree, lambdas)




estimated_mse_sklearn = np.zeros(numlamdas)
kfold = KFold(7)
poly = PolynomialFeatures(4, include_bias=False)
for i, lmb in enumerate(lambdas):
    ridge = SkRidge(alpha=lmb)

    Xa = poly.fit_transform(X)
    Xa = scaler.fit_transform(Xa)
    y_mean = np.mean(y)
    y_scaled = y - y_mean
    estimated_mse_folds = cross_val_score(ridge, Xa,y_scaled , scoring='neg_mean_squared_error', cv=kfold )
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)



mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '16',
    'xtick.labelsize': '14',
    'ytick.labelsize': '14',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})






fix1 , ax1 = plt.subplots(figsize=(10,5))
ax1.set_title("Heatmap of scores")
ax1.set_xlabel("Lambda")
ax1.set_xticks(np.arange(0,len(lambdas),1),labels=lambdas.round(6), rotation=90)
ax1.set_yticks(np.arange(0,maxdegree,1), labels=polydegree)
ax1.set_ylabel(r"Degree")
im = ax1.imshow(estimatedFoldsR2Ridge[:,:,2], cmap="plasma")
cbar = ax1.figure.colorbar(im, ax=ax1) 
cbar.ax.set_ylabel("R2", rotation=-90, va="bottom")
plt.savefig("../runsAndAdditions/heatmapCrossval.png")
# plt.show()









fix1 , ax1 = plt.subplots(figsize=(10,10))
ax1.set_title("MSE for various k")
ax1.set_xlabel("k",size=20)
ax1.set_ylabel(r"MSE")
ax1.plot(karray, bestk)
plt.savefig("../runsAndAdditions/MSEoverK.png")










plt.figure(figsize=(10,10))

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'sKlearn\'s cross_val_score')
plt.plot(np.log10(lambdas), estimatedFoldsRidge[3,:,3], 'r--', label = 'Our KFold')
plt.xlabel(r'$log_{10}(\lambda)$',size=20)
plt.ylabel('mse', size=20)
plt.title('k = 7, Degree = 4')
plt.legend()

plt.savefig('../runsAndAdditions/crossvalOursVsSklearn.png')









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





