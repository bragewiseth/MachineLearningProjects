import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split




def MSE(y_data,y_model):
    """
    Calculates the mean squared error
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n





def R2(y_data,y_model):
    """
    Calculates the R2 score
    """
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_model))**2)





def fit_beta(X,y):
    """
    Fits beta for OLS
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y




def fit_beta_ridge(X,y,Lambda):
    """
    Fits beta for ridge regression
    """
    p = X.shape[1]
    return np.linalg.inv(X.T @ X + (Lambda * np.eye(p))) @ X.T @ y







def FrankeFunction(x,y):
    """
    Franke function
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4







def plotFrankefunction(xx,yy, z, figsize=(10,10), subplot=(1,1,1),title=None ):
    """
    Plots the FrankeFunction on a 3D surface. You need to call makeFigure() first
    and then pass the figure to this function. You need to call plt.show() after
    """    
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    ax = fig.add_subplot(subplot[0],subplot[1], subplot[2],projection='3d')
    ax.set_title(title, fontsize=16)
        # Plot the surface.
    surf = ax.plot_surface(xx, yy, z, cmap="plasma", linewidth=0, antialiased=False,alpha=0.8)

    ax.set_zlim(-0.10, 1.40)
    # remove z tick numbers but keep grid
    ax.set_zticks([])


    # ax.zaxis.set_major_locator(LinearLocator(5))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Add a color bar which maps values to colors.
    return ax







def readData(path):
    """
    reads csv file 
    """
    df = pd.read_csv(path)
    X = df[["x","y"]].to_numpy()
    y = df["z"].to_numpy()
    return X, y





def printGrid(trainerror, testerror, trainR2, testR2, polydegree, lamdas):
    for i in range(len(polydegree)):
        testi  = np.argmin(testerror[i])    # find index of minimum test error (best fit)
        traini = np.argmin(trainerror[i])   # find index of minimum train error (best fit)
        testR2i  = np.argmax(testR2[i])     # find index of maximum test R2 (best fit)
        trainR2i = np.argmax(trainR2[i])    # find index of maximum train R2 (best fit)
        print("Degree of polynomial = ", polydegree[i])
        print("Best test error = {:25} \tfor λ = {}".format(testerror[i,testi],  lamdas[testi]))
        print("Best train error = {:24} \tfor λ = {}".format(trainerror[i,traini],  lamdas[traini]))
        print("Best test R2 = {:26} \tfor λ = {}".format(testR2[i,testR2i],  lamdas[testR2i]))
        print("Best train R2 = {:25} \tfor λ = {}".format(trainR2[i,trainR2i],  lamdas[trainR2i]))     







class MyStandardScaler:
    """
    Implements the same basic functionality as sklearn.preprocessing.StandardScaler
    without the fancy stuff
    """
    def __init__(self):
        self.mean_ = None
        self.var_ = None

    def fit(self, X):
        p = X.shape[1]
        self.mean_ = np.zeros(p)
        self.var_ = np.zeros(p)
        for i in range(p):
            self.mean_[i] = np.mean(X[:, i])
            self.var_[i] = np.var(X[:, i])

    def transform(self, X, with_std=True, with_mean=True):
        if self.mean_ is None or self.var_ is None:
            raise ValueError("Call fit first")

        if with_mean:
            X = X - self.mean_.reshape(1, -1)
        if with_std and self.var_.all() != 0:
            X /= np.sqrt(self.var_).reshape(1, -1)
        return X

    def fit_transform(self, X, with_std=True, with_mean=True):
        self.fit(X)
        return self.transform(X, with_std, with_mean)







class OLS:
    """
    Implements the same basic functionality as sklearn.linear_model.LinearRegression
    without the fancy stuff
    """
    def __init__(self):
        pass


    def fit(self,X,y):
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y


    def predict(self,X):
        return X @ self.beta







class Ridge:
    """
    Implements the same basic functionality as sklearn.linear_model.Ridge
    without the fancy stuff
    """
    def __init__(self, alpha=0.0):
        self.alpha = alpha


    def fit(self,X, y):
        p = X.shape[1]
        self.beta = np.linalg.inv(X.T @ X + (self.alpha * np.eye(p))) @ X.T @ y

    def predict(self, X):
        return X @ self.beta
