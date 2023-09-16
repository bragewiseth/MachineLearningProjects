import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def R2(y_data,y_model):
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_model))**2)


def fit_beta(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def fit_beta_ridge(X, y, l):
    p = X.shape[1]
    return np.linalg.inv(X.T @ X + (l * np.eye(p))) @ X.T @ y


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4



def makeFigure(figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    return fig


def plotFrankefunction(xx,yy, z, fig, subplot=(1,1,1),title=None ):
    ax = fig.add_subplot(subplot[0],subplot[1], subplot[2],projection='3d')
    ax.title.set_text(title)
        # Plot the surface.
    surf = ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, # type: ignore
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Add a color bar which maps values to colors.
    return ax







class MyStandardScaler:
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

