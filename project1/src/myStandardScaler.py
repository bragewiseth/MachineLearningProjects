import numpy as np


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
        if with_std and self.var_ != 0:
            X /= np.sqrt(self.var_).reshape(1, -1)
        return X

    def fit_transform(self, X, with_std=True, with_mean=True):
        self.fit(X)
        return self.transform(X, with_std, with_mean)



