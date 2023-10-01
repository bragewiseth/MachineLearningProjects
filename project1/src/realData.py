from matplotlib.ticker import FormatStrFormatter, LinearLocator
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import MSE, R2, Ridge , OLS, printGrid
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge as SkRidge
from sklearn.model_selection import cross_val_score




ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
# Load the terrain
terrain = imageio.imread(os.path.join(ROOT_DIR, 'DataFiles', 'SRTM_data_Norway_2.tif'))
terrain = np.array(terrain)
N = 1000
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x = np.linspace(0,np.shape(terrain)[0]-1, np.shape(terrain)[0], dtype=int)
y = np.linspace(0,np.shape(terrain)[1]-1, np.shape(terrain)[1], dtype=int)
x_mesh, y_mesh = np.meshgrid(x,y)
z = terrain[x_mesh,y_mesh]
X = np.concatenate((x_mesh.ravel().reshape(-1,1), y_mesh.ravel().reshape(-1,1)), axis=1)







x_train, x_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2, shuffle=True)
k = 5


maxdegree = 15
scaler = StandardScaler()
poly = PolynomialFeatures(maxdegree,include_bias=False)
numfeatures = int(((maxdegree+1) **2 + (maxdegree-1)) / 2)
numlamdas = 50
lambdas = np.logspace(1,-6,numlamdas)
polydegree = np.zeros(maxdegree)




k = 6
kfold = KFold(n_splits = k)
scores_KFold_R2OLS = np.zeros((maxdegree, numlamdas, k))
scores_KFoldOLS = np.zeros((maxdegree, numlamdas, k))
scores_KFold_R2Ridge = np.zeros((maxdegree, numlamdas, k))
scores_KFoldRidge = np.zeros((maxdegree, numlamdas, k))
scores_KFold_R2Lasso = np.zeros((maxdegree, numlamdas, k))
scores_KFoldLasso = np.zeros((maxdegree, numlamdas, k))


for i, lmb in enumerate(lambdas):
    ridge = Ridge()
    ols = OLS()
    lasso = Lasso(alpha = lmb )     
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
            polydegree[degree] = degree + 1
            k += 1


estimated_mse_KFoldOLS = np.mean(scores_KFoldOLS, axis = 2)
estimated_mse_KFold_R2OLS = np.mean(scores_KFold_R2OLS, axis = 2)
estimated_mse_KFoldRidge = np.mean(scores_KFoldRidge, axis = 2)
estimated_mse_KFold_R2Ridge = np.mean(scores_KFold_R2Ridge, axis = 2)
estimated_mse_KFoldLasso = np.mean(scores_KFoldLasso, axis = 2)
estimated_mse_KFold_R2Lasso = np.mean(scores_KFold_R2Lasso, axis = 2)




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



findParmas(estimated_mse_KFoldOLS, estimated_mse_KFold_R2OLS, polydegree )
findParmas(estimated_mse_KFoldRidge, estimated_mse_KFold_R2Ridge, polydegree, lambdas)
findParmas(estimated_mse_KFoldLasso, estimated_mse_KFold_R2Lasso, polydegree, lambdas)





estimated_mse_sklearn = np.zeros(numlamdas)
poly = PolynomialFeatures(5, include_bias=False)
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


plt.figure(figsize=(10,10))

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'sKlearn\'s cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFoldRidge[maxdegree-1], 'r--', label = 'Our KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse', size=20)
plt.title('k = {}, Degree = {}'.format(k,maxdegree))
plt.legend()

plt.savefig('../runsAndAdditions/crossvalOursVsSklearn.png')

































# plot the best model


z = np.array(terrain[x_mesh,y_mesh],dtype=float) #* 0.001 # km

X = np.concatenate((x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)), axis=1)
X = poly.transform(X)
X = scaler.transform(X)
zpred = model.predict(X) + y_train_mean



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_title("ff", fontsize=16)
    # Plot the surface.
surf = ax.plot_surface(x_mesh, y_mesh, zpred.reshape(1000,1000), cmap=cm.coolwarm, # type: ignore
                    linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(0, 2000)
ax.set_xlabel('m')
ax.set_ylabel('m')
ax.set_zlabel('km')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)
# Add a color bar which maps values to colors.
plt.show()

# Show the terrain





fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_title("ff", fontsize=16)
    # Plot the surface.
surf = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm, # type: ignore
                    linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(0, 6000)
ax.set_xlabel('m')
ax.set_ylabel('m')
ax.set_zlabel('km')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)
# Add a color bar which maps values to colors.
plt.show()







plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(zpred, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
