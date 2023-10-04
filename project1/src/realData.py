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
# terrain = imageio.imread(os.path.join(ROOT_DIR, 'data', 'SRTM_data_Norway_1.tif'))
terrain = imageio.imread(os.path.join(ROOT_DIR, 'data', 'SRTM_data_Norway_2.tif'))
terrain = np.array(terrain)
N = 1800
n = 1000 
terrain = terrain[:N,:N]
# Creates mesh of image pixels
x0 = np.linspace(0,np.shape(terrain)[0]-1, np.shape(terrain)[0], dtype=int)
x1 = np.linspace(0,np.shape(terrain)[1]-1, np.shape(terrain)[1], dtype=int)
x0sample = np.random.randint(0, N, n)
x1sample = np.random.randint(0, N, n)
x_mesh, y_mesh = np.meshgrid(x0,x1)
terrain = terrain[x_mesh,y_mesh]
y = terrain[x0sample,x1sample].ravel().reshape(-1,1)
X = np.concatenate((x0sample.reshape(-1,1), x1sample.reshape(-1,1)), axis=1)


# we introduce a final test set since we have more data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
k = 8


lambdas = [0.00001 ,0.0001, 0.001, 0.01,0.1, 1, 10 ] 
degrees = [3,5,8,10,12,15,18,20,22]
numlamdas = len(lambdas)
numdegrees = len(degrees)
polydegree = np.zeros(numdegrees)

scaler = StandardScaler()
k = 8
kfold = KFold(n_splits = k)
scores_KFold_R2OLS = np.zeros((numdegrees, numlamdas, k))
scores_KFoldOLS = np.zeros((numdegrees, numlamdas, k))
scores_KFold_R2Ridge = np.zeros((numdegrees, numlamdas, k))
scores_KFoldRidge = np.zeros((numdegrees, numlamdas, k))
scores_KFold_R2Lasso = np.zeros((numdegrees, numlamdas, k))
scores_KFoldLasso = np.zeros((numdegrees, numlamdas, k))


for i, lmb in enumerate(lambdas):
    ridge = Ridge()
    ols = OLS()
    lasso = Lasso(alpha = lmb )     
    for j , degree in enumerate(degrees):
        poly = PolynomialFeatures(degree, include_bias=False)
        k = 0
        for trainind, testind in kfold.split(x_train):
            x_train_fold, x_val_fold = x_train[trainind], x_train[testind]
            y_train_fold, y_val_fold = y_train[trainind], y_train[testind]
            X_train = poly.fit_transform(x_train_fold)
            X_train = scaler.fit_transform(X_train)
            X_test = poly.transform(x_val_fold)
            X_test = scaler.transform(X_test)  # type: ignore
            y_train_mean = np.mean(y_train_fold)
            y_train_scaled = y_train_fold - y_train_mean
            ridge.fit(X_train, y_train_scaled, alpha=lmb)
            ols.fit(X_train, y_train_scaled)
            # lasso.fit(X_train, y_train_scaled)
            scores_KFoldOLS[j,i,k] = MSE(y_val_fold, ols.predict(X_test) + y_train_mean)
            scores_KFold_R2OLS[j,i,k] = R2(y_val_fold, ols.predict(X_test) + y_train_mean)
            scores_KFoldRidge[j,i,k] = MSE(y_val_fold, ridge.predict(X_test) + y_train_mean)
            scores_KFold_R2Ridge[j,i,k] = R2(y_val_fold, ridge.predict(X_test) + y_train_mean)
            # scores_KFoldLasso[j,i,k] = MSE(y_val_fold, lasso.predict(X_test) + y_train_mean)
            # scores_KFold_R2Lasso[j,i,k] = R2(y_val_fold, lasso.predict(X_test) + y_train_mean)
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



findParmas(estimated_mse_KFoldOLS, estimated_mse_KFold_R2OLS, degrees )
findParmas(estimated_mse_KFoldRidge, estimated_mse_KFold_R2Ridge, degrees, lambdas)
findParmas(estimated_mse_KFoldLasso, estimated_mse_KFold_R2Lasso, degrees, lambdas)





mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '16',
    'xtick.labelsize': '14',
    'ytick.labelsize': '14',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})



fix1 , ax1 = plt.subplots(figsize=(10,10))
ax1.set_title(r"OLS - $\mathbf{\beta}$ and model complexity")
ax1.set_xlabel("Lambda")
ax1.set_xticks(np.arange(len(lambdas)),labels=lambdas)
ax1.set_yticks(np.arange(len(degrees)), labels=degrees)
ax1.set_ylabel(r"values of $\beta$")
im = ax1.imshow(estimated_mse_KFold_R2Ridge, cmap="plasma")
cbar = ax1.figure.colorbar(im, ax=ax1) 
cbar.ax.set_ylabel("label", rotation=-90, va="bottom")
# plt.savefig("../runsAndAdditions/tullll.png")
plt.show()





ols = OLS()
poly = PolynomialFeatures(12, include_bias=False) 
X_train = poly.fit_transform(x_train) 
X_train = scaler.fit_transform(X_train)
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean
ols.fit(X_train,y_train_scaled )
A = np.concatenate((x_mesh.ravel().reshape(-1,1), y_mesh.ravel().reshape(-1,1)), axis=1)
A = poly.fit_transform(A)
A = scaler.transform(A)
zpred = ols.predict(A) + y_train_mean



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_title("ff", fontsize=16)
    # Plot the surface.
zpred[zpred>1300]= np.nan
zpred[zpred<0]= np.nan
surf = ax.plot_surface(x_mesh, y_mesh, zpred.reshape(N,N), cmap="plasma", linewidth=0, antialiased=True)


# Customize the z axis.
ax.set_zlim(0, 2000)
ax.set_xlabel('m')
ax.set_ylabel('m')
ax.set_zlabel('km')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
# Add a color bar which maps values to colors.
plt.show()

# Show the terrain





fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.set_title("ff", fontsize=16)
    # plot the surface.
surf = ax.plot_surface(x_mesh, y_mesh, terrain, cmap='plasma', linewidth=0, antialiased=True)

# customize the z axis.
ax.set_zlim(0, 2000)
ax.set_xlabel('m')
ax.set_ylabel('m')
ax.set_zlabel('km')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# fig.colorbar(surf, shrink=0.5, aspect=5)
# add a color bar which maps values to colors.
plt.show()







plt.figure()
plt.title('terrain over norway 1')
plt.imshow(terrain, cmap='plasma')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
