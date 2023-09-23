from matplotlib.ticker import FormatStrFormatter, LinearLocator
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from utils import FrankeFunction, makeData, MSE, R2, OLS, makeFigure, plotFrankefunction, Ridge
import matplotlib as mpl



ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
# Load the terrain
terrain = imageio.imread(os.path.join(ROOT_DIR, 'DataFiles', 'SRTM_data_Norway_2.tif'))
terrain = np.array(terrain)
N = 1000
terrain = terrain[:N,:N]
# Creates mesh of image pixels
rng = np.random.default_rng()
x = np.linspace(0,np.shape(terrain)[0]-1, np.shape(terrain)[0], dtype=int)
y = np.linspace(0,np.shape(terrain)[1]-1, np.shape(terrain)[1], dtype=int)
# x = rng.integers(0,np.shape(terrain)[0]-1, np.shape(terrain)[0])
# y = rng.integers(0,np.shape(terrain)[1]-1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)
z = terrain[x_mesh,y_mesh]

X = np.concatenate((x_mesh.ravel().reshape(-1,1), y_mesh.ravel().reshape(-1,1)), axis=1)
maxdegree = 50
scaler = StandardScaler()
poly = PolynomialFeatures(maxdegree,include_bias=False)

x_train, x_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2, shuffle=True)
y_train_mean = np.mean(y_train)
X_train = poly.fit_transform(x_train)
X_test = poly.transform(x_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # type: ignore
y_train_mean = np.mean(y_train)
y_train_scaled = y_train - y_train_mean
model = OLS()


model.fit(X_train, y_train_scaled)












x = np.linspace(0,np.shape(terrain)[0]-1, np.shape(terrain)[0], dtype=int)
y = np.linspace(0,np.shape(terrain)[1]-1, np.shape(terrain)[1], dtype=int)
x_mesh, y_mesh = np.meshgrid(x,y)
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