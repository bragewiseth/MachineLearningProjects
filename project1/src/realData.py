from matplotlib.ticker import FormatStrFormatter, LinearLocator
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
import os


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
# Load the terrain
terrain = imageio.imread(os.path.join(ROOT_DIR, 'DataFiles', 'SRTM_data_Norway_2.tif'))
terrain = np.array(terrain)
N = 1000
m = 5 # polynomial order
terrain = terrain[:N,:N]
# Creates mesh of image pixels
rng = np.random.default_rng()
# x = rng.integers(0,np.shape(terrain)[0]-1, np.shape(terrain)[0])
# y = rng.integers(0,np.shape(terrain)[1]-1, np.shape(terrain)[1])
x = np.linspace(0,np.shape(terrain)[0]-1, np.shape(terrain)[0], dtype=int)
y = np.linspace(0,np.shape(terrain)[1]-1, np.shape(terrain)[1], dtype=int)
x_mesh, y_mesh = np.meshgrid(x,y)


z = np.array(terrain[x_mesh,y_mesh],dtype=float) #* 0.001 # km
print(z)
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(1,1,1,projection='3d')
# ax.set_title("ff", fontsize=16)
#     # Plot the surface.
# surf = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm, # type: ignore
#                     linewidth=0, antialiased=True)

# # Customize the z axis.
# ax.set_zlim(0, 6)
# ax.set_xlabel('m')
# ax.set_ylabel('m')
# ax.set_zlabel('km')
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# # fig.colorbar(surf, shrink=0.5, aspect=5)
# # Add a color bar which maps values to colors.
# plt.show()

# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
