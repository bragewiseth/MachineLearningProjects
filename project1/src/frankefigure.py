import matplotlib
import matplotlib.pyplot as plt
from utils import FrankeFunction, plotFrankefunction, readData 
import numpy as np


"""
This file is used to generate the Franke function plot in the report.
"""



# plot 
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '12',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})


x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
z = FrankeFunction(xx,yy)

plotFrankefunction(xx,yy,z, (8,8), (1,1,1) ,"True function")
plt.show()


X,y = readData("../data/syntheticData.csv")
fig = plt.figure(figsize=(8,8))
fig.tight_layout()
ax = fig.add_subplot(projection='3d')
ax.set_title("data", fontsize=16)
# Plot the surface.
surf = ax.scatter(X[:,0],X[:,1], y) 
                       

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)


# fig.colorbar(surf, shrink=0.5, aspect=5)
# Add a color bar which maps values to colors.


plt.show()
