import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import FrankeFunction, plotFrankefunction, readData 
import numpy as np
from matplotlib import cm

"""
This file is used to generate the Franke function plot in the report.
"""



# plot 
mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': '20',
    'xtick.labelsize': '18',
    'ytick.labelsize': '18',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})




x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
z = FrankeFunction(xx,yy)

plotFrankefunction(xx,yy,z, (8,8), (1,1,1) ,"True function")
plt.savefig("../runsAndAdditions/trueFunction.png")





X,y = readData("../data/syntheticData.csv")
fig = plt.figure(figsize=(8,8))
# fig.tight_layout()
ax = fig.add_subplot(projection='3d')


# ax.set_title("Data", fontsize=16)
ax.plot_surface(xx, yy, z, cmap="bone", linewidth=0, antialiased=False, alpha=0.2)
surf = ax.scatter(X[:,0],X[:,1], y , c=y, cmap='winter', alpha=1) 
ax.set_zlim(-0.10, 1.40)
ax.set_zticks([])
ax.view_init(35, -35)

    # ax.zaxis.set_major_locator(LinearLocator(5))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("../runsAndAdditions/synthDataSide.png")
