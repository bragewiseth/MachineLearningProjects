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
    'font.size': '16',
    'xtick.labelsize': '14',
    'ytick.labelsize': '14',
    # 'text.usetex': True,
    'pgf.rcfonts': True,
})




x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
xx,yy = np.meshgrid(x,y)
z = FrankeFunction(xx,yy)

plotFrankefunction(xx,yy,z, (8,8), (1,1,1) ,"True function")
# plt.savefig("../runsAndAdditions/frankefigure.png")
plt.show()





X,y = readData("../data/syntheticData.csv")
fig = plt.figure(figsize=(8,8))
fig.tight_layout()
ax = fig.add_subplot(projection='3d')

# ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.2))
# ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.2))
# ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.2))
ax.set_title("Data", fontsize=16)
ax.plot_surface(xx, yy, z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.4)
ax.scatter(X[:,0],X[:,1], y , c=y, cmap="cool", alpha=1) 
ax.set_zlim(-0.10, 1.40)
# plt.savefig("../runsAndAdditions/frankefigurenoise.png")
plt.show()
