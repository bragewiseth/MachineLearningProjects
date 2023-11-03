import numpy as np
import ADAMLL as ada
import matplotlib.pyplot as plt
import matplotlib as mpl




def plot3Dmesh(xx, yy, z, nrows=1, ncols=1, index=1, fig=None , title=None  ):
    """
    Plots the 3D surface
    """    
    if fig is None:
        fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(nrows, ncols, index, projection='3d')
    ax.set_title(title, fontfamily='DM Sans')
    ax.set_xticks([-1,0,1], labels=[-1,0,1])
    ax.set_yticks([-1,0,1], labels=[-1,0,1])
    ax.set_xlabel('hello')
    ax.set_ylabel('world')
    surf = ax.plot_surface(xx, yy, z, cmap='winter', linewidth=0, antialiased=False, alpha=0.8)
    ax.set_zlim(-0.10, 1.40)
    ax.set_zticks([0,1], labels=[0,1])
    ax.view_init(30, 60)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig, ax



x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x,y)
z = np.sin(xx*yy * 10) * np.exp(-xx**2 - yy**2) * 0.3
plot3Dmesh(xx, yy, z, title=r'3D plot $\sin(xy)$' )
print(ada.MSE(1,2))
print( ada.__name__ )
plt.show()
