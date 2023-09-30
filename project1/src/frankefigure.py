import matplotlib
import matplotlib.pyplot as plt
from utils import FrankeFunction, plotFrankefunction, makeFigure
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

fig = makeFigure((8,8))
fig1 = makeFigure((8,8))
plotFrankefunction(xx,yy,z, fig, (1,1,1) ,"True function")
data = readData() 
fig = plt.figure(figsize=(8,8))



plt.show()
