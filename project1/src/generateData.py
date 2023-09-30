import numpy as np
import pandas as pd
from utils import FrankeFunction


np.random.seed(9282)
n = 100
rand = 0.1

x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
z = FrankeFunction(x, y) + rand * np.random.randn(n)
x = x.reshape(-1,1)
y = y.reshape(-1,1)
z = z.ravel().reshape(-1,1)
# write data to csv using pandas
df = pd.DataFrame(np.concatenate((x,y,z), axis=1))
# fist two collums is x and last is y
df.to_csv("../data/syntheticData.csv", header=["x", "y", "z"], index=False)
