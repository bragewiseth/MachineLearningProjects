import ADAMLL.something as ada # import the ADAMLL library
import timeit 
import jax.numpy as np # import numpy

a = np.arange(10000).reshape(100, 100) # create a 100x100 matrix
b = np.arange(10000).reshape(100, 100) # create a 100x100 matrix


print(ada.multiply_matrices_jit(a, b)) # run the multiply_matrices_jit function
# c = timeit.timeit(ada.multiply_matrices(a, b)) # run the multiply_matrices function
