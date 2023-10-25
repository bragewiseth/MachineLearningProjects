import numpy as np



def ELU(x, alpha=1):
    """
    Exponential Linear Unit
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def ReLU(x):
    """
    Rectified Linear Unit
    """
    return np.where(x > 0, x, 0)

def leaky_ReLU(x, alpha=0.01):
    """
    Leaky Rectified Linear Unit
    """
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    """
    Hyperbolic tangent
    """
    return np.tanh(x)

def logistic(x):
    """
    Logistic function
    """
    return 1 / (1 + np.exp(-x))
