import jax.numpy as np
import jax
from .util import forward, MSE
from .activations import sigmoid





def backprop( X,t, params, activation=sigmoid ):
    """
    Backpropagation algorithm
    ## Parameters
        X, T : ndarray
            input data and targets
    ## Returns
        :tuple
        ( first layer weights gradient, first layer bias gradient, last layer weights gradient, last layer bias gradient )
    """
    a0, a1 = forward(X, params, activation ) 
    output_error = (a1 - t)
    hidden_error =  output_error @ params.w1.T * a0 * (1 - a0) 

    gw0 = X.T @ hidden_error             
    gb0 = np.sum(hidden_error, axis=0)   
    gw1 = a0.T @ output_error         
    gb1 = np.sum(output_error, axis=0)     

    return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]



def backprop_auto( X,t, params, activation=sigmoid, loss=MSE ):
    a0, a1 = forward(X, params, activation )
    dloss = jax.grad(loss)(a1, t)
    output_error = (a1 - t)
    hidden_error =  output_error @ params.w1.T * jax.grad(activation)(a0)

    gw0 = X.T @ hidden_error
    gb0 = np.sum(hidden_error, axis=0)
    gw1 = a0.T @ output_error * dloss
    gb1 = np.sum(output_error, axis=0)

    return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]





def gradients(X, t, params):
    w = params[0]['w']
    b = params[0]['b']
    y = np.dot(X , w) + b
    wgrad = (2/X.shape[0]) * np.dot(X.T , (y - t))
    bgrad = 2/X.shape[0] * np.sum(y - t)
    return [{'w': wgrad, 'b': bgrad}]

