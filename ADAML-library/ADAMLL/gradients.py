import jax.numpy as np
import jax 


def backpropagation( X,t, params, activ, activ_diff):
    """
    Backpropagation algorithm
    ## Parameters
        X, T : ndarray
            input data and targets
    ## Returns
        :tuple
        ( first layer weights gradient, first layer bias gradient, last layer weights gradient, last layer bias gradient )
    """
    a0, a1 = forward(X, params, activ)
    output_error = (a1 - t)
    hidden_error =  output_error @ params.w1.T * activ_diff(a0)

    grads.w0 = X.T @ hidden_error                # * 1/X.shape[0] # can be baked into the learning rate
    grads.b0 = np.sum(hidden_error, axis=0)      # * 1/X.shape[0] # can be baked into the learning rate
    grads.w1 = a0.T @ output_error               # * 1/X.shape[0] # can be baked into the learning rate
    grads.b1 = np.sum(output_error, axis=0)      # * 1/X.shape[0] # can be baked into the learning rate

    return grads, params 



@jax.jit
def gradients(X, t, N, w):
    return 2/N * X.T @ (X @ w - t)