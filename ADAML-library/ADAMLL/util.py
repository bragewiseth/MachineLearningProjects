import jax.numpy as np
import jax


@jax.jit
def MSE(y_data:np.ndarray ,y_model:np.ndarray  ) -> float:
    """
    Calculates the mean squared error
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n




@jax.jit
def R2(y_data:np.ndarray ,y_model:np.ndarray ) -> float:
    """
    Calculates the R2 score
    """
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_model))**2)


@jax.jit
def CE(y, t):
    """
    Cross entropy loss function
    """
    return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))



def init_network_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [{'w': jax.random.normal(k, (in_size, out_size)) * np.sqrt(2 / in_size),
             'b': np.zeros(out_size)}
            for k, in_size, out_size in zip(keys, layer_sizes[:-1], layer_sizes[1:])]




def forward(params, x, activation):
    a = x
    for param in params[:-1]:
        a = activation(np.dot(a, param['w']) + param['b'])
    last_layer = params[-1]
    return np.dot(a, last_layer['w']) + last_layer['b']