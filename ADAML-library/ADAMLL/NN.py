import jax.numpy as np
import jax
from .util import  MSE, CE
from .activations import sigmoid, softmax, relu, tanh, leaky_relu, elu, swish, mish, gelu, softplus, softsign






# INITIALIZATION

def init_network_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [{'w': jax.random.normal(k, (in_size, out_size)) * np.sqrt(2 / in_size),
             'b': np.zeros(out_size)}
            for k, in_size, out_size in zip(keys, layer_sizes[:-1], layer_sizes[1:])]






# BACKWARD PROPAGATION




def backprop_one_hidden_auto( X,t, params, activations , loss=MSE ):
    a0, a1 = activations
    dloss = jax.grad(loss)(a1, t)
    output_error = (a1 - t)
    hidden_error =  output_error @ params.w1.T * jax.grad(sigmoid) (a0)

    gw0 = X.T @ hidden_error
    gb0 = np.sum(hidden_error, axis=0)
    gw1 = a0.T @ output_error * dloss
    gb1 = np.sum(output_error, axis=0)

    return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]





def backprop_one_hidden( params, t, activations):
    X, a0, a1 = activations
    output_error = (a1 - t)
    hidden_error =  output_error @ params[1]['w'].T * a0 * (1 - a0) 

    gw0 = X.T @ hidden_error
    gb0 = np.sum(hidden_error, axis=0)
    gw1 = a0.T @ output_error
    gb1 = np.sum(output_error, axis=0)
    gw0, gb0, gw1, gb1 = jax.tree_map(lambda p: p / X.shape[0], (gw0, gb0, gw1, gb1))
    return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]


def single_layer_gradients(_, t, activations ):
    X, y = activations
    wgrad = (2/X.shape[0]) * np.dot(X.T , (y - t))
    bgrad = 2/X.shape[0] * np.sum(y - t)
    return [{'w': wgrad, 'b': bgrad}]



def build_backpropagation(architecture, loss=MSE):
    pass


# FORWARD PROPAGATION



def forward_propagate(network, inputs):
    activations = [inputs] # List to hold all activations, layer by layer
    for layer in network:
        net_input = np.dot(activations[-1], layer['w']) + layer['b']
        activations.append(sigmoid(net_input))
    return activations



def build_forward_propagation(architecture):
    pass