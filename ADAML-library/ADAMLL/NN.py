import jax.numpy as np
import jax
from .util import  MSE, CE, print_message
from .activations import sigmoid, softmax, relu, tanh, leaky_relu, elu, swish, mish, gelu, softplus, softsign
from . import optimizers






# INITIALIZATION

def init_network_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [{'w': jax.random.normal(k, (in_size, out_size)) * np.sqrt(2 / in_size),
             'b': np.zeros(out_size)}
            for k, in_size, out_size in zip(keys, layer_sizes[:-1], layer_sizes[1:])]



# NETWORK CLASS

class NN():

    def __init__(self ,architecture=[ [2,2, 1], [sigmoid, sigmoid] ],
                eta=0.1, epochs=100, tol=0.001, optimizer='sgd', alpha=0,
                 gamma=0, epsilon=0.0001,  beta1=0.9, beta2=0.999, backwards=None, loss=MSE):
        self.eta = eta
        self.epochs = epochs
        self.tol = tol
        self.optimizer = optimizer
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.architecture = architecture
        if backwards == None:
            self.backwards = build_backwards(architecture, loss)
        else:
            self.backwards = backwards
        self.forward = build_forward(architecture)



    def init_optimizer(self, params):
        match self.optimizer:
            case 'sgd':
                optimizer = optimizers.create_update_sgd(self.eta, self.gamma)
                optimizerState = optimizers.init_SGD_state(params)
                return optimizer, optimizerState
            case 'adagrad':
                optimizer = optimizers.create_update_adagrad(self.eta, self.epsilon )
                optimizerState = optimizers.init_adagrad_state(params)
                return optimizer, optimizerState
            case 'rmsprop':
                optimizer = optimizers.create_update_rmsprop(self.eta, self.epsilon, self.gamma)
                optimizerState = optimizers.init_rmsprop_state(params)
                return optimizer, optimizerState
            case 'adam':
                optimizer = optimizers.create_update_adam(self.eta, self.beta1, self.beta2, self.epsilon)
                optimizerState = optimizers.init_adam_state(params)
                return optimizer, optimizerState
            case _:
                raise ValueError(f"Unknown optimizer {self.optimizer}")



    def fit(self, X, t, X_val, t_val, batch_size=None ):
        N,n = X.shape
        self.architecture[0][0] = n
        if batch_size is None:
            batch_size = N
        key = jax.random.PRNGKey(1234)
        params = init_network_params(self.architecture[0], key)
        update_params, opt_state = self.init_optimizer(params) 
        batches = int(N/batch_size)
        loss = np.zeros(self.epochs)


        @jax.jit # one step of gradient descent jitted to make it zoom
        def step(params, opt_state, X, t):
            activations = self.forward(params, X)
            grads = self.backwards(params, t, activations, self.alpha)
            params, opt_state = update_params(params, grads, opt_state)
            return params, opt_state


        for e in range(self.epochs):
            for _ in range(batches):

                key, subkey = jax.random.split(key)
                random_index = batch_size * jax.random.randint(subkey, minval=0, maxval=batches, shape=())
                X_batch = X[random_index:random_index+batch_size]
                t_batch = t[random_index:random_index+batch_size]

                params, opt_state = step(params, opt_state, X_batch, t_batch)

                current_loss = CE(self.forward(params, X_val)[-1], t_val)

                # clip gradients
                if not np.isfinite(current_loss).all():
                    params = jax.tree_map(lambda p: np.clip(p, -100, 100), params)

                loss = loss.at[e].set(current_loss)

                # Early stopping condition
                if e > 10 and np.abs(loss[e-10] - loss[e]) < self.tol:
                    loss = loss.at[e+1:].set(loss[e]) 
                    break


        print_message(f"Training stopped after {e} epochs")
        return loss , params








# BACKWARD PROPAGATION




def backprop_one_hidden_auto( X,t, params, activations, alpha , loss=MSE ):
    """cross entropy with sigmoid - sigmoid autoiff"""
    a0, a1 = activations
    dloss = jax.grad(loss)(a1, t)
    output_error = (a1 - t)
    hidden_error =  output_error @ params.w1.T * jax.grad(sigmoid) (a0)

    gw0 = X.T @ hidden_error
    gb0 = np.sum(hidden_error, axis=0)
    gw1 = a0.T @ output_error * dloss
    gb1 = np.sum(output_error, axis=0)

    return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]





def backprop_one_hidden( params, t, activations, alpha):
    """Cross entropy with sigmoid - sigmoid"""
    X, a0, a1 = activations
    output_error = (a1 - t)
    hidden_error =  output_error @ params[1]['w'].T * a0 * (1 - a0) 

    gw0 = X.T @ hidden_error
    gb0 = np.sum(hidden_error, axis=0)
    gw1 = a0.T @ output_error
    gb1 = np.sum(output_error, axis=0)
    gw0, gb0, gw1, gb1 = jax.tree_map(lambda p: p / X.shape[0], (gw0, gb0, gw1, gb1))
    return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]


def single_layer_gradients(_, t, activations, alpha ):
    X, y = activations
    wgrad = (2/X.shape[0]) * np.dot(X.T , (y - t))
    bgrad = 2/X.shape[0] * np.sum(y - t)
    return [{'w': wgrad, 'b': bgrad}]



def build_backwards(architecture, loss=MSE):
    acitvation_functions = architecture[1]

    def last_layer_activation(params, t, activations):
        X, a0, a1 = activations
        dloss = jax.grad(loss)(a1, t)
        output_error = (a1 - t)
        hidden_error =  output_error @ params[-1]['w'].T * jax.grad(acitvation_functions[-1]) (a0)

        gw0 = X.T @ hidden_error
        gb0 = np.sum(hidden_error, axis=0)
        gw1 = a0.T @ output_error * dloss * jax.grad(acitvation_functions[-1]) (a1)
        gb1 = np.sum(output_error, axis=0)

        return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]  

    def last_layer_no_activation(params, t, activations):
        X, a0, a1 = activations
        output_error = (a1 - t)
        hidden_error =  output_error @ params[-1]['w'].T * a0 * (1 - a0) 

        gw0 = X.T @ hidden_error
        gb0 = np.sum(hidden_error, axis=0)
        gw1 = a0.T @ output_error
        gb1 = np.sum(output_error, axis=0)
        gw0, gb0, gw1, gb1 = jax.tree_map(lambda p: p / X.shape[0], (gw0, gb0, gw1, gb1))
        return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]

    if len(architecture[0]) -  len(acitvation_functions ) > 1:
        return last_layer_activation
    else:
        return last_layer_no_activation


# FORWARD PROPAGATION




def build_forward(architecture):
    acitvation_functions = architecture[1]

    @jax.jit
    def last_layer_activation(network, inputs):
        activations = [inputs]
        for i in range(len(network)): 
            activations.append(acitvation_functions[i](np.dot(activations[-1], network[i]['w']) + network[i]['b']))
        return activations  
    @jax.jit
    def last_layer_no_activation(network, inputs):
        activations = [inputs]
        for i in range(len(network) - 1):
            activations.append(acitvation_functions[i](np.dot(activations[-1], network[i]['w']) + network[i]['b']))

        activations.append(np.dot(activations[-1], network[-1]['w']) + network[-1]['b'])
        return activations

    if len(architecture[0]) -  len(acitvation_functions ) > 1:
        return last_layer_no_activation
    else:
        return last_layer_activation
