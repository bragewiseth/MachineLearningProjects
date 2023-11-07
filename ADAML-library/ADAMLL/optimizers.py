import jax
import jax.numpy as np




def create_update_sgd(eta, gamma):
    def sgd_update(params, grads, state ):
        def update_one_param(param, grad, momentum ):
            new_momentum = gamma * momentum + eta * grad
            new_param = param - new_momentum
            return new_param, new_momentum
                # Update all parameters and states separately
        new_params = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[0], params, grads, state
        )
        new_state = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[1], params, grads, state
        )
        return new_params, new_state
    return sgd_update



def init_SGD_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)



# with momentum
def create_update_adagrad(eta=0.001,  epsilon=1e-8):
    def adagrad_update(params, grads, state ):
        def update_one_param(param, grad, state):
            v = state
            v = v + grad ** 2
            # Update parameter
            new_param = param - eta * grad / (np.sqrt(v) + epsilon)
            return new_param, v

        # Update all parameters and states
        new_params = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[0], params, grads, state
        )
        new_state = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[1], params, grads, state
        )
        return new_params, new_state 
    return adagrad_update



def init_adagrad_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment







def create_update_adam(eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    def adam_update(params, grads, state ):
        def update_one_param(param,grad, state):
            m, v, t = state
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** (t + 1))
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** (t + 1))
            # Update parameter
            t = t + 1
            new_param = param - eta * m_hat / (np.sqrt(v_hat) + epsilon)
            return new_param, m, v

        # Update all parameters and states
        new_params = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[0], params, grads, state
        )
        new_state = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[1], params, grads, state
        )
        return new_params, state 
    return adam_update



def init_adam_state(params):
    ms = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize first moment
    vs = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment
    ts = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize time step
    return ms, vs, ts





def create_update_rmsprop(eta=0.001, gamma=0.9, epsilon=1e-8):
    def rmsprop_update(params, grads, state ):
        def update_one_param(param, grad, state):
            v = state
            # Update biased second raw moment estimate
            v = gamma * v + (1 - gamma) * (grad ** 2)
            # Update parameter
            new_param = param - eta * grad / (np.sqrt(v) + epsilon)
            return new_param, v

        # Update all parameters and states
        new_params = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[0], params, grads, state
        )
        new_state = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[1], params, grads, state
        )
        return new_params, state 
    return rmsprop_update

def init_rmsprop_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment


