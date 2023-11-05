import jax
import jax.numpy as np




def create_update_sgd(eta, gamma):
    def sgd_update(params, grads, state ):
        def update_one_param(param, grad, velocity):
            new_velocity = gamma * velocity + eta * grad
            new_param = param - new_velocity
            return new_param, new_velocity
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








def create_update_ridge(eta, momentum, weight_decay):
    def ridge_update(params, grads, state):
        def update_one_param(param, grad, velocity):
            new_velocity = momentum * velocity + eta * grad + weight_decay * grad
            new_param = param - new_velocity
            return new_param, new_velocity
        new_params = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[0], params, grads, state
        )
        new_state = jax.tree_map(
            lambda p, g, s: update_one_param(p, g, s)[1], params, grads, state
        )
        return new_params, new_state
    return ridge_update

def init_ridge_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)











def init_adam_state(params):
    ms = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize first moment
    vs = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment
    ts = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize time step
    return ms, vs, ts

def create_update_adam(eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    def adam_update(params, grads, state ):
        ms, vs, ts = state
        def update_one_param(param, m, v, t, grad):
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** (t + 1))
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** (t + 1))
            # Update parameter
            new_param = param - eta * m_hat / (np.sqrt(v_hat) + epsilon)
            return new_param, m, v

        # Update all parameters and states
        new_params, new_ms, new_vs = jax.tree_multimap(update_one_param, params, ms, vs, ts, grads)
        new_ts = jax.tree_map(lambda t: t + 1, ts)  # Update time step
        return new_params, (new_ms, new_vs, new_ts)
    return adam_update


def create_update_rmsprop(eta=0.001, gamma=0.9, epsilon=1e-8):
    def rmsprop_update(params, grads, state):
        vs = state
        def update_one_param(param, v, grad):
            new_v = gamma * v + (1 - gamma) * (grad ** 2)
            new_param = param - eta * grad / (np.sqrt(new_v) + epsilon)
            return new_param, new_v
        new_params, new_vs = jax.tree_multimap(update_one_param, params, vs, grads)
        return new_params, new_vs
    return rmsprop_update

def init_rmsprop_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)
