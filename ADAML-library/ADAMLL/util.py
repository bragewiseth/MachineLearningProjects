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
