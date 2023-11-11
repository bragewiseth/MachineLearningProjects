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


@jax.jit
def accuracy(y, t):
    """
    Calculates the accuracy of a classification model
    """
    return np.mean(y == t) 





def one_hot_encode(y, n_classes):
    """
    One hot encode a vector
    """
    return np.eye(n_classes)[y]




def confusion_matrix(y, t):
    """
    Calculates the confusion matrix
    """
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    n_classes = np.max(t) + 1

    matrix = np.zeros((n_classes, n_classes))

    for i in range(len(y)):
        matrix[t[i]][y[i]] += 1

    return matrix





def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    return f'Progress: [{arrow}{padding}] {int(fraction*100)}%'



def print_message(message):
    print(f"\r{message: <70}", end='')
