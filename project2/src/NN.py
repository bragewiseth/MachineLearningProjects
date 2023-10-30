import jax.numpy as np
import numpy as onp
from jax import grad


def logistic_diff(a):
    return a * (1 - a)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(y, t):
    """
    Cross entropy
    """
    return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))


def MSE(y, t):
    """
    Mean squared error
    """
    return 0.5 * np.mean((y - t)**2)



def loss(w0,w1, b0, b1, X, T):
    """
    Loss function
    """
    a0 = logistic(X @ w0 + b0)
    a1 = logistic(a0 @ w1 + b1)
    return cross_entropy(a1, T)




class NN():
    """
    Neural network with one hidden layer
    """

    def __init__(self, dim_hidden = 6, eta=0.001, epochs = 100, tol=0.01, n_epochs_no_update=10):
        # Intialize the hyperparameters
        self.dim_hidden = dim_hidden
        self.eta = eta
        self.epochs = epochs
        self.tol = tol
        self.n_epochs_no_update = n_epochs_no_update

        self.activ = logistic
        self.activ_diff = logistic_diff




    def fit(self, X_train, t_train, X_val=None, t_val=None,  batch_size=5):
        (N, m) = X_train.shape
        batches = int(N/batch_size)
        dim_in = m 
        dim_out = t_train.shape[1]
        self.init_weights_and_biases(dim_in, dim_out)

        if (X_val is None) or (t_val is None): 
            for e in range(self.epochs):
                for _ in range(batches):
                    random_index = batch_size*onp.random.randint(batches)
                    w0g, b0g, w1g, b1g = self.backpropagation\
                    (
                        X_train[random_index:random_index+batch_size],
                        t_train[random_index:random_index+batch_size],
                    )

        else:
            self.loss = np.zeros(self.epochs)
            self.accuracies = np.zeros(self.epochs)

            for e in range(self.epochs):
                for _ in range(batches):
                    random_index = batch_size*onp.random.randint(batches)
                    w0g, b0g, w1g, b1g = self.backpropagation\
                    (
                        X_train[random_index:random_index+batch_size],
                        t_train[random_index:random_index+batch_size],
                    )
                self.loss[e] = MSE(self.weights, X_val, t_val)
                # self.accuracies[e]= accuracy(self.predict(X_val), t_val)


                if e > self.n_epochs_no_update and np.abs(self.loss[e-self.n_epochs_no_update] - self.loss[e]) < self.tol:
                    self.loss[e:] = self.loss[e]
                    print(f"Early stopping at epoch {e}")
                    return
                print("\rDid not converge")




    def forward(self, X):
        """
        Forward pass through the network
        returns ( hidden layer, output layer )
        """
        a0 = self.activ(X @ self.w0 + self.b0)  # hidden layer
        a1 = self.activ(a0 @ self.w1 + self.b1) # output layer
        return a0, a1




    def predict(self, X):
        forw = self.forward(X)
        return np.argmax(forw[1], axis=1) 




    def init_weights_and_biases(self, dim_in, dim_out):
        self.w0 = onp.random.randn(dim_in, self.dim_hidden)
        self.b0 = np.zeros(self.dim_hidden) + 0.01
        self.w1 = onp.random.randn(self.dim_hidden, dim_out)
        self.b1 = np.zeros(dim_out) + 0.01




    def backpropagation(self, X, T ):
        """
        Backpropagation algorithm
        ## Parameters
            X, T : ndarray
                input data and targets
        ## Returns
            :tuple
            ( first layer weights gradient, first layer bias gradient, last layer weights gradient, last layer bias gradient )
        """
        a0, a1 = self.forward(X)
        output_error = (a1 - T)
        hidden_error =  output_error @ self.w1.T * self.activ_diff(a0)

        w0_gradient = X.T @ hidden_error                # * 1/X.shape[0] # can be baked into the learning rate
        b0_gradient = np.sum(hidden_error, axis=0)      # * 1/X.shape[0] # can be baked into the learning rate
        w1_gradient = a0.T @ output_error               # * 1/X.shape[0] # can be baked into the learning rate
        b1_gradient = np.sum(output_error, axis=0)      # * 1/X.shape[0] # can be baked into the learning rate

        return w0_gradient, b0_gradient, w1_gradient, b1_gradient






model = NN(dim_hidden=2, epochs=1)
X = onp.random.randn(10, 3)
T = onp.random.randn(10, 1)
model.fit(X, T, batch_size=10)
w0g, b0g, w1g, b1g = model.backpropagation(X, T)
jaxgrads = grad(loss, argnums=(0,1,2,3))
print("w0g", w0g / X.shape[0])
print("jax w0g", jaxgrads(model.w0, model.w1, model.b0, model.b1, X, T)[0])
print("---------------------------------")
print("b0g", b0g / X.shape[0])
print("jax b0g", jaxgrads(model.w0, model.w1, model.b0, model.b1, X, T)[1])
print("---------------------------------")
print("w1g", w1g / X.shape[0])
print("jax w1g", jaxgrads(model.w0, model.w1, model.b0, model.b1, X, T)[2])
print("---------------------------------")
print("b1g", b1g / X.shape[0])
print("jax b1g", jaxgrads(model.w0, model.w1, model.b0, model.b1, X, T)[3])