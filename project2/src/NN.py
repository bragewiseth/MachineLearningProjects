import numpy as np
from activationFunctions import *


class NN():
    """
    A multi-layer neural network with one hidden layer
    """
    
    def __init__(self, dim_hidden = 6, eta=0.001, epochs = 100, X_val=None, t_val =None, tol=0.01, n_epochs_no_update=10):
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
                    random_index = batch_size*np.random.randint(batches)
                    self.backpropagation(
                        X_train[random_index:random_index+batch_size], 
                        t_train[random_index:random_index+batch_size], N
                    )

        else:
            self.loss = np.zeros(self.epochs)
            self.accuracies = np.zeros(epochs)

            for e in range(self.epochs):
                for _ in range(batches):
                    random_index = batch_size*np.random.randint(batches)
                    self.backpropagation( 
                        X_train[random_index:random_index+batch_size],
                        t_train[random_index:random_index+batch_size], N
                    )
                self.loss[e] = MSE(self.weights, X_val, t_val)
                self.accuracies[e]= accuracy(self.predict(X_val), t_val)


                if e > self.n_epochs_no_update and np.abs(self.loss[e-self.n_epochs_no_update] - self.loss[e]) < self.tol:
                    self.loss[e:] = self.loss[e]
                    print(f"Early stopping at epoch {e}")
                    return
                print("\rDid not converge")


        
            
    def forward(self, X):
        hidden_activations = self.activ(X @ self.weights1)
        outputs = hidden_outs @ self.weights2
        return hidden_outs, outputs
    



    def predict(self, X):
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5).astype('int')
    



    def predict_probability(self, X):
        return self.forward(Z)[1][:, 0]
        




    def init_weights_and_biases(self, dim_in, dim_out):
        self.hidden_weights = np.random.randn(dim_in, self.dim_hidden)
        self.hidden_bias = np.zeros(self.dim_hidden) + 0.01
        self.output_weights = np.random.randn(self.dim_hidden, dim_out)
        self.output_bias = np.zeros(dim_out) + 0.01





    def backpropagation(self, X, T, N):
        # One epoch
        hidden_outs, outputs = self.forward(X)
        # The forward step
        out_deltas = (outputs - T)
        # The delta term on the output node
        hiddenout_diffs = out_deltas @ self.weights2.T
        # The delta terms at the output of the jidden layer
        hiddenact_deltas = (hiddenout_diffs[:, 1:] * self.activ_diff(hidden_outs[:, 1:])) # first index is bias hence [:, 1:]

        self.weights2 -= self.eta * hidden_outs.T @ out_deltas
        self.weights1 -= self.eta * X.T @ hiddenact_deltas 
