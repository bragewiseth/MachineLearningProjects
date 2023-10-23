class MLPBinaryLinRegClass():
    # A multi-layer neural network with one hidden layer
    
    def __init__(self, bias=-1, dim_hidden = 6):
        # Intialize the hyperparameters
        self.bias = bias
        self.dim_hidden = dim_hidden
        
        self.activ = logistic
        self.activ_diff = logistic_diff
        
    def fit(self, X_train, t_train, eta=0.001, epochs = 100, X_val=None, t_val =None, tol=0.01, n_epochs_no_update=5):
    
        self.eta = eta   
        X = add_bias(X_train, self.bias) 
        T = t_train.reshape(-1,1)
        dim_in = X_train.shape[1] 
        dim_out = T.shape[1]
        
        # Itilaize the wights
        self.weights1 = (np.random.rand( dim_in + 1, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand( self.dim_hidden+1, dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        
    
        if (X_val is None) or (t_val is None):
            for e in range(epochs):
                self.backpropagation(X, T)


        else:
            self.loss = np.zeros(epochs)
            self.accuracies = np.zeros(epochs)
            self.epochs_ran = epochs
            # loop trough first n_epochs_no_update
            for e in range(n_epochs_no_update):
                self.backpropagation(X, T)     
                self.loss[e] =      MSE(self.predict_probability(X_val), t_val)
                self.accuracies[e]= accuracy(self.predict(X_val), t_val)

            # loop trough rest
            for e in range(n_epochs_no_update, epochs):
                self.backpropagation(X, T)      
                self.loss[e] =      MSE(self.predict_probability(X_val), t_val)
                self.accuracies[e] =accuracy(self.predict(X_val), t_val)
                # if no update exit training
                if self.loss[e-n_epochs_no_update] - self.loss[e] < tol:
                    self.epochs_ran = e
                    self.loss = self.loss[:e]
                    self.accuracies = self.accuracies[:e]
                    break
        
            
    def forward(self, X):
        hidden_activations = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden_activations, self.bias)
        outputs = hidden_outs @ self.weights2
        return hidden_outs, outputs
    
    def predict(self, X):
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5).astype('int')
    
    def predict_probability(self, X):
        Z = add_bias(X, self.bias)
        return self.forward(Z)[1][:, 0]
        
    

    def backpropagation(self, X, T):
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








class MLPMultiLogRegClass():
    # A multi-layer neural network with one hidden layer
    
    def __init__(self, bias=-1, dim_hidden = 6):
        # Intialize the hyperparameters
        self.bias = bias
        self.dim_hidden = dim_hidden
        
        self.activ = logistic
        self.activ_soft = softmax
        self.activ_diff = logistic_diff

        
        
    def fit(self, X_train, t_train, eta=0.001, epochs = 100, X_val=None, t_val=None, tol=0.01, n_epochs_no_update=5):
    
        self.eta = eta    
        T = one_hot_encoding(t_train)
        X = add_bias(X_train, self.bias)
        dim_in = X_train.shape[1] 
        dim_out = T.shape[1]
        
        # Itilaize the wights
        self.weights1 = (np.random.rand( dim_in + 1, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand( self.dim_hidden+1, dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        
        

        if (X_val is None) or (t_val is None):
            for e in range(epochs):
                self.backpropagation(X, T)

        else:
            T_val = one_hot_encoding(t_multi_val)
            self.loss = np.zeros(epochs)
            self.accuracies = np.zeros(epochs)
            self.epochs_ran = epochs
            # loop trough first n_epochs_no_update
            for e in range(n_epochs_no_update):
                self.backpropagation(X, T)     
                self.loss[e] =      MSE(self.predict_probability(X_val), T_val)
                self.accuracies[e]= accuracy(self.predict(X_val), T_val)

            # loop trough rest
            for e in range(n_epochs_no_update, epochs):
                self.backpropagation(X, T)      
                self.loss[e] =      MSE(self.predict_probability(X_val), T_val)
                self.accuracies[e] =accuracy(self.predict(X_val), T_val)
                # if no update exit training
                if self.loss[e-n_epochs_no_update] - self.loss[e] < tol:
                    self.epochs_ran = e
                    self.loss = self.loss[:e]
                    self.accuracies = self.accuracies[:e]
                    break
        
            
    def forward(self, X):
        hidden_activations = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden_activations, self.bias)
        outputs = self.activ_soft(hidden_outs @ self.weights2)
        return hidden_outs, outputs
    
    def predict(self, X):
        Z = add_bias(X, self.bias)
        return self.forward(Z)[1].argmax(axis=1) 
    
    def predict_probability(self, X):
        Z = add_bias(X, self.bias)
        return self.forward(Z)[1]
        
    

    def backpropagation(self, X, T):
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