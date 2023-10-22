r"""°°°
# Week 41
°°°"""
# |%%--%%| <hGEkvQzFu2|nYv1ZH37Ti>
r"""°°°
### OLS gradient descent
°°°"""
# |%%--%%| <nYv1ZH37Ti|DB89Iq22ja>

import numpy as np

# |%%--%%| <DB89Iq22ja|85EG4cKh4X>

class NumpyLinRegClass():

    def __init__(self, bias=-1, dim_hidden=0):
        self.bias=bias
        
    
    def fit(self, X_train, t_train, X_val=None, t_val=None, eta = 0.1, epochs=10, tol=0.001):

        if self.bias:
            X = add_bias(X_train, self.bias)

        (N, m) = X.shape
        
        self.weights = weights = np.zeros(m)
        
        if (X_val is None) or (t_val is None): 
            for e in range(epochs):
                weights -= eta / N *  X.T @ (X @ weights - t_train)  
    
        else:
            self.loss = np.zeros(epochs)
            self.accuracies = np.zeros(epochs)
            for e in range(epochs):
                weights -= eta / N *  X.T @ (X @ weights - t_train)   
                self.loss[e] =          MSE(X @ self.weights, t_train)
                self.accuracies[e] =    accuracy(self.predict(X_val), t_val)



    def predict(self, X, threshold=0.5):
        if self.bias:
            X = add_bias(X, self.bias)

        ys = X @ self.weights
        return ys > threshold

# |%%--%%| <85EG4cKh4X|B4JxqCncb8>

class NumpyLogReg():


    def __init__(self, bias=-1, dim_hidden=0):
        self.bias=bias

        
    def fit(self, X_train, t_train, X_val=None, t_val=None, eta = 0.1, epochs=10, tol=0.01, n_epochs_no_update=5):
        
        (N, m) = X_train.shape
        if self.bias:
            X = add_bias(X_train, self.bias)
        

        self.weights = weights = np.zeros(m+1)
        

        if (X_val is None) or (t_val is None):       
            for e in range(epochs):
                weights -= eta / N *  X.T @ (self.forward(X) - t_train)  
        
        else:
            self.loss = np.zeros(epochs)
            self.val_loss = np.zeros(epochs)
            self.accuracies = np.zeros(epochs)
            self.val_accuracies = np.zeros(epochs)
            self.epochs_ran = epochs
            # loop trough first n_epochs_no_update
            for e in range(n_epochs_no_update):
                weights -= eta / N *  X.T @ (self.forward(X) - t_train)      
                self.loss[e] =          CE(self.predict_probability(X_train), t_train)
                self.val_loss[e] =      CE(self.predict_probability(X_val), t_val)
                self.val_accuracies[e]= accuracy(self.predict(X_val), t_val)
                self.accuracies[e] =    accuracy(self.predict(X_train), t_train)
            # loop trough rest
            for e in range(n_epochs_no_update, epochs):
                weights -= eta / N *  X.T @ (self.forward(X) - t_train)      
                self.loss[e] =          CE(self.predict_probability(X_train), t_train)
                self.val_loss[e] =      CE(self.predict_probability(X_val), t_val)
                self.val_accuracies[e] =accuracy(self.predict(X_val), t_val)
                self.accuracies[e] =    accuracy(self.predict(X_train), t_train)
                # if no update exit training
                if self.loss[e-n_epochs_no_update] - self.loss[e] < tol:
                    self.epochs_ran = e
                    self.loss = self.loss[:e]
                    self.val_loss = self.val_loss[:e]
                    self.val_accuracies = self.val_accuracies[:e]
                    self.accuracies = self.accuracies[:e]
                    break

            
    
    def forward(self, X):
        return logistic(X @ self.weights)
    

    def predict(self, x, threshold=0.5):
        if self.bias:
            x = add_bias(x,self.bias)
        return (self.forward(x) > threshold).astype('int')
    
    def predict_probability(self, x):
        if self.bias:
            x = add_bias(x,self.bias)
        return (self.forward(x))

# |%%--%%| <B4JxqCncb8|DqNArB72Bc>

class NumpyLogRegMulti():


    def __init__(self, bias=-1, dim_hidden=0):
        self.bias=bias

        
    def fit(self, X_train, t_train, X_val=None, t_val=None, eta = 0.1, epochs=10, tol=0.01, n_epochs_no_update=5):
        
        (N, m) = X_train.shape
        t = one_hot_encoding(t_train)

        if self.bias:
            X = add_bias(X_train, self.bias)
        

        self.weights = weights = np.zeros((m+1, t.shape[1]))
        

        if (X_val is None) or (t_val is None):      
            for e in range(epochs):
                weights -= eta / N *  X.T @ (self.forward(X) - t)  


        else: # if validation is provided: calculate loss and accuracies for each epoch
            t_val_t = one_hot_encoding(t_val)
            self.loss = np.zeros(epochs)
            self.val_loss = np.zeros(epochs)
            self.accuracies = np.zeros(epochs)
            self.val_accuracies = np.zeros(epochs)
            self.epochs_ran = epochs
            # loop trough first n_epochs_no_update
            for e in range(n_epochs_no_update):
                weights -= eta / N *  X.T @ (self.forward(X) - t)      
                self.loss[e] =          sum(CE(self.predict_probability(X_train), t))
                self.val_loss[e] =      sum(CE(self.predict_probability(X_val), t_val_t))
                self.val_accuracies[e]= accuracy(self.predict(X_val), t_val)
                self.accuracies[e] =    accuracy(self.predict(X_train), t_train)
            # loop trough rest
            for e in range(n_epochs_no_update, epochs):
                weights -= eta / N *  X.T @ (self.forward(X) - t)      
                self.loss[e] =          sum(CE(self.predict_probability(X_train), t))
                self.val_loss[e] =      sum(CE(self.predict_probability(X_val), t_val_t))
                self.val_accuracies[e] =accuracy(self.predict(X_val), t_val)
                self.accuracies[e] =    accuracy(self.predict(X_train), t_train)
                # if no update exit training
                if self.loss[e-n_epochs_no_update] - self.loss[e] < tol:
                    self.epochs_ran = e
                    self.loss = self.loss[:e]
                    self.val_loss = self.val_loss[:e]
                    self.val_accuracies = self.val_accuracies[:e]
                    self.accuracies = self.accuracies[:e]
                    break

            
    
    
    def forward(self, X):
        return logistic(X @ self.weights)
    
    def predict(self, x, threshold=0.5):
        if self.bias:
            x = add_bias(x,self.bias)
        return self.forward(x).argmax(axis=1)
    
    def predict_probability(self, x):
        if self.bias:
            x = add_bias(x,self.bias)
        return (self.forward(x))
    

    def score(self,x,t):
        return accuracy(self.predict(x), t)

# |%%--%%| <DqNArB72Bc|v33OdFFpCC>

# Importing various packages
import numpy as np
import pandas as pd

np.random.seed(2020)

learning_rate = [0.0001, 0.001, 0.01, 0.1]
momentum = [0.0, 0.1, 0.5, 0.9]
change = 0.0
mse = []

# the number of datapoints
for i in range(len(momentum)):
    mse_tmp = []
    for j in range(len(learning_rate)):
        polydegree = 2
        n = 100
        x = 2*np.random.rand(n,1)
        y = 4+3*x**2+np.random.randn(n,1)

        X = np.c_[np.ones((n,1)), x, x**2] 

        beta = np.random.randn(polydegree+1,1)

        eta = learning_rate[j]#learning rate
        Niterations = 1000

        for iter in range(Niterations):
            gradient = (2.0/n)*X.T @ (X @ beta-y)
            change = eta*gradient + momentum[i]*change
            beta -= change

        
        ypredict = X.dot(beta)

        mse_tmp.append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
        
    mse.append(mse_tmp)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)
print("\nmin mse:",np.min(mse))

# |%%--%%| <v33OdFFpCC|kzCHQ2bAKH>
r"""°°°
We see that a low learning rate results in bad MSEs probably due to a relatively low number of iterations and it not converging. With higher momentum it drastically improves while the higher learning rates at best get minor improvements.
°°°"""
# |%%--%%| <kzCHQ2bAKH|1hmrs3xxtO>
r"""°°°

### Ridge gradient descent
°°°"""
# |%%--%%| <1hmrs3xxtO|tgQYpkD7jC>

# Importing various packages
import numpy as np
import pandas as pd


learning_rate = [ 0.001, 0.01, 0.1, 0.25]
momentum = [0.0, 0.1, 0.5, 0.9]
change = 0.0
mse = []
lamda_list = [0.0001, 0.001, 0.01, 0.1, 1.0]


# the number of datapoints
for i in range(len(momentum)):
    mse_tmp = []
    for j in range(len(learning_rate)):
        mse_tmp_tmp = []
        for k in range(len(lamda_list)):
            polydegree = 2
            n = 100
            x = 2*np.random.rand(n,1)
            y = 4+3*x**2+np.random.randn(n,1)

            X = np.c_[np.ones((n,1)), x, x**2] 

            beta = np.random.randn(polydegree+1,1)

            eta = learning_rate[j]#learning rate
            Niterations = 1000

            lamda = lamda_list[k]

            for iter in range(Niterations):
                gradient = (2.0/n)*X.T@(X@beta - y) + 2*lamda*beta
                change = eta*gradient + momentum[j]*change
                beta -= change

            ypredict = X.dot(beta)

            mse_tmp_tmp.append(np.round((1.0/n)*np.sum((y - X@beta)**2), 3))
        
        mse_tmp.append(mse_tmp_tmp)
        
    mse.append(mse_tmp)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

for i in range(len(mse)):
    panda = pd.DataFrame(data=mse[i])
    print("momentum = ", momentum[i])  
    print(panda)

print("\nmin mse:",np.min(mse))

# |%%--%%| <tgQYpkD7jC|P0po2K104T>
r"""°°°
Skrive noe her om det over:::::
°°°"""
# |%%--%%| <P0po2K104T|Fl8bpQVJiR>
r"""°°°
### OLS stochastic gradient descent
°°°"""
# |%%--%%| <Fl8bpQVJiR|QFuOldn0gU>

# Importing various packages
import numpy as np
import pandas as pd

mse = []
epochs = [1, 10, 50, 100, 500, 1000]

polydegree = 2
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x**2+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x**2] 

H = (2.0/n)* X.T @ X
EigValues, EigVectors = np.linalg.eig(H)

for n_epochs in epochs:

    beta = np.random.randn(polydegree+1,1)
    eta = 0.01/np.max(EigValues)
    M = 5   #size of each minibatch
    m = int(n/M) #number of minibatches
    t0, t1 = 5, 50

    def learning_schedule(t):
        return t0/(t+t1)

    for epoch in range(n_epochs):
    # selects a random mini-batch at every epoch. it does not garanty that all the data will be used
        for i in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
            eta = learning_schedule(epoch*m+i)
            beta = beta - eta*gradients
    
    mse.append((1.0/n)*np.sum((y - X@beta)**2))

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)

print("\nmin mse:",np.min(mse))

# |%%--%%| <QFuOldn0gU|VNUdbFqOMI>

Skrive noe her om det over:::::

# |%%--%%| <VNUdbFqOMI|WHiHDgg3xb>
r"""°°°
### Ridge stochastic gradient descent
°°°"""
# |%%--%%| <WHiHDgg3xb|ST8hV3veeN>

# Importing various packages
import numpy as np
import pandas as pd

mse = []
epochs = [1, 10, 50, 100, 500]

mse = []
lamda_list = [0.0001, 0.001, 0.01, 0.1, 1.0]

polydegree = 2
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x**2+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x**2] 

H = (2.0/n)* X.T @ X
EigValues, EigVectors = np.linalg.eig(H)

for lamda in lamda_list:
    mse_tmp = []
    for n_epochs in epochs:
        beta = np.random.randn(polydegree+1,1)
        eta = 0.01/np.max(EigValues)
        M = 5   #size of each minibatch
        m = int(n/M) #number of minibatches
        t0, t1 = 5, 50

        def learning_schedule(t):
            return t0/(t+t1)

        for epoch in range(n_epochs):
        # selects a random mini-batch at every epoch. it does not garanty that all the data will be used
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                gradients = (2.0/n)*X.T@(X@beta - y) + 2*lamda*beta
                eta = learning_schedule(epoch*m+i)
                beta = beta - eta*gradients
        
        mse_tmp.append((1.0/n)*np.sum((y - X@beta)**2))
    mse.append(mse_tmp)

beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

print("mse.hape", np.shape(mse))

panda = pd.DataFrame(data=mse)
print(panda)

print("\nmin mse:",np.min(mse))

# |%%--%%| <ST8hV3veeN|i5YSdKTNFT>

# Importing various packages
import numpy as np
import pandas as pd

np.random.seed(2020)

mse = []
epochs = [1, 50, 100, 500, 1000,5000]

polydegree = 2
n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x**2+np.random.randn(n,1)
X = np.c_[np.ones((n,1)), x, x**2] 

H = (2.0/n)* X.T @ X
EigValues, EigVectors = np.linalg.eig(H)
Ms = [1, 5,10,20,40,50]
delta  = 1e-8
rho = 0
for M in Ms:
    msetmp = []
    for n_epochs in epochs:
    
        beta = np.random.randn(polydegree+1,1)
        eta = 0.01/np.max(EigValues)
        #size of each minibatch
        m = int(n/M) #number of minibatches
        t0, t1 = 5, 50
        beta1 = 0.9
        beta2 = 0.999
        iter = 0
        np.random.seed(2020)
        for epoch in range(n_epochs):
            first_moment = 0.0
            second_moment = 0.0
            iter += 1
            # selects a random mini-batch at every epoch. it does not guarantee that all the data will be used
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                gradients = (2.0/M)* xi.T @ ((xi @ beta)-yi)
                first_moment = beta1*first_moment + (1-beta1)*gradients
                second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
                first_term = first_moment/(1.0-beta1**iter)
                second_term = second_moment/(1.0-beta2**iter)
                # Scaling with rho the new and the previous results
                update = eta*first_term/(np.sqrt(second_term)+delta)
                beta -= update
        
        msetmp.append((1.0/n)*np.sum((y - X@beta)**2))

    mse.append(msetmp)
beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
mse_predict = (1.0/n)*np.sum((y - X@beta_linreg)**2)
print("MSE (linreg): ", mse_predict)

panda = pd.DataFrame(data=mse)
print(panda)

print(np.min(mse))


# |%%--%%| <i5YSdKTNFT|u4G4kpWunj>
r"""°°°
MSE for adam with various minibatch sizes verticaly and number of epochs horizontaly 
For smaller minibatches it looks like relatively few epochs are needed before the MSE starts converging towards a good result while larger minibatches seems to need a lot more epochs to aproach similar results.
°°°"""