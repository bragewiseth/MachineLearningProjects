import jax.numpy as np
import jax
from matplotlib import pyplot as plt
import pandas as pd
from jax.lib import xla_bridge
import ADAMLL as ada
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as onp
print("jax backend {}".format(xla_bridge.get_backend().platform))

key = jax.random.PRNGKey(2024)


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train).reshape(-1,1)
y_test = np.asarray(y_test).reshape(-1,1)

print("X_train shape {}".format(X_train.shape))
print("y_train shape {}".format(y_train.shape))
print("X_test shape {}".format(X_test.shape))
print("y_test shape {}".format(y_test.shape))



#The NN has a single output node, the number of input nodes matches the number of features in the data



activations = [ada.activations.sigmoid, ada.activations.tanh, ada.activations.relu]
n_neurons = [1,3,8,21,55]
n_hidden_layers = [1,2,3,4,5]
eta = 0.01
accuracy = []


heatmap = onp.zeros((len(n_neurons), len(n_hidden_layers)))


for i in range(len(n_hidden_layers)):
    for j in range(len(n_neurons)):
        architecture = [[n_neurons[j], ada.activations.sigmoid] for k in range(n_hidden_layers[i])]
        architecture.append([1, ada.activations.sigmoid])
        network = ada.NN.Model(architecture=architecture, eta=eta, epochs=300, optimizer='sgd', loss=ada.CE)
        #fitting the data and finding the accuracy
        l,_ = network.fit(X_train,y_train, X_test, y_test)
        heatmap[j,i] = ada.accuracy(network.classify(X_test), y_test)

fig = sns.heatmap(heatmap, annot=True, xticklabels=n_neurons, yticklabels=n_hidden_layers)
plt.xlabel("neurons")
plt.ylabel("hidden_layers")
plt.show()


for func in activations:
    print("activation function {}".format(func))
    #creating the network using the best architecture from the test above
    network = ada.NN.Model(architecture=[[1, func] for i in range(4)], eta=eta, epochs=300, optimizer='sgd', loss=ada.CE)
    
    #fitting the data and finding the accuracy
    l,_ = network.fit(X_train,y_train, X_test, y_test)
    accuracy.append(ada.accuracy(network.classify(X_test), y_test))
    print(l)

plt.bar(["sigmoid", "tanh", "relu"], accuracy)
plt.show()