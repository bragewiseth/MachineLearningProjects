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




activations = [ada.activations.sigmoid, ada.activations.tanh, ada.activations.relu]
n_neurons = [1,3]
n_hidden_layers = [1,2]
eta = 0.001
loss = []


import pandas as pd
heatmap= []

for i in n_hidden_layers:
    accuracy = []
    for j in n_neurons:
        #The NN has a single output node, the number of input nodes matches the number of features in the data
        #the NN becomes a logistic regression model
        network = ada.NN.Model(architecture=[[j, ada.activations.sigmoid] for k in range(i)], eta=eta, epochs=300, optimizer='sgd', loss=ada.CE)
        #fitting the data and finding the accuracy
        l,_ = network.fit(X_train,y_train, X_test, y_test)
        accuracy.append(ada.accuracy(network.classify(X_test), y_test))
        loss.append(l)
    heatmap.append(accuracy)

df = pd.DataFrame(heatmap, index=n_neurons, columns=n_hidden_layers)
print("Heatmap")
print(df)
sns.heatmap(df)
plt.show()
accuracy = []


for func in activations:
    #The NN has a single output node, the number of input nodes matches the number of features in the data
    #the NN becomes a logistic regression model
    network = ada.NN.Model(architecture=[[1, func]], eta=eta, epochs=300, optimizer='sgd', loss=ada.CE)
    
    #fitting the data and finding the accuracy
    l,_ = network.fit(X_train,y_train, X_test, y_test)
    accuracy.append(ada.accuracy(network.classify(X_test), y_test))
    loss.append(l)

#plt.plot([str(activation) for activation in activations], accuracy)

plt.figure()
plt.plot(loss[0], label='eta=0.1')
plt.show()