import ADAMLL as ada
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
#sns.heatmap(np.corrcoef(X_train.T), fmt='.2f', cmap='Blues')
plt.show()

etas = [0.1, 0.01, 0.001, 0.0001]
accuracy = []
for eta in etas:
    #The NN has a single output node, the number of input nodes matches the number of features in the data
    #the NN becomes a logistic regression model
    network = ada.NN.Model(architecture=[[1, ada.activations.sigmoid]], eta=eta, epochs=100, optimizer='sgd', loss=ada.CE)
    params = network.fit(X_train,y_train, X_test, y_test)[0]
    accuracy.append(ada.util.accuracy(network.forward(network, X_test)[0], y_test))

plt.plot(etas, accuracy)
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.show()