import ADAMLL as ada
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split 
import jax.numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)
#sns.heatmap(np.corrcoef(X_train.T), fmt='.2f', cmap='Blues')
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
plt.show()


etas = [0.1, 0.01, 0.001, 0.0001]
accuracy = []
for eta in etas:
    #The NN has a single output node, the number of input nodes matches the number of features in the data
    #the NN becomes a logistic regression model
    network = ada.NN.Model(architecture=[[1, ada.activations.sigmoid]], eta=eta, epochs=100, optimizer='sgd', loss=ada.CE)
    loss,_=network.fit(X_train,y_train, X_test, y_test)
    print(loss)
    #print("--------" + str(network.classify(X_test).shape))
    #print("--------" + str(y_test.shape))
    accuracy.append(network.score(X_test,y_test))

plt.plot(etas, accuracy)
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.show()