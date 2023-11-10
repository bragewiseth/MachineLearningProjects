import ADAMLL as ada
from sklearn.datasets import load_breast_cancer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

x, t = cancer.data, cancer.target
print(x.shape)
print(t.shape)
#sns.heatmap(np.corrcoef(x.T), annot=True, fmt='.2f', cmap='Blues')
sns.heatmap(x[:30], fmt='.2f', cmap='Blues')
plt.show()
object = ada.NN.NN(architecture=[[1, ada.activations.sigmoid]], eta=0.1, epochs=100, optimizer='sgd', backwards=None, loss=ada.util.MSE)

print(object.fit(x, t))