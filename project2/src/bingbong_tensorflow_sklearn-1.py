#Copied from the fys-stk4155 repository: https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter4.html
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score   


def cross_val_sklearn(X_train, y_train, model, kfold):
    k = 0
    score = onp.zeros(kfold.n_splits)
    for train_ind, test_ind in kfold.split(X_train):
        model.fit(X_train[train_ind], y_train[train_ind])
        score[k] = ada.accuracy(model.predict(X_train[test_ind]), y_train[test_ind])
        k += 1
    return score.mean(), score.std(), loss


Test_accuracy = onp.zeros((len(n_nodes), len(n_layers)))

for i in range(len(n_nodes)):
    for j in range(len(n_layers)):
        dnn = MLPClassifier(hidden_layer_sizes=(tuple([n_nodes[i] for k in range(n_layers[j])])), activation='logistic',
                            learning_rate_init=eta, max_iter=epochs, momentum=gamma,) 
        dnn.fit(X_train, y_train)
        y_pred = dnn.predict(X_test)
        scoremean, scorestd, loss = cross_val_sklearn(X_train, y_train, dnn, kfold)
        Test_accuracy[i, j] = scoremean
        
plt.figure()
ax = sns.heatmap(Test_accuracy, annot=True, fmt=".4f", cmap="rocket", vmax=1.0, vmin=0.3)
ax.set_title("Accuracy")
ax.set_xlabel("Nodes")
ax.set_xticklabels(n_layers)
ax.set_ylabel("Layers")
ax.set_yticklabels(n_nodes)
#plt.savefig("../runsAndFigures/accuracy_layers_nodes.png",bbox_inches='tight')
plt.show()