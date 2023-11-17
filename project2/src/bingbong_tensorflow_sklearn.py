import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pickle
import os 


"""Load breast cancer dataset"""

onp.random.seed(0)        #create same seed for random number every time

cancer=load_breast_cancer()      #Download breast cancer dataset

inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
outputs=cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
labels=cancer.feature_names[0:30]

print('The content of the breast cancer dataset is:')      #Print information about the datasets
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

x=inputs      #Reassign the Feature and Label matrices to other variables
y=outputs



# Generate training and testing datasets

#Select features relevant to classification (texture,perimeter,compactness and symmetery) 
#and add to input matrix
     

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,random_state=0)

y_train=to_categorical(y_train)     #Convert labels to categorical when using categorical cross entropy
y_test=to_categorical(y_test)

# %%

# Define tunable parameters"

eta= 0.001                  #Define vector of learning rates (parameter to SGD optimiser)
lamda=0.01                                  #Define hyperparameter
n_layers= [1, 2, 3, 5]                             #Define number of hidden layers in the model
n_nodes= [1, 10, 50, 100]       #Define number of neurons per layer
epochs=100                                   #Number of reiterations over the input data
batch_size=100                              #Number of samples per gradient update

# %%

"""Define function to return Deep Neural Network model"""

def NN_model(inputsize,n_layers,n_nodes,eta,lamda):
    model=Sequential()      
    for i in range(n_layers):       #Run loop to add hidden layers to the model
        if (i==0):                  #First layer requires input dimensions
            model.add(Dense(n_nodes,activation='sigmoid',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize))
        else:                       #Subsequent layers are capable of automatic shape inferencing
            model.add(Dense(n_nodes,activation='sigmoid',kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(2,activation='sigmoid'))  #2 outputs - ordered and disordered (softmax for prob)
    sgd=optimizers.SGD(lr=eta)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

    
Train_accuracy=onp.zeros((len(n_nodes),len(n_layers)))      #Define matrices to store accuracy scores as a function
Test_accuracy=onp.zeros((len(n_nodes),len(n_layers)))       #of learning rate and number of hidden neurons for 

for i in range(len(n_nodes)):     #run loops over hidden neurons and learning rates to calculate 
    for j in range(len(n_layers)):      #accuracy scores 
        DNN_model=NN_model(X_train.shape[1],n_layers[j],n_nodes[i],eta,lamda)
        DNN_model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
        #Train_accuracy[i,j]=DNN_model.evaluate(X_train,y_train)[1]
        Test_accuracy[i,j]=DNN_model.evaluate(X_test,y_test)[1]


               
plt.figure()
ax = sns.heatmap(Test_accuracy, annot=True, fmt=".4f", cmap="rocket", vmax=1.0, vmin=0.3)
ax.set_title("Accuracy")
ax.set_xlabel("Nodes")
ax.set_xticklabels(n_nodes)
ax.set_ylabel("Layers")
ax.set_yticklabels(n_layers)
#plt.savefig("../runsAndFigures/accuracy_layers_nodes.png",bbox_inches='tight')
plt.show()

"""
#Copied from the fys-stk4155 repository: https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/week42.html#the-breast-cancer-data-now-with-keras
import numpy as onp
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import pickle
import os 

X_train = onp.asarray(X_train)
X_test = onp.asarray(X_test)
y_train = onp.asarray(y_train).reshape(-1,1)
y_test = onp.asarray(y_test).reshape(-1,1)

                      #Number of samples per gradient update



"Define function to return Deep Neural Network model"

def NN_model(inputsize,n_layers,n_nodes,eta,lamda):
    model=Sequential()      
    for i in range(n_layers):       #Run loop to add hidden layers to the model
        if (i==0):                  #First layer requires input dimensions
            model.add(Dense(n_nodes,activation='relu',kernel_regularizer=regularizers.l2(lamda),input_dim=inputsize))
        else:                       #Subsequent layers are capable of automatic shape inferencing
            model.add(Dense(n_nodes,activation='relu',kernel_regularizer=regularizers.l2(lamda)))
    model.add(Dense(1,activation='sigmoid'))  #2 outputs - ordered and disordered (softmax for prob)
    sgd=optimizers.SGD(lr=eta)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model

    
Train_accuracy=onp.zeros((len(n_nodes),len(etas)))      #Define matrices to store accuracy scores as a function
Test_accuracy=onp.zeros((len(n_nodes),len(etas)))       #of learning rate and number of hidden neurons for 

for i in range(len(n_nodes)):
    for J in range(len(n_layers)):     #run loops over hidden neurons and learning rates to calculate 
        DNN_model=NN_model(X_train.shape[1],n_layers[J],n_nodes[i],0.01, 0.0001)
        DNN_model.fit(X_train,y_train,epochs=100,batch_size=batch_size,verbose=0)
        Train_accuracy[i,j]=DNN_model.evaluate(X_train,y_train)[1]
        Test_accuracy[i,j]=DNN_model.evaluate(X_test,y_test)[1]
               

plt.figure()
ax = sns.heatmap(Test_accuracy, annot=True, fmt=".4f", cmap="rocket", vmax=1.0, vmin=0.3)
ax.set_title("Accuracy")
ax.set_xlabel("Nodes")
ax.set_xticklabels(n_nodes)
ax.set_ylabel("Layers")
ax.set_yticklabels(n_layers)
#plt.savefig("../runsAndFigures/accuracy_layers_nodes.png",bbox_inches='tight')
plt.show()
"""

"""

#Copied from the fys-stk4155 repository: https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter4.html


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score   
# store models for later use
Test_accuracy = onp.zeros((len(n_nodes), len(n_layers)))

for i in range(len(n_nodes)):
    for j in range(len(n_layers)):
        dnn = MLPClassifier(hidden_layer_sizes=(tuple([n_nodes[i] for k in range(n_layers[j])])), activation='logistic',
                            learning_rate_init=eta, max_iter=epochs, momentum=gamma,) 
        dnn.fit(X_train, y_train)
        y_pred = dnn.predict(X_test)
        Test_accuracy[i][j] = accuracy_score(y_test, y_pred) 
        
plt.figure()
ax = sns.heatmap(Test_accuracy, annot=True, fmt=".4f", cmap="rocket", vmax=1.0, vmin=0.3)
ax.set_title("Accuracy")
ax.set_xlabel("Nodes")
ax.set_xticklabels(n_nodes)
ax.set_ylabel("Layers")
ax.set_yticklabels(n_layers)
#plt.savefig("../runsAndFigures/accuracy_layers_nodes.png",bbox_inches='tight')
plt.show()

"""