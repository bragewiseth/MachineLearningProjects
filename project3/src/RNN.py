import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import numpy as onp

dx=0.1 #space increment
dt=0.05 #time increment
tmin=0.0 #initial time
tmax=2.0 #simulate until
xmin=-5.0 #left bound
xmax=5.0 #right bound...assume packet never reaches boundary
c=1.0 #speed of sound
rsq=(c*dt/dx)**2 #appears in finite diff sol

nx = int((xmax-xmin)/dx) + 1 #number of points on x grid
nt = int((tmax-tmin)/dt) + 2 #number of points on t grid
u = np.zeros((nt,nx)) #solution to WE

#set initial pulse shape
def init_fn(x):
    val = np.exp(-(x**2)/0.25)
    if val<.001:
        return 0.0
    else:
        return val

for a in range(0,nx):
    u[0,a]=init_fn(xmin+a*dx)
    u[1,a]=u[0,a]

#simulate dynamics
for t in range(1,nt-1):
    for a in range(1,nx-1):
        u[t+1,a] = 2*(1-rsq)*u[t,a]-u[t-1,a]+rsq*(u[t,a-1]+u[t,a+1])


#--------------------perdicting with RNN--------------------
# ----- koden jeg brukte en dag på å skrive...
# dataset


# test train split


# input_dim = u.shape[1]  # the amount of features
# input_length = 1  # the amount of time steps

# # model
# model = keras.models.Sequential()
# model.add(layers.SimpleRNN(32, input_shape=(input_length, input_dim), activation='relu'))#input and hidden layer
# #model.add(layers.Dense(32, activation='relu'))#hidden layer
# model.add(layers.Dense(u.shape[1], activation='relu'))#output layer
# model.compile(optimizer="RMSprop", loss="mse")

# print(model.summary())

# X_train = u[:-1]#all but last two
# y_train = u[1:]#all but first two

# print(X_train.shape)

# # Reshape X_train to be 3D [samples, timesteps, features]
# X_train = X_train.reshape((X_train.shape[0], input_length, input_dim))

# model.fit(X_train, y_train, epochs=100, batch_size=32)

#-----------koden uiogpt brukte 10 sek på å skrive...
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(u.shape)
# reshape data for RNN input
data = u.reshape((nt, nx, 1))
print(data.shape)

exit()
# define sizes
input_size = 1
output_size = 1
hidden_layer_size = 50

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(hidden_layer_size, return_sequences=True, input_shape=[None, input_size]),
    tf.keras.layers.Dense(output_size)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# prepare data for training
X_train = data[:-1]
Y_train = data[1:]

# train model
history = model.fit(X_train, Y_train, epochs=50, verbose=2, batch_size=1)



#-------------------plotting--------------------

X_train_predict = model.predict(X_train)

fig = plt.figure()
plts = []  # get ready to populate this list the Line artists to be plotted
for i in range(len(X_train_predict)):
    p1, = plt.plot(u[i,:], 'k')  # this is how you'd plot a single line...
    p2, = plt.plot(X_train_predict[i,:], 'r')  # plot the prediction
    plts.append([p1, p2])  # save the line artist for the animation
ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)  # run the animation
#ani.save('wave.gif', writer='pillow')  # optionally save it to a file

plt.show()
