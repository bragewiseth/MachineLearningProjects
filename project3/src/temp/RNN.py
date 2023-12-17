import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import tensorflow as tf
from tensorflow import keras

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


# reshape data for RNN input
data = u.reshape((nt, nx, 1))

# define sizes
input_size = 1
output_size = 1
hidden_layer_size = 50

# define model
model = keras.models.Sequential([
    keras.layers.SimpleRNN(hidden_layer_size, return_sequences=True, input_shape=[None, input_size]),
    keras.layers.Dense(output_size)# return_sequences=True means that the output of the RNN layer will be a sequence rather than a single vector
])

model.compile(optimizer='adam', loss='mean_squared_error')

# prepare data for training
X_train = data[:-1]
Y_train = data[1:]

print("data: " + str(data.shape))
print("X_train: " + str(X_train.shape))

# train model
model.fit(X_train, Y_train, epochs=10, verbose=2, batch_size=1)



#-------------------plotting--------------------

X_train_predict = model.predict(X_train)
print("X_train_predict: " + str(X_train_predict.shape))

fig = plt.figure()
plts = []  # list with all the images to be plotted
for i in range(len(X_train_predict)):
    p1, = plt.plot(u[i,:], 'k')  # plot the simulation
    p2, = plt.plot(X_train_predict[i,:], 'r')  # plot the prediction
    plts.append([p1, p2])  # save the line artist for the animation
ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)  # run the animation
#ani.save('wave.gif', writer='pillow')  # optionally save it to a file

plt.show()
