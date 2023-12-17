import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import tensorflow as tf
from tensorflow import keras


#--------------simulation----------------

dx=0.1 #space increment
dt=0.05 #time increment
tmin=0.0 #initial time
tmax=3.0 #simulate until
xmin=-5.0 #left bound
xmax=5.0 #right bound
ymin=-5.0 #bottom bound
ymax=5.0 #top bound
c=1.0 #speed of sound
rsq=(c*dt/dx)**2 #appears in finite diff sol

nx = int((xmax-xmin)/dx) + 1 #number of points on x grid
ny = int((ymax-ymin)/dx) + 1 #number of points on y grid
nt = int((tmax-tmin)/dt) + 2 #number of points on t grid
u = np.zeros((nt,nx,ny)) #solution to WE

#set initial pulse shape
def init_fn(x, y):
    val = np.exp(-((x**2 + y**2)/0.25))
    if val<.001:
        return 0.0
    else:
        return val

for i in range(nx):
    for j in range(ny):
        u[0,i,j]=init_fn(xmin+i*dx, ymin+j*dx)
        u[1,i,j]=u[0,i,j]

#simulate dynamics
for t in range(1,nt-1):
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u[t+1,i,j] = 2*(1-2*rsq)*u[t,i,j] - u[t-1,i,j] + rsq*(u[t,i-1,j] + u[t,i+1,j] + u[t,i,j-1] + u[t,i,j+1])


fig = plt.figure()
ims = []  #list with all the images to be plotted
for i in range(len(u)):
    im = plt.imshow(u[i,:,:], animated=True)  # imshow for 2D images...
    ims.append([im])  
ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000)  # run the animation
plt.show()

#--------------RNN----------------

# reshape data for RNN input
data = u.reshape((nt, nx*ny))
data = np.expand_dims(data, axis=2)

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

# train model
history = model.fit(X_train, Y_train, epochs=10, verbose=2, batch_size=1)

#--------------Prediction & Plotting----------------

X_train_predict = model.predict(X_train)

# Reshape the predicted data to be 2D for plotting
X_train_predict = X_train_predict.reshape(nt - 1, nx, ny)

# Plotting
fig = plt.figure()
ims = []  #list with all the images to be plotted
for i in range(len(X_train_predict)):
    im = plt.imshow(X_train_predict[i,:,:], animated=True)  # imshow for 2D images...
    ims.append([im])  
ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000)  # run the animation
plt.show()

fig = plt.figure()
ims = []  #list with all the images to be plotted
for i in range(len(X_train_predict)):
    im = plt.imshow(X_train_predict[i,:,:], animated=True)  # imshow for 2D images...
    ims.append([im])  
ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000)  # run the animation
plt.show()

# bruk subplots til å compare de
