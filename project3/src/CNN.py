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


print(u.shape)
# reshape data for RNN input
data = u.reshape((nt, nx, 1))

#model
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(nx, 1)))#
model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.MaxPooling1D(pool_size=2))#halves the size of the input by taking the max of every 2 values
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(101))

# compile model
model.compile(loss='mse', optimizer='adam')


# fit model
model.fit(data, data, epochs=100, verbose=0)



#-------------------plotting--------------------

data_predict = model.predict(data)

print(data.shape)
print(data_predict.shape)

fig = plt.figure()
plts = []  # list with all the images to be plotted
for i in range(len(data_predict)):
    p1, = plt.plot(u[i,:], 'k')  # plot the simulation
    p2, = plt.plot(data_predict[i,:], 'r')  # plot the prediction
    plts.append([p1, p2])  # save the line artist for the animation
ani = animation.ArtistAnimation(fig, plts, interval=50, repeat_delay=3000)  # run the animation
#ani.save('wave.gif', writer='pillow')  # optionally save it to a file

plt.show()