import numpy as np
import keras

#Dense: Defines a layer of connected neurons
#units: number of neurons in the Dense Layer
#Sequential: Successive layers are defined in sequence
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Loss: Function that measures how accurate the predictions are
#Optmizer: Takes the results from loss and decides how to optimize the network
#sgd: Stochastic Gradient Descent
model.compile(optimizer='sgd', loss='mean_squared_error')

# f(x) = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#Fit is used to train the model
#Epochs are the number of iterations being executed
model.fit(xs, ys, epochs=500)

print(model.predict([10]))
#Expected 19, predicted [[18.984869]]
