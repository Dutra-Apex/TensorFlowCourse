import tensorflow as tf

#Adding Convolution to a neural network

#The first line adds a convolution with 64 filters,
#each filter is 3x3, with rectifier activation (removes negative values)
#The input shape is the same as before, but with 1 extra dimension.
#That extra dimensions tells the number of bytes we use for convolution
#Greyscale images only require 1 byte

#The MaxPooling2D layer is a pooling that takes the maximum value
#The 2x2 means that for every 4 pixels, take the highest value

#We then add a new set of convolution and pooling on top
#So that the network can learn yet another feature from the images
#While also further simplifying the image with pooling

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.maxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activaion='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
