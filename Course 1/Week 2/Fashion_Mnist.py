import tensorflow as tf
import keras

#Import a lis of images of clothing from fashion_mnist
#The images are 28x28 and have already been greyscaled
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Builds a model with 3 layers:

#The 1st layer takes 28 x 28 inputs, with is the number of pixels in each image
#Flatten is used to turn the input into a linear array

#The 2nd layer is a Dense layer with 128 neurons

#The 3rd layer outputs 10 neurons, as there are 10 different types of clothing

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])
