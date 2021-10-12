import tensorflow as tf
import keras

#Class that implements the on_epoch_end function
#If loss is less than 0.4, end training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("\nLoss is low, training stops")
            self.model.stop_training = True
            
#Initiates class
callbacks = myCallback()


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#Adds extra callback parameter to fit function
model.fit(training_images, training_labels, epochs = 5, callbacks=[callbacks])
