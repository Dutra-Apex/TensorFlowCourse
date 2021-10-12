# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class myCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(float(logs.get('accuracy')) > 0.99):
                print("Reached 99% accuracy so cancelling training!")
                self.model.stop_training = True
                

    callbacks = myCallBack()
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    x_train = x_train / 255.0
    y_train = y_train / 255.0

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(x_train, y_train, epochs = 15, callbacks=[callbacks]
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
