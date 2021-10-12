#Acceptable loss
x = 0.1

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        #logs.get('accuracy') >= 0.6
        if(logs.get('loss')< x):
            print("\nLoss is low, training stops")
            self.model.stop_training = True
