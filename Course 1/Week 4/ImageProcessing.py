from tensorflow.keras.preprocessing.image \
    import ImageDataGenerator

#Instantiates an Image generator with rescaling casting
train_datagen = ImageDataGenerator(rescale=1./255)

#Points the generator to the directory where subdirectories
#are located as to create the labels
train_generator = train_datagen.flow_from_directory(
    #Directory where you point the generator
    train_dir,  
    #Resizes images
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

#Creates the model
#Note that the input shape for the convolution is 300x300x3,
#The three at the end represents 3 bytes per pixel, RGB.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    #The sigmoid function is great for binary classification, and only requires 1 neuron
    #It's still possible to use foftmax with 2 neurons, but sigmoid is more efficient
    tf.keras.layers.Dense(1, activation='sigmoid')
])
