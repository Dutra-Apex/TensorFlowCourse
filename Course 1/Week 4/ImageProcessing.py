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
