# import statements
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# dimensions of our images.
img_width, img_height = 150, 150

# you need to change this to find the training and validation datasets for your
#train_data_dir='/Users/ShereenElaidi/Desktop/University/AI4SocialGood/Construct-I/dataset/training'
#validation_data_dir = '/Users/ShereenElaidi/Desktop/University/AI4SocialGood/Construct-I//dataset/validation'
#test_data_dir = '/Users/ShereenElaidi/Desktop/University/AI4SocialGood/Construct-I/dataset/test'


train_data_dir='/home/jenisha/third-eye/dataset/training'
validation_data_dir = '/home/jenisha/third-eye/dataset/validation'
test_data_dir = '/home/jenisha/third-eye/dataset/test'

# the sizes of the samples
nb_train_samples = 83
nb_validation_samples = 40
nb_test_samples = 14

epochs = 10
batch_size = 5

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    print("img data format channels_first")
else:
    input_shape = (img_width, img_height, 3)


def model_def():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('hard_sigmoid'))
	return model

model = model_def()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('first_model.h5')
model.save_weights('first_weights.h5')


