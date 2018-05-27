from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf

# dimensions of our images.
img_width, img_height = 150, 150

input_shape = (img_width, img_height, 3)
test_model = Sequential()
test_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))
test_model.add(Conv2D(32, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))
test_model.add(Conv2D(64, (3, 3)))
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(2, 2)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))
test_model.add(Activation('hard_sigmoid'))

test_model = load_model('model.h5')


def predict(basedir, model):
    for i in range(1,5):
        path = basedir + str(i) + '.png'
        img = load_img(path, False, target_size=(img_width, img_height))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = model.predict_classes(x)

        probs = model.predict_proba(x)

        print(probs)


basedir = "/home/jenisha/third-eye/dataset/testing/construction/"

predict(basedir, test_model)

basedir = "/home/jenisha/third-eye/dataset/testing/no_construction/"

predict(basedir, test_model)

print('done')