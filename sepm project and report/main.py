import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow_core.python.keras.engine.sequential import Sequential
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.core import Activation, Flatten, Dense, Dropout
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import json

print("test")

# See if we can use the GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

data_dir = ""

while not os.path.isdir(data_dir):
    data_dir = 'images'

model_name_weights = "model.h5"
model_name_json = "model.json"

# For picking up the label during prediction
label_map = dict()
for dir in os.listdir(data_dir):
    label_map[len(label_map)] = dir

# Sanity check
if os.path.exists(data_dir):
    print("Found training data")
else:
    print("Could not find training data")
    exit(1)


# Build the CNN
def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    training_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    training_generator = training_data_generator.flow_from_directory(
        data_dir,
        target_size=(300, 300),
        batch_size=5,
        class_mode="categorical")

    model.fit_generator(training_generator,	epochs=3)

    return model


model = create_model()

model.save("testing")
