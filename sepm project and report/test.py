import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.preprocessing import image
import numpy as np

CATEGORIES = ['apple', 'baby_back_ribs']

model = tf.keras.models.load_model('testing')


# Prediction loop
select = True
count = 0
while select:
    try:
        image_path = input("Enter full path to an image file: ")

        test_image = image.load_img(image_path, target_size=(300, 300))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image, batch_size=1)
        index = int(result.round())
        print("Prediction: {} {}".format(CATEGORIES[index], result))
    except:
        print("Try again")
        if count>3:
            select = False
        else:
            count= count + 1