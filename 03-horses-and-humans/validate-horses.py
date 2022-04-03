import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = tf.keras.models.load_model('humans-and-horses.h5')

#predicting images
path = os.path.join(os.getcwd(), 'images/training/horses/')

directories = os.listdir(path)

humans = 0
horses = 0

for file in directories:
    img = image.load_img(path + file, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images)

    if classes[0] > 0.5:
        humans += 1
    else:
        horses += 1

print('Found ' + str(humans) + ' humans and ' + str(horses) + ' horses.')
