import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

print(tf.__version__)

# this doesnt work, even the original example has the same issue
# https://github.com/lmoroney/tfbook/blob/master/chapter2/fashion-cnn.py

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    # for some reason the the accuracy looks like 0.0995 instead of the expected 0.95
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks=myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels),(test_images, test_labels ) = mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('-------SUMMARY-------')
print(model.summary())
print('---------------------')

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])


print('-------EVAL-------')
print(model.evaluate(test_images, test_labels))
print('------------------')

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
