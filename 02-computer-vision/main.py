import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks=myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels),(test_images, test_labels ) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

print('-------EVAL-------')
print(model.evaluate(test_images, test_labels))
print('------------------')

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
