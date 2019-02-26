from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

checkpoint_path = "./keras_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)

print(train_images.shape)
print(train_images.shape[0], train_images.shape[1:4])
tf.reset_default_graph()


model = keras.Sequential([
    keras.layers.Conv2D(filters=3, kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu, data_format='channels_last', padding='valid', input_shape= (28,28,1)),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), activation=tf.nn.relu, padding='same'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

try:
    model.load_weights(checkpoint_path)
except:
    print("No model saved.\n")

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

num_epochs = 5
hist = model.fit(train_images, train_labels, epochs=num_epochs)
print(hist.history['loss'])
plt.plot(hist.history['loss'])
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)
print()
model.save_weights(checkpoint_path)