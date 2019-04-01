from __future__ import absolute_import, division, print_function

import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

checkpoint_path = "./keras_model_test/cp.ckpt"
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
    #keras.layers.BatchNormalization(axis=0, momentum=0.99, epsilon=0.001, beta_initializer='zeros', gamma_initializer='ones'),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.DepthwiseConv2D(filters=3, kernel_size=(3,3), stride=(1,1), depth_multiplier=1, activation=tf.nn.relu, padding='same'),
    keras.layers.Conv2D(filters=3, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu, padding='valid'),
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

start = time.time()

#test_loss, test_acc = model.evaluate(test_images, test_labels)
prediction = model.predict(data)     
rst = ''.join(map(str, np.argmax(prediction,axis=1).tolist()))
end = time.time()

print ("\nInference Exe. time: ", end-start, "Sec.")
print('Test accuracy:', test_acc)
print()

#model.save_weights(checkpoint_path)