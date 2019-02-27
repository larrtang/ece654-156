#from __future__ import absolute_import, division, print_function

import requests
import gzip
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data

def get_testset():
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request': 'testdata', 'netid':'ltang23'}
    r = requests.post(url, data=values, allow_redirects=True)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f: 
        f.write(r.content)
    return load_dataset(filename), testset_id


test_images, _id = get_testset()
#print (test_images, _id)

#test_images = load_dataset('images_b56745b1.gz')

checkpoint_path = "./keras_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

test_images = test_images / 255.0
test_images = test_images.reshape(1000,28,28,1)

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

predictions_onhot = model.predict(test_images, batch_size=1000)

predictions = []
for p in predictions_onhot:
    predictions.append(str(np.argmax(p)))

predictions = ''.join(predictions)
print predictions
url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
values = {'request': 'verify', 'netid':'ltang23', 'testset_id': _id, 'prediction': predictions}
r = requests.post(url, data=values, allow_redirects=True)


print()
print(r.text)

