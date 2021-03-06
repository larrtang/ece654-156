from __future__ import absolute_import, division, print_function

import requests
import gzip
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time


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



#print (test_images, _id)

#test_images = load_dataset('images_b56745b1.gz')

checkpoint_path = "./keras_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)



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


for i in range(1):
    test_images, _id = get_testset()
    print("\n----------------------------------------------------------\ndataset ID: ", _id)

    test_images = test_images / 255.0
    test_images = test_images.reshape(1000,28,28,1)

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #model.summary()
    start = time.time()

    predictions_onhot = model.predict(test_images, batch_size=1000)
    end = time.time()
    latency = end-start

    predictions = []
    for p in predictions_onhot:
        predictions.append(str(np.argmax(p)))

    predictions = ''.join(predictions)
    #print predictions
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'
    values = {'request': 'verify', 'netid':'ltang23', 'testset_id': _id, 'prediction': predictions, 'latency': latency, 'team': 'test'}
    r = requests.post(url, data=values, allow_redirects=True)
    
    acc = float(r.text)/1000

    #print()
    print('Latency:', latency)
    print("Test set accuracy (out of 1000):", r.text)

    Score= acc /(5*latency)

    print(Score)

