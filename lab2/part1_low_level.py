from __future__ import absolute_import, division, print_function


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = './LL_model_test/model.ckpt'

# Training Parameters
learning_rate = 0.001
num_steps = 10
batch_size = 300
display_step = 1
train_size = 60000
# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

train_images = train_images.reshape(train_images.shape[0], num_input)
test_images = test_images.reshape(test_images.shape[0], num_input)

# Convert to one hot
train_labels_onehot = np.zeros((train_labels.shape[0], num_classes))
train_labels_onehot[np.arange(train_labels.shape[0]), train_labels] = 1

test_labels_onehot = np.zeros((test_labels.shape[0], num_classes))
test_labels_onehot[np.arange(test_labels.shape[0]), test_labels] = 1

print(train_images.shape)
print(train_labels_onehot)
# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)



def conv2d(x, W, b, strides=1, pad='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=pad)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, pad='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=pad)


def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'], pad='VALID')
    conv1 = maxpool2d(conv1, k=2, pad='VALID')

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.softmax(fc3)

    return fc3



# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs0
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 3], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 3, 3], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([108, 100], stddev=0.1)),

    'wd2': tf.Variable(tf.random_normal([100, 50], stddev=0.1)),

    'wd3': tf.Variable(tf.random_normal([50, num_classes], stddev=0.1))
    # 1024 inputs, 10 outputs (class prediction)
    #'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([3], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([3], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([100], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([50], stddev=0.1)),
    'bd3': tf.Variable(tf.random_normal([10], stddev=0.1))

   # 'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

losses = []
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    saver = tf.train.Saver()
    try:
        saver.restore(sess, model)
    except:
        print('No Model Saved.\n')

    
    for step in range(num_steps):
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        #print(train_images.shape, train_images[step*batch_size:(step+1)*batch_size].shape)
        #print(train_labels_onehot[step*batch_size:(step+1)*batch_size-1,].reshape(31,10).shape)
        # batch_x = train_images[step*batch_size:(step+1)*batch_size,].reshape(batch_size,784)
        # batch_y = train_labels_onehot[step*batch_size:(step+1)*batch_size,].reshape(batch_size,10)
        i = 0
        losses_epoch = []
        while i < train_size:
            
            batch_x = train_images[i:i+batch_size]
            batch_y = train_labels_onehot[i:i+batch_size]
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                    Y: batch_y,
                                                                    keep_prob: 1.0})
                # print("Step " + str(step) + ", Minibatch Loss= " + \
                #     "{:.4f}".format(loss) + ", Training Accuracy= " + \
                #     "{:.3f}".format(acc))
            i += batch_size
            
         #   losses_epoch.append(loss)
        #losses.append(sum(losses_epoch)/len(losses_epoch))
        
        l, a = sess.run([loss_op, accuracy], feed_dict={X: test_images,
                                      Y: test_labels_onehot,
                                      keep_prob: 1.0})
        losses.append(l)

        #print("Optimization Finished!")
        
        # Calculate accuracy for 256 MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_images,
                                        Y: test_labels_onehot,
                                        keep_prob: 1.0}))
    
    save_path = saver.save(sess, model)
                                      
plt.plot(losses)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()