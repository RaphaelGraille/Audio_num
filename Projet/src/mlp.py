'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
import os
import sys

TRAIN_BINARY_FILE_PATH = "data/binary_train_data/"

SAVE_MODEL_PATH ="models/mlp.ckpt"
LOAD_MODEL = False

USE_ALL_TRAIN_DATA = True

# Parameters
learning_rate = 0.001
training_epochs = 500
#batch_size = 100
display_step = 1
save_step = 5

total_batch = 5

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 256
n_hidden_4 = 256
n_input = 129 # MNIST data input (img shape: 28*28)
mask_size = 129 # MNIST total classes (0-9 digits)

binary_file_used = ["train_data_and_labels_1.npy","train_data_and_labels_2.npy","train_data_and_labels_3.npy","train_data_and_labels_4.npy","train_data_and_labels_5.npy"]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, mask_size])


def iterate_binary_files():
    for binary_file in os.listdir(TRAIN_BINARY_FILE_PATH):
        if binary_file in binary_file_used || USE_ALL_TRAIN_DATA:
            print("loading binary file...")
            data_and_labels = np.load(TRAIN_BINARY_FILE_PATH+binary_file)
            data = data_and_labels[0]
            labels = data_and_labels[1]
            print("done!")
            yield data, labels, data.shape[0]


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.sigmoid(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    out_layer = tf.sigmoid(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, mask_size]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b2': tf.Variable(tf.random_normal([n_hidden_3])),
    'b2': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([mask_size]))
}

# Construct model
print("Building model ...")
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(tf.multiply(pred,x),y)),1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    if LOAD_MODEL:
        print("loading model...")
        saver.restore(sess, SAVE_MODEL_PATH)
        print("Done !")
    # Training cycle
    print("Start training ...")
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        #for i in range(total_batch):
        for batch_x, batch_y, batch_size in iterate_binary_files():
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
        if epoch % save_step == 0:
            save_path = saver.save(sess, SAVE_MODEL_PATH)
            print("Model saved in file: %s" % save_path)
    print("Optimization Finished!")

    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))