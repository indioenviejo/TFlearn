

import tensorflow as tf
import os

# Change to the tflearn directory
os.chdir('/home/raghav/mygithub/TensorFlow/')

# File provided by TensorFlow documentation to download mnist data easily
import input_data_mnist

mnist = input_data_mnist.read_data_sets('MNIST_data', one_hot=True)


# Tensor Flow common commands
# Declare Session
session = tf.Session()


# Input Values
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

# Variables to be determined
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# Initialize all the variables & start session
init = tf.initialize_all_variables()
session.run(init)

# Model Output
y = tf.nn.softmax(tf.matmul(x,W) + b)

# Cost Function

cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# Training with learning rate or step length of 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# 1000 iterations
numberOfIterations = 1000
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(numberOfIterations):
	batch = mnist.train.next_batch(50)
	session.run(train_step,feed_dict={x: batch[0], y_: batch[1]})
	accuracyValue = session.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
	print i,accuracyValue
session.close()
   

