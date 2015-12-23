

import tensorflow as tf
import os


if __name__ == '__main__':


	def trainStepLR(eta,cost):
		train_step = tf.train.GradientDescentOptimizer(eta).minimize(cost)
		return train_step

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
	W2 = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	b2 = tf.Variable(tf.zeros([10]))
	
	# Initialize all the variables & start session
	init = tf.initialize_all_variables()
	session.run(init)

	# Model Outputs
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	y2 = tf.nn.softmax(tf.matmul(x,W2) + b2)		
	
	# Cost Function

	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	MSE = (tf.pow((y_-y2),2))  

	# Bokeh initilizations 
	from bokeh.plotting import figure, output_server, cursession, show,ColumnDataSource
	output_server("Softmax Regression")

	#Create the figure
	fig = figure(title = "Softmax Classifier Accuracy")

	# Cost functions with Learning rate
	trainStep_a = trainStepLR(0.01,cross_entropy)
	trainStep_b = trainStepLR(0.01,MSE)

	# 1000 iterations
	numberOfIterations = 1000
	
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	correct_prediction2 = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

	dummyIteration = [0]
	dummyAccuracyVals = [0]
	
	# This is the linear regression line that gets updated for each iteration
	fig.line(dummyIteration,dummyAccuracyVals,name = "accuracy-a",line_width=3,line_color='#3288bd',legend="Cost - Entropy")
	fig.line(dummyIteration,dummyAccuracyVals,name = "accuracy-b",line_width=3,line_color='#ff5706',legend="Cost - MSE")

	show(fig)

	# Bokeh server initializations 
	renderer_a  = fig.select(dict(name="accuracy-a"))
	renderer_b  = fig.select(dict(name="accuracy-b"))
	
	ds = renderer_a[0].data_source
	ds_b = renderer_b[0].data_source

	cursession().store_objects(ds)
	cursession().store_objects(ds_b)
	
	xVals = [0]
	yVals_a = [0]
	yVals_b = [0]

	for i in range(500):
		xVals.extend([i])	
		batch = mnist.train.next_batch(50)
		session.run(trainStep_a,feed_dict={x: batch[0], y_: batch[1]})
		accuracyValue = session.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
		yVals_a.extend([accuracyValue])
		ds.data["x"] = xVals
		ds.data["y"] = yVals_a		
		cursession().store_objects(ds)
		session.run(trainStep_b,feed_dict={x: batch[0], y_: batch[1]})
		accuracyValue2 = session.run(accuracy2,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
		yVals_b.extend([accuracyValue2])
		ds_b.data["x"] = xVals
		ds_b.data["y"] = yVals_b
		cursession().store_objects(ds_b)
	session.close()
	   







