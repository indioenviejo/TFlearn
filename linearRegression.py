


def model(X,w,b):
    return (tf.mul(X,w)+b)

if __name__ == '__main__':

	import tensorflow as tf
	import numpy as np
	import os
	import pandas as pda
	import time

	learningRate = 0.0005
	numberOfIterations = 100

	# Generate Linear Data

	xVals = np.linspace(-20, 20, 1001)
	yVals = 6 * xVals + 50 + np.random.randn(*xVals.shape) * 10.33 # create a y value which is approximately linear but with some random noise

	#Get the maximum value of the x-corodinates
	maXval = np.max(xVals)
	minXval = np.min(xVals)
	plotXVals = list(np.linspace(minXval,maXval,100))
	plotYVals = [np.random.rand(1)[0]*i for i in plotXVals]

	X = tf.placeholder("float")
	Y = tf.placeholder("float")

	# w is a variable . When you train a model, you use variables to hold and update parameters. 
	# When you create a Variable you pass a Tensor as its initial value to the Variable() constructor
	# Here, the values of w,b is initialized to 0.0
	w = tf.Variable(0.0,name = "modelWeights")
	b = tf.Variable(0.0,name = "bias")
	
	# The model output is the linear model as defined by the function model
	modelOutput = model(X,w,b)

	# The least squares function that needs to be minimized
	costFunction = (tf.pow((Y-modelOutput),2))

	# Define the optimizer
	optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(costFunction)

	# Tensor Flow common commands
	# Declare Session
	session = tf.Session()

	# Initialize all the variables & start session
	init = tf.initialize_all_variables()
	session.run(init)

	# Bokeh initilizations 
	from bokeh.plotting import figure, output_server, cursession, show
	output_server("LinearRegression")

	#Create the figure
	fig = figure(title = "Gradient Descent Iterations")
	# Command that actually plots the points
	fig.scatter(xVals,yVals, marker="circle", fill_color="#ee6666", fill_alpha=0.5, size=12)
	
	# This is the linear regression line that gets updated for each iteration
	fig.line(plotXVals,plotYVals,name = "currLRLine",line_width=6)
	show(fig)
	
	# Bokeh server initializations 
	renderer = fig.select(dict(name="currLRLine"))
	ds = renderer[0].data_source
	cursession().store_objects(ds)

	for i in range(numberOfIterations):
		for x,y in zip(xVals,yVals):
			session.run(optimizer,feed_dict={X:x,Y:y})
			wVal = session.run(w)
			bVal = session.run(b)
			plotYVals = [wVal*i+bVal for i in plotXVals]
		# The y value for the line in the bokeh plot needs to be updated 
		# in every iteration and stored in the datasource		
		ds.data["y"] =plotYVals
		cursession().store_objects(ds)
	session.close()
