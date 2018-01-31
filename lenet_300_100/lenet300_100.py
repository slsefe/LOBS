import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
with tf.Session() as sess:
	sess.run(init)
	weights = {
		'fc1': tf.Variable(np.load('original_parameters/w_layer1.npy').astype('float32')),
		'fc2': tf.Variable(np.load('original_parameters/w_layer2.npy').astype('float32')),
		'fc3': tf.Variable(np.load('original_parameters/w_layer3.npy').astype('float32'))
	}
	biases = {		'fc1': tf.Variable(np.load('original_parameters/b_layer1.npy').astype('float32')),
		'fc2': tf.Variable(np.load('original_parameters/b_layer2.npy').astype('float32')),
		'fc3': tf.Variable(np.load('original_parameters/b_layer3.npy').astype('float32'))
	}
	train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
	print('train_accuracy before L-OBS = ', '{:.9f}'.format(train_acc))
	validate_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
	print('validate_accuracy before L-OBS = ', '{:.9f}'.format(validate_acc))
	test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print("test_accuracy before L-OBS =", "{:.9f}".format(test_acc))