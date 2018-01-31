import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 100  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
alpha = 1e6  # Hession矩阵的初始化值，取值为[1e4, 1e8]
learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1
display_batch_step = 10

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def produce_layer_input(x, weights, biases):
	"""
	:param x:
	:param weights:
	:param biases:
	:return:
	"""
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['fc1']), biases['fc1'])
	# x:100x784, weights['fc1']:784x300, biases['fc1']:300x1, layer_1:100x300
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['fc2']), biases['fc2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	# out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return x, layer_1, layer_2


# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['fc1']), biases['fc1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['fc2']), biases['fc2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['fc3']) + biases['fc3']
	return out_layer


def generate_hessian_inverse_fc(hessian_inverse_path, w_layer_path, layer_input_train_dir):
	"""
	This function calculate hessian inverse for a fully-connect layer
	:param hessian_inverse_path: the path store generated hessian inverse
	:param w_layer_path: the layer weights
	:param layer_input_train_dir: layer inputs
	:return:
	"""

	w_layer = np.load(w_layer_path)
	# (784, 300), (300, 100), (100, 10)
	n_hidden_1 = w_layer.shape[0]
	# 784, 300, 100

	# Here we use a recursive way to calculate hessian inverse
	hessian_inverse = alpha * np.eye(n_hidden_1 + 1)
	# (785, 785), (301, 301), (101, 101)

	for input_index, input_file in enumerate(os.listdir(layer_input_train_dir)):
		# 对每一个batch进行操作
		# os.listdir(path)返回一个在给定路径文件夹下所有条目的列表
		# enumrate()用于计数
		# input_index = 0,1,2,3,...,549
		# input_file = batch_0.npy,batch_1.npy,...,batch_549.npy
		layer2_input_train = np.load(layer_input_train_dir + '/' + input_file)
		# 获取每一个batch的层输入fc1:(100,784), fc2:(100,300), fc3:(100,100)
		# 每一行代表一个样本
		# if input_index == 0:
		dataset_size = layer2_input_train.shape[0] * len(os.listdir(layer_input_train_dir))
			# data_size = 100x550 = 55000 = train_data.
		for i in range(layer2_input_train.shape[0]):
			# i = 0,1,2,...,99
			vect_w_b = np.vstack((np.array([layer2_input_train[i]]).T, np.array([[1.0]])))
			# (785, 1) = np.vstack((1, 785).T, (1, 1))
			denominator = dataset_size + np.dot(np.dot(vect_w_b.T, hessian_inverse), vect_w_b)
			# denominator = 55000 + (1,785)*(785,785)*(785,1)
			numerator = np.dot(np.dot(hessian_inverse, vect_w_b), np.dot(vect_w_b.T, hessian_inverse))
			# (785,785) = ((785,785)*(785,1))*((1,785)*(785,785))
			hessian_inverse = hessian_inverse - numerator * (1.00 / denominator)
			# (785,785) = (785,785) - (785,785)/(1,1)

		if input_index % 100 == 0 or input_index +1 ==dataset_size/batch_size:
			# 每隔100次或者结束时打印Hession矩阵逆的batch的计算进度
			print ('[%s] Finish processing batch %s' % (datetime.now(), input_index))

	if not os.path.exists(hessian_inverse_path):
		os.makedirs(hessian_inverse_path)

	np.save(hessian_inverse_path, hessian_inverse)
	print ('[%s]Hessian Inverse Done!' % (datetime.now()))


def edge_cut(hessian_inverse_path, w_layer_path, b_layer_path, prune_save_path, cut_ratio):
	"""
	This function prune weights of biases based on given hessian inverse and cut ratio
	:param hessian_inverse_path:
	:param w_layer_path:
	:param b_layer_path:
	:param prune_save_path:
	:param cut_ratio: The zeros percentage of weights and biases, or, 1 - compression ratio
	:return:
	"""

	# dataset_size = layer2_input_train.shape[0]
	w_layer = np.load(w_layer_path)
	# (784,300), (300,100), (100,10)
	b_layer = np.load(b_layer_path)
	# (300, ), (100, ), (10, )
	n_hidden_1 = w_layer.shape[0]
	# 784, 300, 100
	n_hidden_2 = w_layer.shape[1]
	# 300, 100, 10

	sensitivity = np.array([])

	hessian_inverse = np.load(hessian_inverse_path)
	# (785, 785), (301, 301), (101, 101)
	print ('[%s] Hessian Inverse Done!' %datetime.now())

	gate_w = np.ones([n_hidden_1, n_hidden_2])
	# (784,300), (300,100), (100,10)
	gate_b = np.ones([n_hidden_2])
	# (300, ), (100, ), (10, )

	max_pruned_num = int(n_hidden_1 * n_hidden_2 * cut_ratio)
	print ('[%s] Max prune number : %d' % (datetime.now(), max_pruned_num))

	# Calcuate sensitivity score. Refer to Eq.6.
	for i in range(n_hidden_2):
		# i = 0,1,2,...300 or 100 or 10
		sensitivity = np.hstack((sensitivity, 0.5 * (np.hstack((w_layer.T[i], b_layer[i])) ** 2) / np.diag(hessian_inverse)))
	sorted_index = np.argsort(sensitivity)
	# 返回从小到大排序后的对应索引

	print ('[%s] Sorted index generate completed.' %datetime.now())
	print ('[%s] Starting Pruning!' %datetime.now())
	hessian_inverseT = hessian_inverse.T


	prune_count = 0
	for i in range(n_hidden_1 * n_hidden_2):
		prune_index = [sorted_index[i]]
		x_index = int(prune_index[0] / (n_hidden_1 + 1))  # next layer num
		y_index = int(prune_index[0] % (n_hidden_1 + 1))  # this layer num
		# 要修剪的权重对应的坐标

		if y_index == n_hidden_1:  # b
			if gate_b[x_index] == 1:
				delta_w = (-b_layer[x_index] / hessian_inverse[y_index][y_index]) * hessian_inverseT[y_index]
				gate_b[x_index] = 0
				prune_count += 1
				# Parameters update, refer to Eq.5
				w_layer.T[x_index] = w_layer.T[x_index] + delta_w[0:-1]
				b_layer[x_index] = b_layer[x_index] + delta_w[-1]
		else:
			if gate_w[y_index][x_index] == 1:
				delta_w = (-w_layer[y_index][x_index] / hessian_inverse[y_index][y_index]) * hessian_inverseT[y_index]
				gate_w[y_index][x_index] = 0
				prune_count += 1
				# Parameters update, refer to Eq.5
				w_layer.T[x_index] = w_layer.T[x_index] + delta_w[0:-1]
				b_layer[x_index] = b_layer[x_index] + delta_w[-1]

		w_layer = w_layer * gate_w
		b_layer = b_layer * gate_b

		if prune_count == max_pruned_num:
			print ('[%s] Have prune required weights' %datetime.now())
			break
	print ('[%s] Prune Finish. compression ratio: %.3f' %(datetime.now(), 1 - float(np.count_nonzero(w_layer)) / w_layer.size))

	if not os.path.exists(prune_save_path):
		os.makedirs(prune_save_path)

	np.save("%s/weights" %prune_save_path, w_layer)
	np.save("%s/biases" % prune_save_path, b_layer)


# Launch the graph
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
# tf weights and biases
# 预训练模型的权重和偏置
weights = {
	'fc1': tf.Variable(np.load('original_parameters/w_layer1.npy').astype('float32')),
	'fc2': tf.Variable(np.load('original_parameters/w_layer2.npy').astype('float32')),
	'fc3': tf.Variable(np.load('original_parameters/w_layer3.npy').astype('float32'))
}
biases = {
	'fc1': tf.Variable(np.load('original_parameters/b_layer1.npy').astype('float32')),
	'fc2': tf.Variable(np.load('original_parameters/b_layer2.npy').astype('float32')),
	'fc3': tf.Variable(np.load('original_parameters/b_layer3.npy').astype('float32'))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
layer_outputs = produce_layer_input(x, weights, biases)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()



# Step 1: Produce layer inputs
with tf.Session() as sess:
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)

	if not os.path.exists('layer_inputs/fc1/'):
		os.makedirs('layer_inputs/fc1/')
	if not os.path.exists('layer_inputs/fc2/'):
		os.makedirs('layer_inputs/fc2/')
	if not os.path.exists('layer_inputs/fc3/'):
		os.makedirs('layer_inputs/fc3/')

	for batch_index in range(total_batch):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		layer_1_inputs, layer_2_inputs, layer_3_inputs = sess.run(layer_outputs, feed_dict={x: batch_x, y: batch_y})

		np.save('layer_inputs/fc1/batch_%d' % batch_index, layer_1_inputs)
		np.save('layer_inputs/fc2/batch_%d' % batch_index, layer_2_inputs)
		np.save('layer_inputs/fc3/batch_%d' % batch_index, layer_3_inputs)

print ('[%s] Layer inputs generate finish.' %datetime.now())

# Step 2: Generate Hessian inverse
generate_hessian_inverse_fc(hessian_inverse_path='hessian_inverse/fc1',
							w_layer_path = 'original_parameters/w_layer1.npy',
							layer_input_train_dir = 'layer_inputs/fc1')

generate_hessian_inverse_fc(hessian_inverse_path='hessian_inverse/fc2',
							w_layer_path = 'original_parameters/w_layer2.npy',
							layer_input_train_dir = 'layer_inputs/fc2')

generate_hessian_inverse_fc(hessian_inverse_path='hessian_inverse/fc3',
						w_layer_path = 'original_parameters/w_layer3.npy',
							layer_input_train_dir = 'layer_inputs/fc3')

# Step 3: Edge cut
edge_cut(hessian_inverse_path='hessian_inverse/fc1.npy',
		 w_layer_path='original_parameters/w_layer1.npy',
		 b_layer_path='original_parameters/b_layer1.npy',
		 prune_save_path = 'pruned_parameters/fc1',
		 cut_ratio=0.9)

edge_cut(hessian_inverse_path='hessian_inverse/fc2.npy',
		 w_layer_path='original_parameters/w_layer2.npy',
		 b_layer_path='original_parameters/b_layer2.npy',
 		 prune_save_path = 'pruned_parameters/fc2',
		 cut_ratio=0.7)

edge_cut(hessian_inverse_path='hessian_inverse/fc3.npy',
		 w_layer_path='original_parameters/w_layer3.npy',
		 b_layer_path='original_parameters/b_layer3.npy',
		 prune_save_path = 'pruned_parameters/fc3',
		 cut_ratio=0.93)

# Step 4: test accuracy
# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	# Training cycle
	# Test the raw accuracy
	weights = {
		'fc1': tf.Variable(np.load('pruned_parameters/fc1/weights.npy').astype('float32')),
		'fc2': tf.Variable(np.load('pruned_parameters/fc2/weights.npy').astype('float32')),
		'fc3': tf.Variable(np.load('pruned_parameters/fc3/weights.npy').astype('float32'))
	}
	biases = {
		'fc1': tf.Variable(np.load('pruned_parameters/fc1/biases.npy').astype('float32')),
		'fc2': tf.Variable(np.load('pruned_parameters/fc1/biases.npy').astype('float32')),
		'fc3': tf.Variable(np.load('pruned_parameters/fc1/biases.npy').astype('float32'))
	}
	train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
	print('train_accuracy after L-OBS = ', '{:.9f}'.format(train_acc))
	validate_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
	print('validate_accuracy after L-OBS = ', '{:.9f}'.format(validate_acc))
	test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
	print("test_accuracy after L-OBS =", "{:.9f}".format(test_acc))