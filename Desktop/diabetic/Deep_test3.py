from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pandas as pd
import numpy as np 
import cv2
import os

sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, 784]) 
y_ = tf.placeholder(tf.float32, shape=[None, 5]) 

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1,200, 200,3])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(inputs=input_layer,filters=32, kernel_size = [4,4] , strides = 2 , padding="same" , activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pool2d(inputs=conv1 , pool_size=[2,2], strides=2,padding = 'valid')
	dropout1 = tf.layers.dropout(inputs=pool1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(inputs=dropout1,ffilters=64, kernel_size = [2,2] , strides = 2 , padding="same" , activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2], strides=2,padding = 'same')
	dropout2 = tf.layers.dropout(inputs=pool2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

	conv3 = tf.layers.conv2d(inputs=dropout2,ffilters=128, kernel_size = [2,2] , strides = 2 , padding="same" , activation=tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2], strides=2,padding = 'same')
	# Dense Layer
	pool3_flat = tf.reshape(pool3, [-1, 5 * 5 * 128])
	dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
	dropout3 = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout3, units=5)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def preprocess_image(img):
	return tf.image.resize_images(img,[28,28])

def read_labeled_image_list(image_list_file):

	df = pd.read_csv(image_list_file)
	
	filenames = []
	labels = []
	for index in range(len(df)):
		filenames.append(df.iloc[index][0]+'.jpeg')
		labels.append(np.int32(df.iloc[index][1]))
	return filenames, labels

def read_images(input_list):
	img_size = 28
	data = []
	for file in input_list:
		path = os.path.join('train/',file)
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (img_size,img_size))
		data.append(np.array(img))
	return np.array(data).astype(np.float32).reshape(-1, img_size*img_size)
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

image_list, label_list = read_labeled_image_list('train.csv')
images = tf.convert_to_tensor(image_list, dtype= tf.string)
labels = tf.one_hot(tf.convert_to_tensor(label_list,dtype = tf.int64) , depth = 5)
batch_size = 20
min_after_dequeue = 100
capacity = min_after_dequeue + 3 * batch_size
batch_data = tf.train.batch([images,labels], batch_size=batch_size,
		capacity=min_after_dequeue + 3 * batch_size, enqueue_many=True, allow_smaller_final_batch=True)


W_conv1 = weight_variable([5, 5, 1, 32]) # 32 feature 5X5 patch 1 channels
b_conv1 = bias_variable([32]) #output channels
x_image = tf.reshape(x, [-1, 28, 28, 1]) #layer size
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for i in range(20000):
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		batch = sess.run(batch_data)
		if i % 100 == 0:
	  		train_accuracy = accuracy.eval(feed_dict={x: read_images(batch[0]), y_: batch[1], keep_prob: 1.0})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		train_step.run(feed_dict={x: read_images(batch[0]), y_: batch[1], keep_prob: 0.5})

	#print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	coord.request_stop()
	coord.join(threads)


