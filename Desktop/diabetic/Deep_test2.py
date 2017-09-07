

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd 
import os 
import cv2
import collections
from tensorflow.python.estimator.inputs.queues import feeding_functions
_TARGET_KEY = '__target_key__'
tf.logging.set_verbosity(tf.logging.INFO)
img_size = 200
batch_size = 20
def print_fn(inputs):
	print "------------>>"+inputs
	return inputs

def load_test():
	np_input_test = np.load('test_data.npy')
	test_data =np.array([i[0] for i in np_input_test]).astype(np.float32).reshape(-1, img_size,img_size,3)
	test_label = np.asarray([i[1] for i in np_input_test],dtype = np.float32)
	return test_data,test_label
def read_labeled_image_list(image_list_file):

	df = pd.read_csv(image_list_file)
	
	filenames = []
	labels = []
	for index in range(len(df)):
		filenames.append(df.iloc[index][0]+'.jpeg')
		labels.append(np.int32(df.iloc[index][1]))
	return np.array(filenames), np.array(labels)

def read_images(input_list):
	data = []
	for file in input_list:
		print file
		path = os.path.join('train/',file)
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR), (img_size,img_size))
		data.append(np.array(img))

	return data

def _get_unique_target_key(features):

	target_key = _TARGET_KEY
	while target_key in features:
		target_key += '_n'
	return target_key


def numpy_input_fn(x,
					y=None,
					batch_size=128,
					num_epochs=1,
					shuffle=None,
					queue_capacity=1000,
					num_threads=1):

	def input_fn():
		ordered_dict_x = collections.OrderedDict(
				sorted(x.items(), key=lambda t: t[0]))

		unique_target_key = _get_unique_target_key(ordered_dict_x)
		if y is not None:
			ordered_dict_x[unique_target_key] = y

		if len(set(v.shape[0] for v in ordered_dict_x.values())) != 1:
			shape_dict_of_x = {k: ordered_dict_x[k].shape
												 for k in ordered_dict_x.keys()}
			shape_of_y = None if y is None else y.shape
			raise ValueError('Length of tensors in x and y is mismatched. All '
											 'elements in x and y must have the same length.\n'
											 'Shapes in x: {}\n'
											 'Shape for y: {}\n'.format(shape_dict_of_x, shape_of_y))

		print ordered_dict_x
		queue = feeding_functions._enqueue_data(	# pylint: disable=protected-access
				ordered_dict_x,
				queue_capacity,
				shuffle=shuffle,
				num_threads=num_threads,
				enqueue_size=batch_size,
				num_epochs=num_epochs)

		features = (queue.dequeue_many(batch_size) if num_epochs is None
								else queue.dequeue_up_to(batch_size))
		if len(features) > 0:
			features.pop(0)

		features = dict(zip(ordered_dict_x.keys(), features))
		if y is not None:
			target = features.pop(unique_target_key)
			return read_images(features), target
		return read_images(features)

	return input_fn

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_tensor = tf.reshape(features["x"], [-1])
	print input_tensor
	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(inputs=input_tensor,filters=32, kernel_size = [4,4] , strides = 2 , padding="same" , activation=tf.nn.relu)

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
	pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])
	dense = tf.layers.dense(inputs=pool3_flat, units=640, activation=tf.nn.relu)
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


def main(unused_argv):
	# Load training and eval data
	image_list, label_list = read_labeled_image_list('train.csv')
	# Create the Estimator
	diabetic_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	# Train the model
	train_input_fn = numpy_input_fn(x={"x": image_list},y=label_list,batch_size= batch_size,num_epochs=None,shuffle=True)
	diabetic_classifier.train(input_fn=train_input_fn,steps=20000)
	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_data},y=test_labels,num_epochs=1,shuffle=False)
	eval_results = diabetic_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()
