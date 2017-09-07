

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
	return filenames, labels

def read_images(input_list):
	data = []
	for file in input_list:
		path = os.path.join('train/',file)
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR), (img_size,img_size))
		data.append(np.array(img))

	return np.array(data).reshape(-1, img_size*img_size*3)
def input_fn(image_list,label_list):
	images = tf.convert_to_tensor(image_list, dtype= tf.string)
	labels = tf.one_hot(tf.convert_to_tensor(label_list,dtype = tf.int64) , depth = 5)

	min_after_dequeue = 500
	capacity = min_after_dequeue + 3 * batch_size
	batch_data = tf.train.batch([images,labels], batch_size=batch_size,
		capacity=min_after_dequeue + 3 * batch_size, enqueue_many=True, allow_smaller_final_batch=True)
	return batch_data[0],batch_data[1]

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_tensor = tf.reshape(features, [-1,])
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
	diabetic_classifier.train(input_fn=input_fn(image_list, label_list),steps=20000)
	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_data},y=test_labels,num_epochs=1,shuffle=False)
	eval_results = diabetic_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()
