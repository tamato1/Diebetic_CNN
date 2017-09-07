
import tensorflow as tf
import pandas as pd
import numpy as np 
import cv2
import os
from datetime import datetime


img_size = 200
x = tf.placeholder(tf.float32, shape=[None, img_size*img_size*3]) 
y_ = tf.placeholder(tf.float32, shape=[None, 5])

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

image_list, label_list = read_labeled_image_list('train.csv')
images = tf.convert_to_tensor(image_list, dtype= tf.string)
labels = tf.one_hot(tf.convert_to_tensor(label_list,dtype = tf.int64) , depth = 5)
batch_size = 20
min_after_dequeue = 500
capacity = min_after_dequeue + 3 * batch_size
batch_data = tf.train.batch([images,labels], batch_size=batch_size,
		capacity=min_after_dequeue + 3 * batch_size, enqueue_many=True, allow_smaller_final_batch=True)


"""____________________________________CNN LAYER ___________________________________________"""
keep_prob = tf.placeholder(tf.float32)
input_img = tf.reshape(x, [-1, img_size, img_size, 3])#input shape [ batch,w , h , channel]


w1 = tf.Variable(tf.truncated_normal([2, 2 , 3, 32], stddev=0.1)) #32 feature 4X4 patch input 1 channels
b1 = tf.Variable(tf.constant(0.1, shape=[32])) #output channels

conv1 = tf.nn.conv2d(input_img, w1, strides=[1,2,2,1] , padding='SAME')#[32,100,100] 
relu1 = tf.nn.relu(conv1 + b1) # Conv2d = conv2d(layersize , weight) +bias
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1] , strides=[1, 2, 2, 1], padding='VALID')#[32,50,50]

w2 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.1))# 32 feature 5X5 patch input 32 channels
b2 = tf.Variable(tf.constant(0.1, shape=[64])) #output channels
conv2 = tf.nn.conv2d(pool1, w2,strides=[1, 2, 2, 1], padding='SAME')#[64,26,26] 
relu2 = tf.nn.relu(conv2 + b2)
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')#[64,14.0,14.0] 
drop2 = tf.nn.dropout(pool2,0.4 if keep_prob!= 1.0 else keep_prob) #Dropout

w3 = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.1))# 32 feature 5X5 patch input 32 channels
b3 = tf.Variable(tf.constant(0.1, shape=[128])) #output channels
conv3 = tf.nn.conv2d(drop2, w3,strides=[1, 2, 2, 1], padding='SAME')#[128,8,8]
relu3 = tf.nn.relu(conv3 + b3)
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')#[128,5,5]

w_fc1 = tf.Variable(tf.truncated_normal([4 * 4 * 128, 640], stddev=0.1))# 32 feature 5X5 patch 1 channels
b_fc1 = tf.Variable(tf.constant(0.1, shape=[640]))#output channels
pool2_flat = tf.reshape(pool3, [-1 , 4*4*128]) #Dense
relu4 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1)# Fully connected
drop4 = tf.nn.dropout(relu4, 0.3 if keep_prob!= 1.0 else keep_prob) #Dropout

w_fc2 = tf.Variable(tf.truncated_normal([640, 5], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[5]))

y_conv = tf.matmul(drop4, w_fc2) + b_fc2 #result

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = train_step.minimize(loss=cross_entropy,global_step=tf.train.get_global_step())
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(20000):
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		batch = sess.run(batch_data)
		if i % 50 == 0:
	  		train_accuracy = accuracy.eval(feed_dict={x: read_images(batch[0]), y_: batch[1], keep_prob: 1.0})
			print(str(datetime.now())+'step %d, training accuracy %g' % (i, train_accuracy))
		train_op.run(feed_dict={x: read_images(batch[0]), y_: batch[1], keep_prob: 0.5})
#	test_data ,  test_label	= load_test()
#	print('test accuracy %g' % accuracy.eval(feed_dict={x: test_data , y_: test_label, keep_prob: 1.0}))
	coord.request_stop()
	coord.join(threads)


