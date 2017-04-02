import tensorflow as tf
import numpy as np 
from layers import *




class ConvolutionalNN(Layers):

	def __init__(self, batch_size, lr, input_size, label_size, phase, num_classes, keep_prob):
		self.batch_size = batch_size
		self.lr = lr
		input_shape = (None,) + tuple(input_size)
		output_shape = (None,) + tuple(label_size)
		self.input = tf.placeholder(tf.float32, shape=input_shape, name='Input')
		self.labels = tf.placeholder(tf.float32, shape=output_shape, name = 'Output')
		self.phase = tf.placeholder(tf.bool, name='phase')
		self.num_classes = num_classes
		self.keep_prob = keep_prob


	def build_model(self):
		filter1 = [5,5,1,16]
		stride1 = [1, 1, 1,1]

		conv_layer1 = conv2d_layer(self.input,filter1, stride1,'SAME','conv_layer1','relu1', self.phase )

		filter2 = [5,5,16,16]
		stride2 = [1, 1, 1,1]

		conv_layer2 = conv2d_layer(conv_layer1,filter2, stride2,'SAME','conv_layer2','relu2', self.phase )

		max_pool1 = tf.nn.max_pool(conv_layer2,ksize=[1,3,3,1],stride=[1,1,1,1],padding='SAME',name='pool1')

		filter3 = [3,3,16,16]
		stride3 = [1, 1, 1,1]

		conv_layer3 = conv2d_layer(max_pool1,filter3, stride3,'SAME','conv_layer3','relu3', self.phase )

		filter4 = [3,3,16,16]
		stride4 = [1, 1, 1,1]

		conv_layer4 = conv2d_layer(conv_layer3,filter4, stride4,'SAME','conv_layer4','relu4', self.phase )

		max_pool2 = tf.nn.max_pool(conv_layer4,ksize=[1,3,3,1],stride=[1,1,1,1],padding='SAME',name='pool2')

		flatten_layer = tf.contrib.layers.flatten(max_pool2)

		hidden_units1 = 100
		fc1_output = fc_layer(flatten_layer, hidden_units1, self.keep_prob, 'weights_1', 'bias_1', 'relu5')

		hidden_units2 = self.num_classes
		fc2_output = fc_layer(fc1_output, hidden_units2, self.keep_prob, 'weights_2', 'bias_2', 'relu6')

		self.output = fc2_output
		self.pred_prob = tf.nn.sigmoid(fc2_output, name='Prob_layer')
		self.pred =  tf.round(tf.nn.sigmoid_cross_entropy_with_logits(labels =self.labels , logits = self.output, name='Pred_layer' ))


	def train_function(self):
		self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels =self.labels , logits = self.output, name='Pred_layer' ))
		self.train_op = tf.train.AdamOptimizer(name='Adam').minimize(self.loss_op)


	def summary(self):
		self.accuracy_op = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.label_batch)tf.float32))

		tf.summary.scalar('Loss', self.loss_op)
		tf.summary.scalar('Accuracy', self.accuracy_op)
		self.summary_op = tf.summary.merge_all()


	def train_one_batch(self, session, input_batch, label_batch, phase):
		feed_dict = {self.input: input_batch, self.labels: label_batch, 
						self.phase: phase}

		_,loss, predictions, accuracies, summaries = session.run([self.train_op, self.loss_op, self.pred, self.accuracy_op, self.summary_op],
									 feed_dict = feed_dict)

		print "Average loss of the current batch is: {0}".format(loss)
		return predictions, accuracies, summaries

	def test_one_batch(self. session,input_batch, label_batch, phase ):
		feed_dict = {self.input: input_batch, self.labels: label_batch, 
						self.phase: phase}

		predictions, accuracies, summaries = session.run([self.pred, self.accuracy_op, self.summary_op], feed_dict = feed_dict)
		return predictions, accuracies, summaries













		