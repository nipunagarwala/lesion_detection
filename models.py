import tensorflow as tf
import numpy as np 
from layers import *




class ConvolutionalNN(Layers):

	def __init__(self, batch_size, lr, input_size, label_size, phase, num_classes, keep_prob):
		super(ConvolutionalNN, self).__init__(keep_prob)
		self.batch_size = batch_size
		self.lr = lr
		input_shape = (None,) + tuple(input_size)
		output_shape = (None,)
		self.input = tf.placeholder(tf.float32, shape=input_shape, name='Input')
		print self.input.get_shape().as_list()
		self.labels = tf.placeholder(tf.int64, shape=output_shape, name = 'Output')
		self.phase = tf.placeholder(tf.bool, name='phase')
		self.num_classes = num_classes
		self.keep_prob = keep_prob


	def build_model(self):
		filter1 = [3,3,1,32]
		stride1 = [1, 1, 1, 1]

		conv_layer1 = self.conv2d_layer(self.input,filter1, stride1,'SAME','conv_layer1', 'filter_1','relu1', self.phase )

		filter2 = [3,3,32,32]
		stride2 = [1, 1, 1,1]

		conv_layer2 = self.conv2d_layer(conv_layer1,filter2, stride2,'SAME','conv_layer2','filter_2','relu2', self.phase )

		max_pool1 = tf.contrib.layers.max_pool2d(conv_layer2,kernel_size=[3,3],stride=[2,2],padding='SAME')

		filter3 = [3,3,32,32]
		stride3 = [1, 1, 1,1]

		conv_layer3 = self.conv2d_layer(max_pool1,filter3, stride3,'SAME','conv_layer3','filter_3','relu3', self.phase )

		filter4 = [3,3,32,32]
		stride4 = [1, 1, 1,1]

		conv_layer4 = self.conv2d_layer(conv_layer3,filter4, stride4,'SAME','conv_layer4', 'filter_4' ,'relu4', self.phase )

		self.cnn_feat = conv_layer4
		max_pool2 = tf.contrib.layers.max_pool2d(conv_layer4,kernel_size=[3,3],stride=[2,2],padding='SAME')

		flatten_layer = tf.contrib.layers.flatten(max_pool2)

		hidden_units1 = 1000
		fc1_output = self.fc_layer(flatten_layer, hidden_units1, self.keep_prob, 'weights_1', 'bias_1', 'relu5')

		hidden_units2 = self.num_classes
		fc2_output = self.fc_layer(fc1_output, hidden_units2, self.keep_prob, 'weights_2', 'bias_2', 'relu6')

		self.output = fc2_output
		self.pred_prob = tf.nn.softmax(fc2_output, name='Prob_layer')
		# self.pred =  tf.nn.sparse_softmax_cross_entropy_with_logits(targets =self.labels , logits = self.output, name='Pred_layer' )
		self.pred = tf.argmax(self.pred_prob,1)

	def build_deeper_model(self):
		filter1 = [3,3,1,32]
		stride1 = [1, 1, 1, 1]

		conv_layer1 = self.conv2d_layer(self.input,filter1, stride1,'SAME','conv_layer1', 'filter_1','relu1', self.phase )

		filter2 = [3,3,32,32]
		stride2 = [1, 1, 1,1]

		conv_layer2 = self.conv2d_layer(conv_layer1,filter2, stride2,'SAME','conv_layer2','filter_2','relu2', self.phase )

		max_pool1 = tf.contrib.layers.max_pool2d(conv_layer2,kernel_size=[3,3],stride=[1,1],padding='SAME')

		filter3 = [3,3,32,32]
		stride3 = [1, 1, 1,1]

		conv_layer3 = self.conv2d_layer(max_pool1,filter3, stride3,'SAME','conv_layer3','filter_3','relu3', self.phase )

		filter4 = [3,3,32,32]
		stride4 = [1, 1, 1,1]

		conv_layer4 = self.conv2d_layer(conv_layer3,filter4, stride4,'SAME','conv_layer4', 'filter_4' ,'relu4', self.phase )

		
		max_pool2 = tf.contrib.layers.max_pool2d(conv_layer4,kernel_size=[3,3],stride=[1,1],padding='SAME')


		filter5 = [3,3,32,32]
		stride5 = [1, 1, 1,1]

		conv_layer5 = self.conv2d_layer(max_pool2,filter5, stride5,'SAME','conv_layer5','filter_5','relu5', self.phase )

		filter6 = [3,3,32,32]
		stride6 = [1, 1, 1,1]

		conv_layer6 = self.conv2d_layer(conv_layer5,filter6, stride6,'SAME','conv_layer6', 'filter_6' ,'relu6', self.phase )

		
		max_pool3 = tf.contrib.layers.max_pool2d(conv_layer6,kernel_size=[3,3],stride=[2,2],padding='SAME')

		filter7 = [3,3,32,32]
		stride7 = [1, 1, 1,1]

		conv_layer7 = self.conv2d_layer(max_pool3,filter7, stride7,'SAME','conv_layer7','filter_7','relu7', self.phase )

		filter8 = [3,3,32,32]
		stride8 = [1, 1, 1,1]

		conv_layer8 = self.conv2d_layer(conv_layer7,filter8, stride8,'SAME','conv_layer8', 'filter_8' ,'relu8', self.phase )

		
		max_pool4 = tf.contrib.layers.max_pool2d(conv_layer8,kernel_size=[3,3],stride=[2,2],padding='SAME')

		self.cnn_feat = max_pool4
		flatten_layer = tf.contrib.layers.flatten(max_pool4)

		hidden_units1 = 1000
		fc1_output = self.fc_layer(flatten_layer, hidden_units1, self.keep_prob, 'weights_1', 'bias_1', 'relu5')

		hidden_units2 = self.num_classes
		fc2_output = self.fc_layer(fc1_output, hidden_units2, self.keep_prob, 'weights_2', 'bias_2', 'relu6')

		self.output = fc2_output
		self.pred_prob = tf.nn.softmax(fc2_output, name='Prob_layer')
		# self.pred =  tf.nn.sparse_softmax_cross_entropy_with_logits(targets =self.labels , logits = self.output, name='Pred_layer' )
		self.pred = tf.argmax(self.pred_prob,1)




	def train_function(self):
		self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels =self.labels , logits = self.output))
		self.train_op = tf.train.AdamOptimizer(name='Adam').minimize(self.loss_op)


	def summary(self):
		self.accuracy_op = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))

		tf.summary.scalar('Loss', self.loss_op)
		tf.summary.scalar('Accuracy', self.accuracy_op)
		self.summary_op = tf.summary.merge_all()


	def train_one_batch(self, session, cur_iter ,input_batch, label_batch, phase):
		feed_dict = {self.input: input_batch, self.labels: label_batch, 
						self.phase: phase}

		_,loss, predictions, accuracies, summaries = session.run([self.train_op, self.loss_op, self.pred, self.accuracy_op, self.summary_op],
									 feed_dict = feed_dict)

		print "Average loss of the current batch in epoch {0} is: {1}".format(cur_iter,loss)
		print "Average accuracy of the current batch is: {0}".format(accuracies)
		return predictions, accuracies, summaries


	def test_one_batch(self, session, input_batch, label_batch, phase ):
		feed_dict = {self.input: input_batch, self.labels: label_batch, 
						self.phase: phase}

		predictions, accuracies, summaries = session.run([self.pred, self.accuracy_op, self.summary_op], feed_dict = feed_dict)
		print "Average accuracy of the current batch is: {0}".format(accuracies)
		return predictions, accuracies, summaries

	def find_cnn_features(self, session,input_batch, phase):
		feed_dict = {self.input: input_batch, self.phase: phase}

		cnn_features = session.run([self.cnn_feat], feed_dict = feed_dict)
		return cnn_features







		