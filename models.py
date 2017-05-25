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
		# conv_layer4 = tf.nn.dropout(conv_layer4, keep_prob=self.keep_prob)
		
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
		# conv_layer8 = tf.nn.dropout(conv_layer8, keep_prob=self.keep_prob)
		
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

	def build_much_deeper_model(self):
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
		# conv_layer4 = tf.nn.dropout(conv_layer4, keep_prob=self.keep_prob)
		
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
		# conv_layer8 = tf.nn.dropout(conv_layer8, keep_prob=self.keep_prob)
		
		max_pool4 = tf.contrib.layers.max_pool2d(conv_layer8,kernel_size=[3,3],stride=[1,1],padding='SAME')

		filter9 = [3,3,32,32]
		stride9 = [1, 1, 1,1]

		conv_layer9 = self.conv2d_layer(max_pool4,filter9, stride9,'SAME','conv_layer9','filter_9','relu9', self.phase )

		filter10 = [3,3,32,32]
		stride10 = [1, 1, 1,1]

		conv_layer10 = self.conv2d_layer(conv_layer9,filter10, stride10,'SAME','conv_layer10', 'filter_10' ,'relu10', self.phase )
		# conv_layer8 = tf.nn.dropout(conv_layer8, keep_prob=self.keep_prob)
		
		max_pool5 = tf.contrib.layers.max_pool2d(conv_layer10,kernel_size=[3,3],stride=[2,2],padding='SAME')

		self.cnn_feat = max_pool5
		flatten_layer = tf.contrib.layers.flatten(max_pool5)

		hidden_units1 = 1000
		fc1_output = self.fc_layer(flatten_layer, hidden_units1, self.keep_prob, 'weights_1', 'bias_1', 'relu5')

		hidden_units2 = self.num_classes
		fc2_output = self.fc_layer(fc1_output, hidden_units2, self.keep_prob, 'weights_2', 'bias_2', 'relu6')

		self.output = fc2_output
		self.pred_prob = tf.nn.softmax(fc2_output, name='Prob_layer')
		# self.pred =  tf.nn.sparse_softmax_cross_entropy_with_logits(targets =self.labels , logits = self.output, name='Pred_layer' )
		self.pred = tf.argmax(self.pred_prob,1)


	def build_residual_model(self):
		filter1 = [3,3,1,32]
		stride1 = [1, 1, 1, 1]
		filter2 = [3,3,32,32]
		stride2 = [1, 1, 1, 1]

		res_layer1 = self.residual_unit(self.input, filter1, stride1,  'conv_layer1', 'filter_1', 'relu1',
					filter2, stride2, 'conv_layer2', 'filter_2', 'relu2', 'SAME',
					down_sample=False,  phase=True, projection=True)

		filter3 = [3,3,32,64]
		stride3 = [1, 1, 1, 1]
		filter4 = [3,3,64,64]
		stride4 = [1, 1, 1, 1]

		res_layer2 = self.residual_unit(res_layer1, filter3, stride3,  'conv_layer3', 'filter_3', 'relu3',
					filter4, stride4, 'conv_layer4', 'filter_4', 'relu4','SAME',
					down_sample=False,  phase=True, projection=True)


		filter5 = [3,3,64,128]
		stride5 = [1, 1, 1, 1]
		filter6 = [3,3,128,128]
		stride6 = [1, 1, 1, 1]

		res_layer3 = self.residual_unit(res_layer2, filter5, stride5,  'conv_layer5', 'filter_5', 'relu5',
					filter6, stride6, 'conv_layer6', 'filter_6', 'relu6', 'SAME',
					down_sample=True,  phase=True, projection=True)

		filter7 = [3,3,128,128]
		stride7 = [1, 1, 1, 1]
		filter8 = [3,3,128,128]
		stride8 = [1, 1, 1, 1]

		res_layer4 = self.residual_unit(res_layer3, filter7, stride7,  'conv_layer7', 'filter_7', 'relu7',
					filter8, stride8, 'conv_layer8', 'filter_8', 'relu8', 'SAME',
					down_sample=False,  phase=True, projection=True)

		filter9 = [3,3,128,64]
		stride9 = [1, 1, 1, 1]
		filter10 = [3,3,64,64]
		stride10 = [1, 1, 1, 1]

		res_layer5 = self.residual_unit(res_layer4, filter9, stride9,  'conv_layer9', 'filter_9', 'relu9',
					filter10, stride10, 'conv_layer10', 'filter_10', 'relu10', 'SAME',
					down_sample=False,  phase=True, projection=True)


		filter11 = [3,3,64,64]
		stride11 = [1, 1, 1, 1]
		filter12 = [3,3,64,64]
		stride12 = [1, 1, 1, 1]

		res_layer6 = self.residual_unit(res_layer5, filter11, stride11,  'conv_layer11', 'filter_11', 'relu11',
					filter12, stride12, 'conv_layer12', 'filter_12', 'relu12', 'SAME',
					down_sample=True,  phase=True, projection=True)

		filter13 = [3,3,64,32]
		stride13 = [1, 1, 1, 1]
		filter14 = [3,3,32,32]
		stride14 = [1, 1, 1, 1]

		res_layer7 = self.residual_unit(res_layer6, filter13, stride13,  'conv_layer13', 'filter_13', 'relu13',
					filter14, stride14, 'conv_layer14', 'filter_14', 'relu14', 'SAME',
					down_sample=False,  phase=True, projection=True)

		filter15 = [3,3,32,32]
		stride15 = [1, 1, 1, 1]
		filter16 = [3,3,32,32]
		stride16 = [1, 1, 1, 1]

		res_layer8 = self.residual_unit(res_layer7, filter15, stride15,  'conv_layer15', 'filter_15', 'relu15',
					filter16, stride16, 'conv_layer16', 'filter_16', 'relu16', 'SAME',
					down_sample=False,  phase=True, projection=True)

		self.cnn_feat = res_layer8
		flatten_layer = tf.contrib.layers.flatten(res_layer8)

		hidden_units1 = 1000
		fc1_output = self.fc_layer(flatten_layer, hidden_units1, self.keep_prob, 'weights_1', 'bias_1', 'relu5')

		hidden_units2 = self.num_classes
		fc2_output = self.fc_layer(fc1_output, hidden_units2, self.keep_prob, 'weights_2', 'bias_2', 'relu6')

		self.output = fc2_output
		self.pred_prob = tf.nn.softmax(fc2_output, name='Prob_layer')
		# self.pred =  tf.nn.sparse_softmax_cross_entropy_with_logits(targets =self.labels , logits = self.output, name='Pred_layer' )
		self.pred = tf.argmax(self.pred_prob,1)



	def train_function(self):
		l2_lambda = 0.1
		l2_cost = 0.0
		# train_vars = tf.trainable_variables()
		# for v in train_vars:
		# 	l2_cost += tf.nn.l2_loss(v)

		main_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels =self.labels , logits = self.output))

		self.loss_op = main_loss + l2_lambda*l2_cost
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







		