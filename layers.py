import tensorflow as tf
import numpy as np 


XAVIER_INIT = tf.contrib.layers.xavier_initializer


class Layers(object):
	def __init__(self, keep_prob):
		self.keep_prob = keep_prob
		# pass

	def conv2d_layer(self, prev_output, filter_dim, stride, padding, conv_name, filter_name, activ_name, phase):
		filter_wt = tf.get_variable(filter_name, filter_dim, initializer=XAVIER_INIT(dtype=tf.float32),
								 trainable=True)
		conv_layer = tf.nn.conv2d(prev_output, filter_wt, stride, padding='SAME', name=conv_name)
		# batch_layer = tf.contrib.layers.batch_norm(inputs=conv_layer,decay=0.99,center=True, scale=True,
		# 			is_training=phase)
		activation = tf.nn.relu(conv_layer, name=activ_name)
		# activation = tf.nn.dropout(activation, keep_prob=self.keep_prob)
		return activation


	def fc_layer(self, prev_output, hidden_units, keep_prob, weight_name, bias_name, activ_name):

		input_shape = prev_output.get_shape().as_list()
		weights = tf.get_variable(weight_name, [input_shape[1], hidden_units ], initializer=XAVIER_INIT(dtype=tf.float32),
								 trainable=True)
		bias= tf.get_variable(bias_name, [hidden_units ],
								 trainable=True)

		result = tf.matmul(prev_output, weights) + bias
		activation = tf.nn.relu(result, name=activ_name)
		output = tf.nn.dropout(activation, keep_prob)
		return output








