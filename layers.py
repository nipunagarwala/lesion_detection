import tensorflow as tf
import numpy as np 


XAVIER_INIT = tf.contrib.layers.xavier_initializer
SEED = 3


class Layers(object):

	def conv2d_layer(self, prev_output, filter_dim, stride, padding, conv_name, activ_name, phase):
		

		conv_layer = tf.nn.conv2d(prev_output, filter_dim, stride, padding='SAME', name=conv_name)
		batch_layer = tf.contrib.layers.batch_norm(inputs=conv_layer,decay=0.99,center=True, scale=True,
					param_initializers=XAVIER_INIT(uniform=True, seed=SEED, dtype=tf.float32),is_training=phase, data_format='NHWC')
		activation = tf.nn.relu(batch_layer, name=activ_name)
		return activation


	def fc_layer(self, prev_output, hidden_units, keep_prob, weight_name, bias_name, activ_name):

		input_shape = prev_output.get_shape().as_list()
		weights = tf.get_variable(weight_name, [input_shape[1], hidden_units ], initializer=XAVIER_INIT(seed=SEED, dtype=tf.float32),
								 trainable=True)
		bias= tf.get_variable(bias_name, [hidden_units ], initializer=XAVIER_INIT(seed=SEED, dtype=tf.float32),
								 trainable=True)

		result = tf.matmul(prev_output, weights) + bias
		activation = tf.nn.relu(result, name=activ_name)
		output = tf.nn.dropout(activation, keep_prob, seed = SEED)
		return output








