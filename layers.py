import tensorflow as tf
import numpy as np 


XAVIER_INIT = tf.contrib.layers.variance_scaling_initializer


class Layers(object):
	def __init__(self, keep_prob):
		self.keep_prob = keep_prob
		# pass

	def conv2d_layer(self, prev_output, filter_dim, stride, padding, conv_name, filter_name, activ_name, phase):
		filter_wt = tf.get_variable(filter_name, filter_dim, initializer=XAVIER_INIT(dtype=tf.float32),
								 trainable=True)
		conv_layer = tf.nn.conv2d(prev_output, filter_wt, stride, padding='SAME', name=conv_name)
		batch_layer = tf.contrib.layers.batch_norm(inputs=conv_layer, center=True, scale=True,
					is_training=phase)
		activation = tf.nn.relu(conv_layer, name=activ_name)
		# activation = tf.nn.dropout(activation, keep_prob=self.keep_prob)
		return activation


	def fc_layer(self, prev_output, hidden_units, keep_prob, weight_name, bias_name, activ_name, use_act = True):

		input_shape = prev_output.get_shape().as_list()
		weights = tf.get_variable(weight_name, [input_shape[1], hidden_units ], initializer=XAVIER_INIT(dtype=tf.float32),
								 trainable=True)
		bias= tf.get_variable(bias_name, [hidden_units ],
								 trainable=True)

		result = tf.matmul(prev_output, weights) + bias

		activation = result
		if use_act:
			activation = tf.nn.relu(result, name=activ_name)
			
		# output = tf.nn.dropout(activation, keep_prob)
		return activation

	def residual_unit(self, prev_output, filter_dim1, stride1,  conv_name1, filter_name1, activ_name1,
					filter_dim2, stride2, conv_name2, filter_name2, activ_name2, padding,
					down_sample, phase, projection=False):

		if down_sample:
			prev_output = tf.contrib.layers.max_pool2d(prev_output,kernel_size=[3,3],stride=[2,2],padding='SAME')

		conv1 = self.conv2d_layer(prev_output, filter_dim1, stride1, padding, conv_name1, filter_name1, activ_name1, phase)
		conv2 = self.conv2d_layer(conv1, filter_dim2, stride2, padding, conv_name2, filter_name2, activ_name2, phase)

		input_depth = filter_dim1[2]
		output_depth = filter_dim1[3]
		input_layer = prev_output

		if input_depth != output_depth:
			if projection:
				# Option B: Projection shortcut
				input_layer = self.conv2d_layer(input_layer, [1, 1, input_depth, output_depth], 
							[1,1,1,1], padding, conv_name1+'_proj', filter_name2+'_proj',
							activ_name2+'_proj', phase)
			else:
				# Option A: Zero-padding
				input_layer = tf.pad(input_layer, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])

		res = conv2 + input_layer
		return res








