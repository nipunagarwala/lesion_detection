import tensorflow as tf
import numpy as np 
import os
import sys
import reader
from preprocessing import *
from models import ConvolutionalNN

class Config():

	def __init__(self):
		batch_size = 32
		lr = 0.001
		# input_size = 
		# label_size = 
		num_classes = 1
		keep_prob = 0.8



def main(args):
	phase = True
	config = Config()
	curModel = ConvolutionalNN(config.batch_size, config.lr, config.input_size, 
				config.label_size, phase, config.num_classes, config.keep_prob)
	curModel.build_model()
	curModel.train_function()
	curModel.summary()

	with tf.Session() as session:
		file_writer = tf.summary.FileWriter(args.ckpt_dir, graph=session.graph, 
									max_queue=10, flush_secs=30)
		if args.train == 'train':
			init_op = tf.global_variables_initializer()
			init_op.run()

			data = load_data(args.data_dir)







if __name__ == '__main__':
	args = parseCommandLine()
	main(args)

