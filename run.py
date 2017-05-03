import tensorflow as tf
import numpy as np 
import os
import sys
import utils
from preprocessing import *
from models import ConvolutionalNN
from os import listdir
from os.path import isfile, join
from scipy import misc
import random

class Config():

	def __init__(self):
		self.batch_size = 32
		self.lr = 0.001
		self.input_size = (480,640, 4)
		self.label_size = (1,)
		self.num_epochs = 30
		self.num_classes = 1
		self.keep_prob = 0.8


def read_files_labels(file_list, args, config):
	label_arr = np.zeros(len(file_list))
	image_arr = np.zeros((len(file_list), config.input_size[0], config.input_size[1]))
	for i in xrange(len(file_list)):
		fname = file_list[i]
		image_arr[i,:,:] = misc.imread(join(args.data_dir, fname),flatten=True)
		if 'normal' in fname:
			label_arr[i] = 0
		if 'internal' in fname:
			label_arr[i] = 1
		if 'boundary' in fname:
			label_arr[i] = 2
		if 'external' in fname:
			label_arr[i] = 3

	return image_arr, label_arr



def run_epoch(session, cur_iter, args, config):
	onlyfiles = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]
	random.shuffle(onlyfiles)
	cur_batch = 0
	num_batches = len(onlyfiles)/config.batch_size 
	for i in xrange(num_batches):
		file_batch, label_batch = read_files_labels(onlyfiles[i*config.batch_size: (i+1)*config.batch_size], args, config)
		train_one_batch(session, file_batch, label_batch, phase=True)
		cur_batch +=1

	file_batch, label_batch = read_files_labels(onlyfiles[cur_batch*config.batch_size:], args, config)
	train_one_batch(session, cur_iter, file_batch, label_batch, phase=True)


def main(args):
	phase = True
	config = Config()
	print "Starting to build the models...."

	curModel = ConvolutionalNN(config.batch_size, config.lr, config.input_size, 
				config.label_size, phase, config.num_classes, config.keep_prob)
	print "Created the constructor...."

	curModel.build_model()
	print "Built the model...."

	curModel.train_function()
	print "Compiled the training function...."

	curModel.summary()
	print "Created the summary statistics operation...."

	saver = tf.train.Saver(max_to_keep=config.num_epochs)
	step = 0
	# gpuconfig = tf.ConfigProto(device_count = {'CPU': 0})
	# gpuconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
	with tf.Session() as session:
		with tf.device("/cpu:0"):
			file_writer = tf.summary.FileWriter(args.ckpt_dir, graph=session.graph, 
										max_queue=10, flush_secs=30)
			i_stopped, found_ckpt = utils.get_checkpoint(args, session, saver)
			if args.train == 'train':
				init_op = tf.global_variables_initializer()
				init_op.run()

				for i in range(i_stopped,config.num_epochs):
					print "Running epoch {0}".format(i)
					run_epoch(session, i, args, config)

		# if args.train == 'test':

				


if __name__ == '__main__':
	args = parseCommandLine()
	main(args)

