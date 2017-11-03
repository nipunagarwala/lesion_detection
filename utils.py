import pickle
import os
import random
import numpy as np
import re
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def save_checkpoint(args, session, saver, i):
	checkpoint_path = os.path.join(args.ckpt_dir, 'model.ckpt')
	saver.save(session, checkpoint_path, global_step=i)

def get_checkpoint(args, session, saver):
	# Checkpoint
	with tf.device("/cpu:0"):
		found_ckpt = False

	# if args.override:
	# 	if tf.gfile.Exists(args.ckpt_dir):
	# 		tf.gfile.DeleteRecursively(args.ckpt_dir)
	# 	tf.gfile.MakeDirs(args.ckpt_dir)

	# check if arags.ckpt_dir is a directory of checkpoints, or the checkpoint itself
		if len(re.findall('model.ckpt-[0-9]+', args.ckpt_dir)) == 0:
			ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(session, ckpt.model_checkpoint_path)
				i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
				print "Found checkpoint for epoch ({0})".format(i_stopped)
				found_ckpt = True
			else:
				print('No checkpoint file found!')
				i_stopped = 0
		else:
			saver.restore(session, args.ckpt_dir)
			i_stopped = int(args.ckpt_dir.split('/')[-1].split('-')[-1])
			print "Found checkpoint for epoch ({0})".format(i_stopped)
			found_ckpt = True



		return i_stopped, found_ckpt

def get_checkpoint_matlab(model_path, session, saver):
	with tf.device("/cpu:0"):
		found_ckpt = False

		if len(re.findall('model.ckpt-[0-9]+', model_path)) == 0:
			ckpt = tf.train.get_checkpoint_state(model_path)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(session, ckpt.model_checkpoint_path)
				i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
				print "Found checkpoint for epoch ({0})".format(i_stopped)
				found_ckpt = True
			else:
				print('No checkpoint file found!')
				i_stopped = 0
		else:
			saver.restore(session, model_path)
			i_stopped = int(model_path.split('/')[-1].split('-')[-1])
			print "Found checkpoint for epoch ({0})".format(i_stopped)
			found_ckpt = True



		return i_stopped, found_ckpt






def plot_graphs():
	train_accuracy = [0.969010416667, 0.970588235294, 0.952205882353, 0.951286764706, 0.972163865546]
	test_accuracy = [0.861041667064, 0.855833333731, 0.856250001987, 0.844791666667, 0.850937501589]
	plt.ylabel('Accuracies')
	plt.xlabel('K-fold Set')
	plt.plot(xrange(1,6,1),train_accuracy, label='Train')
	plt.plot(xrange(1,6,1),test_accuracy, label='Test')
	plt.legend(loc=4)
	plt.savefig('K_fold_results.png')







if __name__ == '__main__':
	plot_graphs()





