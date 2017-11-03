import numpy as np 
import os
import sys
import scipy.io as sio
import scipy.misc
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
from argparse import ArgumentParser
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import random

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
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import KFold


curModel = None
session = None
saver = None
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.8
FINAL_DIM = 40
INT_BIN_CNT = None
BOUND_BIN_CNT = None
EXT_BIN_CNT = None
BIN_RANGE = np.arange(0.0, 1.025, 0.025)

class Config():

	def __init__(self):
		self.batch_size = 64
		self.lr = 0.001
		self.input_size = (FINAL_DIM,FINAL_DIM,1)
		self.label_size = (1,)
		self.num_epochs = 500
		self.num_classes = 3
		self.keep_prob = 0.8

def load_model(model_path):
	global curModel
	global session
	global saver

	phase = True
	config = Config()

	print "Starting to build the models...."

	curModel = ConvolutionalNN(config.batch_size, config.lr, config.input_size, 
				config.label_size, phase, config.num_classes, config.keep_prob)
	print "Created the constructor...."

	curModel.build_deeper_model()
	print "Built the model...."

	curModel.train_function()
	print "Compiled the training function...."

	curModel.summary()
	print "Created the summary statistics operation...."

	saver = tf.train.Saver(max_to_keep=config.num_epochs)
	step = 0

	# return 
	comb_step = 0
	session = tf.Session(config=GPU_CONFIG)
		# with tf.device("/cpu:0"):
	# file_writer = tf.summary.FileWriter(model_path, graph=session.graph, 
	# 							max_queue=10, flush_secs=30)
	
	i_stopped, found_ckpt = utils.get_checkpoint_matlab(model_path, session, saver)

def collect_data_stats(data_path):
	global INT_BIN_CNT
	global BOUND_BIN_CNT
	global EXT_BIN_CNT

	avg_int_im = np.zeros((FINAL_DIM, FINAL_DIM))
	avg_bound_im = np.zeros((FINAL_DIM, FINAL_DIM))
	avg_ext_im = np.zeros((FINAL_DIM, FINAL_DIM))

	num_internal = 0
	num_boundary = 0
	num_external = 0

	file_list = os.listdir(data_path)
	for fname in file_list:
		loaded_im = np.load(os.path.join(data_path, fname))
		if 'internal' in fname:
			avg_int_im += (loaded_im - avg_int_im)/(num_internal+1)
			num_internal += 1
		elif 'boundary' in fname:
			avg_bound_im += (loaded_im - avg_bound_im)/(num_boundary+1)
			num_boundary += 1
		elif 'external' in fname:
			avg_ext_im += (loaded_im - avg_ext_im)/(num_external+1)
			num_external += 1

	INT_BIN_CNT, _ = np.histogram(avg_int_im, BIN_RANGE)
	BOUND_BIN_CNT, _ = np.histogram(avg_bound_im, BIN_RANGE)
	EXT_BIN_CNT, _ = np.histogram(avg_ext_im, BIN_RANGE)
	print INT_BIN_CNT
	INT_BIN_CNT = INT_BIN_CNT.astype(float)
	BOUND_BIN_CNT = BOUND_BIN_CNT.astype(float)
	EXT_BIN_CNT = EXT_BIN_CNT.astype(float)

	INT_BIN_CNT += 1e-3
	BOUND_BIN_CNT += 1e-3
	EXT_BIN_CNT += 1e-3

	INT_BIN_CNT = INT_BIN_CNT/np.sum(INT_BIN_CNT)
	BOUND_BIN_CNT = BOUND_BIN_CNT/np.sum(BOUND_BIN_CNT)
	EXT_BIN_CNT = EXT_BIN_CNT/np.sum(EXT_BIN_CNT)


# def collect_data_stats_cnn_features(data_path):
# 	global INT_BIN_CNT
# 	global BOUND_BIN_CNT
# 	global EXT_BIN_CNT

# 	avg_int_im = np.zeros((FINAL_DIM, FINAL_DIM))
# 	avg_bound_im = np.zeros((FINAL_DIM, FINAL_DIM))
# 	avg_ext_im = np.zeros((FINAL_DIM, FINAL_DIM))

# 	num_internal = 0
# 	num_boundary = 0
# 	num_external = 0

# 	file_list = os.listdir(data_path)
# 	for fname in file_list:
# 		loaded_im = np.load(os.path.join(data_path, fname))
# 		if 'internal' in fname:
# 			avg_int_im += (loaded_im - avg_int_im)/(num_internal+1)
# 			num_internal += 1
# 		elif 'boundary' in fname:
# 			avg_bound_im += (loaded_im - avg_bound_im)/(num_boundary+1)
# 			num_boundary += 1
# 		elif 'external' in fname:
# 			avg_ext_im += (loaded_im - avg_ext_im)/(num_external+1)
# 			num_external += 1



# 	INT_BIN_CNT, _ = np.histogram(avg_int_im, BIN_RANGE)
# 	BOUND_BIN_CNT, _ = np.histogram(avg_bound_im, BIN_RANGE)
# 	EXT_BIN_CNT, _ = np.histogram(avg_ext_im, BIN_RANGE)

# 	INT_BIN_CNT = INT_BIN_CNT.astype(float)
# 	BOUND_BIN_CNT = BOUND_BIN_CNT.astype(float)
# 	EXT_BIN_CNT = EXT_BIN_CNT.astype(float)

# 	INT_BIN_CNT += 1e-3
# 	BOUND_BIN_CNT += 1e-3
# 	EXT_BIN_CNT += 1e-3

# 	INT_BIN_CNT = INT_BIN_CNT/np.sum(INT_BIN_CNT)
# 	BOUND_BIN_CNT = BOUND_BIN_CNT/np.sum(BOUND_BIN_CNT)
# 	EXT_BIN_CNT = EXT_BIN_CNT/np.sum(EXT_BIN_CNT)




def run_iteration(x_loc, y_loc, matImage,im_x_dim, im_y_dim, patch_radius):
	# patch_list = np.zeros((len(x_coords),FINAL_DIM, FINAL_DIM,1 ))
	x_coords = np.atleast_1d(x_loc)
	y_coords = np.atleast_1d(y_loc)

	# center_coords_norm = []
	# for i in range(len(x_correct)):
	# 	center_coords_norm.append([x_correct[i], y_correct[i])

	# image_poly_les = Polygon(center_coords_norm)
	# image_poly_les = image_poly_norm.buffer(0)

	patch_list = np.zeros((x_coords.size,FINAL_DIM, FINAL_DIM,1 ))
	curImage = np.reshape(matImage, (im_y_dim, im_x_dim))
	curImage = curImage.T
	curImage = curImage/np.amax(curImage)

	# curImage = (curImage - np.mean(curImage))/np.std(curImage)

	for i in xrange(x_coords.size):
		x_low = int(x_coords[i]) - patch_radius
		x_high = int(x_coords[i]) + patch_radius
		y_low = int(y_coords[i]) - patch_radius
		y_high = int(y_coords[i]) + patch_radius

		cur_patch = curImage[x_low:x_high, y_low:y_high]
		fitted_patch = fit_canvas(cur_patch)
		patch_list[i, :,:,0] = fitted_patch

	dummy_labels = np.zeros(x_coords.size)

	# probabilities = curModel.get_confidence_one_batch(session, patch_list, dummy_labels, phase=False)

	probabilities = curModel.get_scaled_confidence_one_batch(session, patch_list, dummy_labels, False,
						INT_BIN_CNT, BOUND_BIN_CNT, EXT_BIN_CNT, x_coords.size, BIN_RANGE)

	# print probabilities
	
	return probabilities, x_coords.size, 3

