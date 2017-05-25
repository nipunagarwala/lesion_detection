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


GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.8

class Config():

	def __init__(self):
		self.batch_size = 64
		self.lr = 0.001
		self.input_size = (100,100,1)
		self.label_size = (1,)
		self.num_epochs = 100
		self.num_classes = 4
		self.keep_prob = 0.8


def read_files_labels(file_list, args, config):
	with tf.device("/cpu:0"):
		label_arr = np.zeros(len(file_list))
		image_arr = np.zeros((len(file_list), config.input_size[0], config.input_size[1], 1))
		for i in xrange(len(file_list)):
			fname = file_list[i]
			image_arr[i,:,:, 0] = np.load(join(args.data_dir, fname))
			# image_arr[i,:,:, 0] = (image_arr[i,:,:, 0] - np.mean(image_arr[i,:,:, 0]))/np.std(image_arr[i,:,:, 0])
			if 'normal' in fname:
				label_arr[i] = 0
			elif 'internal' in fname:
				label_arr[i] = 1
			elif 'boundary' in fname:
				label_arr[i] = 2
			elif 'external' in fname:
				label_arr[i] = 3

		return image_arr, label_arr



def run_epoch(session, curModel, cur_iter, args, config, file_writer,file_dataset, label_dataset, num_files, phase):
	
	cur_batch = 0
	num_batches = num_files/config.batch_size 
	step = 0

	tot_acc = 0
	tot_true_labels = []
	tot_pred_labels = []
	permutation = np.random.permutation(label_dataset.shape[0])
	file_set = file_dataset[permutation,:,:,:]
	label_set = label_dataset[permutation]

	for i in xrange(num_batches):
		file_batch = file_set[i*config.batch_size:(i+1)*config.batch_size]
		label_batch = label_set[i*config.batch_size:(i+1)*config.batch_size]
		
		tot_true_labels.extend(label_batch)
		print "The true labels are: "
		print label_batch
		if phase == 'train':
			predictions, accuracies, summaries = curModel.train_one_batch(session, cur_iter, file_batch, label_batch, phase=True)
			print "The predictions are: "
			print predictions
			tot_acc = tot_acc + (accuracies - tot_acc)/(i+1)
		elif phase == 'test':
			predictions, accuracies, summaries = curModel.test_one_batch(session, file_batch, label_batch, phase=False)
			tot_pred_labels.extend(predictions)
			print "The predictions are: "
			print predictions
			tot_acc = tot_acc + (accuracies - tot_acc)/(i+1)

		file_writer.add_summary(summaries, cur_batch)
		cur_batch +=1
		# break

	# return cur_batch
	file_batch = file_set[cur_batch*config.batch_size:]
	label_batch = label_set[cur_batch*config.batch_size:]
	tot_true_labels.extend(label_batch)
	if phase == 'train':
		predictions, accuracies, summaries = curModel.train_one_batch(session, cur_iter, file_batch, label_batch, phase=True)
		tot_acc = tot_acc + (accuracies - tot_acc)/(cur_batch+1)
		tot_pred_labels.extend(predictions)
	elif phase == 'test':
		predictions, accuracies, summaries = curModel.test_one_batch(session, file_batch, label_batch, phase=False)
		tot_pred_labels.extend(predictions)
		cm = confusion_matrix(tot_true_labels, tot_pred_labels)
		class_labels = ['Normal', 'Internal', 'Boundary', 'External']
		plot_confusion_matrix(cm,args,phase, classes=class_labels )
		tot_acc = tot_acc + (accuracies - tot_acc)/(cur_batch+1)

	file_writer.add_summary(summaries, cur_batch)
	print "Total accuracy is: {0}".format(tot_acc)
	return cur_batch


def find_features(session, curModel, cur_iter, args, config, file_writer, phase):
	onlyfiles = None
	with tf.device("/cpu:0"):
		onlyfiles = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

	random.shuffle(onlyfiles)
	cur_batch = 0
	num_batches = len(onlyfiles)/config.batch_size 
	step = 0

	for i in xrange(num_batches):
		file_batch = None
		label_batch = None
		cur_file_list = onlyfiles[i*config.batch_size: (i+1)*config.batch_size]
		with tf.device("/cpu:0"):
			file_batch, label_batch = read_files_labels(cur_file_list, args, config)

		cur_feat_mat = curModel.find_cnn_features(session, file_batch, phase)
		for k in xrange(config.batch_size):
			for c in xrange(32):
				plt.imshow(cur_feat_mat[0][k,:,:,c], cmap='gray')
				plt.colorbar()
				file_name_wo_ext = cur_file_list[k].split('.npy')[0]
				plt.savefig('/scratch/users/nipuna1/lesion_data/feature_maps/' + file_name_wo_ext + '_' + str(c) + '.png')
				plt.clf()

		cur_batch +=1

	file_batch, label_batch = read_files_labels(onlyfiles[cur_batch*config.batch_size:], args, config)
	find_all_feat_mat[cur_batch*config.batch_size:, :, :] = curModel.find_cnn_features(session, file_batch, phase)



def plot_confusion_matrix(cm, args, phase, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(args.ckpt_dir+'Confusion_matrix_' + str(phase) + '.png')

def create_train_test_split(args):
	onlyfiles = None
	with tf.device("/cpu:0"):
		onlyfiles = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

	random.shuffle(onlyfiles)
	train_set = onlyfiles[:8058]
	test_set = onlyfiles[8058:]
	for f in train_set:
		os.rename(join(args.data_dir, f), '/scratch/users/nipuna1/lesion_data/lesion_noupsampled_train/'+f)

	for f in test_set:
		os.rename(join(args.data_dir, f), '/scratch/users/nipuna1/lesion_data/lesion_noupsampled_test/'+f)

def create_kfold_train_test_split(args):
	onlyfiles = None
	with tf.device("/cpu:0"):
		onlyfiles = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

	print("Completed reading dataset")

	random.shuffle(onlyfiles)
	print("Completed shuffling dataset")

	kf = KFold(n_splits=5)
	kf.get_n_splits(onlyfiles)
	print("Completed spliting dataset")

	curK = 1
	for train_index, test_index in kf.split(onlyfiles):
		train_batch = [onlyfiles[i] for i in train_index]
		test_batch = [onlyfiles[i] for i in test_index]
		print("Length of train batch {0}".format(len(train_batch)))
		print("Length of test batch {0}".format(len(test_batch)))
		for f in train_batch:
			cmd = "scp {0} {1}".format(join(args.data_dir, f),'/scratch/users/nipuna1/lesion_data/lesion_noupsampled_kfold/lesion_set_' + str(curK) + '/train/'+f)
			os.system(cmd)
			print("Completed copying training")
			# os.rename(join(args.data_dir, f), '/scratch/users/nipuna1/lesion_data/lesion_noupsampled_kfold/lesion_noupsampled_1/'+f)

		for f in test_batch:
			cmd = "scp {0} {1}".format(join(args.data_dir, f),'/scratch/users/nipuna1/lesion_data/lesion_noupsampled_kfold/lesion_set_' + str(curK) + '/test/'+f)
			os.system(cmd)
			print("Completed copying testing")

		print("Completed Fold {0}".format(curK))
		curK += 1



def main(args):
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

	onlyfiles = None
	with tf.device("/cpu:0"):
		onlyfiles = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

	random.shuffle(onlyfiles)
	num_files = len(onlyfiles)
	file_set, label_set = read_files_labels(onlyfiles, args, config)

	comb_step = 0
	with tf.Session(config=GPU_CONFIG) as session:
		# with tf.device("/cpu:0"):
		file_writer = tf.summary.FileWriter(args.ckpt_dir, graph=session.graph, 
									max_queue=10, flush_secs=30)
		i_stopped, found_ckpt = utils.get_checkpoint(args, session, saver)
		if args.train == 'train':
			if not found_ckpt:
				init_op = tf.global_variables_initializer()
				init_op.run()

			for i in range(i_stopped,config.num_epochs):
				print "Running epoch {0}".format(i)
				tot_steps = run_epoch(session, curModel, i, args, config, file_writer,file_set, label_set, num_files,'train')
				
				print "Saving checkpoint for epoch {0}".format(i)
				utils.save_checkpoint(args, session, saver, i)
				comb_step += tot_steps

		if args.train == 'test':
			run_epoch(session, curModel, 0, args, config, file_writer, 'test')

		if args.train == 'features':
			find_features(session, curModel, 0, args, config, file_writer, 'test')


if __name__ == '__main__':
	args = parseCommandLine()
	# print "Creating the Train/Test split"
	# create_train_test_split(args)
	# create_kfold_train_test_split(args)
	main(args)

