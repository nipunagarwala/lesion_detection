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
FINAL_DIM = 40

class Config():

	def __init__(self):
		self.batch_size = 64
		self.lr = 0.001
		self.input_size = (FINAL_DIM,FINAL_DIM,1)
		self.label_size = (1,)
		self.num_epochs = 500
		self.num_classes = 3
		self.keep_prob = 0.8


def read_files_labels(file_list, args, config):
	with tf.device("/cpu:0"):
		num_normal = 0
		num_internal = 0
		num_boundary = 0
		num_external = 0
		label_arr = np.zeros(len(file_list))
		image_arr = np.zeros((len(file_list), config.input_size[0], config.input_size[1], 1))
		file_final_list = []
		for i in xrange(len(file_list)):
			fname = file_list[i]
			# print "Current image name is {0}".format(file_list[i])
			loaded_im = np.load(join(args.data_dir, fname))
			if np.count_nonzero(loaded_im) == 0:
				continue
			image_arr[i,:,:, 0] = loaded_im
			file_final_list.append(fname)
			# image_arr[i,:,:, 0] = (image_arr[i,:,:, 0] - np.mean(image_arr[i,:,:, 0]))/np.std(image_arr[i,:,:, 0])
			# if 'normal' in fname:
			# 	label_arr[i] = 0
			# 	num_normal += 1
			if 'internal' in fname:
				label_arr[i] = 0
				num_internal += 1
			elif 'boundary' in fname:
				label_arr[i] = 1
				num_boundary += 1
			elif 'external' in fname:
				label_arr[i] = 2
				num_external += 1

		# print("Number of normal tissue examples: {0}".format(num_normal))
		print("Number of internal tissue examples: {0}".format(num_internal))
		print("Number of boundary tissue examples: {0}".format(num_boundary))
		print("Number of external tissue examples: {0}".format(num_external))
		return image_arr[:len(file_final_list)], label_arr[:len(file_final_list)], file_final_list



def run_epoch(session, curModel, cur_iter, args, config, file_writer,file_dataset, label_dataset, file_final_list,
							 num_files, phase):
	
	cur_batch = 0
	num_batches = num_files/config.batch_size 
	step = 0

	tot_acc = 0
	tot_true_labels = []
	tot_pred_labels = []
	permutation = np.random.permutation(label_dataset.shape[0])
	file_set = file_dataset[permutation,:,:,:]
	label_set = label_dataset[permutation]
	name_set = [file_final_list[i] for i in permutation]
	for i in xrange(num_batches):
		file_batch = file_set[i*config.batch_size:(i+1)*config.batch_size]
		label_batch = label_set[i*config.batch_size:(i+1)*config.batch_size]
		name_batch = name_set[i*config.batch_size:(i+1)*config.batch_size]

		# print "The list of files being trained on are: "
		# print name_batch

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
			cmd = "scp {0} {1}".format(join(args.data_dir, f),'/scratch/users/nipuna1/lesion_data/mri_data/mri_kfold/lesion_set_' + str(curK) + '/train/'+f)
			os.system(cmd)
			print("Completed copying training")
			# os.rename(join(args.data_dir, f), '/scratch/users/nipuna1/lesion_data/lesion_noupsampled_kfold/lesion_noupsampled_1/'+f)

		for f in test_batch:
			cmd = "scp {0} {1}".format(join(args.data_dir, f),'/scratch/users/nipuna1/lesion_data/mri_data/mri_kfold/lesion_set_' + str(curK) + '/test/'+f)
			os.system(cmd)
			print("Completed copying testing")

		print("Completed Fold {0}".format(curK))
		curK += 1

def create_random_contour_test(session, curModel, data_path, num_test_images, num_test_patches):

	# Load the dataset
	data = sio.loadmat(data_path)
	imageData = data['NewImage']
	CompROIdata = data['ROIdata']


	# Setup the test images
	image_rad = 5
	bound_buffer = 40

	for i in xrange(num_test_images):
		curImage = imageData[0,i]
		ROIdata = CompROIdata[0,i]

		# curImage = (curImage - np.mean(curImage))/np.std(curImage)

		ymin = int(np.min(ROIdata['ROI_Y'][0,0]))
		ymax = int(np.max(ROIdata['ROI_Y'][0,0]))
		xmin = int(np.min(ROIdata['ROI_X'][0,0]))
		xmax = int(np.max(ROIdata['ROI_X'][0,0]))

		yloc = ROIdata['ROI_Y'][0,0].astype(int)
		xloc = ROIdata['ROI_X'][0,0].astype(int)

		center_coords = []
		for i in range(len(xloc)):
			center_coords.append([xloc[i], yloc[i]])

		image_poly = Polygon(center_coords)
		image_poly = image_poly.buffer(0)

		file_batch = np.zeros((num_test_patches, FINAL_DIM, FINAL_DIM, 1))
		label_batch = np.zeros(num_test_patches)

		minx_lesion, miny_lesion, maxx_lesion, maxy_lesion = image_poly.bounds

		for tp in xrange(num_test_patches):
			x_limit = curImage.shape[0] - bound_buffer
			y_limit = curImage.shape[1] - bound_buffer
			# randX_pt = int(random.uniform(bound_buffer, x_limit))
			# randY_pt = int(random.uniform(bound_buffer, y_limit))

			randX_pt = int(random.uniform(minx_lesion, maxx_lesion))
			randY_pt = int(random.uniform(miny_lesion, maxy_lesion))

			xSmall = randX_pt-image_rad
			xLarge = randX_pt+image_rad
			ySmall = randY_pt-image_rad
			yLarge = randY_pt+image_rad

			test_patch = Polygon([(xSmall,ySmall),(xSmall, yLarge), (xLarge, ySmall),(xLarge, yLarge)])
			test_patch = test_patch.buffer(0)

			cur_patched_image = curImage[xSmall:xLarge, ySmall:yLarge]
			lesion_area_scaled = (image_poly.intersection(test_patch)).area/test_patch.area

			true_label = 0
			if image_poly.intersects(test_patch) and lesion_area_scaled < 0.6: # Boundary Path
				true_label = 1
			elif image_poly.contains(test_patch) or lesion_area_scaled >= 0.6: # Internal patch
				true_label = 0
			else: # External patch
				true_label = 2

			file_batch[tp, :,:,0] = fit_canvas(cur_patched_image)
			label_batch[tp] = true_label

		
		predictions, accuracies, summaries = curModel.test_one_batch(session, file_batch, label_batch, phase=False)
		print("The test accuracy is: {0}".format(accuracies))
		print("The predictions are: {0}".format(predictions))
		print("The true labels are: {0}".format(label_batch))






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

	# return 
	comb_step = 0
	with tf.Session(config=GPU_CONFIG) as session:
		# with tf.device("/cpu:0"):
		file_writer = tf.summary.FileWriter(args.ckpt_dir, graph=session.graph, 
									max_queue=10, flush_secs=30)
		
		i_stopped, found_ckpt = utils.get_checkpoint(args, session, saver)
		if args.train == 'train':
			onlyfiles = None
			with tf.device("/cpu:0"):
				onlyfiles = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

			random.shuffle(onlyfiles)
			num_files = len(onlyfiles)
			file_set, label_set, file_final_list = read_files_labels(onlyfiles, args, config)

			print("Shape of files is: {0}".format(file_set.shape))
			print("Shape of labels is: {0}".format(label_set.shape))
			print("Shape of file list is: {0}".format(len(file_final_list)))

			if not found_ckpt:
				init_op = tf.global_variables_initializer()
				init_op.run()

			for i in range(i_stopped,config.num_epochs):
				print "Running epoch {0}".format(i)
				tot_steps = run_epoch(session, curModel, i, args, config, file_writer,file_set, label_set, file_final_list,
												 len(file_final_list),'train')
				
				print "Saving checkpoint for epoch {0}".format(i)
				utils.save_checkpoint(args, session, saver, i)
				comb_step += tot_steps

		if args.train == 'test':
			# run_epoch(session, curModel, 0, args, config, file_writer,file_set, label_set, num_files, 'test')
			create_random_contour_test(session, curModel, args.data_dir , 10 , 30)

		if args.train == 'features':
			find_features(session, curModel, 0, args, config, file_writer, 'test')


if __name__ == '__main__':
	args = parseCommandLine()
	# print "Creating the Train/Test split"
	# create_train_test_split(args)
	# if args.kfold:
		# create_kfold_train_test_split(args)
		# exit(0)
	main(args)

