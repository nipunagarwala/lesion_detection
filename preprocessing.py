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

# DATA_PATH = '/scratch/users/nipuna1/lesion_data/mri_data/MR4Nipun.mat'
# DATA_PATH = '/scratch/users/nipuna1/lesion_data/liver_lesions.mat'
DATA_PATH = '/scratch/users/nipuna1/lesion_data/CTLiverNew.mat'
# DATA_PATH = '/scratch/users/nipuna1/lesion_data/BrainMR_Norm.mat'
# DATA_PATH = '/scratch/users/nipuna1/lesion_data/FinalLungCT.mat'
NO_UPSAMPLED_DIR = '/scratch/users/nipuna1/lesion_data/lesion_noupsampled_dataset/'
UPSAMPLED_DIR = '/scratch/users/nipuna1/lesion_data/lesion_upsampled_dataset/'
UPSAMPLED_IMAGE_DIR = '/scratch/users/nipuna1/lesion_data/lesion_upsampled_images/'
NO_UPSAMPLED_IMAGE_DIR = '/scratch/users/nipuna1/lesion_data/lesion_noupsampled_images/'

# PATCHED_DATA_DIR = '/scratch/users/nipuna1/lesion_data/lesion_patched_ct_40_dataset/'
# PATCHED_DATA_IMAGES = '/scratch/users/nipuna1/lesion_data/lesion_patched_ct_40_images/'
PATCHED_DATA_DIR = '/scratch/users/nipuna1/lesion_data/lesion_patched_ct_patch_10_dataset/'
PATCHED_DATA_IMAGES = '/scratch/users/nipuna1/lesion_data/lesion_patched_ct_patch_10_images/'
# PATCHED_MRI_DIR = '/scratch/users/nipuna1/lesion_data/lesion_patched_mri_40_dataset/'
# PATCHED_MRI_IMAGES = '/scratch/users/nipuna1/lesion_data/lesion_patched_mri_40_images/'
PATCHED_MRI_DIR = '/scratch/users/nipuna1/lesion_data/lesion_patched_brain_rand_100_dataset/'
PATCHED_MRI_IMAGES = '/scratch/users/nipuna1/lesion_data/lesion_patched_brain_rand_100_images/'
# PATCHED_LUNG_DIR = '/scratch/users/nipuna1/lesion_data/lesion_patched_lung_40_dataset/'
# PATCHED_LUNG_IMAGES = '/scratch/users/nipuna1/lesion_data/lesion_patched_lung_40_images/'
PATCHED_LUNG_DIR = '/scratch/users/nipuna1/lesion_data/lesion_patched_lung_rand_100_dataset/'
PATCHED_LUNG_IMAGES = '/scratch/users/nipuna1/lesion_data/lesion_patched_lung_rand_100_images/'


MRI_NOUPSAMPLED_DIR = '/scratch/users/nipuna1/lesion_data/mri_data/mri_noupsampled_dataset/'
MRI_NOUPSAMPLED_IMAGE_DIR = '/scratch/users/nipuna1/lesion_data/mri_data/mri_noupsampled_images/'

TRAINSET_DIR = UPSAMPLED_DIR
FINAL_DIM = 40

TEST_DIR = '/home/nipuna1/lesion_detection_research/lesion_detection/opp_coord_bound/'

imageData = None
CompROIdata = None

def parseCommandLine():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-p', choices = ["train", "test", "dev", "features"], type = str,
						dest = 'train', required = True, help = 'Training or Testing phase to be run')

	parser.add_argument('-ckpt', dest='ckpt_dir', default='/scratch/users/nipuna1/lesion_ckpt', 
									type=str, help='Set the checkpoint directory')
	parser.add_argument('-data', dest='data_dir', default=TRAINSET_DIR, 
									type=str, help='Set the data directory')
	parser.add_argument('-kfold', dest='kfold', action='store_true')
	parser.add_argument('-upsampled', dest='upsampled', action='store_true')
	parser.add_argument('-no_upsampled', dest='no_upsampled', action='store_true')
	parser.add_argument('-patched', dest='patched', action='store_true')

	args = parser.parse_args()
	return args




def load_data(data_path):
	global imageData
	global CompROIdata
	data = sio.loadmat(data_path)

	imageData = data['NewImage']
	CompROIdata = data['ROIdata']
	print 'Loaded Data'
	print data.keys()
	return data['NewImage'], data['ROIdata']


def create_dataset_mt():
	num_images = imageData.shape[1] #8
	print "Creating threads for dataset creation"
	P = Pool(processes=8)
	P.map(process_function, (i for i in range(0, num_images )) )


def process_function(it):

	curImage = imageData[0, it]
	ROIdata = CompROIdata[0][it]
	scale = 0.75

	lesion_scale = np.linspace(0.6, 0.8, num=3)
	boundary_scale = np.linspace(0.9, 1.1, num=3)
	external_scale = np.linspace(1.3, 1.5, num=3)

	num_xpixels = 20
	num_ypixels = 10
	# plt.imshow(curImage, cmap='gray')
	# plt.savefig(DATASET_DIR + 'original_image_' + str(it) + '.png')

	create_internal_external_data(curImage, ROIdata, scale, lesion_scale, 'internal_lesion' , it, num_xpixels, num_ypixels)
	create_internal_external_data(curImage, ROIdata, scale, boundary_scale, 'boundary_lesion', it, num_xpixels, num_ypixels)
	create_internal_external_data(curImage, ROIdata, scale, external_scale, 'external_lesion', it, num_xpixels, num_ypixels)
	create_normal_tissue_data(curImage, ROIdata, scale, lesion_scale, 'normal_tissue', it, num_xpixels, num_ypixels)


	print 'Completed Lesion and Normal data creation for Liver {}'.format(it)

def create_internal_external_data(curImage, ROIdata, scale, lesion_scale, str_name, it, num_xpixels, num_ypixels):
	# Lesion Images
	ymin = int(np.min(ROIdata['ROI_X'][0,0][0]))
	ymax = int(np.max(ROIdata['ROI_X'][0,0][0]))
	xmin = int(np.min(ROIdata['ROI_Y'][0,0][0]))
	xmax = int(np.max(ROIdata['ROI_Y'][0,0][0]))

	origMask = np.zeros((xmax-xmin, ymax-ymin))
	# print "Original Lesion Boundaries"
	# print xmin, xmax, ymin, ymax
	for i in lesion_scale:
		curMask = scipy.misc.imresize(origMask, i)
		mask_x, mask_y = curMask.shape
		cur_xmin = (xmax+xmin)/2 - mask_x/2
		cur_xmax = (xmax+xmin)/2 + mask_x/2
		cur_ymin = (ymax+ymin)/2 - mask_y/2
		cur_ymax = (ymax+ymin)/2 + mask_y/2
		print "Resized Lesion boundaries"
		print cur_xmin, cur_xmax, cur_ymin, cur_ymax
		upsampled_image, orig_no_canvas_im = upsample_data(curImage, cur_ymin, cur_ymax, cur_xmin, cur_xmax)
		augment_image_list  = augment_data(upsampled_image, orig_no_canvas_im, scale, num_xpixels, num_ypixels)

		augment_image_list.append(upsampled_image)
		save_images(augment_image_list,str_name, it, i)

def create_normal_tissue_data(curImage, ROIdata, scale, lesion_scale, str_name, it, num_xpixels, num_ypixels):
	# Lesion Images
	ymin = int(np.min(ROIdata['Normal_ROIx'][0][0]))
	ymax = int(np.max(ROIdata['Normal_ROIx'][0][0]))
	xmin = int(np.min(ROIdata['Normal_ROIy'][0][0]))
	xmax = int(np.max(ROIdata['Normal_ROIy'][0][0]))

	origMask = np.zeros((xmax-xmin, ymax-ymin))
	# print "Original Lesion Boundaries"
	# print xmin, xmax, ymin, ymax
	for i in lesion_scale:
		curMask = scipy.misc.imresize(origMask, i)
		mask_x, mask_y = curMask.shape
		cur_xmin = (xmax+xmin)/2 - mask_x/2
		cur_xmax = (xmax+xmin)/2 + mask_x/2
		cur_ymin = (ymax+ymin)/2 - mask_y/2
		cur_ymax = (ymax+ymin)/2 + mask_y/2
		print "Resized Lesion boundaries"
		print cur_xmin, cur_xmax, cur_ymin, cur_ymax
		upsampled_image, orig_no_canvas_im = upsample_data(curImage, cur_ymin, cur_ymax, cur_xmin, cur_xmax)
		augment_image_list = augment_data(upsampled_image, orig_no_canvas_im, scale, num_xpixels, num_ypixels)

		augment_image_list.append(upsampled_image)
		save_images(augment_image_list,str_name, it, i)


def upsample_data(curImage, ymin, ymax, xmin, xmax):
	curLesion = curImage[xmin:xmax, ymin:ymax]

	xdist = xmax - xmin
	ydist = ymax - ymin
	scale_len = np.max([xdist, ydist])

	if scale_len <= FINAL_DIM:
		num_scale = FINAL_DIM/scale_len
	else:
		num_scale = FINAL_DIM*1.0/scale_len

	scaled_image = scipy.ndimage.zoom(curLesion, num_scale, order=0)

	final_im = fit_canvas(scaled_image)

	return final_im, scaled_image



def augment_data(imData, orig_no_canvas_im, scale, num_xpixels, num_ypixels):
	rot1, rot2, rot3 = rotate_images(imData)
	trans_im1,trans_im2,trans_im3,trans_im4 = translation(imData, num_xpixels, num_ypixels)
	elast_xcomp, elast_ycomp = elastic_deformation(orig_no_canvas_im, scale)
	# noisy_im = noise_grayscale(orig_no_canvas_im)

	return [rot1, rot2, rot3, trans_im1,trans_im2,trans_im3,
				trans_im4, elast_xcomp, elast_ycomp]


def rotate_images(imData):
	rows, cols = imData.shape
	rotMat = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
	newImage_90 = cv2.warpAffine(imData,rotMat,(cols,rows))
	newImage_180 = cv2.warpAffine(newImage_90,rotMat,(cols,rows))
	newImage_270 = cv2.warpAffine(newImage_180,rotMat,(cols,rows))

	return newImage_90, newImage_180, newImage_270

def translation(imData, num_xpixels, num_ypixels):
	rows, cols = imData.shape
	rotMat1 = np.float32([[1,0,num_xpixels],[0,1, num_ypixels]])
	rotMat2 = np.float32([[1,0,-num_xpixels],[0,1, num_ypixels]])
	rotMat3 = np.float32([[1,0,num_xpixels],[0,1, -num_ypixels]])
	rotMat4 = np.float32([[1,0,-num_xpixels],[0,1, -num_ypixels]])
	trans_image1 = cv2.warpAffine(imData,rotMat1,(cols,rows))
	trans_image2 = cv2.warpAffine(imData,rotMat2,(cols,rows))
	trans_image3 = cv2.warpAffine(imData,rotMat3,(cols,rows))
	trans_image4 = cv2.warpAffine(imData,rotMat4,(cols,rows))

	return trans_image1,trans_image2,trans_image3,trans_image4 

def elastic_deformation(imData, scale):
	cols, rows = imData.shape
	resultImage_1 = cv2.resize(imData,(int(cols*scale),rows))
	resultImage_2 = cv2.resize(imData,(cols,int(rows*scale) ))

	final_elast_col = fit_canvas(resultImage_1)
	final_elast_row = fit_canvas(resultImage_2)

	return final_elast_col, final_elast_row

def noise_grayscale(imData):
	noisy_im = cv2.randu(imData, (0), (0.1))
	actual_im = fit_canvas(noisy_im)
	return noisy_im

def fit_canvas(resultImage):
	cols, rows = resultImage.shape
	col_low = FINAL_DIM/2 - cols/2
	col_high = FINAL_DIM/2 + cols/2
	row_low = FINAL_DIM/2 - rows/2
	row_high = FINAL_DIM/2 + rows/2

	if (col_high - col_low) != cols:
		col_high += 1
	if (row_high - row_low) != rows:
		row_high += 1

	final_im = np.zeros((FINAL_DIM, FINAL_DIM))
	final_im[col_low:col_high, row_low:row_high] = resultImage

	return final_im

def save_images(image_list, im_type, num, scale):
	image_labels = ['upsampled_' + im_type + '_' + str(scale) + '_rot1_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_rot2_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_rot3_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_trans1_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_trans2_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_trans3_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_trans4_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_elast_xcomp_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_elast_ycomp_' + str(num) + '.png',
			# 'upsampled_' + im_type + '_' + str(scale) + '_noisy_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_' + str(num) + '.png']

	np_labels = ['upsampled_' + im_type + '_' + str(scale) + '_rot1_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_rot2_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_rot3_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_trans1_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_trans2_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_trans3_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_trans4_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_elast_xcomp_' + str(num) + '.npy',
			'upsampled_' + im_type + '_' + str(scale) + '_elast_ycomp_' + str(num) + '.npy',
			# 'upsampled_' + im_type + '_' + str(scale) + '_noisy_' + str(num) + '.png',
			'upsampled_' + im_type + '_' + str(scale) + '_' + str(num) + '.npy']

	plt.figure()
	for i in xrange(len(image_list)):
		plt.imshow(image_list[i], cmap='gray')
		plt.colorbar()
		plt.savefig(IMAGE_DIR + image_labels[i])
		np.save(UPSAMPLED_DIR + np_labels[i], image_list[i])
		plt.clf()


def test_contour_creation():
	curImage = imageData[0, 0]
	ROIdata = CompROIdata[0][0]
	ymin = int(np.min(ROIdata['ROI_Y'][0,0]))
	ymax = int(np.max(ROIdata['ROI_Y'][0,0]))
	xmin = int(np.min(ROIdata['ROI_X'][0,0]))
	xmax = int(np.max(ROIdata['ROI_X'][0,0]))

	cnt_list = np.zeros((len(ROIdata['ROI_X'][0,0]), 2))
	cnt_list[:,0] = ROIdata['ROI_X'][0,0][:,0]
	cnt_list[:,1] = ROIdata['ROI_Y'][0,0][:,0]

	ctr = np.array(cnt_list).reshape((-1,2)).astype(np.int32)
	# newIm = cv2.drawContours(curImage.copy(), [ctr], 0, (0,255,0), 1)
	curImage  = np.float32(curImage)
	print curImage.dtype

	curImage = cv2.cvtColor(curImage, cv2.COLOR_BGR2YUV)
	curImage = cv2.equalizeHist(curImage)
	newIm = cv2.ellipse(curImage.copy(),(int((ymin+ymax)/2),int((xmin+xmax)/2) ),
				(np.max([int((xmax-xmin))/2, int((ymax-ymin))/2]), np.min([int((xmax-xmin))/2, int((ymax-ymin))/2])), 
				0.0, 0.0, 360.0, (0,255,255), 2)

	print np.max([int((xmax-xmin))/2, int((ymax-ymin))/2]), np.min([int((xmax-xmin))/2, int((ymax-ymin))/2])

	plt.imshow(newIm, cmap='gray')
	plt.savefig('noupsampled_image_with_contour.png')


def create_nonsampled_data_mt():
	num_images = imageData.shape[1] #8

	print "Creating threads for dataset creation"
	P = Pool(processes=1)
	P.map(create_contours_on_plot, (i for i in range(0, num_images )) )

def create_contours_on_plot(it):
	print "Starting contour creations for Image {0}".format(it+1)
	curImage = imageData[0,it]
	# print curImage.shape
	ROIdata = CompROIdata[0][it]

	# ymin = int(np.min(ROIdata['ROI_X'][0,0][0]))
	# ymax = int(np.max(ROIdata['ROI_X'][0,0][0]))
	# xmin = int(np.min(ROIdata['ROI_X'][0,0][0]))
	# xmax = int(np.max(ROIdata['ROI_X'][0,0][0]))
	ymin = int(np.min(ROIdata['ROI_X'][0,0]))
	ymax = int(np.max(ROIdata['ROI_X'][0,0]))
	xmin = int(np.min(ROIdata['ROI_Y'][0,0]))
	xmax = int(np.max(ROIdata['ROI_Y'][0,0]))

	lesion_scale = np.linspace(0.6, 0.8, num=3)
	booundary_scale = np.linspace(0.9, 1.1, num=3)
	external_scale = np.linspace(1.3, 1.5, num=3)

	print "Finding Ellipse Boundaries"
	print ymin, ymax, xmin, xmax
	cropped_lesion, new_xmin, new_xmax, new_ymin, new_ymax = find_cropped_lesion(curImage,
																 xmin, xmax, ymin, ymax)
	origMask = np.zeros((xmax- xmin, ymax- ymin))
	if (xmax- xmin) <= 15 and (ymax- ymin) <= 15:
		# print("Image {0}:{1} has not been added to dataset".format(str_name, im_num))
		return

	print "Creating ellipses for lesion"
	create_ellipses(cropped_lesion, origMask, lesion_scale, new_xmin, new_xmax, new_ymin, new_ymax, it, 'internal')
	create_ellipses(cropped_lesion, origMask, booundary_scale, new_xmin, new_xmax, new_ymin, new_ymax, it, 'boundary')
	create_ellipses(cropped_lesion, origMask, external_scale, new_xmin, new_xmax, new_ymin, new_ymax, it, 'external')

	# ymin = int(np.min(ROIdata['Normal_ROIy'][0][0]))
	# ymax = int(np.max(ROIdata['Normal_ROIy'][0][0]))
	# xmin = int(np.min(ROIdata['Normal_ROIx'][0][0]))
	# xmax = int(np.max(ROIdata['Normal_ROIx'][0][0]))
	ymin = int(np.min(ROIdata['Normal_ROIy'][0,0]))
	ymax = int(np.max(ROIdata['Normal_ROIy'][0,0]))
	xmin = int(np.min(ROIdata['Normal_ROIx'][0,0]))
	xmax = int(np.max(ROIdata['Normal_ROIx'][0,0]))

	cropped_lesion, new_xmin, new_xmax, new_ymin, new_ymax = find_cropped_lesion(curImage,
																 xmin, xmax, ymin, ymax)
	origMask = np.zeros((xmax- xmin, ymax- ymin))
	print origMask.shape
	create_ellipses(cropped_lesion, origMask, lesion_scale, new_xmin, new_xmax, new_ymin, new_ymax, it, 'normal')


def find_cropped_lesion(curImage, xmin, xmax, ymin, ymax):
	xmid = int((xmin + xmax)/2)
	ymid = int((ymin + ymax)/2)

	xcol_low = xmid - FINAL_DIM/2
	xcol_high = xmid + FINAL_DIM/2
	ycol_low = ymid - FINAL_DIM/2
	ycol_high = ymid + FINAL_DIM/2

	if xcol_high - xcol_low != FINAL_DIM:
		xcol_high += FINAL_DIM - (xcol_high - xcol_low)
	if ycol_high - ycol_low != FINAL_DIM:
		ycol_high += FINAL_DIM - (ycol_high - ycol_low)

	cropped_lesion = curImage[xcol_low:xcol_high, ycol_low:ycol_high]

	new_xmin = xmin - xcol_low
	new_xmax = xmax - xcol_low
	new_ymin = ymin - ycol_low
	new_ymax = ymax - ycol_low

	return cropped_lesion, new_xmin, new_xmax, new_ymin, new_ymax



def create_ellipses(cropped_lesion, origMask, mask_range, xmin, xmax, ymin, ymax, im_num, str_name):
	num_xpixels = 20
	num_ypixels = 10
	scale = 0.75
	for k in mask_range:
		curMask = scipy.misc.imresize(origMask, k)
		mask_x, mask_y = curMask.shape

		if mask_x > FINAL_DIM:
			mask_x = FINAL_DIM
		if mask_y > FINAL_DIM:
			mask_y = FINAL_DIM

		xmid = int((xmin+xmax)/2)
		ymid = int((ymin+ymax)/2)

		cur_xmin = xmid - mask_x/2
		cur_xmax = xmid + mask_x/2
		cur_ymin = ymid - mask_y/2
		cur_ymax = ymid + mask_y/2

		maj_axis = np.max([int((cur_xmax - cur_xmin)/2), int((cur_ymax - cur_ymin)/2)])
		min_axis = np.max([int((cur_xmax - cur_xmin)/2), int((cur_ymax - cur_ymin)/2)])

		# newIm = cv2.ellipse(cropped_lesion.copy(),(xmid,ymid), (maj_axis, min_axis), 
		# 	0.0, 0.0, 360.0, (0,255,255), 1)
		newIm = cropped_lesion[cur_xmin:cur_xmax, cur_ymin:cur_ymax]
		newIm_with_canvas = fit_canvas(newIm)
		augment_image_list = augment_data(newIm_with_canvas, newIm, scale, num_xpixels, num_ypixels)

		augment_image_list.append(newIm_with_canvas)
		save_images_nonupsampled(augment_image_list,str_name, im_num, k)

		# plt.imshow(newIm, cmap='gray')
		# plt.savefig(NO_UPSAMPLED_DIR + 'cropped_lesion_' + str(im_num) + '_with_' + str_name + '_contour_scale_' + str(k) + '.png')

def save_images_nonupsampled(image_list, im_type, num, scale):
	image_labels = ['cropped_lesion_' + im_type + '_' + str(scale) + '_rot1_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_rot2_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_rot3_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans1_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans2_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans3_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans4_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_elast_xcomp_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_elast_ycomp_' + str(num) + '.png',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_' + str(num) + '.png']

	np_labels = ['cropped_lesion_' + im_type + '_' + str(scale) + '_rot1_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_rot2_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_rot3_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans1_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans2_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans3_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_trans4_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_elast_xcomp_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_elast_ycomp_' + str(num) + '.npy',
			'cropped_lesion_' + im_type + '_' + str(scale) + '_' + str(num) + '.npy']

	plt.figure()
	for i in xrange(len(image_list)):
		plt.imshow(image_list[i], cmap='gray')
		plt.colorbar()
		plt.savefig(MRI_NOUPSAMPLED_IMAGE_DIR + image_labels[i])
		np.save(MRI_NOUPSAMPLED_DIR + np_labels[i], image_list[i])
		plt.clf()


def create_patched_data_mt():
	num_images = imageData.shape[1] #8
	print "Creating threads for patched data creation"
	P = Pool(processes=8)
	# num_images = 9
	P.map(create_patched_data, (i for i in xrange(0, num_images)) )
	create_patched_data(num_images)



def create_patched_data(it):
	image_rad = 5
	curImage = imageData[0,it]
	# print curImage.shape

	ROIdata = CompROIdata[0,it]

	ymin = int(np.min(ROIdata['ROI_Y'][0,0]))
	ymax = int(np.max(ROIdata['ROI_Y'][0,0]))
	xmin = int(np.min(ROIdata['ROI_X'][0,0]))
	xmax = int(np.max(ROIdata['ROI_X'][0,0]))


	if not ((xmax - xmin) >= 12 and (ymax-ymin) >= 12):
		return

	yloc = ROIdata['ROI_Y'][0,0].astype(int)
	xloc = ROIdata['ROI_X'][0,0].astype(int)

	print("The y coordinates of the lesion tissue is: {0}".format(yloc))
	print("The x coordinates of the lesion tissue is: {0}".format(xloc))

	center_coords = []
	for i in range(len(xloc)):
		center_coords.append([xloc[i], yloc[i]])

	image_poly = Polygon(center_coords)
	image_poly = image_poly.buffer(0)

	num_boundary_images = 20
	test_patch_list = []

	minx_lesion, miny_lesion, maxx_lesion, maxy_lesion = image_poly.bounds


	counter = 0
	loop_count = 0
	while counter < num_boundary_images: 

		randX_pt = int(random.uniform(minx_lesion, maxx_lesion))
		randY_pt = int(random.uniform(miny_lesion, maxy_lesion))

		xSmall = randX_pt-image_rad
		xLarge = randX_pt+image_rad
		ySmall = randY_pt-image_rad
		yLarge = randY_pt+image_rad

		test_patch = Polygon([(xSmall,ySmall),(xSmall, yLarge), (xLarge, ySmall),(xLarge, yLarge)])
		test_patch = test_patch.buffer(0)

		lesion_area_scaled = (image_poly.intersection(test_patch)).area/test_patch.area

		# print lesion_area_scaled

		if image_poly.intersects(test_patch) and lesion_area_scaled < 0.6:
			# print xSmall, xLarge, ySmall, yLarge
			# print curImage.shape
			cur_patched_image = curImage[xSmall:xLarge, ySmall:yLarge]
			# print cur_patched_image.shape
			final_im = fit_canvas(cur_patched_image)
			# final_im = cur_patched_image
			cur_filename = 'patched_lesion_boundary_' + str(it) + '_sample_' + str(counter)
			save_patched_data(cur_filename,final_im)
			# test_patch_list.append(test_patch)
			# num_boundary_images += 1
			counter += 1
			# print num_boundary_images
			#code to save images to file

	# return
	yloc_norm = ROIdata['Normal_ROIy'][0,0].astype(int)
	xloc_norm = ROIdata['Normal_ROIx'][0,0].astype(int)

	print "The y coordinates of the normal tissue is: {0}".format(yloc_norm)
	print "The x coordinates of the normal tissue is: {0}".format(xloc_norm)


	center_coords_norm = []
	for i in range(len(xloc_norm)):
		center_coords_norm.append([xloc_norm[i][0], yloc_norm[i][0]])

	image_poly_norm = Polygon(center_coords_norm)
	image_poly_norm = image_poly_norm.buffer(0)


	create_inside_outside_patches(curImage, num_boundary_images, image_poly, image_poly_norm, image_rad, it)


def create_inside_outside_patches(curImage, num_boundary_images, image_poly, image_poly_norm, image_rad, im_num):
	minx_lesion, miny_lesion, maxx_lesion, maxy_lesion = image_poly.bounds
	minx_normal, miny_normal, maxx_normal, maxy_normal = image_poly_norm.bounds

	counter = 0
	loop_count = 0
	while counter < num_boundary_images:
		randX_pt = int(random.uniform(minx_lesion, maxx_lesion))
		randY_pt = int(random.uniform(miny_lesion, maxy_lesion))

		pnt = Point(randX_pt, randY_pt)
		if loop_count > 1000:
			break
		if image_poly.contains(pnt):
			xSmall = randX_pt-image_rad
			xLarge = randX_pt+image_rad
			ySmall = randY_pt-image_rad
			yLarge = randY_pt+image_rad
			# counter += 1
			test_patch = Polygon([(xSmall,ySmall),(xSmall, yLarge), (xLarge, ySmall),(xLarge, yLarge)])
			test_patch = test_patch.buffer(0)
			
			lesion_area_scaled = (image_poly.intersection(test_patch)).area/test_patch.area

			if image_poly.contains(test_patch) or lesion_area_scaled >= 0.6:
				cur_patched_image = curImage[xSmall:xLarge, ySmall:yLarge]
				final_im = fit_canvas(cur_patched_image)
				# final_im = cur_patched_image
				cur_filename = 'patched_lesion_internal_' + str(im_num) + '_sample_' + str(counter)
				save_patched_data(cur_filename,final_im)
				counter += 1
			else:
				loop_count += 1


	print("Done with Internal Patches for image {0}!".format(im_num))

	counter = 0
	loop_count = 0
	while counter < num_boundary_images:
		randX_pt = int(random.uniform(minx_normal, maxx_normal))
		randY_pt = int(random.uniform(miny_normal, maxy_normal))

		pnt = Point(randX_pt, randY_pt)
		if loop_count > 1000:
			break
		if image_poly_norm.contains(pnt):
			xSmall = randX_pt-image_rad
			xLarge = randX_pt+image_rad
			ySmall = randY_pt-image_rad
			yLarge = randY_pt+image_rad
			# counter += 1
			test_patch = Polygon([(xSmall,ySmall),(xSmall, yLarge), (xLarge, ySmall),(xLarge, yLarge)])
			if image_poly_norm.contains(test_patch):
				cur_patched_image = curImage[xSmall:xLarge, ySmall:yLarge]
				final_im = fit_canvas(cur_patched_image)
				# final_im = cur_patched_image
				cur_filename = 'patched_lesion_external_' + str(im_num) + '_sample_' + str(counter)
				save_patched_data(cur_filename,final_im)
				counter += 1
			else:
				loop_count += 1

	print("Done with External Patches for image {0}!".format(im_num))



def save_patched_data(filename, image):
	plt.figure()
	plt.imshow(image, cmap='gray')
	plt.colorbar()
	plt.savefig(PATCHED_DATA_IMAGES + filename + '.png')
	np.save(PATCHED_DATA_DIR + filename + '.npy', image)
	plt.close()


def main():
	load_data(DATA_PATH)
	# test_contour_creation()
	# create_dataset_mt()
	# create_contours_on_plot()
	# create_nonsampled_data_mt()
	create_patched_data_mt()



if __name__ == "__main__":
	main()

