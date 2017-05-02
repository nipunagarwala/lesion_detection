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

DATA_PATH = '/scratch/users/nipuna1/lesion_data/liver_lesions.mat'
DATASET_DIR = '/scratch/users/nipuna1/lesion_data/lesion_contour_dataset/'
FINAL_DIM = 100

imageData = None
CompROIdata = None

def parseCommandLine():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-p', choices = ["train", "test", "dev"], type = str,
						dest = 'train', required = True, help = 'Training or Testing phase to be run')

	parser.add_argument('-ckpt', dest='ckpt_dir', default='/scratch/users/nipuna1/lesion_ckpt', 
									type=str, help='Set the checkpoint directory')
	parser.add_argument('-data', dest='data_dir', default=DATASET_DIR, 
									type=str, help='Set the data directory')

	args = parser.parse_args()
	return args

def load_data(data_path):
	global imageData
	global CompROIdata
	data = sio.loadmat(data_path)

	imageData = data['NewImage']
	CompROIdata = data['ROIdata']
	print 'Loaded Data'
	# print data.keys()
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
		augment_image_list  = augment_data(upsampled_image, orig_no_canvas_im, scale, num_xpixels, num_ypixels)

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

	for i in xrange(len(image_list)):
		plt.figure()
		plt.imshow(image_list[i], cmap='gray')
		plt.colorbar()
		plt.savefig(DATASET_DIR + image_labels[i])


def test_contour_creation():
	curImage = imageData[0, 0]
	ROIdata = CompROIdata[0][0]
	ymin = int(np.min(ROIdata['ROI_X'][0,0][0]))
	ymax = int(np.max(ROIdata['ROI_X'][0,0][0]))
	xmin = int(np.min(ROIdata['ROI_Y'][0,0][0]))
	xmax = int(np.max(ROIdata['ROI_Y'][0,0][0]))

	cnt_list = np.zeros((len(ROIdata['ROI_X'][0,0][0]), 2))
	cnt_list[:,0] = ROIdata['ROI_X'][0,0][0]
	cnt_list[:,1] = ROIdata['ROI_Y'][0,0][0]

	ctr = np.array(cnt_list).reshape((-1,2)).astype(np.int32)
	# newIm = cv2.drawContours(curImage.copy(), [ctr], 0, (0,255,0), 1)
	newIm = cv2.ellipse(curImage.copy(),(int((ymin+ymax)/2),int((xmin+xmax)/2) ),
				(np.max([int((xmax-xmin))/2, int((ymax-ymin))/2]), np.min([int((xmax-xmin))/2, int((ymax-ymin))/2])), 
				0.0, 0.0, 360.0, (0,255,255), 2)

	# upsampled_image, orig_no_canvas_im = upsample_data(newIm, ymin, ymax, xmin, xmax)

	print np.max([int((xmax-xmin))/2, int((ymax-ymin))/2]), np.min([int((xmax-xmin))/2, int((ymax-ymin))/2])

	plt.imshow(newIm, cmap='gray')
	plt.savefig(DATASET_DIR + 'upsampled_image_with_contour.png')


def create_contours_on_plot():
	num_images = 1 #imageData.shape[1]
	for i in xrange(0, num_images):
		print "Starting contour creations for Image {0}".format(i+1)
		curImage = imageData[0,i]
		ROIdata = CompROIdata[0][i]

		ymin = int(np.min(ROIdata['ROI_X'][0,0][0]))
		ymax = int(np.max(ROIdata['ROI_X'][0,0][0]))
		xmin = int(np.min(ROIdata['ROI_Y'][0,0][0]))
		xmax = int(np.max(ROIdata['ROI_Y'][0,0][0]))

		lesion_scale = np.linspace(0.6, 0.8, num=3)
		booundary_scale = np.linspace(0.9, 1.1, num=3)
		external_scale = np.linspace(1.3, 1.5, num=3)

		print "Finding Ellipse Boundaries"
		cropped_lesion, new_xmin, new_xmax, new_ymin, new_ymax = find_cropped_lesion(curImage,
																	 xmin, xmax, ymin, ymax)
		origMask = np.zeros((xmax- xmin, ymax- ymin))

		print "Creating ellipses for lesion"
		create_ellipses(cropped_lesion, origMask, lesion_scale, new_xmin, new_xmax, new_ymin, new_ymax, i, 'internal')
		create_ellipses(cropped_lesion, origMask, booundary_scale, new_xmin, new_xmax, new_ymin, new_ymax, i, 'boundary')
		create_ellipses(cropped_lesion, origMask, external_scale, new_xmin, new_xmax, new_ymin, new_ymax, i, 'external')


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
	for k in mask_range:
		curMask = scipy.misc.imresize(origMask, k)
		mask_x, mask_y = curMask.shape

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
		newIm = fit_canvas(newIm)
		plt.imshow(newIm, cmap='gray')
		plt.savefig(DATASET_DIR + 'cropped_lesion_' + str(im_num) + '_with_' + str_name + '_contour_scale_' + str(k) + '.png')




def main():
	load_data(DATA_PATH)
	# test_contour_creation()
	create_dataset_mt()
	# create_contours_on_plot()



if __name__ == "__main__":
	main()

