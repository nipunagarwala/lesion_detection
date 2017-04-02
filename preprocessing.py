import numpy as np 
import os
import sys
import scipy.io as sio
from argparse import ArgumentParser



def parseCommandLine():
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-p', choices = ["train", "test", "dev"], type = str,
						dest = 'train', required = True, help = 'Training or Testing phase to be run')

	parser.add_argument('-ckpt', dest='ckpt_dir', default='/scratch/users/nipuna1/lesion_ckpt', 
									type=str, help='Set the checkpoint directory')
	parser.add_argument('-data', dest='data_dir', default='/scratch/users/nipuna1/lesion_data/liver_lesions.mat', 
									type=str, help='Set the data directory')

	args = parser.parse_args()
	return args

def load_data(data_path):
	data = sio.loadmat(data_path)
	print(data)
	print data.keys()



def main():
	print "HEY"
	load_data('/scratch/users/nipuna1/lesion_data/liver_lesions.mat')



if __name__ == "__main__":
	main()

