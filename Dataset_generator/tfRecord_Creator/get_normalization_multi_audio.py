from __future__ import print_function
from __future__ import division

import glob
import numpy as np

features_file_list = sorted(glob.glob('/home/storage/Data/MULTI_GRID/s*/multi_audio/*.csv'))
#features_file_list = sorted(glob.glob('/home/storage/Data/MULTI_GRID/s*/base_audio/*.csv'))

print('Total number of files = {}'.format(len(features_file_list)))

features=[]

for file_index, csv_file in enumerate(features_file_list):
	print(file_index,csv_file)
	data=np.loadtxt(csv_file,delimiter=',')
	features.append(data)

features = np.concatenate(features)
print(features.shape)

dataset_mean = np.mean(features,axis=0)
dataset_stdev = np.std(features,axis=0)

print(dataset_mean.shape)
print(dataset_stdev.shape)

mean_file = 'dataset_multi_audio_mean.npy'
std_file = 'dataset_multi_audio_stdev.npy'

np.save(mean_file,dataset_mean)
np.save(std_file,dataset_stdev)