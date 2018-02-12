from __future__ import print_function
from __future__ import division

import glob
import numpy as np

features_file_list = sorted(glob.glob('/home/storage/Data/MULTI_GRID/s*/video/*.txt'))
#features_file_list = sorted(glob.glob('/home/lpasa/Data/MULTI_GRID/s*/video/*.txt'))
print('Total number of files = {}'.format(len(features_file_list)))

features=[]

for file_index, txt_file in enumerate(features_file_list):
	print(file_index,txt_file)
	data=np.loadtxt(txt_file)
	features.append(data)

features = np.concatenate(features)
print(features.shape)

dataset_mean = np.mean(features,axis=0)
dataset_stdev = np.std(features,axis=0)

print(dataset_mean.shape)
print(dataset_stdev.shape)
print(dataset_stdev)

mean_file = 'dataset_video_mean.npy'
std_file = 'dataset_video_stdev.npy'

np.save(mean_file,dataset_mean)
np.save(std_file,dataset_stdev)
