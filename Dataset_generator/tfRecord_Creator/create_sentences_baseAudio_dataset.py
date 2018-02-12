from __future__ import print_function
from __future__ import division

import glob
import numpy as np
import tensorflow as tf

def serialize_sequence(audio_sequence,labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(audio_sequence)
    labels_length = len(labels)

    ex.context.feature["feat_length"].int64_list.value.append(sequence_length)
    ex.context.feature["label_length"].int64_list.value.append(labels_length)

    # Feature lists for the two sequential features of our example
    fl_audio_feat = ex.feature_lists.feature_list["audio_feat"]
    fl_audio_labels = ex.feature_lists.feature_list["audio_labels"]

    for audio_feat in audio_sequence:
    	fl_audio_feat.feature.add().float_list.value.extend(audio_feat)

    for label in labels:
    	fl_audio_labels.feature.add().float_list.value.append(label)
    	
    return ex

# load dataset mean and std
dataset_mean=np.load('dataset_audio_base_mean.npy')
dataset_std=np.load('dataset_audio_base_stdev.npy')

# destination folders
train_dir='/home/storage/Data/MULTI_GRID/soloBaseAudioTfRec/TRAIN_CTC_SENTENCES/'
val_dir='/home/storage/Data/MULTI_GRID/soloBaseAudioTfRec/VAL_CTC_SENTENCES/'
test_dir='/home/storage/Data/MULTI_GRID/soloBaseAudioTfRec/TEST_CTC_SENTENCES/'

# load dictionary
f=open('./dictionary.txt','r')
dictionary=f.read()

phonemes=dictionary.replace('\n',' ').split(' ')
phonemes = [ph for ph in sorted(set(phonemes)) if ph is not '']
print('Number of phonemes = ',len(phonemes))
print(phonemes)


# to import all the csv files in the GRID folder

features_file_list = sorted(glob.glob('/home/storage/Data/MULTI_GRID/s*/base_audio/*.csv'))

print('Total number of files = {}'.format(len(features_file_list)))

# prepare indeces for cross validation
indeces=np.arange(len(features_file_list))
np.random.seed(3)
np.random.shuffle(indeces)

# cross validation split
train_percent=0.6
val_percent=0.2
test_percent=0.2

print(len(features_file_list))

num_sentences_train = int(len(features_file_list)*train_percent)
num_sentences_val = int(len(features_file_list)*val_percent)
num_sentences_test = len(features_file_list) - num_sentences_train - num_sentences_val

print('num sentences train = ',num_sentences_train)
print('num sentences val   = ',num_sentences_val)
print('num sentences test  = ',num_sentences_test)

train_indeces = indeces[:num_sentences_train]
val_indeces = indeces[num_sentences_train:(num_sentences_train+num_sentences_val)]
test_indeces = indeces[(num_sentences_val+num_sentences_train):]

train_counter=0
val_counter=0
test_counter=0

# creation sentences fot ctc 
for file_index, csv_file in enumerate(features_file_list):

	print('file {:s}'.format(csv_file))

	features=np.loadtxt(csv_file,delimiter=',')

	labels_file=csv_file.replace('/base_audio/','/transcription/').replace('.csv','.transcription')

	f = open(labels_file,'r')
	labels=f.read()
	labels = labels.replace('\n','').replace('SP','').split(',')
	labels = [lab for lab in labels if lab is not '']
	print('labels : ',labels)
	labels = [phonemes.index(ph) for ph in labels]
	print('labels : ',labels)
	labels=np.asarray(labels)
	print(labels.shape)
	print('')

	features = np.subtract(features,dataset_mean) / dataset_std

	if file_index in train_indeces:
		sentence_file=train_dir+'sequence_full_{:05d}.tfrecords'.format(train_counter)
		train_counter+=1
	if file_index in val_indeces:
		sentence_file=val_dir+'sequence_full_{:05d}.tfrecords'.format(val_counter) 
		val_counter+=1
	if file_index in test_indeces:
		sentence_file=test_dir+'sequence_full_{:05d}.tfrecords'.format(test_counter) 
		test_counter+=1

	fp = open(sentence_file,'w')
	writer = tf.python_io.TFRecordWriter(fp.name)

	serialized_sentence = serialize_sequence(features,labels)

	# write to tfrecord
	writer.write(serialized_sentence.SerializeToString())
	writer.close()

	fp.close()