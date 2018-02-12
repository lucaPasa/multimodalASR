import glob
import numpy as np
import tensorflow as tf


def serialize_sequence(audio_sequence, video_sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    audio_sequence_length = len(audio_sequence)
    video_sequence_length = len(video_sequence)
    labels_length = len(labels)

    ex.context.feature["audio_length"].int64_list.value.append(audio_sequence_length)
    ex.context.feature["video_length"].int64_list.value.append(video_sequence_length)
    ex.context.feature["label_length"].int64_list.value.append(labels_length)

    # Feature lists for the two sequential features of our example
    fl_audio_feat = ex.feature_lists.feature_list["audio_feat"]
    fl_video_feat = ex.feature_lists.feature_list["video_feat"]
    fl_labels = ex.feature_lists.feature_list["labels"]

    for audio_feat in audio_sequence:
        fl_audio_feat.feature.add().float_list.value.extend(audio_feat)

    for video_feat in video_sequence:
        fl_video_feat.feature.add().float_list.value.extend(video_feat)

    for label in labels:
        fl_labels.feature.add().float_list.value.append(label)

    return ex


# load dataset mean and std
# dataset_audio_base_mean=np.load('dataset_audio_base_mean.npy')
# dataset_audio_base_std=np.load('dataset_audio_base_stdev.npy')

dataset_multi_audio_mean = np.load('dataset_multi_audio_mean.npy')
dataset_multi_audio_std = np.load('dataset_multi_audio_stdev.npy')

dataset_video_mean = np.load('dataset_video_mean.npy')
dataset_video_std = np.load('dataset_video_stdev.npy')

# destination folders
train_dir = '/home/storage/Data/MULTI_GRID/multiModalTfRec/TRAIN_CTC_SENTENCES/'
val_dir = '/home/storage/Data/MULTI_GRID/multiModalTfRec/VAL_CTC_SENTENCES/'
test_dir = '/home/storage/Data/MULTI_GRID/multiModalTfRec/TEST_CTC_SENTENCES/'

f = open('./dictionary.txt', 'r')
dictionary = f.read()

phonemes = dictionary.replace('\n', ' ').split(' ')
phonemes = [ph for ph in sorted(set(phonemes)) if ph is not '']
print('Number of phonemes = ', len(phonemes))
print(phonemes)

features_file_list_audio = sorted(glob.glob('/home/storage/Data/MULTI_GRID/s*/multi_audio/*.csv'))
features_file_list_video = sorted(glob.glob('/home/storage/Data/MULTI_GRID/s*/video/*.txt'))

assert len(features_file_list_audio) == len(features_file_list_video), "#multi_audio != #video"
print('Total number of files = {}'.format(
    len(features_file_list_audio)))  # it has to be equal to len(features_file_list_video)

# prepare indices for cross validation
indices = np.arange(len(features_file_list_audio))  # same of indices_video = np.arange(len(features_file_list_video))

np.random.seed(3)
np.random.shuffle(indices)

# cross validation split
train_percent = 0.6
val_percent = 0.2
test_percent = 0.2

print(len(features_file_list_audio))

num_sentences_train = int(len(features_file_list_audio) * train_percent)
num_sentences_val = int(len(features_file_list_audio) * val_percent)
num_sentences_test = len(features_file_list_audio) - num_sentences_train - num_sentences_val

print('num sentences train = ', num_sentences_train)
print('num sentences val   = ', num_sentences_val)
print('num sentences test  = ', num_sentences_test)

train_indices = indices[:num_sentences_train]
val_indices = indices[num_sentences_train:(num_sentences_train + num_sentences_val)]
test_indices = indices[(num_sentences_val + num_sentences_train):]

train_counter = 0
val_counter = 0
test_counter = 0

for file_index, (csv_file_audio, txt_file_video) in enumerate(zip(features_file_list_audio, features_file_list_video)):

    print('audio {:s}, video {:s}'.format(csv_file_audio, txt_file_video))

    features_audio = np.loadtxt(csv_file_audio, delimiter=',')
    features_video = np.loadtxt(txt_file_video)

    print features_audio.shape
    print features_video.shape

    # label path
    labels_file = csv_file_audio.replace('/multi_audio/', '/transcription/').replace('.csv', '.transcription')
    f = open(labels_file, 'r')

    labels = f.read()
    labels = labels.replace('\n', '').replace('SP', '').split(',')
    labels = [lab for lab in labels if lab is not '']
    print('labels : ', labels)
    labels = [phonemes.index(ph) for ph in labels]
    print('labels : ', labels)
    labels = np.asarray(labels)
    print(labels.shape)
    print('')
    features_audio = np.subtract(features_audio, dataset_multi_audio_mean) / dataset_multi_audio_std
    features_video = np.subtract(features_video, dataset_video_mean) / dataset_video_std
    print features_video.shape
    raw_input()

    if file_index in train_indices:
        sentence_file = train_dir + 'sequence_full_{:05d}.tfrecords'.format(train_counter)
        train_counter += 1
    if file_index in val_indices:
        sentence_file = val_dir + 'sequence_full_{:05d}.tfrecords'.format(val_counter)
    val_counter += 1
    if file_index in test_indices:
        sentence_file = test_dir + 'sequence_full_{:05d}.tfrecords'.format(test_counter)
        test_counter += 1

    fp = open(sentence_file, 'w')
    writer = tf.python_io.TFRecordWriter(fp.name)

    serialized_sentence = serialize_sequence(features_audio, features_video, labels)
    # write to tfrecord
    writer.write(serialized_sentence.SerializeToString())
    writer.close()

    fp.close()
