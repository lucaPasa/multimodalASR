import tensorflow as tf
from os import listdir
from os.path import isfile, join
import numpy as np
from math import floor
from math import ceil
import tensorflow as tf
from os import listdir
from os.path import isfile, join


class dataManager:
    def __init__(self, single_audio_frame_size=123, single_video_frame_size=134):
        self.batch_size_ph = tf.placeholder(tf.int64, shape=[])
        self.n_epoch_ph = tf.placeholder(tf.int64, shape=[])
        self.buffer_size_ph = tf.placeholder(tf.int64, shape=[])
        self.single_audio_frame_size = single_audio_frame_size
        self.single_video_frame_size = single_video_frame_size
        self.n_sample = 0

    def get_dataset(self, folderPath):
        fileList = [join(folderPath, f) for f in listdir(folderPath) if isfile(join(folderPath, f))]
        dataset = tf.data.TFRecordDataset(fileList)
        dataset = dataset.map(self.read_data_format)
        self.n_sample = len(fileList)
        return dataset

    def get_iterator(self, dataset):
        dataset = dataset.shuffle(self.buffer_size_ph).repeat(self.n_epoch_ph)
        batch_dataset = dataset.padded_batch(self.batch_size_ph, padded_shapes=([], [None, None],[], [None, None], [], [None]))

        iterator = batch_dataset.make_initializable_iterator()
        return batch_dataset, iterator

    def read_data_format(self, sample):
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(sample,
                                                                           context_features=dict(
                                                                               audio_length=tf.FixedLenFeature((),
                                                                                                               dtype=tf.int64),
                                                                               video_length=tf.FixedLenFeature((),
                                                                                                               dtype=tf.int64),
                                                                               label_length=tf.FixedLenFeature((),
                                                                                                               dtype=tf.int64)),
                                                                           sequence_features={
                                                                               "audio_feat": tf.FixedLenSequenceFeature(
                                                                                   (self.single_audio_frame_size),
                                                                                   dtype=tf.float32),
                                                                               "video_feat": tf.FixedLenSequenceFeature(
                                                                                   (self.single_video_frame_size),
                                                                                   dtype=tf.float32),
                                                                               "labels": tf.FixedLenSequenceFeature((),
                                                                                                                    dtype=tf.float32)})

        return context_parsed['audio_length'], sequence_parsed['audio_feat'], context_parsed['video_length'], \
               sequence_parsed['video_feat'], context_parsed['label_length'], sequence_parsed['labels']


# (,(self.n_batch,self.single_frame_size))


'''

'''
def align_audio_video(audio_seq,video_seq):
    #audio and video seq spam the same time, so by taking the audio len unit e dividing the time by the audio length we can easily compute the video time step position
    #we know that the video sample frequenci is lower than audio frequecy

    p_ceil_floor=float(audio_seq.shape[0])/video_seq.shape[0]%1
    ceil_n_audio_frame_per_audio=int(ceil(float(audio_seq.shape[0])/video_seq.shape[0]))
    floor_n_audio_frame_per_audio = int(floor(float(audio_seq.shape[0]) / video_seq.shape[0]))
    new_video=[]
    for video_frame in video_seq:
        for i in range(np.random.choice([ceil_n_audio_frame_per_audio, floor_n_audio_frame_per_audio], p=[p_ceil_floor, 1 - p_ceil_floor])):
            new_video.append(video_frame)
    new_video=np.asarray(new_video)
    if new_video.shape[0] != audio_seq.shape[0] and new_video.shape[0] > audio_seq.shape[0]:
        new_video=new_video[0:audio_seq.shape[0],:]
    else:
        for _ in range(audio_seq.shape[0] - new_video.shape[0]):
            new_video=np.vstack([new_video,new_video[-1]])

    return new_video

def video_batch_align(audio_batch, video_batch):
    new_video_batch=[]
    for i,(audio_seq, video_seq) in enumerate(zip(audio_batch, video_batch)):
        new_video_seq=align_audio_video(audio_seq,video_seq)
        new_video_batch.append(new_video_seq)

    return np.asarray(new_video_batch)


if __name__ == '__main__':
    path = '/home/storage/Data/MULTI_GRID/multiModalTfRec/TRAIN_CTC_SENTENCES/'
    n_batch = 4
    n_epoch = 5
    buffer_size = 2

    dm = dataManager()

    ds = dm.get_dataset(path)

    ds, it = dm.get_iterator(ds)
    with tf.Session() as sess:
        sess.run(it.initializer,
                 feed_dict={dm.batch_size_ph: n_batch, dm.n_epoch_ph: n_epoch, dm.buffer_size_ph: 1000000})
        while (True):
            try:

                x_a_len, x_a, x_v_len, x_v, y_len, y = sess.run(it.get_next())
                x_v=video_batch_align(x_a,x_v)
                print x_a.shape
                print x_v.shape
                # print x_v
                raw_input()
                # res = sess.run(it.get_next())


            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break

    # dataset = tf.data.Dataset.range(5)
    # dataset=dataset.shuffle(2).repeat(2)
    # iterator = dataset.make_initializable_iterator()
    # next_element = iterator.get_next()
    #
    # with tf.Session() as sess:
    #
    #     sess.run(iterator.initializer)
    #     while(True):
    #         try:
    #             print sess.run(next_element)
    #
    #         except tf.errors.OutOfRangeError:
    #             print("End of dataset")
    #             break
