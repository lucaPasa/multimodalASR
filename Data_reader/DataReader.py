import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


num_examples_training=594
num_epochs=10
batch_size=10

def read_my_file_format(filename_queue, feat_dimension=123):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized_example,
                                                                       context_features={
                                                                           "feat_length": tf.FixedLenFeature([],dtype=tf.int64),
                                                                           "label_length": tf.FixedLenFeature([],dtype=tf.int64)},
                                                                       sequence_features={
                                                                           "audio_feat": tf.FixedLenSequenceFeature([feat_dimension], dtype=tf.float32),
                                                                           "audio_labels": tf.FixedLenSequenceFeature([], dtype=tf.float32)})


    return tf.to_int32(context_parsed['feat_length']), tf.to_int32(context_parsed['label_length']),sequence_parsed['audio_feat'], tf.to_int32(sequence_parsed['audio_labels']), key


def get_input_batch(dataPath, num_epochs=10 ,batch_size=10,feat_dimension=123,num_examples=594, queueCapacity=5000,queueThread=5):
    fileList = [dataPath+'sequence_full_{:05d}.tfrecords'.format(i) for i in
                      range(num_examples)]
    for f in fileList:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    fileQueue = tf.train.string_input_producer(fileList, num_epochs=num_epochs, shuffle=True)
    # convert the data in queue
    sequence_length, labels_length, audio_features, audio_labels, file_key = read_my_file_format(fileQueue,feat_dimension)
    # create a batch
    audio_features_batch, audio_labels_batch, seq_length_batch, labels_length_batch, files_batch = tf.train.batch(
        [audio_features, audio_labels, sequence_length, labels_length, file_key],
        batch_size=batch_size,
        num_threads=queueThread,
        capacity=queueCapacity,
        dynamic_pad=True,
        enqueue_many=False)
    return audio_features_batch, audio_labels_batch, seq_length_batch, labels_length_batch, files_batch



if __name__ == '__main__':
    with tf.Graph().as_default():
        #code to get a batch
        audio_features, audio_labels, seq_length, lab_length, files = get_input_batch(dataPath='/home/storage/Data/MULTI_GRID/soloVideoTfRec/TRAIN_CTC_VIDEO_SENTENCES/',num_epochs=10,batch_size=594,feat_dimension=134,num_examples=num_examples_training)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            '''
            Since Python can kill the queue_runners threads while they are still running (in this case will be raised the "Skipping cancelled enqueue attempt with queue not closed" error message),
            it is important ot to avoid them by stopping threads manually by using the following pattern:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            <do stuff>
            coord.request_stop()
            coord.join(threads)
             
            '''
            coord = tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            #print files.eval()
            print (audio_labels.eval()).shape
            # print np.max(audio_labels.eval())
            coord.request_stop()
            coord.join(threads=threads)

            # # variables initializer
            # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())