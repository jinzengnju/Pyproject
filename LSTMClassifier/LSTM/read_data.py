#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf


def read_TfReacords(filename_queue):
    reader=tf.TFRecordReader()
    _,serilized_example=reader.read(filename_queue)
    features = tf.parse_single_example(serilized_example,
                                       features={
                                           'fact': tf.FixedLenFeature([], tf.string),
                                           'law': tf.FixedLenFeature([183],tf.int64)
                                       })
    fact = features['fact']
    law = features['law']
    return fact,law


def _generate_text_and_label_batch(fact,law,batch_size,shuffle):
    num_preprocess_threads=2
    if shuffle:
        fact_batch,law_batch=tf.train.shuffle_batch([fact,law],
                                batch_size=batch_size,min_after_dequeue=200,num_threads=num_preprocess_threads,capacity=1000+3*batch_size)
    else:
        fact_batch,law_batch=tf.train.batch([fact,law],
                                batch_size=batch_size,min_after_dequeue=200,num_threads=num_preprocess_threads,capacity=1000+3*batch_size)
    return fact_batch,law_batch
def inputs(data_dir,batch_size):
    filenames=[data_dir]
    with tf.name_scope('input'):
        filename_queue=tf.train.string_input_producer(filenames)
        fact,law=read_TfReacords(filename_queue)
    fact_batch,law_batch =_generate_text_and_label_batch(fact,law,batch_size,shuffle=True)
    return fact_batch,law_batch