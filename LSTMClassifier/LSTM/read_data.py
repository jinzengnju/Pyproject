#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf
import jieba
import numpy as np
from LSTMClassifier.LSTM.data_utils import *
def cut_text(alltext):
    count = 0
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append([word for word in jieba.cut(text) if len(word)>1])
    return train_text

def read_TfReacords(filename_queue,FLAGS):
    reader=tf.TFRecordReader()
    _,serilized_example=reader.read(filename_queue)
    features = tf.parse_single_example(serilized_example,
                                       features={
                                           'fact': tf.FixedLenFeature([], tf.string),
                                           'law': tf.VarLenFeature(tf.int64)
                                       })
    fact = features['fact']
    law = tf.sparse_tensor_to_dense(features['law'])
    one_hot_index = law
    y_one_hot = np.zeros(FLAGS.num_class)
    y_one_hot.flat[one_hot_index] = 1
    return cut_text(fact),y_one_hot

def _generate_text_and_label_batch(fact,law,batch_size,shuffle):
    num_preprocess_threads=2
    if shuffle:
        fact_batch,law_batch=tf.train.shuffle_batch([fact,law,batch_size],
                                batch_size=batch_size,min_after_dequeue=1000,num_threads=num_preprocess_threads,capacity=1000+3*batch_size)
    else:
        fact_batch,law_batch=tf.train.batch([fact,law,batch_size],
                                batch_size=batch_size,min_after_dequeue=1000,num_threads=num_preprocess_threads,capacity=1000+3*batch_size)
    return fact_batch,law_batch
def inputs(data_dir,batch_size,vocab_dict):
    filenames=[data_dir]
    with tf.name_scope('input'):
        filename_queue=tf.train.string_input_producer(filenames)
        fact,law=read_TfReacords(filename_queue)
    fact_batch,law_batch =_generate_text_and_label_batch(fact,law,batch_size,shuffle=True)
    fact_batch_new=data_to_token_ids(fact_batch,vocab_dict)
    return fact_batch_new,law_batch