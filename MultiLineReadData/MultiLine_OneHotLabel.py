#!/usr/bin/python
# -*- coding:UTF-8 -*-
#一共5列，最后两列是标签，one-hot编码
import tensorflow as tf
filenames=['A.csv']
filename_queue=tf.train.string_input_producer(filenames,shuffle=False)

reader=tf.TextLineReader()
key,value=reader.read(filename_queue)

#定义decoder
record_defaults=[[1],[1],[1],[1],[1]]
col1,col2,col3,col4,col5=tf.decode_csv(value,record_defaults=record_defaults)
features=tf.stack([col1,col2,col3])
label=tf.stack([col4,col5])

example_batch,label_batch=tf.train.shuffle_batch([features,label],batch_size=2,capacity=200,min_after_dequeue=100,num_threads=2)

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        e_val,l_val=sess.run([example_batch,label_batch])
        print(e_val,l_val)
    coord.request_stop()
    coord.join(threads)