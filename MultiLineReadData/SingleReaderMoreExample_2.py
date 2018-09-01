#!/usr/bin/python
# -*- coding:UTF-8 -*-
#单个reader，多个样本
import tensorflow as tf
filenames=['A.csv','B.csv','C.csv']
filename_queue=tf.train.string_input_producer(filenames,shuffle=False)

reader=tf.TextLineReader()
key,value=reader.read(filename_queue)

example,label=tf.decode_csv(value,record_defaults=[['null'],['null']])
example_batch,label_batch=tf.train.batch([example,label],batch_size=5)

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        print(example_batch.eval(),label_batch.eval())
    #这种形式的输出example和label是不能相对应的
    coord.request_stop()
    coord.join(threads)