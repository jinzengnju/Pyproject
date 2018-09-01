#!/usr/bin/python
# -*- coding:UTF-8 -*-
#单个reader，多个样本
import tensorflow as tf
filenames=['A.csv','B.csv','C.csv']
filename_queue=tf.train.string_input_producer(filenames,shuffle=False)

reader=tf.TextLineReader()
key,value=reader.read(filename_queue)
#key标志着读到第几行
example,label=tf.decode_csv(value,record_defaults=[['null'],['null']])

example_bach,label_batch=tf.train.batch([example,label],batch_size=5)
#这里的tf.train.batch会额外创建一个队列，会话取出的数据是从该队列取出的
#设置每个batch有5个样本
with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        e_val,l_val=sess.run([example_bach,label_batch])
        #这里的batch有5个数据
        print(e_val,l_val)
    coord.request_stop()
    coord.join(threads)