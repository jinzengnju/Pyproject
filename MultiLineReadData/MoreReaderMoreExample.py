#!/usr/bin/python
# -*- coding:UTF-8 -*-
#多个reader，多个样本
import tensorflow as tf
filenames=['A.csv','B.csv','C.csv']
filename_queue=tf.train.string_input_producer(filenames,shuffle=False)

reader=tf.TextLineReader()
key,value=reader.read(filename_queue)
record_defaults=[['null'],['null']]

#定义了多种解码器，每个解码器都和一个reader连接,value是从reader读来的，解码器是对value进行解码的
example_list=[tf.decode_csv(value,record_defaults=record_defaults) for _ in range(2)]
#Reader设置为2
#使用tf.train.batch_join( )，可以使用多个reader，并行读取数据。每个redaer使用一个线程。读取的数据存入batch_join创建的队列中

example_batch,label_batch=tf.train.batch_join(example_list,batch_size=5)
with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        e_val,l_val=sess.run([example_batch,label_batch])
        print(e_val,l_val)
    coord.request_stop()
    coord.join(threads)