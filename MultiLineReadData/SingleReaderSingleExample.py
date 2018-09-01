#!/usr/bin/python
# -*- coding:UTF-8 -*-
#单个reader，单个样本
import tensorflow as tf
filenames=['A.csv','B.csv','C.csv']

filename_queue=tf.train.string_input_producer(filenames,shuffle=False)
#文件名队列中的文件顺序打乱

reader=tf.TextLineReader()
#从文件名队列中读取数据
key,value=reader.read(filename_queue)

#定义decoder
example,label=tf.decode_csv(value,record_defaults=[['null'],['null']])
#解析每一行数据

example_batch,label_batch=tf.train.shuffle_batch([example,label],batch_size=1,capacity=200,min_after_dequeue=100,num_threads=2)
#输出一个batch的数据

with tf.Session() as sess:
    #创建线程协调器
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    #取出10个batch的数据
    for i in range(10):
        e_val,l_val=sess.run([example_batch,label_batch])
        #循环每次调用一次example_batch,label_batch，就会从线程队列中取出数据
        print(e_val,l_val)
    coord.request_stop()
    coord.join(threads)
