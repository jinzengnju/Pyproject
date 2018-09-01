#!/usr/bin/python
# -*- coding:UTF-8 -*-

import tensorflow as tf


def read_myfile_format(filename_queue):
    reader=tf.TFRecordReader()
    _,serilized_example=reader.read(filename_queue)

    features=tf.parse_single_example(serilized_example,
                                     features={
                                         'fact': tf.FixedLenFeature([], tf.string),
                                         'accusation': tf.VarLenFeature(tf.string),
                                         'law': tf.VarLenFeature(tf.int64),
                                         'time': tf.FixedLenFeature([], tf.int64)
                                     })
    fact=features['fact']
    accu=tf.sparse_tensor_to_dense(features['accusation'],default_value='')
    law=tf.sparse_tensor_to_dense(features['law'])
    time_label=features['time']
    return fact,accu,law,time_label

def input_pipline(filenames,batch_size,num_epochs=1):
    filename_queue=tf.train.string_input_producer([filenames],num_epochs=num_epochs)
    fact,accu,law,time_label=read_myfile_format(filename_queue)
    facts, accus, laws, time_labels=tf.train.shuffle_batch([fact,accu,law,time_label],batch_size=batch_size,num_threads=2,capacity=1000+3*batch_size,min_after_dequeue=1000)
    return facts, accus, laws, time_labels

#如果你的任务是多标记问题(并且标记列表的长度不一),那么需要在存tfrecords文件时将标记的长度统一(pad补齐),这样才能在读取解析tfrecords文件之后使用tf.train.shuffle_batch函数
#https://blog.csdn.net/yaoqi_isee/article/details/77526497
#下面这篇博客生成批数据不是通过string_input_producer来实现的,而是调用了TfrecordDataSet来实现.感觉底层应该还是用了队列
#https://blog.csdn.net/he_wen_jie/article/details/80269256
#上面这两个博客很经典
#http://www.soaringroad.com/?p=672
#https://www.jianshu.com/p/f580f4fc2ba0

def run_training():
    with tf.Graph().as_default(),tf.Session() as sess:
        facts, accus, laws, time_labels=input_pipline("train.tfrecords",5)
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #sess.run(fetches,feed_dict)Runs operations and evaluates tensors in fetches
        try:
            while not coord.should_stop():
                myfacts, myaccus, mylaws, mytime_labels=sess.run([facts, accus, laws, time_labels])
                print(myfacts, myaccus, mylaws, mytime_labels)
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.should_stop()
        coord.join(threads)
        sess.close()

if __name__=='__main__':
    run_training()



