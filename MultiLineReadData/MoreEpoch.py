#!/usr/bin/python
# -*- coding:UTF-8 -*-
#epoch
import tensorflow as tf
filenames=['A.csv','B.csv','C.csv']

filename_queue=tf.train.string_input_producer(filenames,shuffle=False,num_epochs=3)
#文件名重复三次放入文件名队列，且顺序打乱
reader=tf.TextLineReader()

key,value=reader.read(filename_queue)

record_defaults=[['null'],['null']]
example_list=[tf.decode_csv(value,record_defaults=record_defaults) for _ in range(2)]
example_batch,label_batch=tf.train.batch_join(example_list,batch_size=1)

init_local_op=tf.local_variables_initializer()
#注意：指定 num_epochs 时，在初始化模型参数的时候选用local_variables_initializer
#tf 会将 num_epoch 作为 local variable
#设置了变量variable需要进行初始化
with tf.Session() as sess:
    sess.run(init_local_op)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            e_val,l_val=sess.run([example_batch,label_batch])
            print(e_val,l_val)
    except tf.errors.OutOfRangeError:
        print("Epoches complete")
    finally:
        coord.request_stop()
    coord.join(threads)
    coord.request_stop()
    coord.join(threads)
