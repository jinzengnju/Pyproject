#!/usr/bin/python
# -*- coding:UTF-8 -*-
#队列文件使用范式
import tensorflow as tf
files=tf.train.match_filenames_once("Records/output.tfrecords")
filename_queue=tf.train.string_input_producer(files,shuffle=False)

reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)

#解析文件读取样例
features=tf.parse_single_example(
    serialized_example,
    #从文件名队列读取的一行数据用于decode
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)
    }
)

#decode_raw后的dtype一定要和toString之前相同
#https://blog.csdn.net/u012436149/article/details/76714126
decoded_images=tf.decode_raw(features['image_raw'],tf.uint8)
retyped_images=tf.cast(decoded_images,tf.float32)
labels=tf.cast(features['label'],tf.int32)
#pixels=tf.cast(features['pixels'],tf.int32)
images=tf.reshape(retyped_images,[784])

#将文件以每100个打包
min_after_dequeue=10000
batch_size=100
capacity=min_after_dequeue+3*batch_size
image_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
def inference(input_tensor,weights1,biases1,weights2,biases2):
    layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
    return tf.matmul(layer1,weights2)+biases2

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
REGULATION_RATE=0.0001
TRAINING_STEPS=5000

weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

weights2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

y=inference(image_batch,weights1,biases1,weights2,biases2)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=label_batch)
cross_entropy_mean=tf.reduce_mean(cross_entropy)

regularizer=tf.contrib.layer.l2_regularizer(REGULATION_RATE)
regularization=regularizer(weights1)+regularizer(weights2)
loss=cross_entropy+regularization

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(TRAINING_STEPS):
        if i%1000==0:
            print("After %d training step(s),loss is %g"%(i,sess.run(loss)))
        sess.run(train_step)

    coord.request_stop()
    coord.join(threads)

# tf.some_reader()


