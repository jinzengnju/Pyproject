#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import os

x=tf.placeholder(tf.float32,shape=[None,1])
y=4*x+4
global_steps=tf.Variable(0,trainable=False)

w=tf.Variable(tf.random_normal([1],-1,1))
#设置了变量，必须对变量进行初始化
b=tf.Variable(tf.zeros([1]))
y_predict=w*x+b


loss=tf.reduce_mean(tf.square(y-y_predict))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss,global_step=global_steps)

isTrain=True
train_steps=100
checkpoint_steps=20
checkpoint_dir='checkpoints'

#定义模型存储对象saver对象
saver=tf.train.Saver(max_to_keep=3)

x_data=np.reshape(np.random.rand(10).astype(np.float32),(10,1))

def save_model(sess,global_step):
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path=os.path.join(checkpoint_dir,'model.ckpt')
    saver.save(sess,checkpoint_path,global_step=global_step)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if isTrain:
        for i in range(train_steps):
            sess.run(train,feed_dict={x:x_data})
            #print(sess.run(global_steps))
            if(i+1)%checkpoint_steps==0:
                save_model(sess,global_steps)
    else:
        ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            pass
        print(sess.run(w))
        print(sess.run(b))