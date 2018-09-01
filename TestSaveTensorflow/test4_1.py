import tensorflow as tf
import os
w1=tf.Variable(tf.random_normal(shape=[2]),name="w1")
w2=tf.Variable(tf.random_normal(shape=[5]),name="w2")

ckpt_dir="checkpoints"

#指定了要保存和恢复的变量

saver=tf.train.Saver([w1,w2])
sess=tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
checkpoint_path=os.path.join(ckpt_dir,"model.ckpt")

saver.save(sess,checkpoint_path,global_step=1000)