import tensorflow as tf
import os
w1=tf.placeholder("float",name="w1")
w2=tf.placeholder("float",name="w2")
b1=tf.Variable(2.0,name='bias')

w3=tf.add(w1,w2)
w4=tf.multiply(w3,b1,name="op_to_restore")
sess=tf.Session()

sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
print(sess.run(w4,feed_dict={w1:4,w2:8}))

ckpt_dir="checkpoints"
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
checkpoint_path=os.path.join(ckpt_dir,"model.ckpt")
saver.save(sess,checkpoint_path,global_step=1000)