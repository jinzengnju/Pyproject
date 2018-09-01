import tensorflow as tf
import os
v1=tf.get_variable("v1",shape=[3],initializer=tf.zeros_initializer)
v2=tf.get_variable("v2",shape=[5],initializer=tf.zeros_initializer)

inc_v1=v1.assign(v1+1)
dec_v2=v2.assign(v2-1)

init_op=tf.global_variables_initializer()

saver=tf.train.Saver()
ckpt_dir="checkpoints"
with tf.Session() as sess:
    sess.run(init_op)
    # print(sess.run(inc_v1))
    # print(sess.run(dec_v2))
    inc_v1.op.run()
    dec_v2.op.run()
    #print(inc_v1.eval())这里会多加一次
    print(v1.read_value().eval())
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")
    print("Saving the model")
    save_path=saver.save(sess, checkpoint_path)

    print("Model saved in path:%s"%save_path)
