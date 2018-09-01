import tensorflow as tf

with tf.Session() as sess:
    saver=tf.train.import_meta_graph("checkpoints/model.ckpt-1000.meta")
    saver.restore(sess,tf.train.latest_checkpoint("checkpoints"))
    print(sess.run('w1:0'))