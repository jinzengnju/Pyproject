import tensorflow as tf
import os
sess=tf.Session()

saver=tf.train.import_meta_graph("checkpoints/model.ckpt-1000.meta")
saver.restore(sess,tf.train.latest_checkpoint("./checkpoints"))

graph=tf.get_default_graph()
w1=graph.get_tensor_by_name("w1:0")
w2=graph.get_tensor_by_name("w2:0")

feed_dict={w1:13.0,w2:17.0}
op_to_restore=graph.get_tensor_by_name("op_to_restore:0")

add_on_op=tf.multiply(op_to_restore,2)
print(sess.run(add_on_op,feed_dict))

# saver=tf.train.Saver(max_to_keep=1)
# for i in range(100):
#     batch_xs,batch_ys=mnist.train.next_batch(100)
#     sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
#     val_loss,val_acc=sess.run([loss,acc],feed_dict={x:})
#     saver.save(sess,checkpoint_path,global_step=i+1)
# sess.close()