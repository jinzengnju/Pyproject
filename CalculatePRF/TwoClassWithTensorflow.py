#!/usr/bin/python
# -*- coding:UTF-8 -*-
#这个文件主要是用Tensorflow来实现准确率、召回率以及F值的计算
#参考连接：https://blog.csdn.net/sinat_35821976/article/details/81334181
#https://blog.csdn.net/sinat_35821976/article/details/80765145
import tensorflow as tf


def tf_confusion_metrics(predict, real, session, feed_dict):
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)
    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )
    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op], feed_dict)
    tpr = float(tp) / (float(tp) + float(fn))
    fpr = float(fp) / (float(fp) + float(tn))
    fnr = float(fn) / (float(tp) + float(fn))
    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))
    f1_score = (2 * (precision * recall)) / (precision + recall)

#如何调用上述函数
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./model.ckpt.meta')
    saver.restore(sess, './model.ckpt')  # .data文件
    pred = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    y_ = graph.get_operation_by_name('y_').outputs[0]
    #y = sess.run(pred, feed_dict={x: test_x, y_: test_y})
    #最后一行y，即网络输出是tensor,而实际标签test_y的类型也是ndarray。但是在函数def tf_confusion_metrics(predict, real,
    # session, feed_dict)中predict、real需要为tensor类型，feed_dict中的数据不能为tensor类型，因此需要讲y转为ndarry类型，
    # 然后将test_y转为tensor类型，然后调用函数。具体实现如下：

def get_PRF(y,test_y):
    #其中输入的y是一个tensor，而实际标签test_y的类型也是ndarray
    predictLabel = tf.constant(y)
    predictLabel = predictLabel.eval()  # 将tensor转为ndarray
    realLabel = tf.convert_to_tensor(test_y)  # 将ndarray转为tensor
    tf_confusion_metrics(y, realLabel, sess, feed_dict={predict: predictLabel, real: test_y})