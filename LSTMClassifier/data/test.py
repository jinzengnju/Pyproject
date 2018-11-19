#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf

def get(sess):
    a = tf.constant(1)
    return a.eval(session=sess)


if __name__=='__main__':
    with tf.Session() as sess:
        print(get(sess))
