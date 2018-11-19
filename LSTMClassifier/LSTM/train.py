#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import sys
import tensorflow as tf
import numpy as np
import json
from LSTMClassifier.LSTM.read_data import *
import time
from LSTMClassifier.LSTM.model import Model
from LSTMClassifier.LSTM.data_utils import *

#Configs
tf.app.flags.DEFINE_string("rnn_unit",'lstm',"Type of RNN unit:rnn|gru|lstm.")
tf.app.flags.DEFINE_float("learning_rate",1e-5,"Learning Rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor",0.99,"Learning rate decays by this much")
tf.app.flags.DEFINE_float("max_gradient_norm",5.0,"Clip gradients to this norm")
tf.app.flags.DEFINE_integer("batch_size",5,"Batch size to use during training")
tf.app.flags.DEFINE_integer("num_hidden_units",300,"Number of hidden units in each RNN unit")
tf.app.flags.DEFINE_integer("num_layers",2,"NUmber of layers in the model")
tf.app.flags.DEFINE_float("dropout",0.5,"Amount to drop during training")
tf.app.flags.DEFINE_integer("en_vocab_size",10000,"English vocabulary size")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoints","Directory to save the model checkpoints")
tf.app.flags.DEFINE_integer("num_classes","2"," ")
tf.app.flags.DEFINE_string("input_traindata","/home/jin/data/cail_0518/temp/TFrecords/train.tfrecords","训练数据路径")
tf.app.flags.DEFINE_string("input_validdata","/home/jin/data/cail_0518/temp/TFrecords/test.tfrecords","验证数据路径")
tf.app.flags.DEFINE_integer("valid_step",20,'')
tf.app.flags.DEFINE_integer("valid_batch",5,'')

FLAGS=tf.app.flags.FLAGS

def create_model(sess,FLAGS):
    text_model=Model(FLAGS)
    ckpt=tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring old model parameters from %s"%ckpt.model_checkpoint_path)
        text_model.saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        print("Create new Model")
        sess.run(tf.global_variables_initializer())
    return text_model


def save_model(model,sess,step_index):
    if not os.path.isdir(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)
    checkpoint_path = os.path.join(FLAGS.ckpt_dir, "model.ckpt")
    print("Saving the model and global_step is:",step_index)
    model.saver.save(sess, checkpoint_path, global_step=model.global_step)



def train(vocab_dict,law_num):
    with tf.Graph().as_default(), tf.Session() as sess:
        train_fact, train_laws = inputs(FLAGS.input_traindata, FLAGS.batch_size)
        valid_fact,valid_laws=inputs(FLAGS.input_validdata,FLAGS.batch_size)
        FLAGS.num_classes=law_num
        model =create_model(sess,FLAGS)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        try:
            step=0
            start_time = time.time()
            while not coord.should_stop():#这里是永远不会停止的，因为epoch设置的是NOne
                train_fact_v,train_law_v=sess.run([train_fact, train_laws])
                train_fact_val,train_seq_lens=get_X_with_word_index(train_fact_v,vocab_dict)
                _, loss, accuracy = model.step(sess, train_fact_val,train_seq_lens,train_law_v,dropout=FLAGS.dropout,
                                               forward_only=False)

                if step%(FLAGS.valid_step)==0:
                    time_use = time.time() - start_time
                    print("***********************************************")
                    step_index=sess.run(model.global_step)
                    save_model(model,sess,step_index)
                    print('Step %d:loss=%.2f(%.3sec)'%(step_index,loss,time_use))
                    for _ in range(FLAGS.valid_batch):
                        valid_loss=0
                        valid_fact_v, valid_law_v = sess.run([valid_fact,valid_laws])
                        valid_fact_val, valid_seq_lens = get_X_with_word_index(valid_fact_v, vocab_dict)
                        loss, accuracy = model.step(sess, valid_fact_val, valid_seq_lens, valid_law_v, dropout=FLAGS.dropout,
                                                       forward_only=True)
                        valid_loss+=loss
                    print("valid loss=%.3f and accuracy=%.3f"%(valid_loss/FLAGS.valid_batch,accuracy))
                    start_time=time.time()
                step+=1
        except tf.errors.OutOfRangeError:
            print('Done training for %d steps'%step)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()




def preProcess():
    law_num = getClassNum("law")
    f_read1=open('/home/jin/Pypro/Pyproject/LSTMClassifier/data/vocab.dict','r')
    vocab_dict=json.load(f_read1)
    f_read1.close()
    return law_num,vocab_dict


if __name__=="__main__":
    law_num,vocab_dict=preProcess()
    train(vocab_dict,law_num)