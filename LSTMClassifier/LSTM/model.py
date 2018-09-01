#!/usr/bin/python
# -*- coding:UTF-8 -*-
import tensorflow as tf

def rnn_cell(FLAGS,dropout):
    if FLAGS.rnn_unit=='rnn':
        rnn_cell_type=tf.nn.rnn_cell.BasicRNNCell
    elif FLAGS.rnn_unit=='gru':
        rnn_cell_type=tf.nn.rnn_cell.GRUCell
    elif FLAGS.rnn_unit=='lstm':
        rnn_cell_type=tf.nn.rnn_cell.BasicLSTMCell
    else:
        raise Exception("choose a valid RNN unit type")

    single_cell=rnn_cell_type(FLAGS.num_hidden_units)
    single_cell=tf.nn.rnn_cell.DropoutWrapper(single_cell,output_keep_prob=1-dropout)

    stacked_cell=tf.nn.rnn_cell.MultiRNNCell([single_cell]*FLAGS.num_layers)

    return stacked_cell

def rnn_inputs(FLAGS,input_data):
    with tf.variable_scope('rnn_inputs',reuse=True):
        W_input=tf.get_variable("W_input",[FLAGS.en_vocab_size,FLAGS.num_hidden_units])
    embeddings=tf.nn.embedding_lookup(W_input,input_data)
    return embeddings

def rnn_softmax(FLAGS,outputs):
    with tf.variable_scope('rnn_softmax',reuse=True):
        W_softmax=tf.get_variable("W_softmax",[FLAGS.num_hidden_units,FLAGS.num_classes])
        b_softmax=tf.get_variable("b_softmax",[FLAGS.num_classes])
    logits=tf.matmul(outputs,W_softmax)+b_softmax
    return logits

def length(data):
    relevant=tf.sign(tf.abs(data))
    length=tf.reduce_sum(relevant,reduction_indices=1)
    length=tf.cast(length,tf.int32)
    return length

class Model(object):
    def __init__(self,FLAGS):
        self.inputs_X=tf.placeholder(tf.int32,shape=[None,None],name='inputs_X')
        self.targets_y=tf.placeholder(tf.float32,shape=[None,None],name='targets_y')
        self.seq_lens=tf.placeholder(tf.int32,shape=[None,],name='seq_lens')
        self.dropout=tf.placeholder(tf.float32)

        stacked_cell=rnn_cell(FLAGS,self.dropout)

        with tf.variable_scope('rnn_inputs'):
            W_input=tf.get_variable("W_input",[FLAGS.en_vocab_size,FLAGS.num_hidden_units])
        inputs=rnn_inputs(FLAGS,self.inputs_X)

        all_outputs,state=tf.nn.dynamic_rnn(cell=stacked_cell,inputs=inputs,sequence_length=self.seq_lens,dtype=tf.float32)
        outputs=state[-1][1]

        with tf.variable_scope('rnn_softmax'):
            W_softmax=tf.get_variable("W_softmax",[FLAGS.num_hidden_units,FLAGS.num_classes])
            b_softmax=tf.get_variable("b_softmax",[FLAGS.num_classes])

        logits=rnn_softmax(FLAGS,outputs)
        probabilities=tf.nn.softmax(logits)
        self.accuracy=tf.equal(tf.argmax(self.targets_y,1),tf.argmax(logits,1))

        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=self.targets_y))

        self.lr=tf.Variable(0.0,trainable=False)
        trainable_vars=tf.trainable_variables()
        grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,trainable_vars),FLAGS.max_gradient_norm)
        optimizer=tf.train.AdamOptimizer(self.lr)
        self.train_optimizer=optimizer.apply_gradients(zip(grads,trainable_vars))

        sampling_outputs=all_outputs[0]

        sampling_logits=rnn_softmax(FLAGS,sampling_outputs)
        self.sampling_probablities=tf.nn.softmax(sampling_logits)

        self.global_step=tf.Variable(0,trainable=False)
        self.saver=tf.train.Saver(tf.all_variables())

    def step(self,sess,batch_X,batch_seq_lens,batch_y=None,dropout=0.0,forward_only=True,sampling=False):
        input_feed={self.inputs_X:batch_X,
                    self.targets_y:batch_y,
                    self.seq_lens:batch_seq_lens,
                    self.dropout:dropout}
        if forward_only:
            if not sampling:
                output_feed=[self.loss,self.accuracy]
            elif sampling:
                input_feed={self.inputs_X:batch_X,
                    self.seq_lens:batch_seq_lens,
                    self.dropout:dropout}
                output_feed=[self.sampling_probablities]
        else:
            output_feed=[self.train_optimizer,self.loss,self.accuracy]
        outputs=sess.run(output_feed,input_feed)
        if forward_only:
            if not sampling:
                return outputs[0],outputs[1]
            elif sampling:
                return outputs[0]
        else:
            return outputs[0],outputs[1],outputs[2]
