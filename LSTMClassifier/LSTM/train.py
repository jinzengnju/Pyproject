#!/usr/bin/python
# -*- coding:UTF-8 -*-
import os
import sys
import tensorflow as tf
import numpy as np
import json
from LSTMClassify.data_utils import(
    generate_epoch,
    train_and_valid_data,
    create_vacabulary,
    read_train_or_valid_Data,
    getClassNum

)
from LSTMClassify.model import (Model,)

#Configs
tf.app.flags.DEFINE_string("rnn_unit",'lstm',"Type of RNN unit:rnn|gru|lstm.")
tf.app.flags.DEFINE_float("learning_rate",1e-5,"Learning Rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor",0.99,"Learning rate decays by this much")
tf.app.flags.DEFINE_float("max_gradient_norm",5.0,"Clip gradients to this norm")
tf.app.flags.DEFINE_integer("num_epochs",100,"Number of epoches during training")
tf.app.flags.DEFINE_integer("batch_size",64,"Batch size to use during training")
tf.app.flags.DEFINE_integer("num_hidden_units",300,"Number of hidden units in each RNN unit")
tf.app.flags.DEFINE_integer("num_layers",2,"NUmber of layers in the model")
tf.app.flags.DEFINE_float("dropout",0.5,"Amount to drop during training")
tf.app.flags.DEFINE_integer("en_vocab_size",10000,"English vocabulary size")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoints","Directory to save the model checkpoints")
tf.app.flags.DEFINE_integer("num_classes",2,"Number of classification classes")

FLAGS=tf.app.flags.FLAGS

def create_model(sess,FLAGS):
    text_model=Model(FLAGS)
    ckpt=tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Restoring old model parameters from %s"%ckpt.model_checkpoint_path)
        text_model.saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        print("Create new Model")
        sess.run(tf.initialize_all_variables())
    return text_model



def train(train_path,valid_path,what):
    print(FLAGS.num_classes)
    accu_num = getClassNum("accu")
    law_num = getClassNum("law")


    f_read1=open('vocab.dict','r')
    vocab_dict=json.load(f_read1)
    f_read1.close()
    f_read2 = open('rev_vocab.dict', 'r')
    rev_vocab_dict = json.load(f_read2)
    f_read2.close()
    # print(json.dumps(vocab_dict,ensure_ascii=False))
    # print(json.dumps(rev_vocab_dict,ensure_ascii=False))
    # print(type(vocab_dict["被告"]))
    # print(rev_vocab_dict)



    # train_alltext, train_accu_label, train_law_label, train_time_label=read_train_or_valid_Data(train_path)
    #
    # valid_alltext, valid_accu_label, valid_law_label, valid_time_label= read_train_or_valid_Data(valid_path)



    # f_write1=open('vocab.dict','w')
    # json.dump(vocab_dict,f_write1,ensure_ascii=False)
    # f_write1.close()
    #
    # f_write2=open("rev_vocab.dict",'w')
    # json.dump(rev_vocab_dict,f_write2,ensure_ascii=False)
    # f_write2.close()



    # if(what=="accu"):
    #     train_X,train_y,train_seq_lens=train_and_valid_data(train_alltext,vocab_dict,train_accu_label,accu_num)
    #     valid_X, valid_y, valid_seq_lens = train_and_valid_data(valid_alltext, vocab_dict, valid_accu_label,accu_num)
    # if(what=="law"):
    #     train_X, train_y, train_seq_lens = train_and_valid_data(train_alltext, vocab_dict, train_law_label,law_num)
    #     valid_X, valid_y, valid_seq_lens = train_and_valid_data(valid_alltext, vocab_dict, valid_law_label,law_num)
    # if (what == "time"):
    #     train_X, train_y, train_seq_lens = train_and_valid_data(train_alltext, vocab_dict, train_time_label)
    #     valid_X, valid_y, valid_seq_lens = train_and_valid_data(valid_alltext, vocab_dict, valid_time_label)

    with tf.Session() as sess:
        model =create_model(sess,FLAGS)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)

        #Train results
        #一般用for循环来遍历一个生成器,generate_epoch这个生成器会把一个epoch的数据切成batch并放在生成器中
        for epoch_num,epoch in enumerate(generate_epoch(train_X,train_y,train_seq_lens,FLAGS.num_epochs,FLAGS.batch_size)):
            print("Epoch:",epoch_num)
            sess.run(tf.assign(model.lr,FLAGS.learning_rate*(FLAGS.learning_rate_decay_factor**epoch_num)))

            train_loss=[]
            train_accuracy=[]

            for batch_num,(batch_X,batch_y,batch_seq_lens) in enumerate(epoch):
                print("batch_num:%d"%batch_num)
                _,loss,accuracy=model.step(sess,batch_X,batch_seq_lens,batch_y,dropout=FLAGS.dropout,forward_only=False,
                                           sampling=False)

                train_loss.append(loss)
                train_accuracy.append(accuracy)

            print
            print("EPOCH %i SUMMARY"%epoch_num)
            print("Training loss %.3f"%np.mean(train_loss))
            print("Training accuracy %.3f"%np.mean(train_accuracy))
            print("-------------------------")
            #Valid results
            for valid_epoch_num,valid_epoch in enumerate(generate_epoch(valid_X,valid_y,valid_seq_lens,num_epochs=1,batch_size=FLAGS.batch_size)):
                valid_loss=[]
                valid_accuracy=[]

                for valid_batch_num,(valid_batch_X,valid_batch_y,valid_batch_seq_lens) in enumerate(valid_epoch):
                    loss,accuracy=model.step(sess,valid_batch_X,valid_batch_seq_lens,valid_batch_y,dropout=0.0,forward_only=True,sampling=False)
                    valid_loss.append(loss)
                    valid_accuracy.append(accuracy)

            print("validation loss %.3f"%np.mean(valid_loss))
            print("Valid accuracy %.3f"%np.mean(valid_accuracy))
            print("------------------------------------------")

            #Save cheackpoint every epoch
            if not os.path.isdir(FLAGS.ckpt_dir):
                os.makedirs(FLAGS.ckpt_dir)
            checkpoint_path=os.path.join(FLAGS.ckpt_dir,"model.ckpt")
            print("Saving the model")
            model.saver.save(sess,checkpoint_path,global_step=model.global_step)

# def sample():
#     X,y=load_data_and_labels()
#     vocab_list,vocab_dict,rev_vocab_dict=create_vacabulary(X)
#     X,seq_lens=data_to_token_ids(X,vocab_dict)
#
#     test_sentence="It was the worst movie I have ever seen"
#     test_sentence=get_tokens(clean_str(test_sentence))
#     test_sentence,seq_len=data_to_token_ids([test_sentence],vocab_dict)
#
#     test_sentence=test_sentence[0]
#     test_sentence=test_sentence+[PAD_ID]*(max(len(sentence) for sentence in X)-len(test_sentence))
#     test_sentence=np.array(test_sentence).reshape([1,-1])
#
#     with tf.Session() as sess:
#         model=create_model(sess,FLAGS)
#         probabilities=model.step(sess,batch_X=test_sentence,batch_seq_lens=np.array(seq_len),forward_only=True,sampling=True)
#         for index,prob in enumerate(probabilities[:seq_len[0]]):
#             print(rev_vocab_dict[test_sentence[0][index]],prob[1])

if __name__=="__main__":
    # if sys.argv[1]=="train":
    train("/home/jin/Contest/data_train.json","/home/jin/Contest/data_valid.json","accu")
    # if sys.argv[1]=="sample":
    #     sample()
































