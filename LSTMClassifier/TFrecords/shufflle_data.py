#!/usr/bin/python
# -*- coding:UTF-8 -*-
from sklearn.utils import shuffle
import numpy as np
import json
import tensorflow as tf

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('origin_file', 'train', 'This is a string')
tf.app.flags.DEFINE_string('shuffle_file','shuffle_train', 'This is the rate in training')

def shuffle_method1(data):
    #data是numpy数组,也可以是list
    return shuffle(data)

def shuffle_method2(df):
    #这里的df是DataFrame结构型的数据
    df.sample(frac=1)

def shuffle_method3(data,label):
    shuffle_index=np.random.permutation(np.arange(len(data)))
    #如果data是列表，不是numpy数组，必须强制转换为numpy数组才能执行下面的东西
    data,label=data[shuffle_index],label[shuffle_index]

def get_shuffledata(input_path,output_path):
    fin=open(input_path,'r',encoding='utf8')
    alltext=[]
    line=fin.readline()
    while line:
        one_text={}
        d=json.loads(line)
        one_text['fact']=d['fact']
        one_text['law']=d['meta']['relevant_articles']
        alltext.append(one_text)
        line=fin.readline()
    alltext=shuffle(alltext)
    fin.close()
    #dump和dumps的区别
    #dump是將dict转为str后还要存入文件，与文件操作相关
    #dumps转为字符串str就结束了，和文件操作无关
    fout=open(output_path,'w',encoding='utf8')
    for one_text in alltext:
        json.dump(one_text,fout,ensure_ascii=False)
        fout.write('\n')

def main(unuse_args):
    get_shuffledata(FLAGS.origin_file,FLAGS.shuffle_file)


if __name__=='__main__':
    tf.app.run()


