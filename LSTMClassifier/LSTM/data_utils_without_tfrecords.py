#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
import re
import jieba
import json
import itertools
from collections import Counter

def init():
	f = open('law.txt', 'r', encoding = 'utf8')
	law = {}
	lawname = {}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
		law[line.strip()] = len(law)
		line = f.readline()
	f.close()


	f = open('accu.txt', 'r', encoding = 'utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()
		accu[line.strip()] = len(accu)
		line = f.readline()
	f.close()


	return law, accu, lawname, accuname

law, accu, lawname, accuname = init()


def getClassNum(kind):
    global law
    global accu

    if kind == 'law':
        return len(law)
    if kind == 'accu':
        return len(accu)


def getName(index, kind):
    global lawname
    global accuname
    if kind == 'law':
        return lawname[index]

    if kind == 'accu':
        return accuname[index]


def gettime(time):
    # 将刑期用分类模型来做
    v = int(time['imprisonment'])

    if time['death_penalty']:
        return 0
    if time['life_imprisonment']:
        return 1
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    else:
        return 8


def getlabel(d, kind):
    global law
    global accu

    # 做单标签
    if kind == 'law':
        # 返回多个类的第一个
        return law[str(d['meta']['relevant_articles'][0])]
    if kind == 'accu':
        return accu[d['meta']['accusation'][0]]

    if kind == 'time':
        return gettime(d['meta']['term_of_imprisonment'])



_PAD="_PAD"
_GO="_GO"
_EOS="EOS"
_UNK="_UNK"
_START_VOCAB=[_PAD,_GO,_EOS,_UNK]
PAD_ID=0
GO_ID=1
EOS_ID=2
UNK_ID=3
#accu:左边是案由代码，右边是第几个案由

def cut_text(alltext):
    count = 0
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append([word for word in jieba.cut(text) if len(word)>1])

    return train_text

def read_train_or_valid_Data(path):
    fin = open(path, 'r', encoding='utf8')

    alltext = []

    accu_label = []
    law_label = []
    time_label = []

    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])

        accu_label.append(getlabel(d, 'accu'))
        law_label.append(getlabel(d, 'law'))
        time_label.append(getlabel(d, 'time'))
        line = fin.readline()
    fin.close()
    alltext=cut_text(alltext)


    return alltext, accu_label, law_label, time_label



def create_vacabulary(X,max_vocabulay_size=10000):
    vocab={}
    for sentence in X:
        for word in sentence:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1


    vocab_list=_START_VOCAB+sorted(vocab,key=vocab.get,reverse=True)

    vocab_list=vocab_list[:max_vocabulay_size]


    vocab_dict=dict((x,y) for (y,x) in enumerate(vocab_list))
    #key is word,value is index
    rev_vocab_dict={v:k for k,v in vocab_dict.items()}
    # key is index,value is word

    return vocab_list,vocab_dict,rev_vocab_dict

def sentence_to_token_ids(sentence,vocab_dict):
    return [vocab_dict.get(word,UNK_ID) for word in sentence]

def data_to_token_ids(X,vocab_dict):
    max_len=max(len(sentence) for sentence in X)
    seq_lens=[]
    data_as_tokens=[]
    for line in X:
        token_ids=sentence_to_token_ids(line,vocab_dict)
        #Padding
        data_as_tokens.append(token_ids+[PAD_ID]*(max_len-len(token_ids)))

        seq_lens.append(len(token_ids))
    return data_as_tokens,seq_lens

def train_and_valid_data(alltext,vocab_dict,y,num):
    alltext,seq_lens=data_to_token_ids(alltext,vocab_dict)

    alltext=np.array(alltext)
    seq_lens=np.array(seq_lens)
    data_size=len(alltext)


    one_hot_index=np.arange(len(y))*num+y
    y_one_hot=np.zeros((len(y),num))
    y_one_hot.flat[one_hot_index]=1


    shuffle_indices=np.random.permutation(np.arange(data_size))


    alltext,y,seq_lens=alltext[shuffle_indices],y_one_hot[shuffle_indices],seq_lens[shuffle_indices]

    return alltext,y,seq_lens

def generate_epoch(X,y,seq_lens,num_epochs,batch_size):
    for epoch_num in range(num_epochs):
        yield generate_batch(X,y,seq_lens,batch_size)

def generate_batch(X,y,seq_lens,batch_size):
    data_size=len(X)
    num_batches=(data_size//batch_size)
    for batch_num in range(num_batches):
        start_index=batch_num*batch_size
        end_index=min((batch_num+1)*batch_size,data_size)
        yield X[start_index:end_index],y[start_index:end_index],seq_lens[start_index:end_index]