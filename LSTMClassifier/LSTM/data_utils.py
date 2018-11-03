#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
import json
def init():
	f = open('law.txt', 'r', encoding = 'utf8')
	law = {}
	lawname = {}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
        #lawname的键是law文件行号索引，对应值是law的名称即第几条
		law[line.strip()] = len(law)
        #law的键是第几条，值是对应的类标记
		line = f.readline()
	f.close()
	f = open('accu.txt', 'r', encoding = 'utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()
        #accuname的键是accu文件的行号索引（也即类标记），对应的值是案由名称
		accu[line.strip()] = len(accu)
        #accu对应的键是俺有名称，对应的值是类标记种类
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
        return law[str(d['meta']['relevant_articles'])]
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

def create_vacabulary(X,max_vocabulay_size=10000):
    vocab={}
    for sentence in X:
        for word in sentence:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1
    vocab_list=_START_VOCAB+sorted(vocab,key=vocab.get(),reverse=True)
    vocab_list=vocab_list[:max_vocabulay_size]
    #这里的vocab_list只会保存max_vocabulay_size个词
    vocab_dict=dict((x,y) for (y,x) in enumerate(vocab_list))
    #key is word,value is index即是第几个词
    rev_vocab_dict={v:k for k,v in vocab_dict.items()}
    # key is index,value is word
    return vocab_list,vocab_dict,rev_vocab_dict


def sentence_to_token_ids(sentence,vocab_dict):
    return [vocab_dict.get(word,UNK_ID) for word in sentence]

#这里传入的X是分词后的结果
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

