#!/usr/bin/python
# -*- coding:UTF-8 -*-
import jieba
import json
import argparse
_PAD="_PAD"
_GO="_GO"
_EOS="EOS"
_UNK="_UNK"
_START_VOCAB=[_PAD,_GO,_EOS,_UNK]
PAD_ID=0
GO_ID=1
EOS_ID=2
UNK_ID=3

def create_vacabulary(X,max_vocabulay_size):
    vocab={}
    for sentence in X:
        for word in sentence:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1
    vocab_temp=sorted(vocab.items(),key=lambda item:item[1],reverse=True)
    vocab_list=_START_VOCAB+[e[0] for e in vocab_temp]
    vocab_list=vocab_list[:max_vocabulay_size]
    #这里的vocab_list只会保存max_vocabulay_size个词
    vocab_dict=dict((x,y) for (y,x) in enumerate(vocab_list))
    #key is word,value is index即是第几个词
    rev_vocab_dict={v:k for k,v in vocab_dict.items()}
    # key is index,value is word
    return vocab_list,vocab_dict,rev_vocab_dict


def cut_text(alltext):
    count = 0
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append([word for word in jieba.cut(text) if len(word)>1])
    return train_text

def get_dict(input_corpus,en_vocab_size):
    fin = open(input_corpus, 'r', encoding='utf8')
    alltext = []
    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        line = fin.readline()
    fin.close()
    alltext = cut_text(alltext)
    vocab_list, vocab_dict, rev_vocab_dict=create_vacabulary(alltext,en_vocab_size)
    return vocab_dict

def write_dict():
    vocab = get_dict(args.input_corpus, args.en_vocab_size)
    f_write = open(args.output_dict_path, 'w')
    json.dump(vocab, f_write, ensure_ascii=False)


paser=argparse.ArgumentParser(description='manual to this script')
paser.add_argument('--input_corpus',type=str,default=None)
paser.add_argument('--en_vocab_size',type=int,default=10000)
paser.add_argument('--output_dict_path',type=str,default=None)
args=paser.parse_args()

if __name__=='__main__':
    write_dict()