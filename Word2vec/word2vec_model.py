#!/usr/bin/python
# -*- coding:UTF-8 -*-
from gensim.models import Word2Vec
import jieba
import json
import argparse

def get_corpus(path):
    fin = open(path, 'r', encoding='utf8')
    alltext = []
    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        line = fin.readline()
    fin.close()
    alltext = cut_text(alltext)
    return alltext
def cut_text(alltext):
    train_text = []
    for text in alltext:
        train_text.append([word for word in jieba.cut(text) if len(word) > 1])
    return train_text

def train_word2vec(sentensces):
    model=Word2Vec(sentences=sentensces,size=150,window=5,min_count=3,workers=2)
    model.save("graduate/w2v_model")

def load_model():

    model=Word2Vec.load("w2v_model")
    try:
        print(model['nmb'])
    except KeyError:
        print("meiyou")

paser=argparse.ArgumentParser(description='manual to this script')
paser.add_argument('--input_corpus',type=str,default=None)
args=paser.parse_args()

if __name__=='__main__':
    all_sentence=get_corpus(args.input_corpus)
    train_word2vec(sentensces=all_sentence)
    #load_model()
