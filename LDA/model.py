#!/usr/bin/python
# -*- coding:UTF-8 -*-
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import jieba
import re
import argparse
import json

def get_context(input_corpus):
    fin = open(input_corpus, 'r', encoding='utf8')
    alltext = []
    line = fin.readline()
    while line:
        d = json.loads(line)
        alltext.append(d['fact'])
        line = fin.readline()
    fin.close()
    return alltext

def cut_text(alltext):
    train_text = []
    for text in alltext:
        text = re.sub('[^(\\u4e00-\\u9fa5)]', '', text)
        text = re.sub('(?i)[^a-zA-Z0-9\u4E00-\u9FA5]', '', text)
        train_text.append([word for word in jieba.cut(text) if len(word)>1])
    return train_text

def train_model(input_path,model_path,dict_path):
    alltext=get_context(input_path)
    train_word=cut_text(alltext)
    dictionary=corpora.Dictionary(train_word)
    dictionary.save(dict_path)
    corpus=[dictionary.doc2bow(text) for text in train_word]
    lda=LdaModel(corpus=corpus,id2word=dictionary,num_topics=256,chunksize=3000,passes=2,iterations=50)
    lda.save(model_path)

def test_newdoc(model_path,dict_path):
    #other_texts为分词后的二维列表
    lda=LdaModel.load(model_path)
    dictionary=Dictionary.load(dict_path)
    print(dictionary)
    #other_corpus=[dictionary.doc2ow(text) for text in other_texts]
    #unseen_doc=other_corpus[0]
    #vector=lda[unseen_doc]

paser = argparse.ArgumentParser(description='manual to this script')
paser.add_argument('--input_corpus', type=str, default=None)
paser.add_argument('--dict_path', type=str, default=None)
paser.add_argument('--model_path', type=str, default=None)
args = paser.parse_args()

if __name__=='__main__':
    train_model(args.input_corpus,args.model_path,args.dict_path)
    #test_newdoc(args.model_path,args.dict_path)
