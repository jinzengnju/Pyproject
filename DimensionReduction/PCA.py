#!/usr/bin/python
# -*- coding:UTF-8 -*-
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD,KernelPCA
def read_data():
    corpus=[]
    labels=[]
    file=open('/home/jin/data/fengyi/data1-6.txt')
    for line in file.readlines():
        corpus.append(line.strip().split('||')[1])
        labels.append([int(i) for i in line.strip().split('||')[3].split(',')])
    tfidf_vector_model=joblib.load("/home/jin/data/fengyi/tfidfvectorizer.m")
    tfidf_corpus=tfidf_vector_model.transform(corpus)
    print("矩阵转换")
    #tfidf_corpus=tfidf_corpus.toarray()
    print(type(tfidf_corpus))
    return tfidf_corpus
def get_newData(corpus):
    pca=TruncatedSVD(n_components=1000)
    newData=pca.fit_transform(corpus)
    print(newData.shape)

def KernalPCA(corpus):
    kpca=KernalPCA(kernal='cosine',n_components=1000)
    X_transformed=kpca.fit_transform(X)
    # 核PCA是先利用核技巧把数据投射到高维空间，然后利用PCA进行降维。这种方法对非线性数据进行降维。
    # 核函数有：polynomial、rbf、sigmoid、cosine、precomputed

if __name__=='__main__':
    corpus=read_data()
    get_newData(corpus)
