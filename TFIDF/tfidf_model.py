#!/usr/bin/python
# -*- coding:UTF-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def mini_data_tfidf(word_list1):
    count_vec = CountVectorizer(token_pattern=r"(?u)\b[^/]+\b")
    #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i个文本下的词频
    X_count_train = count_vec.fit_transform(word_list1)
    X_count_train = X_count_train.toarray()
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(X_count_train)
    word = count_vec.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

def large_data_tfidf(word_list1):
    #当数据较大时，count_vec.fit_transform(word_list1)"这里触发memoryerror
    #对数据量进行了去重，然后继续用fit_transform的方式，这时候我们的词库已经锁定了。
    #之后就是在保证数据量还是130W+的前提下如何解决词频统计的问题，这时候我就用到了countvectorize里的transform,就是用已经锁定好的词库，
    # 来对句子transform一次词频矩阵,并运用迭代器。
    word_list2 = list(set(word_list1))  # 对word_list1进行去重
    count_vec = CountVectorizer(token_pattern=r"(?u)\b[^/]+\b")#这一步主要是为了建立词库
    X_count_train = count_vec.fit_transform(word_list2)
    X_count_train = X_count_train.toarray()
    for i in range(len(word_list1)):
        X_count_test=count_vec.transform(word_list1[i])
        print(X_count_test.toarray())
    #注意点：不能在fit_transform那里进行迭代。因为它只支持一次性输入，也就是说我第一次放进的数据获得的词库
    #会被第二次放进的数据获得的词库所覆盖。类似于fit_transform就是我们的训练器，transform就是我们的测试器。




