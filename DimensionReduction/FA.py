#!/usr/bin/python
# -*- coding:UTF-8 -*-
#因子分析法
from  sklearn import decomposition
def FA_method(X):
    fa=decomposition.FactorAnalysis()
    fa.fit_transform(X)
    #因子分析是另一种降维方法，其假定变量中有一些公共因子，公共因子的数目是少于变量的，这样就达到降维的目的