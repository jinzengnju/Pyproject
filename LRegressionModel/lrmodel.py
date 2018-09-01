#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
import os
from numpy.linalg import det
from numpy.linalg import inv
from numpy import mat
from numpy import random
import math
#使程序输出不是科学计数法的形式，而是浮点型的数据输出
np.set_printoptions(suppress=True)

def init_data():
    path=os.path.abspath('..')
    data=np.loadtxt(os.path.join(path,'data/data_csv'),delimiter=',')
    return data

def ridge_regression(x_train, y_train, lam=0.2):
    x_mat = mat(x_train)
    [m, n] = np.shape(x_mat)
    print("行:%d,列:%d"%(m,n))
    x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))

    y_mat = mat(y_train).T
    print("标签的行数:%d"%np.shape(y_mat)[0])

    weight = mat(random.rand(n + 1, 1))
    xTx = x_mat.T * x_mat + lam * mat(np.eye(n+1))
    if det(xTx) == 0.0:
        print("the det of xTx is zero!")
        return
    weight = xTx.I * x_mat.T * y_mat
    return weight

def lr_predict(self, x_test):
    m = len(x_test)
    x_mat = np.hstack((mat(x_test).T, np.ones((m, 1))))
    return x_mat * self.weight

def main():
    data = init_data()
    x_train = data[:,1:] #得到的是二维数组,每行代表一条数据
    y_train = data[:,0] #得到的是一维数组
    weight=ridge_regression(x_train, y_train)
    print(weight)
    print(np.shape(weight))


if __name__=='__main__':
    main()
