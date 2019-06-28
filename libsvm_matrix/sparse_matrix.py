#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
import _pickle as pkl

def libsvm_2_coo(libsvm_data, shape):
    coo_rows = []
    coo_cols = []
    coo_data = []
    n = 0
    for x, d in libsvm_data:
        coo_rows.extend([n] * len(x))
        coo_cols.extend(x)
        coo_data.extend(d)
        n += 1
    coo_rows = np.array(coo_rows)
    coo_cols = np.array(coo_cols)
    coo_data = np.array(coo_data)
    temp= coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)
    return temp

def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        print(type(indices))
        values = csr_mat.data
        shape = csr_mat.shape
        print(shape)
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs


if __name__=="__main__":
    y=[0,1,1]
    X=[[1,2,3],
       [4,5,6],
       [7,8,9]]
    D = [[0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6],
         [0.7, 0.8, 0.9]]
    y = np.reshape(np.array(y), [-1])
    X = libsvm_2_coo(zip(X, D), (3, 10)).tocsr()
    csr_2_input(X)



