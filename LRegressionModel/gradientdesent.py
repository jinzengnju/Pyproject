#!/usr/bin/python
# -*- coding:UTF-8 -*-

#梯度下降代码
#训练数据10个,每个是2维度的,加上常数项1,就3维
import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
def batchGradientDecsent(x,y,theta,alpha,m,maxIterations):
    X_trains=x.transpose()
    #要对训练数据进行转置
    for i in range(0,maxIterations):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        gradient=np.dot(X_trains,loss)/m
        #对所有的样本进行球和,然后除以样本数
        #得到的loss是10*1的,需要用3*10的矩阵乘以loss,才能得到theta三个维度的梯度.这里就是为什么前面需要转置的原因
        theta=theta-alpha*gradient
    return  theta

def StochasticGradientDescent(x, y, theta, alpha, m, maxIterations):
    data=[]
    for i in range(10):
        data.append(i)
    X_trains=x.transpose
    #将样本数据变为3*10(一般情况下aT默认为列向量),每一列是一个训练样本
    for i in range(0,maxIterations):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        #这里算出的loss是一个10维的列向量,包含所有样本数据的loss,下面会随机采样一个数据
        index=random.sample(data,1)
        #任意选取一个样本点,得到他的下标,便于下面找到X_trains的对应列
        index1=index[0]
        #因为返回回来的是list,取出一个第一维度的数即可
        gradient=loss[index1]*x[index1]
        #只取这一个点进行更新计算,得到的gradient是一个1*3的行向量,所以下面需要转置
        theta=theta-alpha*gradient.T
    return theta

def predict(x,theta):
    m,n=np.shape(x)
    X_test=np.ones((m,n+1))
    X_test[:,:-1]=x
    res=np.dot(X_test,theta)
    return res

trainData = np.array([[1.1,1.5,1],[1.3,1.9,1],[1.5,2.3,1],[1.7,2.7,1],[1.9,3.1,1],[2.1,3.5,1],[2.3,3.9,1],[2.5,4.3,1],[2.7,4.7,1],[2.9,5.1,1]])
trainLabel = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.1
maxIteration = 5000
#下面返回的theta就是学到的theta
theta = batchGradientDecsent(trainData, trainLabel, theta, alpha, m, maxIteration)
print ("theta = ",theta)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print (predict(x, theta))
theta = StochasticGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
print ("theta = ",theta)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print (predict(x, theta))



path=u'./samples.mat'
mat=sio.loadmat(path)
dataset=mat['samples']
batch_size=1

def random_get_samples(mat,batch_size):
    batch_id=random.sample(range(mat.shape[0]),batch_size)
    #从mat.shape[0]个数据中随机选择batch_size个
    ret_batch=mat[batch_id,0]
    ret_line=mat[batch_id,1]
    #返回训练数据的X和y
    return ret_batch,ret_line

params = {
    'w1': np.random.normal(size=(1)),
    'b': np.random.normal(size=(1))
}
def predict(x):
    return params['w1']*x+params['b']
learning_rate = 0.001
for i in range(3000):
    batch,line=random_get_samples(dataset,batch_size)
    y_pred=predict(batch)
    y_pred=np.reshape(y_pred,(batch_size,1))
    line=np.reshape(line,(batch_size,1))
    batch = np.reshape(batch, (batch_size, 1))
    delta = line - y_pred
    #deta维度为batch_size*1,batch的维度为batch_size*1
    params['w1'] = params['w1'] + learning_rate * np.sum(delta * batch) / batch_size
    params['b'] = params['b'] + learning_rate * np.sum(delta) / batch_size
    if i % 100 == 0:
        print(np.sum(np.abs(line - y_pred)) / batch_size)

print(params['w1'])
print(params['b'])
x = dataset[:, 0]
line = dataset[:, 1]
y = params['w1']*x+params['b']
plt.figure(1)
plt.plot(x, line, 'b--')
plt.plot(x, y, 'r--')
plt.show()




