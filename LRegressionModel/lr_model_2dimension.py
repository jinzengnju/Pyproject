#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np

def init_data():
    data=np.loadtxt('data.csv',delimiter=',')
    return  data

def linear_regression():
    learning_rate=0.01
    initial_b=0
    initial_m=0
    num_iter=1000

    data=init_data()
    [b,m]=optimizer(data,initial_b,initial_m,learning_rate,num_iter)

    print(b,m)
    return b,m

#batch梯度下降
def optimizer(data,initial_b,initial_m,learning_rate,num_iter):
    b=initial_b
    m=initial_m

    for i in range(num_iter):
        #我们这里用的是batch梯度下降,如果要用mini-batch梯度下降需要修改代码
        b,m=compute_gradient(b,m,data,learning_rate)
        if i%100==0:
            print(i,computer_error(b,m,data))
    return [b,m]


def compute_gradient(b_cur, m_cur, data, learning_rate):
    b_gradient=0
    m_gradient=0
    N=float(len(data))

    #batch梯度下降,遍历完整个数据集再进行一次梯度更新
    for i in range(0,len(data)):
        x=data[i,0]
        y=data[i,1]

        b_gradient+=-(2/N)*(y-((m_cur*x)+b_cur))
        m_gradient+=-(2/N)*x*(y-((m_cur*x)+b_cur))
        #求出的梯度有一个符号,但是在梯度更新的时候要减去梯度,所以又会变成加号

    new_b=b_cur-(learning_rate*b_gradient)
    new_m=m_cur-(learning_rate*m_gradient)
    return [new_b,new_m]


def computer_error(b,m,data):
    totalError=0
    x=data[:,0]
    y=data[:,1]

    totalError=(y-m*x-b)**2
    totalError=np.sum(totalError,axis=0)
    return totalError/len(data)

def optimizer_two(data,initial_b,initial_m,learning_rate,num_iter):
    b=initial_b
    m=initial_m
    while True:
        before=computer_error(b,m,data)
        b,m=compute_gradient(b,m,data,learning_rate)
        after=computer_error(b,m,data)
        if abs(after-before)<0.00000001:
            break
        return [b,m]
def compute_gradient_two(b_cur, m_cur, data, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    delta = 0.0000001
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        # 利用导数的定义来计算梯度
        #这里的error并不是实际的损失函数(平均损失),是整个样本集上的损失
        b_gradient = (error(x, y, b_cur + delta, m_cur) - error(x, y, b_cur - delta, m_cur)) / (2 * delta)
        m_gradient = (error(x, y, b_cur, m_cur + delta) - error(x, y, b_cur, m_cur - delta)) / (2 * delta)
    #为什么要除以N,原因是我们的损失函数中是平均损失函数,除了样本个数的
    b_gradient = b_gradient / N
    m_gradient = m_gradient / N
    new_b = b_cur - (learning_rate * b_gradient)
    new_m = m_cur - (learning_rate * m_gradient)
    return [new_b, new_m]


def error(x, y, b, m):
    return (y - (m * x) - b) ** 2

if __name__=='__main__':
    linear_regression()

