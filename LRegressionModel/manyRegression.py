#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
from numpy.linalg import det
#计算行列式
from numpy.linalg import inv
from numpy import mat
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self):
        pass

    #这里的优化方式直接采用了闭式求解,没有采用梯度下降的方法
    def train(self,x_train,y_train):
        #用mat函数转换之后才可以进行一些线性代数的操作,比如计算行列式或者逆矩阵等
        #线性代数中向量一般默认的是行向量,加上转置就是列向量了
        x_mat=mat(x_train).T
        #数组转化为矩阵,并求转置赋值给x_mat
        y_mat=mat(y_train).T
        [m,n]=x_mat.shape
        x_mat=np.hstack((x_mat,mat(np.ones(m,1))))
        #hstack是左右合并,m表示样本个数,也就是每行对应我们的一个样本,拼接了一个维度代表常数项
        self.weight=mat(random.rand(n+1,1))
        if det(x_mat.T*x_mat)==0:
            print('the det of xTx is 0')
            return
        else:
            self.weight=inv(x_mat.T*x_mat)*x_mat.T*y_mat
        return  self.weight

#局部加权线性回归算法代价极高,对于每一个要预测的点,都要根据整个训练集合重新训练一个线性回归模型.具体是先根据核方法计算每个样本权重,再根据权重和最小二乘损失函数偏导求解线性回归系数
    def locally_weighted_linear_regression(self,test_point,x_train,y_train,k=1.0):
        x_mat = mat(x_train).T
        [m, n] = x_mat.shape
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        y_mat = mat(y_train).T
        #标签也转化为一个行向量
        #将一个测试数据一维数组转化为矩阵
        test_point_mat = mat(test_point)
        #为什么要拼接,原因是系数除了w还有b,拼接的数据对应和常数项b相乘
        test_point_mat = np.hstack((test_point_mat, mat([[1]])))

        self.weight = mat(np.zeros((n + 1, 1)))
        #weight是线性回归模型的权重系数
        weights = mat(np.eye((m)))
        #这里的weights是每个样本的权重,可以由高斯核函数获得,默认为m维度的对角矩阵
        test_data = np.tile(test_point_mat, [m, 1])
        #test_point_mat在第0维度重复m次,1维重复一次形成矩阵,目的是为了方便与训练数据矩阵计算得到距离

        distances = (test_data - x_mat) * (test_data - x_mat).T / (n + 1)
        distances = np.exp(distances / (-2 * k ** 2))
        #取出的对角线元素就是要与测当前样本test_point时,其他样本的权重
        weights = np.diag(np.diag(distances))
        # weights = distances * weights
        xTx = x_mat.T * (weights * x_mat)
        if det(xTx) == 0.0:
            print('the det of xTx is equal to zero.')
            return
        self.weight = xTx.I * x_mat.T * weights * y_mat
        return test_point_mat * self.weight

    def ridge_regression(self, x_train, y_train, lam=0.2):
        x_mat = mat(x_train).T#加转置默认转为列向量
        [m, n] = np.shape(x_mat)
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        y_mat = mat(y_train).T
        self.weight = mat(random.rand(n + 1, 1))
        xTx = x_mat.T * x_mat + lam * mat(np.eye(n))
        if det(xTx) == 0.0:
            print("the det of xTx is zero!")
            return
        self.weight = xTx.I * x_mat.T * y_mat
        return self.weight

    def lasso_regression(self, x_train, y_train, eps=0.01, itr_num=100):
        x_mat = mat(x_train).T
        [m, n] = np.shape(x_mat)
        x_mat = (x_mat - x_mat.mean(axis=0)) / x_mat.std(axis=0)
        x_mat = np.hstack((x_mat, mat(np.ones((m, 1)))))
        #标准化后再拼接
        y_mat = mat(y_train).T
        y_mat = (y_mat - y_mat.mean(axis=0)) / y_mat.std(axis=0)
        self.weight = mat(random.rand(n + 1, 1))
        best_weight = self.weight.copy()
        for i in range(itr_num):
            print(self.weight.T)
            lowest_error = np.inf
            for j in range(n + 1):
                #有n+1个参数需要学习
                for sign in [-1, 1]:
                    weight_copy = self.weight.copy()
                    weight_copy[j] += eps * sign
                    #将第j个参数上下微调eps
                    y_predict = x_mat * weight_copy
                    error = np.power(y_mat - y_predict, 2).sum()
                    if error < lowest_error:
                        lowest_error = error
                        best_weight = weight_copy
            self.weight = best_weight
        return self.weight

    def lwlr_predict(self, x_test, x_train, y_train, k=1.0):
        m = len(x_test)
        y_predict = mat(np.zeros((m, 1)))
        for i in range(m):
            #这里对每个待测样本都会训练一个局部线性回归模型
            y_predict[i] = self.locally_weighted_linear_regression(x_test[i], x_train, y_train, k)
        return y_predict

    def lr_predict(self, x_test):
        m = len(x_test)
        x_mat = np.hstack((mat(x_test).T, np.ones((m, 1))))
        return x_mat * self.weight

    def plot_lr(self, x_train, y_train):
        x_min = x_train.min()
        x_max = x_train.max()
        y_min = self.weight[0] * x_min + self.weight[1]
        y_max = self.weight[0] * x_max + self.weight[1]
        plt.scatter(x_train, y_train)
        plt.plot([x_min, x_max], [y_min[0, 0], y_max[0, 0]], '-g')
        plt.show()

    def plot_lwlr(self, x_train, y_train, k=1.0):
        x_min = x_train.min()
        x_max = x_train.max()
        x = np.linspace(x_min, x_max, 1000)
        y = self.lwlr_predict(x, x_train, y_train, k)
        plt.scatter(x_train, y_train)
        plt.plot(x, y.getA()[:, 0], '-g')
        plt.show()

    def plot_weight_with_lambda(self, x_train, y_train, lambdas):
        weights = np.zeros((len(lambdas),))
        for i in range(len(lambdas)):
            self.ridge_regression(x_train, y_train, lam=lambdas[i])
            weights[i] = self.weight[0]#这里的weight是一个(n+1)*1的矩阵
        plt.plot(np.log(lambdas), weights)
        #观察lamda和weight的关系,是否lamda越大,weight越小
        plt.show()

def main():
    data = pd.read_csv('/home/LiuYao/Documents/MarchineLearning/regression.csv')
    data = data / 30
    x_train = data['x'].values
    y_train = data['y'].values
    regression = LinearRegression()
    # regression.train(x_train, y_train)
    # y_predict = regression.predict(x_train)
    # regression.plot(x_train, y_train)
    # print '相关系数矩阵：', np.corrcoef(y_train, np.squeeze(y_predict))
    # y_predict = regression.lwlr_predict([[15],[20]], x_train, y_train, k=0.1)
    # print y_predict
    # regression.ridge_regression(x_train, y_train, lam=3)
    # regression.plot_lr(x_train, y_train)
    regression.lasso_regression(x_train, y_train, itr_num=1000)
    regression.plot_lr(x_train, y_train)

if __name__ == '__main__':
    main()