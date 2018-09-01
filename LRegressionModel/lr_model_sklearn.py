#!/usr/python/bin
# -*- coding:UTF-8 -*-
from sklearn.linear_model import Ridge,RidgeCV,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def init_data():
    path=os.path.abspath('..')
    data=np.loadtxt(os.path.join(path,'data/Feature30/data_csv'),delimiter=',')
    return data

def ridgecv():
    data = init_data()
    X = data[:,1:]
    #得到的是二维数组,每行代表一条数据,岭回归训练的时候X是一个二维数组,每行代表一条数据.Y是一个一位数组
    Y = data[:,0] #得到的是一维数组
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    # 利用train_test_split方法，将X,y随机划分问，训练集（X_train），训练集标签（X_test），测试卷（y_train）
    alphas = [0.03,0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 1, 1.5, 2]
    ridgecv = RidgeCV(alphas,store_cv_values=True)
    ridgecv.fit(X_train,y_train)
    # #自己通过计算选择
    # smallest_idx=ridgecv.cv_values_.mean(axis=0).argmin()
    #要使用cv_values_需要在建立ridgecv指定store_cv_values=True
    # #在第0维度上进行平均,也就是计算不同的alpha下,在所有样本上的平均误差,最后形成12维数组,因为alpha可以取到12个
    # print("自己计算",alphas[smallest_idx])
    smallest_idx = ridgecv.cv_values_.mean(axis=0).argmin()
    f,ax=plt.subplots(figsize=(7,5))
    ax.set_title(r"various values of a")
    xy=(alphas[smallest_idx],ridgecv.cv_values_.mean(axis=0)[smallest_idx])
    xytext=(xy[0]+.01,xy[1]+.1)
    ax.annotate(r'choose this a',xy=xy,xytext=xytext,arrowprops=dict(facecolor='black',shrink=0,width=0))
    #https://blog.csdn.net/qq_30638831/article/details/79938967
    ax.plot(alphas,ridgecv.cv_values_.mean(axis=0))
    plt.show()

    print("sklearn指定最优alpha值:",ridgecv.alpha_)
    print(ridgecv.coef_)
    print(ridgecv.intercept_)
    test_Y_pred=ridgecv.predict(X)
    print("测试集MSE:", mean_squared_error(Y, test_Y_pred))
    #print后面接了一个逗号表示字符串的链接,后面就不用强行转化类型了

def ridge_with_single_search():
    data=init_data()
    X = data[:, 1:]
    Y = data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    alphas = [0,0.03,0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 1, 1.5, 2]
    _ridge = Ridge()
    for a in alphas:
        _ridge.set_params(alpha=a)
        _ridge.fit(X_train,y_train)
        test_y_pred=_ridge.predict(X_test)
        print("alpha is %f,测试集MSE is %f"%(a,mean_squared_error(y_test, test_y_pred)))

def lr():
    data = init_data()
    X = data[:, 1:]
    # 得到的是二维数组,每行代表一条数据,岭回归训练的时候X是一个二维数组,每行代表一条数据.Y是一个一位数组
    Y = data[:, 0]  # 得到的是一维数组
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    lrmodel=LinearRegression()
    lrmodel.fit(X_train,y_train)
    y_test_pred=lrmodel.predict(X_test)
    print("测试集MSE:", mean_squared_error(y_test, y_test_pred))


    # alphas = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 1, 1.5, 2]
    # ridgecv = RidgeCV(alphas,store_cv_values=True)
    # ridgecv.fit(X_train_ploy, y_train)
    # smallest_idx = ridgecv.cv_values_.mean(axis=0).argmin()
    # f, ax = plt.subplots(figsize=(7, 5))
    # ax.set_title(r"various values of a")
    # xy = (alphas[smallest_idx], ridgecv.cv_values_.mean(axis=0)[smallest_idx])
    # xytext = (xy[0] + .01, xy[1] + .1)
    # ax.annotate(r'choose this a', xy=xy, xytext=xytext, arrowprops=dict(facecolor='black', shrink=0, width=0))
    # # https://blog.csdn.net/qq_30638831/article/details/79938967
    # ax.plot(alphas, ridgecv.cv_values_.mean(axis=0))
    # plt.show()
    # print("sklearn指定最优alpha值:", ridgecv.alpha_)
    # print(ridgecv.coef_)
    # print(ridgecv.intercept_)
    # test_Y_pred = ridgecv.predict(X_test_ploy)
    # print("测试集MSE:", mean_squared_error(y_test, test_Y_pred))
from sklearn.preprocessing import PolynomialFeatures
import json
def MyNonlinear():
    data = init_data()
    X = data[:,1:]
    Y = data[:,0] #得到的是一维数组

    poly_reg = PolynomialFeatures(degree=2)
    X_ploy=poly_reg.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_ploy, Y, test_size=0.3, random_state=42)
    alphas = [10.0]
    ridgecv = RidgeCV(alphas,store_cv_values=True)
    ridgecv.fit(X_train,y_train)
    smallest_idx = ridgecv.cv_values_.mean(axis=0).argmin()
    f,ax=plt.subplots(figsize=(7,5))
    ax.set_title(r"various values of a")
    xy=(alphas[smallest_idx],ridgecv.cv_values_.mean(axis=0)[smallest_idx])
    xytext=(xy[0]+.01,xy[1]+.1)
    ax.annotate(r'choose this a',xy=xy,xytext=xytext,arrowprops=dict(facecolor='black',shrink=0,width=0))
    #https://blog.csdn.net/qq_30638831/article/details/79938967
    ax.plot(alphas,ridgecv.cv_values_.mean(axis=0))
    plt.show()

    print("sklearn指定最优alpha值:",ridgecv.alpha_)
    #print(X_ploy)
    print("features:",poly_reg.get_feature_names())

    print("升维后长度:",poly_reg.n_output_features_)
    print("权重长度",ridgecv.coef_.__len__())
    print("截距",ridgecv.intercept_)

    f=open("feature_weights",'w')
    for e in ridgecv.coef_:
        f.write(str(e)+'\n')

    test_pred=ridgecv.predict(X_ploy)
    print("测试集MSE:", mean_squared_error(Y, test_pred))
    for e in test_pred:
        print(e)
    f.close()


if __name__=='__main__':
    #ridgecv()
    #ridge_with_single_search()
    #lr()
    MyNonlinear()


# ax = plt.gca()
# ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
# ax.plot(alphas, coefs)
# ax.set_xscale('log')
#把当前的图形x轴设置为对数坐标
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show()

