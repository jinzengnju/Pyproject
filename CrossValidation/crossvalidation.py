#!/usr/bin/python
# -*- coding:UTF-8 -*-


#https://blog.csdn.net/cymy001/article/details/78578665
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
iris=load_iris()


#随机种子为0或者不填时,每次调用train_test_split生成的数据都是不同的


#对数据集进行指定次数的交叉验证并为每次验证效果评测,默认是以 scoring=’f1_macro’进行评测的
# from sklearn.model_selection import cross_val_score
# clf=svm.SVC(kernel='linear',C=1)
# scores=cross_val_score(clf,iris.data,iris.target,cv=5)
#用交叉验证的目的是为了得到可靠稳定的模型
#对C=1的SVC模型进行5折交叉验证,那么就会训练出5个模型,每个模型上都会有一个f1score,最终返回的是一个5维数组score

def simple_gridsearch():
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
    best_score=0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            #对于每种参数组合都训练一个svc
            svm=SVC(gamma=gamma,C=C)
            svm.fit(X_train,y_train)
            #在测试集合上测试
            score=svm.score(X_test,y_test)
            if score>best_score:
                best_score=score
                best_params={'C':C,'gamma':gamma}
    print("best score: ", best_score)
    print("best parameters: ", best_params)

def gridsearch_with_crossvalidation():
    X_trainval, X_test, y_trainval, y_test=train_test_split(iris.data,iris.target,random_state=0)
    X_train, X_valid, y_train, y_valid=train_test_split(X_trainval,y_trainval,random_state=1)
    best_score=0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm=SVC(gamma=gamma,C=C)
            scores=cross_val_score(svm,X_trainval,y_trainval,cv=5)
            score=np.means(scores)
            if score>best_score:
                best_score=score
                best_params={'C':C,'gamma':gamma}
    print('网格搜索for循环<有cross_val_score交叉验证>获得的最好参数组合:',best_params)
    print(' ')
    svmf=SVC(**best_params)
    svmf.fit(X_trainval,y_trainval)
    print('网格搜索<有交叉验证>获得的最好估计器,在训练验证集上没做交叉验证的得分:', svmf.score(X_trainval, y_trainval))
    print(' ')
    #没有交叉验证的的得分直接是svmf.score,有交叉验证的得分是cross_val_score
    scores=cross_val_score(svmf, X_trainval, y_trainval, cv=5)
    print('网格搜索<有交叉验证>获得的最好估计器,在训练验证集上做交叉验证的平均得分:', np.mean(scores))  # 交叉验证的平均accuracy
    print(' ')
    print('网格搜索<有交叉验证>获得的最好估计器,在测试集上的得分:', svmf.score(X_test, y_test))  #####

#构造参数字典,代替双层for循环进行网格搜索
def params_search():
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(iris.data, iris.target,
                                                                  random_state=0)  # default=0.25
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # 网格搜索+交叉验证
    grid_search.fit(X_trainvalid, y_trainvalid)
    print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合', grid_search.best_params_)
    print(' ')
    print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分', grid_search.score(X_trainvalid, y_trainvalid))  #####
    print(' ')
    print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分', grid_search.best_score_)  # ?????
    # print(' ')
    # print('BEST_ESTIMATOR:',grid_search.best_estimator_)   #对应分数最高的估计器
    print(' ')
    print('GridSearchCV交叉验证网格搜索获得的最好估计器,在测试集上的得分', grid_search.score(X_test, y_test))  #####

from sklearn.model_selection import GridSearchCV
#嵌套交叉验证：字典参数+cross_val_score
def params_search2():
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)
    #这里是针对整个数据集合
    # 选定网格搜索的每一组超参数,对训练集与测试集的交叉验证(cross_val_score没指定数据集合分割的默认情况)
    print("Cross-validation scores: ", scores)
    print("Mean cross-validation score: ", scores.mean())


def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    # for each split of the data in the outer cross-validation
    # (split method returns indices)
    for training_samples, test_samples in outer_cv.split(X, y):
        # find best parameter using inner cross-validation:网格搜索外层cv
        best_parms = {}
        best_score = -np.inf
        # iterate over parameters
        for parameters in parameter_grid:
            # accumulate score over inner splits
            cv_scores = []
            # iterate over inner cross-validation
            for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
                # build classifier given parameters and training data交叉验证内层cv
                clf = Classifier(**parameters)
                clf.fit(X[inner_train], y[inner_train])
                # evaluate on inner test set
                score = clf.score(X[inner_test], y[inner_test])
                cv_scores.append(score)
            # compute mean score over inner folds
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # if better than so far, remember parameters
                best_score = mean_score
                best_params = parameters
        # build classifier on best parameters using outer training set
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return outer_scores

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid, StratifiedKFold
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html#sklearn.model_selection.ParameterGrid
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
#ParameterGrid是按给定参数字典分配训练集与测试集,StratifiedKFold是分层分配训练集与测试集
nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid))










