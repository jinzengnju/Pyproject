#!/usr/bin/python
# -*- coding:UTF-8 -*-
import numpy as np
import pandas as pd

#用了pandas来保存或加载数据
def load_data_with_numpy():
    a=np.loadtxt('/home/jin/Pypro/PyProject/data/data',delimiter='\t')
    print(a)

def load_data_with_pandas():
    pass


#我们可以用一个np.array来创建DataFrame,可以选择指定列标签和行标签
#pandas允许每一列的数据dtype可以不同,而我们享用numpy导出数据时,要求所有列的数据dtype是相同的
#pandas中的一列数据相当于一个样本,行索引相当于属性
def write_csv():
    file_read=open('/home/jin/Pypro/PyProject/data/Feature31/feature30')
    file_write=open('/home/jin/Pypro/PyProject/data/Feature31/data_csv','w')
    for line in file_read:
        temp=line.split(' ')
        example_list = []
        print(line)
        for i in range(31):
            if(i==0):
                example_list.append(temp[i])
            else:
                example_list.append(temp[i].split(':')[1])
            #获取元素数据类型
            #print(type(example_list[0]))
        file_write.write(','.join(example_list))
    file_read.close()
    file_write.close()

def readByPandas():
    data=pd.read_csv("/home/jin/Pypro/PyProject/data/data1",header=None)
    print(data)
    #通过pandas读取的数据与通过numpy读取的请做对比,pandas不会要求每列的数据类型一致,而numpy则会要求一致.
    #如果存csv的时候是存的整型,读csv指定了dtype为float型,那么用numpy读取的时候会将所有数据转化为float


def readByNumpy():
    data=np.loadtxt("/home/jin/Pypro/PyProject/data/data_csv",dtype=np.float,delimiter=',')
    return data


    # d1=pd.DataFrame(data)
    # print(d1)
    #通过numpy读取的数据ndarray,可以转化为DataFrame格式,然后又可以用pandas的方法.注意:这种方式转化的pandas,所有元素数据类型是一致的,因为中途经过了numpy读取


def readByPandas_dealWithLog():
    data=pd.read_csv("/home/jin/下载/linden_plugin_test_log10",header=None,delimiter='\t')
    data.to_csv("/home/jin/下载/change",header=None)

def test_pandas():
    d2 = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [5, 6, 7, 8],
    })
    d2.to_csv('/home/jin/Pypro/PyProject/data/data1')

    d2 = pd.DataFrame({
        'c': [9, 10, 11, 12],
        'd': [13, 14, 15, 16]
    })
    d2.to_csv('/home/jin/Pypro/PyProject/data/data1')
    #该函数说明to_csv方法文件保存是覆盖式保存,只会保存最后一次to_csv的内容
import math
def entropy_compute(arr):
    a=0.0
    for i in arr:
        a+=i*math.log(i,2)
    return a

if __name__=='__main__':
    #print(entropy_compute([0.366,0.271,0.233,0.136]))
    #write_csv()
    #readByPandas()
    #readByNumpy()
    data=readByPandas_dealWithLog()