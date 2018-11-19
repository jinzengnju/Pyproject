import tensorflow as tf

def method1():
    NUM_CLASSES = 10 # 10分类
    labels = [1,1,2,3] # sample label
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1) # 增加一个维度
    #这里的labels是Tesnor张量
    indices = tf.expand_dims(tf.range(0, batch_size,1), 1) #生成索引
    #indices也是一个4*1的tensor张量
    concated = tf.concat([indices, labels], 1)  # 作为拼接
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0) # 生成one-hot编码的标签
    #将稀疏矩阵转换成密集矩阵，其中索引在concated中，值为1.其他位置的值为默认值0.
    #https://blog.csdn.net/qq_22812319/article/details/83374125这个更详细

# def method2():
#     b=tf.one_hot(indices,4,1,0)
#其中indices为类别索引，如[0,7,4,5,6,3]或者numpy数组，可以是numpy数组或者list
#这种方法不适合多标记分类，因为当indeces为[[1,2],[3],[3,5],[8,6,9]]会报错。
#为什么，因为要求输入的indices为denseTensor，[[1,2],[3],[3,5],[8,6,9]]为SparseTensor

if __name__=='__main__':
    method1()