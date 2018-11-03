from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


import tensorflow as tf
import json

tf.app.flags.DEFINE_string('directory','/home/jin/Pypro/PyProject/TestTfrecords','output_directory')
FLAGS=tf.app.flags.FLAGS


def gettime(time):
    # 将刑期用分类模型来做
    v = int(time['imprisonment'])

    if time['death_penalty']:
        return 0
    if time['life_imprisonment']:
        return 1
    elif v > 10 * 12:
        return 2
    elif v > 7 * 12:
        return 3
    elif v > 5 * 12:
        return 4
    elif v > 3 * 12:
        return 5
    elif v > 2 * 12:
        return 6
    elif v > 1 * 12:
        return 7
    else:
        return 8


def getlabel(d, kind):
    global law
    global accu

    # 做单标签
    #lawname通过law转化为类标记
    if kind == 'law':
        # 返回多个类的第一个
        return law[str(d['meta']['relevant_articles'])]
    if kind == 'accu':
        return accu[d['meta']['accusation']]

    if kind == 'time':
        return gettime(d['meta']['term_of_imprisonment'])


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
#value的值必须是一个列表

def convert_to(input_filename,output_filename):
    print(FLAGS.directory)
    fin=open(input_filename,'r',encoding='utf8')
    fout_path = os.path.join(FLAGS.directory, output_filename + '.tfrecords')
    print('Writing', fout_path)
    writer = tf.python_io.TFRecordWriter(fout_path)
    line=fin.readline()
    while line:
        print(line)
        d=json.loads(line)
        # print(type(d['meta']['accusation']))
        print(type(d['meta']['accusation']));
        fact=d['fact']
        accusation=[tf.compat.as_bytes(e) for e in d['meta']['accusation']]
        #将数据转为bytes字节形式保存，因为上面是_bytes_feature
        law=d['meta']['relevant_articles']
        time=getlabel(d,'time')
        example=tf.train.Example(features=tf.train.Features(feature={
            # 'fact':_bytes_feature(d['fact'].encode()),
            # 'accusation':tf.train.Feature(bytes_list=tf.train.BytesList(value=[e.encode() for e in d['meta']['accusation']])),
            # 'law': tf.train.Feature(
            #     int64_list=tf.train.Int64List(value=d['meta']['relevant_articles'])),
            # 'time':_int64_feature(getlabel(d,'time'))
            'fact': _bytes_feature(tf.compat.as_bytes(fact)),
            #先将fact转为字节bytes object存储，因为上面是bytes fature
            'accusation': _bytes_feature(accusation),
            'law': _int64_feature(law),
            'time': _int64_feature(time)

        }))
        writer.write(example.SerializeToString())
        line=fin.readline()
    fin.close()
    writer.close()

def read_from_tfrecords(filename):
    filename=[filename]
    filename_queue=tf.train.string_input_producer(filename,shuffle=False)
    reader=tf.TFRecordReader()
    _,serilized_example=reader.read(filename_queue)
    features=tf.parse_single_example(
        serilized_example,
        features={
            'fact':tf.FixedLenFeature([],tf.string),
            'accusation':tf.VarLenFeature(tf.string),
            'law':tf.VarLenFeature(tf.int64),
            'time':tf.FixedLenFeature([],tf.int64)

        }
    )
    #对于长度可变的列表,需要用VarLenFeature来读
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        #start_queue_runners运行需要在定义好graph图之后,真正的sess.run()之前
        #其作用是把queue里边的内容初始化，不跑这句一开始string_input_producer那里就没用，整个读取流水线都没用了
        for i in range(6):
            fact,accu=sess.run([features['fact'],tf.sparse_tensor_to_dense(features['accusation'], default_value='')])
            #decode_raw是对某个属性进行转换,不是对整个example转换.image=tf.deocode_raw(features['train/image'],tf.float32)
            #需要注意,虽然读进来的fact是bytes二进制的形式,但仍然可以直接用jieba分词,分出来是中文的
            #print(bytes.decode(fact),bytes.decode(accu))
            print(bytes.decode(fact),[bytes.decode(e) for e in accu])

        coord.request_stop()
        coord.join(threads)
        # tf.train.Coordinator.should_stop()
        # 如果线程应该停止，返回True
        # tf.train.Coordinator.request_stop()
        # 请求停止线程
        # tf.train.Coordinator.join()
        # 等待直到指定线程停止



if __name__=='__main__':
    #convert_to('data.json','train')
    read_from_tfrecords('train.tfrecords')