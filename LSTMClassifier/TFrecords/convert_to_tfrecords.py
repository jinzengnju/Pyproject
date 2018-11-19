from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import json
import numpy as np

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_path','/home/jin/data/cail_0518/temp/train.json','')
tf.app.flags.DEFINE_string('output_path','/home/jin/data/cail_0518/temp/TFrecords','')
tf.app.flags.DEFINE_integer('class_num','183','')

def getlabel(d, kind):
    global law
    #lawname通过law转化为类标记
    if kind == 'law':
        return [law[str(e)] for e in d['law']]
    #將列表转为字符串

def init():
	f = open('law.txt', 'r', encoding = 'utf8')
	law = {}
	lawname = {}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
		law[line.strip()] = len(law)
		line = f.readline()
	f.close()


	f = open('accu.txt', 'r', encoding = 'utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()
		accu[line.strip()] = len(accu)
		line = f.readline()
	f.close()


	return law, accu, lawname, accuname

law, accu, lawname, accuname = init()


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

def get_classnum(lawname):
    print(lawname.__len__())


def convert_to(input_filename,output_filename):
    class_num=FLAGS.class_num
    mlb=MultiLabelBinarizer(classes=[e for e in np.arange(class_num)])
    fin=open(input_filename,'r',encoding='utf8')
    writer = tf.python_io.TFRecordWriter(output_filename)
    line=fin.readline()
    while line:
        d=json.loads(line)
        fact=d['fact']
        law=getlabel(d,'law')
        print(law)
        law_label=mlb.fit_transform([law])[0].tolist()#因为下面_int64_feature只能接受list数据
        print(law_label)
        example=tf.train.Example(features=tf.train.Features(feature={
            'fact': _bytes_feature(tf.compat.as_bytes(fact)),
            #先将fact转为字节bytes object存储，因为上面是bytes fature
            'law': _int64_feature(law_label)
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
            'law':tf.FixedLenFeature([183],tf.int64)

        }
    )
    #对于长度可变的列表,需要用VarLenFeature来读
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        #start_queue_runners运行需要在定义好graph图之后,真正的sess.run()之前
        #其作用是把queue里边的内容初始化，不跑这句一开始string_input_producer那里就没用，整个读取流水线都没用了
        for i in range(1000):
            law,fact=sess.run([features['law'],features['fact']])
            #decode_raw是对某个属性进行转换,不是对整个example转换.image=tf.deocode_raw(features['train/image'],tf.float32)
            #需要注意,虽然读进来的fact是bytes二进制的形式,但仍然可以直接用jieba分词,分出来是中文的
            #print(bytes.decode(fact),bytes.decode(accu))
            #print(type(fact))
            #sess.run后返回的law是numpy数组。而fact则是bytes对象,不能用shape属性
            print(law,bytes.decode(fact))

        coord.request_stop()
        coord.join(threads)
        # tf.train.Coordinator.should_stop()
        # 如果线程应该停止，返回True
        # tf.train.Coordinator.request_stop()
        # 请求停止线程
        # tf.train.Coordinator.join()
        # 等待直到指定线程停止

def main(unuse_args):
    convert_to(FLAGS.input_path,FLAGS.output_path)

if __name__=='__main__':
    tf.app.run()
    #get_classnum(lawname)
    #convert_to('/home/jin/data/cail_0518/temp/shuffle_test_temp_json.json','test')
    #read_from_tfrecords('/home/jin/data/cail_0518/temp/TFrecords/train.tfrecords')