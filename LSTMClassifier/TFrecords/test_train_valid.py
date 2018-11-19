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

def read_from_tfrecords(filename,num_epoch):
    filename=[filename]
    filename_queue=tf.train.string_input_producer(filename,shuffle=False,num_epochs=num_epoch)
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
    return features['law'],features['fact']



def main(unuse_args):
    with tf.Session() as sess:
        train_law, train_fact = read_from_tfrecords('/home/jin/data/cail_0518/temp/TFrecords/train.tfrecords',1)
        valid_law, valid_fact = read_from_tfrecords('/home/jin/data/cail_0518/temp/TFrecords/test.tfrecords',1)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                law_train, fact_train = sess.run([train_law,train_fact])
                print("train:",bytes.decode(fact_train))
                law_valid,fact_valid=sess.run([valid_law, valid_fact])
                print("test:", bytes.decode(fact_valid))
        except tf.errors.OutOfRangeError:
            print('Done training for %d epoches,%d steps')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__=='__main__':
    tf.app.run()
    #get_classnum(lawname)
    #convert_to('/home/jin/data/cail_0518/temp/shuffle_test_temp_json.json','test')
    #read_from_tfrecords('/home/jin/data/cail_0518/temp/TFrecords/train.tfrecords')