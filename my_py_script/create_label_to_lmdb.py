#!usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import lmdb
import getopt
import numpy

caffe_root = '/home/lab704/caffe-m'
sys.path.insert(0, caffe_root + '/python')
import caffe

p_list = ''
p_output = ''
base_lmdb = ''

def this_help():
    print 'usage:'
    print '-h \t\t---show help'
    print '-l \t\t---list file'
    print '-b \t\t---base lmdb'
    print '-o \t\t---output dir'
    print 'example:'
    print 'python this.py -l train.txt -o ./lmdb_output -b path/to/tarin_lmdb'


def get_base_lmdb_key(base_db, f_txt):
    all_labels = []
    key_labels = []
    lmdb_env = lmdb.open(base_db, readonly=True)  # 打开数据文件
    lmdb_txn = lmdb_env.begin()  # 生成处理句柄
    lmdb_cursor = lmdb_txn.cursor()  # 生成迭代器指针
    label_fr = open(f_txt, 'r')
    label_list = label_fr.readlines()
    for key, value in lmdb_cursor:  # 循环获取数据
        key_labels.append(key.split('_')[1])
    for _key in key_labels:
        for line in label_list:
            line = line.strip().split(' ')
            if _key == line[0]:
                all_labels.append(line)
    return numpy.array(all_labels)

def myArrayConverter(arr):
    convertArr = []
    for s in arr.ravel():    
        try:
            value = numpy.float32(s)
        except ValueError:
            value = s
        convertArr.append(value)
    return numpy.array(convertArr,dtype=object).reshape(arr.shape[0], 1, 1)

def label2lmdb(label_list, db_out):
    label_db = lmdb.open(db_out, map_size=int(1e12))
    key = 0
    with label_db.begin(write=True) as in_txn:
        for label in label_list:
            print label
            label_data = label[1:]
            l_data = myArrayConverter(label_data)
            print l_data
            #print l_data, l_data.shape
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = label_data.shape[0]
            datum.height = 1
            datum.width = 1
            datum.data = l_data.tostring()
            datum.label = 0
            key_str = '{:08}_{}'.format(key, label[0])
            in_txn.put(key_str.encode('ascii'), datum.SerializeToString())
            key += 1

if __name__ == '__main__':
    try:
        if len(sys.argv) < 6:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'l:o:b:h')
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '-l':
                p_list = value
                print '%-20s%s' % ('list txt:', value)
            if op == '-b':
                base_lmdb = value
                print '%-20s%s' % ('base lmdb:', value)
            elif op == '-o':
                p_output = value
                print '%-20s%s' % ('outdir:', value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    np_list = get_base_lmdb_key(base_lmdb, p_list)
    label2lmdb(np_list, p_output)
