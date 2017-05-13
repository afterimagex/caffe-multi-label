# -*- coding: utf-8 -*-

import sys 
caffe_root = '/home/lab704/caffe-m'
sys.path.insert(0, caffe_root + '/python')
import caffe
import lmdb
import numpy
from caffe.proto import caffe_pb2  

lmdbpath = sys.argv[1]
#lmdbpath = '/home/lab210/WS/FaceAttr/ResNet-18/data/lmdb/label_val'
#lmdbpath = '/home/lab210/WS/FaceAttr/ResNet-18/data/lmdb/image_val'
#lmdbpath = '/home/lab210/WS/FaceAttr/scripts/multi_label/test/lmdb/label_val'
lmdb_env = lmdb.open(lmdbpath, readonly=True) # 打开数据文件  
lmdb_txn = lmdb_env.begin() # 生成处理句柄  
lmdb_cursor = lmdb_txn.cursor() # 生成迭代器指针  
datum = caffe_pb2.Datum() # caffe 定义的数据类型  

for key, value in lmdb_cursor: # 循环获取数据
    datum.ParseFromString(value)
    ch = datum.channels
    height = datum.height
    width = datum.width
    label = datum.label
    #print datum.data
    flat_x = numpy.fromstring(datum.data, dtype=numpy.uint8)
    #flat_x = numpy.fromstring(datum.data)
    data_type = flat_x.dtype
    #flat_f = datum.labels
    #x = flat_x.reshape(datum.channels, datum.height, datum.width)
    print '{:<20}{:<8}{:<8}{:<8}{:<8}{:<8}{:<40}'.format('key', 'channel', 'height', 'width', 'label', 'type', 'data')
    print '{:<20}{:<8}{:<8}{:<8}{:<8}{:<8}{:<40}'.format(key, ch, height, width, label, data_type, flat_x)
    print ''

