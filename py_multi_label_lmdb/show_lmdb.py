# -*- coding: utf-8 -*-
import sys
import numpy as np
import lmdb
caffe_root = '/home/lab704/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe
import PIL.Image
import getopt
import cv2
from StringIO import StringIO
 
lmdbpath = ''


def this_help():
    print 'usage:'
    print '-d \t\t---lmdb'
    print 'example:'
    print 'python this.py -d /path/to/lmdb'

# 打开 lmdb 数据库, 指定好位置
def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for index, value in cursor:
        datum.ParseFromString(value)
        ch = datum.channels
    	height = datum.height
    	width = datum.width
    	label = datum.label
        data = caffe.io.datum_to_array(datum)
        image = data.transpose(1, 2, 0)
        yield image, (index, ch, height, width, label)

if __name__ == '__main__':
    try:
        if len(sys.argv) < 3:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'd:')
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '-d':
                lmdbpath = value
                print '%-20s%s' % ('lmdb file:', value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    for im, (key, c, h, w, l)in read_lmdb(lmdbpath):
        cv2.imshow('lmdb', im)
        print 'key_str: {}\nchannels: {}\nheight: {}\nwidth: {}\nlabel: {}\n'.format(key, c, h, w, l)
        enter_key = cv2.waitKey()
        if enter_key == 113 or enter_key == 27:
            sys.exit()
