# -*- coding: utf-8 -*-
import sys
import numpy as np
import lmdb
caffe_root = '/home/lab704/caffe-m'
sys.path.insert(0, caffe_root + '/python')
import caffe
import PIL.Image
import getopt
import cv2
import time

def this_help():
    print 'usage:'
    print '{:<20}{:<20}'.format('-b', 'path to lmdb')
    print '{:<20}{:<20}'.format('-d', 'daploy file')
    print '{:<20}{:<20}'.format('-w', 'caffemodel file')
    print '{:<20}{:<20}'.format('-c', 'mean file')
    print 'example:'
    print 'python this.py -b /path/to/lmdb'

# 打开 lmdb 数据库, 指定好位置
def read_lmdb(lmdb_file):
    cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    for index, value in cursor:
        datum.ParseFromString(value)
        ch = datum.channels
    	height = datum.height
    	width = datum.width
    	label = datum.labels
        image = caffe.io.datum_to_array(datum)   #(c, h, w)
        yield image, (index, ch, height, width, label)

if __name__ == '__main__':
    caffe.set_mode_gpu()
    lmdb_path = ''
    deploy_file = ''
    weight_file = ''
    mean_file = ''
    try:
        if len(sys.argv) < 8:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'b:d:w:c:')
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '-b':
                lmdb_path = value
                print '%-20s%s' % ('lmdb file:', value)
            if op == '-d':
                deploy_file = value
                print '{:<20}{:<20}'.format('deploy file:', value)
            if op == '-w':
                weight_file = value
                print '{:<20}{:<20}'.format('weights file:', value)
            if op == '-c':
                mean_file = value
                print '{:<20}{:<20}'.format('mean file:', value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    
    net = caffe.Net(deploy_file, weight_file, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_channel_swap('data', (2, 1, 0)) 
    
    all_count = 0.0
    error_count = 0.0
    for img, (key, c, h, w, l)in read_lmdb(lmdb_path):
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        starttime = time.time()
        out = net.forward()
        endtime = time.time()
        for d_key in out:
            scores = out[d_key][0]
        #image = img.transpose(1, 2, 0)
        #cv2.imshow('123',image)
        l_label = [str(int(x)) for x in l]
        r_label =  ['1' if v>0.6 else '0' for v in scores]
        p_label = []
        for i in range(len(r_label)):
            all_count += 1
            if l_label[i] == r_label[i]:
                p_label.append('+')
            else:
                p_label.append('-')
                error_count += 1
        print '{:-<20}[{}]{:-<20}'.format('',time.ctime(),'')
        print '{:<20}{:<20}'.format('key_str:', key)
        #print '{:<20}{:<20}'.format('channels:', c)
        #print '{:<20}{:<20}'.format('height:', h)
        #print '{:<20}{:<20}'.format('width:', w)
        print '{:<20}{:<20}'.format('time:', endtime - starttime) 
        print '{:<20}{:<20}'.format('Really labels:', l_label)
        print '{:<20}{:<20}'.format('Predict labels:', r_label)
        print '{:<20}{:<20}'.format('Acc labels:', p_label)
        print '{:<20}{:<.4f}% [{}/{}]'.format('Error Rate:', (error_count / all_count) * 100, error_count, all_count)
        print '{:<20}{:<.4f}% [{}/{}]'.format('Correct Rate:', (1 - error_count / all_count) * 100, all_count - error_count, all_count)
        enter_key = cv2.waitKey()
        if enter_key == 113 or enter_key == 27:
            sys.exit() 
