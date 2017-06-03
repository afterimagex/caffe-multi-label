#!usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import lmdb
import getopt
import numpy as np
import random
from time import ctime
from PIL import Image

caffe_root = '/home/lab704/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe

SHUFFLE = False

def this_help():
    print '{:-<40}'.format('')
    print '{:<20}'.format('usage:')
    print '{:<20}{:<10}'.format('-l', 'label txt')
    print '{:<20}{:<10}'.format('-i', 'image dir')
    print '{:<20}{:<10}'.format('-o', 'output dir')
    print '{:<20}{:<10}'.format('--help', 'show help')
    print '{:<20}{:<10}'.format('--shuffle', 'shuffle image')
    print '{:<20}'.format('example:')
    print '{} {}'.format('python this.py', '-l label.txt -i image_dir -o lmdb_train/')
    print '{:-<40}'.format('')


def label2lmdb(label_in, img_dir, db_out):
    global SHUFFLE
    tmp_list = []
    im_name = []
    label = []
    labels = open(label_in, 'r').readlines()
    
    if SHUFFLE:
        random.shuffle(labels)

    for index, value in enumerate(labels):
        tmp_list = value.strip().split(' ')
        im_name.append(tmp_list[0])
        label.append(map(int, tmp_list[1:]))
    X = np.array(label, dtype=np.uint8).reshape(len(label), len(label[0]), 1)
    #map_size = X.nbytes * 1000
    #print '{:<20}{:<20}'.format('map_size:', map_size)
    label_db = lmdb.open(db_out+'/label', map_size=int(1e12))
    with label_db.begin(write=True) as l_txn:
        for index, ni in enumerate(X):
            datum = caffe.proto.caffe_pb2.Datum()
            nni = ni.reshape(ni.shape[0], 1, 1)
            datum.channels = nni.shape[0]
            datum.height = nni.shape[1]
            datum.width = nni.shape[2]
            datum.data = nni.tostring()
            datum.label = 0
            str_id = '{:0>8}{}{}'.format(index, '_', im_name[index])
            l_txn.put(str_id.encode('ascii'), datum.SerializeToString())
    
    img_db = lmdb.open(db_out+'/image', map_size=int(1e12))
    with img_db.begin(write=True) as i_txn:
        for index, img in enumerate(im_name):
            datum = caffe.proto.caffe_pb2.Datum()
            im = Image.open(img_dir + '/' + img)
            im = im.resize((224, 224), Image.ANTIALIAS)
            im_array = np.array(im, dtype=np.uint8)
            im_array = im_array[:,:,::-1]
            im_array = im_array.transpose((2,0,1))
            datum.data = im_array.tostring()
            datum.channels = im_array.shape[0]
            datum.height = im_array.shape[1]
            datum.width = im_array.shape[2]
            datum.label = 0
            str_id = '{:0>8}{}{}'.format(index, '_', img)
            print '[{:<20}] {:<20} {:<20}'.format(ctime(),str_id, im_array.shape)
            i_txn.put(str_id.encode('ascii'), datum.SerializeToString())

def check_dir(_dir):
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    else:
        print 'dir is not empty'
        sys.exit()

if __name__ == '__main__':
    p_list = ''
    p_output = ''
    p_image = ''
    try:
        if len(sys.argv) < 5:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'i:l:o:', ['help', 'shuffle'])
        for op, value in opts:
            if op == '--help':
                this_help()
                sys.exit()
            if op == '-l':
                p_list = value
                print '%-20s%s' % ('list txt:', value)
            if op == '-i':
                p_image = value
                print '%-20s%s' % ('image dir:', value)
            if op == '-o':
                p_output = value
                check_dir(p_output)
                print '%-20s%s' % ('outdir:', value)
            if op == '--shuffle':
                SHUFFLE = True
                print '%-20s%s' % ('shuffle:', 'True')
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    label2lmdb(p_list, p_image, p_output)
