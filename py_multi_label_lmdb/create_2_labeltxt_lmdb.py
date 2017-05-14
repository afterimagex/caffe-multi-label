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
    print '{:<}'.format('@label1.txt: 00001.jpg 1 2 3 4 ...')
    print '{:<}'.format('@label2.txt: 00001.jpg 1 2 3 ...')
    print '{:<10}'.format('usage:')
    print '{:<10}{:<10}'.format('--help', 'show help')
    print '{:<10}{:<10}'.format('--shuffle', 'shuffle image')
    print '{:<10}{:<10}'.format('-l', 'label1.txt')
    print '{:<10}{:<10}'.format('-t', 'label2.txt')
    print '{:<10}{:<10}'.format('-i', 'image dir')
    print '{:<10}{:<10}'.format('-o', 'output')
    print '{:<10}'.format('example:')
    print '{} {}'.format('python this.py', '-l label.txt -i image_dir -o lmdb_train/')
    print '{:-<40}'.format('')

def label2lmdb(label1_in, label2_in, img_dir, db_out):
    global SHUFFLE
    labels1 = open(label1_in, 'r').readlines()
    labels2 = open(label2_in, 'r').readlines()
    if not len(labels1) == len(labels2):
        print '{}'.format('label1.txt not match label2.txt')
        sys.exit()
    N = len(labels1)
    labels = [0 for i in range(N)]
    for index in range(N):
        l1 = labels1[index].strip().split(' ')
        l2 = labels2[index].strip().split(' ')
        if l1[0] == l2[0]:
            labels[index] = (str(l1[0]), [int(x) for x in l1[1:]], [int(x) for x in l2[1:]])
        else:
            print 'label1 {:<20} {} label2 {:<20} ignore...'.format(l1[0],'not match', l2[0])
    if SHUFFLE:
        random.shuffle(labels)
    (key, arr1, arr2)  = map(list, zip(*labels))
    ndarr1 = np.array(arr1, dtype=np.uint8)
    ndarr2 = np.array(arr2, dtype=np.uint8)

    map_size = ndarr1.nbytes * 10000
    label1_db = lmdb.open(db_out+'/label1', map_size=int(map_size))
    with label1_db.begin(write=True) as txn1:
        for index, nd in enumerate(ndarr1):
            ndarr = nd.reshape(len(nd), 1, 1)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = ndarr.shape[0]
            datum.height = ndarr.shape[1]
            datum.width = ndarr.shape[2]
            datum.data = ndarr.tostring()
            datum.label = 0
            str_id = '{:0>8}_{}'.format(index, key[index])
            txn1.put(str_id.encode('ascii'), datum.SerializeToString())
            print '[{:<20}] {:<20} {:<20} label1'.format(ctime(), str_id, ndarr.shape)

    map_size = ndarr2.nbytes * 10000
    label2_db = lmdb.open(db_out+'/label2', map_size=int(map_size))
    with label2_db.begin(write=True) as txn2:
        for index, nd in enumerate(ndarr2):
            ndarr = nd.reshape(len(nd), 1, 1)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = ndarr.shape[0]
            datum.height = ndarr.shape[1]
            datum.width = ndarr.shape[2]
            datum.data = ndarr.tostring()
            datum.label = 0
            str_id = '{:0>8}_{}'.format(index, key[index])
            txn2.put(str_id.encode('ascii'), datum.SerializeToString())
            print '[{:<20}] {:<20} {:<20} label2'.format(ctime(), str_id, ndarr.shape)
        
    img_db = lmdb.open(db_out+'/image', map_size=int(1e12))
    with img_db.begin(write=True) as i_txn:
        for index, img in enumerate(key):
            datum = caffe.proto.caffe_pb2.Datum()
            im = Image.open('{}/{}'.format(img_dir, img))
            im_array = np.array(im, dtype=np.uint8)
            im_array = im_array[:,:,::-1]
            im_array = im_array.transpose((2,0,1))
            datum.channels = im_array.shape[0]
            datum.height = im_array.shape[1]
            datum.width = im_array.shape[2]
            datum.data = im_array.tostring()
            datum.label = 0
            str_id = '{:0>8}_{}'.format(index, key[index])
            i_txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print '[{:<20}] {:<20} {:<20} image'.format(ctime(), str_id, im_array.shape)
            del im

def check_dir(_dir):
    if not os.path.isdir(_dir):
        os.makedirs(_dir)
    else:
        print 'dir is not empty'
        sys.exit()

if __name__ == '__main__':
    p_label1 = ''
    p_label2 = ''
    p_output = ''
    p_image = ''
    try:
        if len(sys.argv) < 5:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'i:l:t:o:', ['help', 'shuffle'])
        for op, value in opts:
            if op == '--help':
                this_help()
                sys.exit()
            if op == '-l':
                p_label1 = value
                print '%-20s%s' % ('list txt:', value)
            if op == '-t':
                p_label2 = value
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
    label2lmdb(p_label1, p_label2, p_image, p_output)
