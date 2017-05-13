# -*- coding: utf-8 -*-
import sys
import numpy as np
caffe_root = '/home/lab704/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe
import getopt

def this_help():
    print '{:-<40}'.format('')
    print '{:<10}{:<10}'.format('descript:', 'convert binaryproto to npy for mean file')
    print '{:<10}'.format('usage:')
    print '{:<10}{:<10}'.format('-h', 'show help')
    print '{:<10}{:<10}'.format('-b', 'binaryproto')
    print '{:<10}{:<10}'.format('-p', 'npy')
    print '{:<10}'.format('example:')
    print '{} {}'.format('python this.py', '-b mean.binaryproto -p mean.npy')
    print '{:-<40}'.format('')

def argv_opt():
    file_mean = ''
    try:
        if len(sys.argv) < 5:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'hb:p:')
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '-b':
                mean_bin = value
                print '{:<10}{:<10}'.format ('bin:', mean_bin)
            if op == '-p':
                mean_npy = value
                print '{:<10}{:<10}'.format ('npy:', mean_npy)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    return mean_bin, mean_npy

if __name__ == '__main__':
    m_bin, m_npy = argv_opt()
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(m_bin, 'rb') .read()
    blob.ParseFromString(data)
    array = np.array(caffe.io.blobproto_to_array(blob))
    mean_npy = array[0]
    np.save(m_npy, mean_npy)
    
