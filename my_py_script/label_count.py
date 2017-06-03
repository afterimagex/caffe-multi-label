# -*- coding: utf-8 -*-
import sys
import os
import getopt

n_TRAIN = 0
n_VAL = 0
s_OUTPUT = ''
s_TYPE = ''


def t_help():
    print 'usage:'
    print '-t \t\t---label list'
    print 'example:'
    print 'python get_image_list_from_celeba.py -t 2500 -v 170 -o ./txt -p full'

def get_argv():
    _txt = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hl:', ['help'])
        for op, value in opts:
            if op in ['-h', '--help']:
                this_help()
                sys.exit()
            if op == '-l':
                _txt = value
                print '%-20s%s' % ('label file:', value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    return _txt


if __name__ == '__main__':
    if len(sys.argv) < 2:
        t_help()
        exit()
    label = get_argv()
    labels = open(label, 'r').readlines()
    length = len(labels[0].strip().split(' ')[1:])
    count = [0 for x in range(length)]
    count_x = [0 for x in range(length)]
    for index, line in enumerate(labels):
        content = line.strip().split(' ')[1:]
        for i, v in enumerate(content):
            if v == '1':
                count[i] += 1
                count_x[i] = (i, count[i])
    #print count, len(count)
    print count_x
