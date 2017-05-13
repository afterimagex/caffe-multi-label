# -*- coding: utf-8 -*-

import os
import sys
import shutil
import getopt
from time import ctime
from PIL import Image

CELEBA = '/home/lab704/caffe/data/celeba'

p_LIST = ''
p_OUTPUT = ''
RESIZE = False
WIDTH = 224
HEIGHT = 224

def t_help():
    print 'usage:'
    print '-h \t\t---show help'
    print '-l \t\t---list file'
    print '-o \t\t---output dir'
    print '-r \t\t---resize image(default --width 224 --height 224)'
    print 'example:'
    print 'python create_image_from_attr_list.py -l train.txt -o ./output -r --width=224 --height=224'
    
def check_dir(p_dir):
    if os.path.isdir(p_dir):
        print 'dir does not empty'
        sys.exit()
    else:
        os.makedirs(p_dir)


def resize_image(img, out):
    global WIDTH, HEIGHT
    im = Image.open(img)
    im_out = im.resize((int(WIDTH), int(HEIGHT)), Image.ANTIALIAS)
    im_out.save(out)
    im.close()
    im_out.close()
    return out
        
def createImage(t_list, p_outdir, resize):
    t_cursor = 0
    fr_list = open(t_list, 'r')
    # 生成train image
    t_lines = len(fr_list.readlines())
    fr_list.seek(0)
    for line in fr_list:
        t_cursor += 1
        img_name = line.strip().split(' ')[0]
        if resize:
            resize_image(CELEBA + '/img_align_celeba/' + img_name,
                         p_outdir +'/'+ img_name)
        else:
            shutil.copy(CELEBA + '/img_align_celeba/' + img_name,
                         p_outdir +'/'+ img_name)
        if t_cursor % 10 == 0:
            print('[{}] copy train {} ... [{}/{}]'.format(ctime(), img_name, t_cursor, t_lines))
    fr_list.close()
    print('\n%s Create Image Done! \n' % ctime())


if __name__ == '__main__':
    try:
        if len(sys.argv) < 4:
            t_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'l:o:rh', ['width=', 'height='])
        for op, value in opts:
            if op == '-h':
                t_help()
                sys.exit()
            if op == '-l':
                p_LIST = value
                print '%-20s%s' % ('list txt:', value)
            elif op == '-o':
                p_OUTPUT = value
                print '%-20s%s' % ('outdir:', value)
            elif op == '-r':
                RESIZE = True
                print '%-20s%s' % ('resize:', 'True')
            elif op == '--width':
                WIDTH = value
                print '%-20s%s' % ('width:', value)
            elif op == '--height':
                HEIGHT = value
                print '%-20s%s' % ('height:', value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    check_dir(p_OUTPUT)
    createImage(p_LIST, p_OUTPUT, RESIZE)
