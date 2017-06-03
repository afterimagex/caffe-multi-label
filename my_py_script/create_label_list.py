# -*- coding: utf-8 -*-
import sys
import os

CELEBA = '/home/lab704/caffe/data/celeba/anno/list_attr_celeba.txt'
n_MAX = 200000

n_TRAIN = 0
n_VAL = 0
s_OUTPUT = ''
s_TYPE = ''


def t_help():
    print 'usage:'
    print '-t \t\t---number of train image'
    print '-v \t\t---number of val image'
    print '-o \t\t---output dir'
    print '-p \t\t---type of list (full/part/none/point)'
    print 'example:'
    print 'python get_image_list_from_celeba.py -t 2500 -v 170 -o ./txt -p full'

def check_dir(p_dir):
    if os.path.isdir(p_dir):
        print 'dir does not empty'
        exit()
    else:
        os.makedirs(p_dir)

def creat_attr_list(p_celeba, a_part, s_out):
    max_train = n_MAX / int(n_TRAIN)
    max_val = n_MAX / int(n_VAL)
    with open(p_celeba, 'r') as fr:
        with open(s_OUTPUT + '/headers.txt', 'w') as fw_label:
            # 跳过第一行
            fr.readline()
            # 从第二行读取内容
            # 生成label
            la_l = []
            line = fr.readline().strip().replace('  ', ' ').split(' ')
            for iv in a_part[1:]:
                la_l.append(line[iv-1])
            char_tmp = '\n'.join(la_l)
            fw_label.write(char_tmp)
        with open(s_out + "/train.txt", 'w') as fw_train:
            fr.seek(0)
            t_cursor = 0
            fr.readline()
            fr.readline()
            # write number 
            #fw_train.write(str(n_TRAIN)+' ')
            # write label
            #var_t = []
            #line = fr.readline().strip().replace('  ', ' ').split(' ')
            #for iv in a_part[1:]:
            #    var_t.append(line[iv-1])
            #char_tmp = ' '.join(var_t)
            #fw_train.write(char_tmp + '\r\n')
            #---
            for t in range(n_MAX):
                t_cursor += 1
                var_t = []
                line = fr.readline()
                # 生成train list
                if t_cursor % max_train == 0:
                    var = line.strip().replace('  ', ' ').split()
                    for iv in a_part:
                        var_t.append(var[iv])
                    char_tmp = ' '.join(var_t)
                    fw_train.write(char_tmp + '\r\n')
        with open(s_out + "/val.txt", 'w') as fw_val:
            fr.seek(0)
            v_cursor = 1
            fr.readline()
            fr.readline()
            # write number
            #fw_val.write(str(n_VAL)+' ')
            # write label
            #var_t = []
            #line = fr.readline().strip().replace('  ', ' ').split(' ')
            #for iv in a_part[1:]:
            #    var_t.append(line[iv-1])
            #char_tmp = ' '.join(var_t)
            #fw_val.write(char_tmp + '\r\n')
            #---
            for t in range(n_MAX):
                v_cursor += 1
                var_t = []
                line = fr.readline()
                # 生成train list
                if v_cursor % max_val == 0:
                    var = line.strip().replace('  ', ' ').split()
                    for iv in a_part:
                        var_t.append(var[iv])
                    char_tmp = ' '.join(var_t)
                    fw_val.write(char_tmp + '\r\n')

def creat_points_list(p_celeba, a_part, s_out):
    max_train = n_MAX / int(n_TRAIN)
    max_val = n_MAX / int(n_VAL)
    source_width = 178
    source_height = 218
    post_width = 224.0
    post_height = 224.0
    scale_x = post_width/source_width
    scale_y = post_height/source_height
    with open(p_celeba, 'r') as fr:
        with open(s_OUTPUT + '/headers.txt', 'w') as fw_label:
            # 跳过第一行
            fr.readline()
            # 从第二行读取内容
            # 生成label
            line = fr.readline().strip().replace('  ', ' ').split(' ')
            char_tmp = '\n'.join(line)
            fw_label.write(char_tmp)
        with open(s_out + "/train.txt", 'w') as fw_train:
            fr.seek(0)
            t_cursor = 0
            fr.readline()
            fr.readline()
            # write number 
            #fw_train.write(str(n_TRAIN)+' ')
            # write label
            #var_t = []
            #line = fr.readline().strip().replace('  ', ' ').split(' ')
            #for iv in a_part[1:]:
            #    var_t.append(line[iv-1])
            #char_tmp = ' '.join(var_t)
            #fw_train.write(char_tmp + '\r\n')
            #---
            for t in range(n_MAX):
                t_cursor += 1
                var_t = []
                var_nx = []
                var_ny = []
                var_n = []
                line = fr.readline()
                # 生成train list
                if t_cursor % max_train == 0:
                    var = line.strip().replace('   ', ' ').replace('  ', ' ').split()
                    for iv in a_part:
                        var_t.append(var[iv])
                    for point_x in var_t[1::2]:
                        #var_nx.append((float(float(point_x) * scale_x))/post_width)
                        var_nx.append(int(float(point_x) * scale_x))
                    for point_y in var_t[2::2]:
                        #var_ny.append((float(float(point_y) * scale_y))/post_height)
                        var_ny.append(int(float(point_y) * scale_y))
                    var_n.append(var_t[0])
                    for i in range((len(a_part)-1)/2):
                        var_n.append('{}'.format(var_nx[i]))
                        var_n.append('{}'.format(var_ny[i]))
                    char_tmp = ' '.join(var_n)
                    fw_train.write(char_tmp + '\r\n')
        with open(s_out + "/val.txt", 'w') as fw_val:
            fr.seek(0)
            v_cursor = 1
            fr.readline()
            fr.readline()
            # write number
            #fw_val.write(str(n_VAL)+' ')
            # write label
            #var_t = []
            #line = fr.readline().strip().replace('  ', ' ').split(' ')
            #for iv in a_part[1:]:
            #    var_t.append(line[iv-1])
            #char_tmp = ' '.join(var_t)
            #fw_val.write(char_tmp + '\r\n')
            #---
            for t in range(n_MAX):
                v_cursor += 1
                var_t = []
                var_nx = []
                var_ny = []
                var_n = []
                line = fr.readline()
                # 生成train list
                if v_cursor % max_val == 0:
                    var = line.strip().replace('   ', ' ').replace('  ', ' ').split()
                    for iv in a_part:
                        var_t.append(var[iv])
                    for point_x in var_t[1::2]:
                        #var_nx.append((float(float(point_x) * scale_x))/post_width)
                        var_nx.append(int(float(point_x) * scale_x))
                    for point_y in var_t[2::2]:
                        #var_ny.append((float(float(point_y) * scale_y))/post_height)
                        var_ny.append(int(float(point_y) * scale_y))
                    var_n.append(var_t[0])
                    for i in range((len(a_part)-1)/2):
                        var_n.append('{}'.format(var_nx[i]))
                        var_n.append('{}'.format(var_ny[i]))
                    char_tmp = ' '.join(var_n)
                    fw_val.write(char_tmp + '\r\n')

def create_attr(n_MAX, n_TRAIN, n_VAL, s_OUTPUT, s_type='full'):
    global CELEBA
    if s_type == 'full':
        a_part = range(41)
        creat_attr_list(CELEBA, a_part, s_OUTPUT)
    elif s_type == 'part':
        #a_part = [0, 9, 16, 21, 22, 32]  # black hair Male mouth open Smiling glasses
        a_part = [0, 5, 6, 9, 10, 12, 16, 17, 18, 21, 22, 25, 26, 27, 32, 34 ,35, 36, 37, 38, 40]
        creat_attr_list(CELEBA, a_part, s_OUTPUT)
    elif s_type == 'none':
        a_part = [0, 1]
        creat_attr_list(CELEBA, a_part, s_OUTPUT)   
    elif s_type == 'point':
        a_part = range(11)
        CELEBA = '/home/lab704/caffe/data/celeba/anno/list_landmarks_align_celeba.txt'
        creat_points_list(CELEBA, a_part, s_OUTPUT)
    else:
        print 'type error'
        exit()


if __name__ == '__main__':
    if len(sys.argv) < 7:
        t_help()
        exit()

    for i, args in enumerate(sys.argv):
        if args == '-t':
            n_TRAIN = sys.argv[i + 1]
        if args == '-v':
            n_VAL = sys.argv[i + 1]
        if args == '-o':
            s_OUTPUT = sys.argv[i + 1]
        if args == '-f':
            s_TYPE = sys.argv[i + 1]
        if args == '-p':
            s_TYPE = sys.argv[i + 1]

    check_dir(s_OUTPUT)
    create_attr(n_MAX, n_TRAIN, n_VAL, s_OUTPUT, s_TYPE)
