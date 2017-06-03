# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
caffe_root = '/home/lab704/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe
import getopt


def this_help():
    print '{:-<40}'.format('')
    print '{:<10}'.format('usage:')
    print '{:<10}{:<10}'.format('-h', 'show help')
    print '{:<10}{:<10}'.format('--i=', 'image')
    print '{:<10}{:<10}'.format('--d=', 'deploy')
    print '{:<10}{:<10}'.format('--m=', 'caffemodel')
    print '{:<10}{:<10}'.format('--c=', 'mean_file')
    print '{:<10}'.format('example:')
    print '{} {}'.format('python this.py', '--i=image.jpg --d=deploy.prototxt --m=train.caffemodel')
    print '{:-<40}'.format('')

def argv_opt():
    file_mean = ''
    try:
        if len(sys.argv) < 4:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], '', ['i=','m=','d=','c='])
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '--i':
                file_image = value
                print '{:<10}{:<10}'.format ('image:', file_image)
            if op == '--m':
                file_model = value
                print '{:<10}{:<10}'.format ('model:', file_model)
            if op == '--d':
                file_deploy = value
                print '{:<10}{:<10}'.format ('deploy:', file_deploy)
            if op == '--c':
                file_mean = value
                print '{:<10}{:<10}'.format ('mean_file:', file_mean)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
    return file_image, file_deploy, file_model, file_mean
    
def analysis(out_dict):
    for layer_name in out_dict:
        print '{:<20}{:<20}'.format('layer name:', layer_name)
        for value in out_dict[layer_name][0]:
            if float(value) >= 0.6:
                t = 1
            else:
                t = 0
            print '{:<20}{:<20}{:<20}'.format('', value, t)

def vis_square(data, padsize=1, padval=0 ):
    data -= data.min()
    data /= data.max()
    #让合成图为方
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #合并卷积图到一个图像中
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print data.shape
    plt.imshow(data)#,  cmap='gray')
    plt.axis('off')
    plt.show()
    plt.axis('off')

if __name__ == '__main__':
    f_im, f_dp, f_md, f_ma = argv_opt()
    net = caffe.Net(f_dp, f_md, caffe.TEST)
    tffm = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    tffm.set_transpose('data', (2,0,1))
    if f_ma != '':
        tffm.set_mean('data', np.load(f_ma).mean(1).mean(1))
    tffm.set_raw_scale('data', 255)
    tffm.set_channel_swap('data', (2,1,0))
    im = caffe.io.load_image(f_im)
    net.blobs['data'].data[...] = tffm.preprocess('data', im)
    print '{:<20}{:<20}'.format('image:', f_im)
    print '{:<20}{:<20}'.format('data-shape:',net.blobs['data'].data[0].shape)
    out = net.forward()
    filters = net.params['conv1'][0].data[:,0]
    #filters = net.params['res5c_branch2c'][0].data[:64,0]
    #feature = net.blobs['res5c_branch2c'].data[0,0:64]
    #feature = net.blobs['res2a_branch1'].data[0,0:64]
    feature = net.blobs['res2b_branch2c'].data[0]
    #feature = net.blobs['conv1'].data[0]
    vis_square(feature, padval=0)
