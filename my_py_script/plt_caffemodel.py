import numpy as np
import matplotlib.pyplot as plt
import os
import getopt
import sys

caffe_root = '/home/lab704/caffe'
sys.path.append(caffe_root + '/python')
import caffe

type_flags = 0

def this_help():
    print 'usage:'
    print '{:<10}{:>5}'.format('-d', '--deploy file')
    print '{:<10}{:>5}'.format('-m', '--weight file')
    #print '{:<10}{:>5}'.format('-t', '--type')
    print '{:<10}{:>5}'.format('--show', '--show type(default=0)')
    print '{:*<10}{:>5}'.format('', '--0 show terminal')
    print '{:*<10}{:>5}'.format('', '--1 show layer numpy.array')
    print '{:*<10}{:>5}'.format('', '--2 show matplotlib image')
    print '{:<10}{:>5}'.format('--layer=', '--which layer(default=-1)')
    print 'example:'
    print '{} {} {} {} {} {}'.format('python', 'this.py', '-d', 'path/to/deploy', '-m', 'path/to/caffemodel')

def show_feature(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.show()

def show_layer(net, layer, out_type):
    if layer == -1:
        return
    t_dict = {}
    n_dict = {}
    for index, (x, y) in enumerate(net.params.items()):
        t_dict[str(index)] = y[0].data
        t_dict[x] = y[0].data
        n_dict[str(index)] = x
    weight = t_dict[layer]
    print '{:<10} {:<10} {:<10} {:<10}'.format('info:', layer, n_dict[layer], weight.shape)
    if out_type == 1:
        print weight
    elif out_type == 2:
        (n, c, h, w) = weight.shape
        if layer != '0':
            ts = weight.reshape(n*c, h, w)
        else:
            ts = weight.transpose(0, 2, 3, 1)
        show_feature(ts)
    elif out_type == 3: 
        (n, c, h, w) = weight.shape
        if layer != '0':
            ts = weight.reshape(n*c, h, w)
        else:
            ts = weight.transpose(0, 2, 3, 1)
        print weight
        show_feature(ts)

if __name__ == '__main__':
    caffe.set_mode_gpu()
    try:
        if len(sys.argv) < 5:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'hd:m:', ['show=', 'layer='])
        n_layer = -1
        out_type = 0
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '-d':
                deploy_path = str(value)
                print '{:<10} {:>5}'.format('deploy:', value)
            if op == '-m':
                weight_path = str(value)
                print '{:<10} {:>5}'.format('weight:', value)
            if op == '--show':
                type_list = value.strip().replace(' ', '').split(',')
                if '1' in type_list:
                    out_type += 1
                if '2' in type_list:
                    out_type += 2
                    plt.rcParams['figure.figsize'] = (8, 8)
                    plt.rcParams['image.interpolation'] = 'nearest'
                    plt.rcParams['image.cmap'] = 'gray'
            if op == '--layer':
                n_layer = str(value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()

    net = caffe.Net(deploy_path, weight_path, caffe.TEST)
    print '{:<10}{:<20}{:<20}'.format('[index]', '[layer]', '[(n, c, h, w)]')
    for index, (x, y) in enumerate(net.params.items()):
        print '{:<10}{:<20}{:<20}'.format(index, x, y[0].data.shape)
    #net.forward()
    #show_feature(net.blobs['conv1'].data[0])
    show_layer(net, n_layer, out_type)

