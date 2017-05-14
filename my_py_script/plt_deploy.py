import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import getopt
import sys

caffe_root = '/home/lab704/caffe'
sys.path.append(caffe_root + '/python')
import caffe

layer_dict = {}

def this_help():
    print 'usage:'
    print '{:<10}{:>5}'.format('-d', '--deploy file')
    print '{:<10}{:>5}'.format('-i', '--image file')
    print '{:<10}{:>5}'.format('--layer=', '--which layer(default=-1)')
    print 'example:'
    print '{} {} {} {} {} {}'.format('python', 'this.py', '-d', 'path/to/deploy', '-i', 'path/to/cat.jpg')

def get_layer_image(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data
    
def get_net_parameter(net):
    global layer_dict
    print '{:^10}{:^20}{:^20}'.format('[index]', '[layer]', '[(n, c, h, w)]')
    l_data = []
    for index, (x, y) in enumerate(net.params.items()):
        print '{:^10}{:<20}{:<20}'.format(index, x, y[0].data.shape)
        layer_dict[index] = x
        
def put_image_into_net(im, net):
    im_input = im[np.newaxis,:,:,:].transpose(0, 3, 1, 2)
    print 'input image shape: ', im_input.shape
    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input
    
def show_origin_image(img, net):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.set_title('origin image')
    ax1.set_xlabel('%s' % str(net.blobs['data'].data.shape))
    ax2.imshow(get_layer_image(net.blobs['data'].data[0]), cmap='gray')
    ax2.set_title('data-blob')
    ax2.set_xlabel('gary image 3 channel')
    
def show_layer_image(c_layer, net):
    if len(net.params[layer_dict[c_layer]][0].data.shape) < 4:
        print 'shape length should be = 4, can not show this layer'
        sys.exit()
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(get_layer_image(net.params[layer_dict[c_layer]][0].data[:,0]), cmap='gray')
    ax1.set_title('weights(filter)')
    ax1.set_xlabel('%s' % str(net.params[layer_dict[c_layer]][0].data.shape))
    ax2.imshow(get_layer_image(net.blobs[layer_dict[c_layer]].data[0]), cmap='gray')
    ax2.set_title('post %s images' % layer_dict[c_layer])
    ax2.set_xlabel('%s' % str(net.blobs[layer_dict[c_layer]].data.shape))
    

if __name__ == '__main__':
    global layer_dict
    image_path = ''
    deploy_path = ''
    c_layer = 0
    try:
        if len(sys.argv) < 5:
            this_help()
            sys.exit()
        opts, args = getopt.getopt(sys.argv[1:], 'hi:d:', ['layer='])
        for op, value in opts:
            if op == '-h':
                this_help()
                sys.exit()
            if op == '-i':
                image_path = str(value)
                print '{:<10} {:>5}'.format('image:', value)
            if op == '-d':
                deploy_path = str(value)
                print '{:<10} {:>5}'.format('deploy:', value)
            if op == '--layer':
                c_layer = int(value)
    except getopt.GetoptError:
        print 'getopt error'
        sys.exit()
        
    img = caffe.io.load_image(image_path)
    net = caffe.Net(deploy_path, caffe.TEST)
    put_image_into_net(img, net)
    get_net_parameter(net)
    #show_net_parameter()
    net.forward()
    
    #plt.rcParams['image.cmap'] = 'gray'
    show_origin_image(img, net)
    show_layer_image(c_layer, net)
    plt.show()
    

