import sys
import cv2
import dlib
import numpy
from PIL import Image
import matplotlib.pyplot as plt
caffe_root = '/home/lab210/WS/caffe'
sys.path.append(caffe_root + '/python')
import caffe


#deploy_file = 'models/deploy.prototxt'
#models_file = 'models/final40-20w_iter_18750.caffemodel'
#mean_file = 'models/mean.npy'
#net = caffe.Classifier(
#    deploy_file,
#    models_file,
#    image_dims=(256,256),
#    raw_scale=255,
#    mean=numpy.load(mean_file).mean(1).mean(1),
#    channel_swap=(2,1,0)
#)

f_dp = 'models/deploy.prototxt'
f_md = 'models/final40-20w_iter_18750.caffemodel'
f_ma = 'models/mean.npy'

net = caffe.Net(f_dp, f_md, caffe.TEST)
tffm = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
tffm.set_transpose('data', (2,0,1))
tffm.set_mean('data', numpy.load(f_ma).mean(1).mean(1))
tffm.set_raw_scale('data', 255)
tffm.set_channel_swap('data', (2,1,0))

def classify_image(image):
    global net
    global tffm
    #scores = net.predict([image],oversample=True)
    net.blobs['data'].data[...] = tffm.preprocess('data', image)
    scores = net.forward()['fc40']
    labels = []
    with open('models/headers.txt') as f:
        for l in f.readlines():
            labels.append(l.strip())
    #print scores
    meta = zip(labels, scores[0])
    meta_max = sorted(meta, key=lambda item:-item[1])
    #meta_min = sorted(meta, key=lambda item:item[1])
    return meta_max[:5]
    #print meta
    
def run_detect():
    c_color = (255, 0, 0)
    p_color = (0, 0, 255)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    capture = cv2.VideoCapture("rtsp://admin:drrobot1@192.168.0.191:554/play3.sdp")
    while capture.isOpened():
        ret, img = capture.read()
        dets = detector(img, 1)
        for d in dets:
            shape = predictor(img, d)
            pt1 = (d.left()-20, d.top()-20)
            pt2 = (d.right()+20, d.bottom()+20)
            box = (d.left()-20, d.top()-20, d.right()+20, d.bottom()+20)
            region = img.crop(box)
            cv2.imwrite('tmp.jpg', region)
            p1_text = (d.right(), d.top())
            cv2.rectangle(img, pt1, pt2, color=c_color, thickness=2)
            cv2.putText(img, 'Smiling', p1_text, 0, 0.5, (0, 255, 255), 2)
            cv2.putText(img, 'Smiling Male Glasses LongHair BigNose', p1_text, 0, 0.5, (0, 255, 255), 2)
            #face_point = numpy.array([[p.x, p.y] for p in shape.parts()])
            #for index in range(68):
            #    point = (face_point[index][0], face_point[index][1])
            #    cv2.circle(img, point, radius=2, color=p_color, thickness=2)
        cv2.imshow('video', img)
        ch = cv2.waitKey(33)
        if ch == 'q' or ch == 27:
            capture.release()
            sys.exit()
            
    
def pic_detect():
    c_color = (255, 0, 0)
    p_color = (0, 0, 255)
    plt.figure()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    img = cv2.imread('12.jpeg')
    image = Image.open('12.jpeg')
    dets = detector(img, 1)
    for index, d in enumerate(dets):
        shape = predictor(img, d)
        pt1 = (d.left()-50, d.top()-90)
        pt2 = (d.right()+50, d.bottom()+10)
        box = (d.left()-50, d.top()-90, d.right()+50, d.bottom()+10)
        region = image.crop(box)
        region = region.resize((224,224), Image.ANTIALIAS)
        region.save('tmp/tmp.jpg')
        im = caffe.io.load_image('tmp/tmp.jpg')
        result = classify_image(im)
        p1_text = (d.right(), d.top())
        cv2.rectangle(img, pt1, pt2, color=c_color, thickness=4)
        cv2.putText(img, '{}'.format(index+1), (d.left()-20, d.top()-25), 0, 2, p_color, 4)
        
        print '{:-<13} fece: {}'.format('',index+1)
        for a, b in result:
            print '{:<20}: {:0<20}'.format(a,b)
        #plt.text(-10,-40*index, result)
        #cv2.putText(img, '{}'.format('asdf'), (20,20), 0, 1, (0, 0, 255), 2)
        #face_point = numpy.array([[p.x, p.y] for p in shape.parts()])
        #for index in range(68):
        #    point = (face_point[index][0], face_point[index][1])
        #    cv2.circle(img, point, radius=2, color=p_color, thickness=2)
    #cv2.imshow('picture', img)
    #cv2.waitKey()
    b,g,r = cv2.split(img)
    ims = cv2.merge([r,g,b])
    plt.imshow(ims)
    plt.show()
            
def caffe_detect():
    f_dp = 'models/deploy.prototxt'
    f_md = 'models/models/final40-20w_iter_18750.caffemodel'
    f_ma = 'models/mean.npy'
    nets = caffe.Net(f_dp, f_md, caffe.TEST)
    tffm = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    tffm.set_transpose('data', (2,0,1))
    tffm.set_mean('data', np.load(f_ma).mean(1).mean(1))
    tffm.set_raw_scale('data', 255)
    tffm.set_channel_swap('data', (2,1,0))
    #imgs = Image.open(f_im).resize((224,224), Image.ANTIALIAS)
    #imgs.save('tmp.jpg')
    #im = caffe.io.load_image('tmp.jpg')
    #net.blobs['data'].data[...] = tffm.preprocess('data', im)
    return net

if __name__ == '__main__':
    pic_detect()
