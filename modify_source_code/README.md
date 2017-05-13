# Modify Caffe Source Code For Support Multi-label

replace these 4 file:
- 'caffe/include/caffe/util/io.hpp'
- 'caffe/src/caffe/util/io.cpp'
- 'caffe/src/caffe/layers/data_layer.cpp'
- 'caffe/src/caffe/proto/caffe.proto'

# usage

`build/tools/convert_imageset image/ label/txt out/lmdb/`

same as the original

# note

add datum.labels

all label save into datum.labels

Data layer top[0] unchange

Data layer top[1] is labels [reshape (batch_size, labels_size(), 1, 1)]

can not use Accuracy layer

I don't know how to solve it
