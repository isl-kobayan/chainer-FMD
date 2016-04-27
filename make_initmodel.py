import argparse
import os
import cPickle as pickle

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer.functions import caffe

parser = argparse.ArgumentParser(
    description='make hdf5 data file of caffemodel')
parser.add_argument('caffemodel', default=None,
                    help='Path to caffemodel')
args = parser.parse_args()

dirname = os.path.dirname(args.caffemodel)
filename = os.path.basename(args.caffemodel)
fname, ext = os.path.splitext(filename)

hdf5filepath = os.path.join(dirname, fname + '.hdf5')
pklfilepath = os.path.join(dirname, fname + '.pkl')
model = caffe.CaffeFunction(args.caffemodel)
serializers.save_hdf5(hdf5filepath, model)
pickle.dump(model, open(pklfilepath, 'wb'))
