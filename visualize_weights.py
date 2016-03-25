#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
import datetime
import locale
import os

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/models')

import fmdio
import models

parser = argparse.ArgumentParser(
    description='Learning Flickr Material Database')
#parser.add_argument('--trainfile', '-f', default='image_list.txt',
#                    help='Path to training image-label list file')
parser.add_argument('--modelfile', '-i', 
                    help='Path to model file')
parser.add_argument('--val', '-v', default='test_list.txt',
                    help='Path to validation image-label list file')
parser.add_argument('--layername', '-l', default='conv1',
                    help='layer name (example: conv1, fc6, etc)')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alexbn, googlenetbn, overfeat)')
#parser.add_argument('--batchsize', '-B', type=int, default=25,
#                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=20,
                    help='Validation minibatch size')
parser.add_argument('--epoch', '-E', default=1, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=10, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--out', '-o', default='model',
                    help='Path to save model on each validation')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
    print('use GPU')
else:
    print('use CPU only')

xp = cuda.cupy if args.gpu >= 0 else np



# Prepare dataset
#train_list = fmdio.load_image_list(args.trainfile)
val_list = fmdio.load_image_list(args.val)
mean_image = pickle.load(open(args.mean, 'rb'))
#N_list = len(train_list)
N_listv = len(val_list)
#args.batchsize = args.val_batchsize
assert len(val_list) % args.val_batchsize == 0
lblmax = np.asarray(val_list)[:,1].astype(np.int32).max() + 1
print(lblmax)
print(args.val_batchsize)

if args.arch == 'nin':
    model = models.NIN()
elif args.arch == 'alex':
    model = models.Alex()
elif args.arch == 'alexbn':
    model = models.AlexBN()
elif args.arch == 'googlenet':
    model = models.GoogLeNet()
elif args.arch == 'googlenet2':
    model = models.GoogLeNet2()
elif args.arch == 'googlenetbn':
    model = models.GoogLeNetBN()
elif args.arch == 'caffealex':
    model = models.CaffeAlex()
elif args.arch == 'caffegooglenet':
    model = models.CaffeGoogLeNet()
else:
    raise ValueError('Invalid architecture name')

print(model.__class__.__name__)

# Prepare model
#model = None
#with open(args.modelfile, 'rb') as f:
#    model = pickle.load(f)

#assert model != None

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
#if args.resume:
#    print('Load optimizer state from', args.resume)
#    serializers.load_hdf5(args.resume, optimizer)


nowt=datetime.datetime.today()
layername_without_special_char = args.layername.replace('/', '_')
outdir=os.path.dirname(args.modelfile)+'/Weights_'+layername_without_special_char
os.mkdir(outdir)
#args.out=outdir + '/' + args.out


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
#optimizer = optimizers.Adam()#MomentumSGD(lr=0.01, momentum=0.9)
#optimizer.setup(model)
#print(dir(model.conv1.W.data), model.conv1.W.data.shape, model.conv1.W.data.__class__)
#n1, n2, h, w = model.conv1.W.shape
n1, n2, h, w = model[args.layername].W.data.shape
W=model[args.layername].W.data
Wmax=np.max(W)
Wmin=np.min(W)
Wrange=Wmax-Wmin
print('max, min = ', Wmax, Wmin)
if n2 == 3:
    for i in six.moves.range(0, n1):
        imgdata = (W[i].transpose(1, 2, 0)-Wmin)*255/Wrange
        img = np.ndarray(imgdata.shape, dtype=np.float32)
        img[:] = imgdata
        WImg=Image.fromarray(np.uint8(img))
        WImg.save(outdir + '/w'+str(i)+'.png')
else:
    for i in six.moves.range(0, n1):
        for j in six.moves.range(0, n2):
            imgdata = (W[i][j]-Wmin)*255/Wrange
            img = np.ndarray(imgdata.shape, dtype=np.float32)
            img[:] = imgdata
            WImg=Image.fromarray(np.uint8(img))
            WImg.save(outdir + '/w'+str(i)+'_'+str(j)+'.png')


# Save final model
#pickle.dump(model, open(args.out, 'wb'), -1)
