#!/usr/bin/env python
""" visualizing filters of convolutional layer.

Prerequisite: To run this program, prepare model's weights data with hdf-5 format.
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
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alexbn, googlenetbn, overfeat)')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--layername', '-l', default='conv1',
                    help='layer name (example: conv1, fc6, etc)')
parser.add_argument('--val_batchsize', '-b', type=int, default=20,
                    help='Validation minibatch size')
parser.add_argument('--scale', '-s', type=int, default=1,
                    help='filter scale')
parser.add_argument('--pad', '-p', type=int, default=1,
                    help='filter padding')
parser.add_argument('--cols', '-c', type=int, default=1,
                    help='columns')
parser.add_argument('--finetune', '-f', default=False, action='store_true',
                    help='Visualize filters before fine-tuning if True (default: False)')
args = parser.parse_args()

# Prepare model
model = models.getModel(args.arch)
if model is None:
    raise ValueError('Invalid architecture name')
if args.finetune:
    print ('finetune')
    model.set_finetune()

print(model.__class__.__name__)

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)

layername_without_special_char = args.layername.replace('/', '_')
outdir = os.path.dirname(args.initmodel) + '/Weights_' + layername_without_special_char
if not os.path.exists(outdir):
    os.makedirs(outdir)

out_ch, in_ch, height, width = model[args.layername].W.data.shape
W = model[args.layername].W.data
Wmax = np.max(W)
Wmin = np.min(W)
Wrange = Wmax - Wmin
print('max, min = ', Wmax, Wmin)

cols = args.cols
pad = args.pad
scale = args.scale
rows = (int)((out_ch + cols - 1) / cols)

w_step = width * scale + pad
h_step = height * scale + pad

# all filters
all_img_width = w_step * cols + pad
all_img_height = h_step * rows + pad
all_img = Image.new('RGB', (all_img_width, all_img_height), (255, 255, 255))

# if number of input channels is 3, visualize filter with RGB
if in_ch == 3:
    for i in six.moves.range(0, out_ch):
        filter_data = (W[i].transpose(1, 2, 0) - Wmin) * 255 / Wrange
        img = Image.fromarray(np.uint8(filter_data))
        if args.scale > 1:
            img = img.resize((width * scale, height * scale), Image.NEAREST)
        all_img.paste(img, (pad + (i % cols) * w_step, pad + (int)(i / rows) * h_step))
        img.save(outdir + '/w' + str(i) + '.png')
    all_img.save(outdir + '/filters.png')
else:
    for i in six.moves.range(0, out_ch):
        for j in six.moves.range(0, in_ch):
            filter_data = (W[i][j] - Wmin) * 255 / Wrange
            img = Image.fromarray(np.uint8(filter_data))
            img.save(outdir + '/w' + str(i) + '_' + str(j) + '.png')
