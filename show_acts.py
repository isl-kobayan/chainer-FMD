#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学習済みモデルに画像を入れ、最後の層の各ニューロンを活性させる画像は何か調べます。

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
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import chainer.functions as F

import fmdio
import models

parser = argparse.ArgumentParser(
    description='Learning convnet from Flickr Material Database')
parser.add_argument('--val', '-v', default='val.txt',
                    help='Path to validation image-label list file')
parser.add_argument('--label', '-l', default='num2label.txt',
                    help='Path to validation image-label list file')
parser.add_argument('--mean', '-m', default='ilsvrc_2012_mean.npy', #'mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alex, alexbn, vgg16, googlenet, googlenet2, googlenetbn)')
parser.add_argument('--batchsize', '-B', type=int, default=10,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=10,
                    help='Validation minibatch size')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--out', '-o', default=None,
                    help='Path to save model on each validation')
parser.add_argument('--outstate', '-s', default='state',
                    help='Path to save optimizer state on each validation')
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
#train_list = fmdio.load_image_list(args.train, args.root)
num2label = fmdio.load_num2label(args.label)
val_list = fmdio.load_image_list(args.val, args.root)

val_size = len(val_list)

assert val_size % args.val_batchsize == 0

# Prepare model
model = models.getModel(args.arch)
if model is None:
    raise ValueError('Invalid architecture name')

mean_image = None
if hasattr(model, 'getMean'):
    mean_image = model.getMean()
else:
    #mean_image = pickle.load(open(args.mean, 'rb'))
    mean_image = np.load(args.mean)

assert mean_image is not None

print(model.__class__.__name__)


nowt = datetime.datetime.today()
outdir = './test_' + args.arch + '_bs' + str(args.val_batchsize) + '_' + nowt.strftime("%Y%m%d-%H%M")
if args.out is None:
    outdir = os.path.dirname(args.initmodel)
else:
    outdir = args.out
    os.mkdir(outdir)

lblmax = np.asarray(val_list)[:,1].astype(np.int32).max() + 1
print(lblmax)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)

print(model.links)
print(model.namedlinks)

# ------------------------------------------------------------------------------
# This example consists of three threads: data feeder, logger and trainer.
# These communicate with each other via Queue.
data_q = queue.Queue(maxsize=1)
res_q = queue.Queue()

def feed_data():
    # Data feeder
    i = 0
    count = 0

    x_batch = np.ndarray(
        (args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
    y_batch = np.ndarray((args.batchsize,), dtype=np.int32)
    val_x_batch = np.ndarray(
        (args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
    val_y_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)
    val_i_batch = np.ndarray((args.val_batchsize,), dtype=np.int32)
    batch_pool = [None] * args.batchsize
    val_batch_pool = [None] * args.val_batchsize
    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put('val')
    j = 0
    for idx, pl in enumerate(val_list):
        path, label = pl
        val_batch_pool[j] = pool.apply_async(
            fmdio.read_image, (path, model.insize, mean_image, True, False))
        val_y_batch[j] = label
        val_i_batch[j] = idx
        j += 1

        if j == args.val_batchsize:
            for k, x in enumerate(val_batch_pool):
                val_x_batch[k] = x.get()
            data_q.put((val_x_batch.copy(), val_y_batch.copy(), val_i_batch.copy()))
            j = 0
    pool.close()
    pool.join()
    data_q.put('end')


def log_result():
    # Logger
    actsfilename=outdir+'/acts.csv'
    actsargfilename=outdir+'/actsargs.csv'
    trainlogfilename=outdir+'/train.log'
    testlogfilename=outdir+'/test_result.log'
    tablefilename=outdir+'/table.csv'
    compfilename=outdir+'/comp.log'
    confusion_matrix=np.zeros(lblmax*lblmax, dtype=np.int32)
    train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    begin_at = time.time()
    val_begin_at = None
    #with open(compfilename, 'w') as f:
    #    label_col_str = ''
    #    for labelname in num2label:
    #        label_col_str = label_col_str + '\t' + labelname
    #    f.write('path\ttrue_label\tpred_label' + label_col_str +'\n')
    allacts=None
    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break
        elif result == 'val':
            print(file=sys.stderr)
            train = False
            val_count = val_loss = val_accuracy = 0
            val_begin_at = time.time()
            continue

        loss, accuracy, lastacts, y_true, list_indices, acts = result
        last_activities = chainer.Variable(xp.asarray(lastacts), volatile='on')
        probability_map = F.softmax(last_activities)
        y_pred = lastacts.reshape(len(lastacts), -1).argmax(axis=1)
        confusion_indices = xp.asarray(y_true)*lblmax + y_pred
        #print(acts.shape)
        #print(acts.__class__.__name__)
        #print(dir(acts))

        if not train:
            val_count += args.val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / args.val_batchsize, val_count,
                        datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            val_accuracy += accuracy
            for idx in confusion_indices:
                confusion_matrix[int(idx)]+=1
            res_str = ''
            acts_str = ''
            if allacts is None:
                allacts = acts.get().reshape((len(acts), -1))
            else:
                allacts = np.r_[allacts, acts.get().reshape((len(acts), -1))]
            #print(allacts.shape)

            for idx, list_idx in enumerate(list_indices):
                path, label = val_list[list_idx]
                res_str = res_str + path + '\t' + num2label[label] + '\t' + num2label[int(y_pred[idx])]
                acts_str = acts_str + path + '\t' + num2label[label] + '\t' + num2label[int(y_pred[idx])]
                thisacts=acts[idx].get().reshape(len(acts[idx]))
                fmdio.append_acts(actsfilename, path, thisacts)

                for label_idx in six.moves.range(lblmax):
                    res_str = res_str + '\t' + str(probability_map.data[idx, label_idx])
                    #acts_str = acts_str +
                    #print(thisacts.shape)
                res_str = res_str + '\n'
                acts_str = acts_str + '\n'
            #with open(compfilename, 'a') as f:
            #    f.write(res_str)
            if val_count == val_size:
                mean_loss = val_loss * args.val_batchsize / val_size
                mean_error = 1 - val_accuracy * args.val_batchsize / val_size
                print(file=sys.stderr)
                print(json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss}))
                #with open(testlogfilename, 'a') as f:
                #    f.write(json.dumps({'type': 'val', 'iteration': train_count,
                #                        'error': mean_error, 'loss': mean_loss})+'\n')
                sys.stdout.flush()
                print(confusion_matrix.reshape((lblmax,lblmax)))
                #np.savetxt(tablefilename, confusion_matrix.reshape((lblmax,lblmax)), delimiter=",", fmt='%d')
                #fmdio.save_confusion_matrix(tablefilename, confusion_matrix, num2label)
        del last_activities, probability_map
    fmdio.get_act_table(actsargfilename, val_list, allacts)
def train_loop():
    # Trainer
    graph_generated = False
    ep = 1
    while True:
        while data_q.empty():
            time.sleep(0.1)
        inp = data_q.get()
        if inp == 'end':  # quit
            res_q.put('end')
            break
        elif inp == 'val':  # start validation
            res_q.put('val')
            model.train = False
            continue

        volatile = 'off' if model.train else 'on'
        x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
        t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)
        indices = np.asarray(inp[2])
        y_true = np.asarray(inp[1])
        model.test(x, t)
        res_q.put((float(model.loss.data), float(model.accuracy.data), model.lastacts.data, y_true, indices, model.acts.data))
        del x, t

# Invoke threads
feeder = threading.Thread(target=feed_data)
feeder.daemon = True
feeder.start()
logger = threading.Thread(target=log_result)
logger.daemon = True
logger.start()

train_loop()
feeder.join()
logger.join()
