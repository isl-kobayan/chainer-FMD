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

import fmdio
import models

parser = argparse.ArgumentParser(
    description='Learning convnet from Flickr Material Database')
parser.add_argument('--train', '-f', default='train_list.txt',
                    help='Path to training image-label list file')
parser.add_argument('--val', '-v', default='test_list.txt',
                    help='Path to validation image-label list file')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alex, alexbn, googlenet, googlenet2, googlenetbn, caffealex, caffegooglenet, vggnet, chainervggnet)')
parser.add_argument('--opt', '-p', default='adam',
                    help='optimizer \
                    (adam, adadelta, adagrad, momentumsgd, rmsprop)')
parser.add_argument('--initlr', default=0.01, type=float,
                    help='Initialize the learning rate from given value')
parser.add_argument('--lrstep', default=0.97, type=float,
                    help='Set learning rate scale')
parser.add_argument('--batchsize', '-B', type=int, default=10,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=10,
                    help='Validation minibatch size')
parser.add_argument('--val_interval', '-i', type=int, default=0,
                    help='Validation interval')
parser.add_argument('--epoch', '-E', default=10, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--loaderjob', '-j', default=20, type=int,
                    help='Number of parallel data loading processes')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--out', '-o', default='model',
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
train_list = fmdio.load_image_list(args.train, args.root)
val_list = fmdio.load_image_list(args.val, args.root)
mean_image = None
if args.arch == 'googlenet' or args.arch == 'googlenet2' or args.arch == 'caffegooglenet':
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123
else:
    mean_image = pickle.load(open(args.mean, 'rb'))

assert mean_image is not None

train_size = len(train_list)
val_size = len(val_list)

if args.val_interval == 0:
    args.val_interval = train_size# / args.batchsize
iterates_per_epoch = train_size / args.batchsize
assert train_size % args.batchsize == 0
assert val_size % args.val_batchsize == 0

# Prepare model
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
elif args.arch == 'vggnet':
    model = models.VGGNet()
elif args.arch == 'chainervggnet':
    model = models.ChainerVGGNet()
else:
    raise ValueError('Invalid architecture name')

print(model.__class__.__name__)
print('total epoch : ' + str(args.epoch))

nowt = datetime.datetime.today()
outdir = './results/' + args.arch + '_bs' + str(args.batchsize) + '_' + nowt.strftime("%Y%m%d-%H%M")
os.mkdir(outdir)
args.out = outdir + '/' + args.out + '_' + args.arch
args.outstate = outdir + '/' + args.outstate

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
#optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
# Setup optimizer
if args.opt == 'adam':
    optimizer = optimizers.Adam()
elif args.opt == 'momentumsgd':
    optimizer = optimizers.MomentumSGD(lr=args.initlr, momentum=0.9)
elif args.opt == 'adadelta':
    optimizer = optimizers.AdaDelta()
elif args.opt == 'adagrad':
    optimizer = optimizers.AdaGrad()
elif args.opt == 'rmsprop':
    optimizer = optimizers.RMSprop()
else:
    raise ValueError('Invalid optimizer name')
print(optimizer.__class__.__name__)
optimizer.setup(model)
print(dir(model['loss3_fc'].W))
print(model['loss3_fc'].W.data)
print(model['loss3_fc'].b.data)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)


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

    batch_pool = [None] * args.batchsize
    val_batch_pool = [None] * args.val_batchsize
    pool = multiprocessing.Pool(args.loaderjob)
    data_q.put('train')
    for epoch in six.moves.range(1, 1 + args.epoch):
        print('epoch', epoch, file=sys.stderr)
        if args.opt == 'momentumsgd':
            print('learning rate', optimizer.lr, file=sys.stderr)
        perm = np.random.permutation(len(train_list))
        for idx in perm:
            path, label = train_list[idx]
            batch_pool[i] = pool.apply_async(fmdio.read_image, (path, model.insize, mean_image, False, True))
            y_batch[i] = label
            i += 1

            if i == args.batchsize:
                for j, x in enumerate(batch_pool):
                    x_batch[j] = x.get()
                data_q.put((x_batch.copy(), y_batch.copy()))
                i = 0

            count += 1
            if count % args.val_interval == 0:
                data_q.put('val')
                j = 0
                for path, label in val_list:
                    val_batch_pool[j] = pool.apply_async(
                        fmdio.read_image, (path, model.insize, mean_image, True, False))
                    val_y_batch[j] = label
                    j += 1

                    if j == args.val_batchsize:
                        for k, x in enumerate(val_batch_pool):
                            val_x_batch[k] = x.get()
                        data_q.put((val_x_batch.copy(), val_y_batch.copy()))
                        j = 0
                data_q.put('train')

        if args.opt == 'momentumsgd':
            optimizer.lr *= args.lrstep
    pool.close()
    pool.join()
    data_q.put('end')


def log_result():
    # Logger
    trainlogfilename=outdir+'/train.log'
    testlogfilename=outdir+'/test.log'
    train_count = 0
    train_cur_loss = 0
    train_cur_accuracy = 0
    begin_at = time.time()
    val_begin_at = None
    while True:
        result = res_q.get()
        if result == 'end':
            print(file=sys.stderr)
            break
        elif result == 'train':
            print(file=sys.stderr)
            train = True
            if val_begin_at is not None:
                begin_at += time.time() - val_begin_at
                val_begin_at = None
            continue
        elif result == 'val':
            print(file=sys.stderr)
            train = False
            val_count = val_loss = val_accuracy = 0
            val_begin_at = time.time()
            continue

        loss, accuracy = result
        if train:
            train_count += 1
            duration = time.time() - begin_at
            throughput = train_count * args.batchsize / duration
            sys.stderr.write(
                '\rtrain {} updates ({} samples) time: {} ({} images/sec)'
                .format(train_count, train_count * args.batchsize,
                        datetime.timedelta(seconds=duration), throughput))

            train_cur_loss += loss
            train_cur_accuracy += accuracy
            if train_count % iterates_per_epoch == 0:
                mean_loss = train_cur_loss / iterates_per_epoch
                mean_error = 1 - train_cur_accuracy / iterates_per_epoch
                print(file=sys.stderr)
                print(json.dumps({'type': 'train', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss}))
                with open(trainlogfilename, 'a') as f:
                    f.write(json.dumps({'type': 'train', 'iteration': train_count,
                                        'error': mean_error, 'loss': mean_loss})+'\n')
                sys.stdout.flush()
                train_cur_loss = 0
                train_cur_accuracy = 0
        else:
            val_count += args.val_batchsize
            duration = time.time() - val_begin_at
            throughput = val_count / duration
            sys.stderr.write(
                '\rval   {} batches ({} samples) time: {} ({} images/sec)'
                .format(val_count / args.val_batchsize, val_count,
                        datetime.timedelta(seconds=duration), throughput))

            val_loss += loss
            val_accuracy += accuracy
            if val_count == val_size:
                mean_loss = val_loss * args.val_batchsize / val_size
                mean_error = 1 - val_accuracy * args.val_batchsize / val_size
                print(file=sys.stderr)
                print(json.dumps({'type': 'val', 'iteration': train_count,
                                  'error': mean_error, 'loss': mean_loss}))
                with open(testlogfilename, 'a') as f:
                    f.write(json.dumps({'type': 'val', 'iteration': train_count,
                                        'error': mean_error, 'loss': mean_loss})+'\n')
                sys.stdout.flush()


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
        elif inp == 'train':  # restart training
            res_q.put('train')
            model.train = True
            continue
        elif inp == 'val':  # start validation
            res_q.put('val')
            serializers.save_hdf5(args.out, model)
            serializers.save_hdf5(args.outstate, optimizer)
            if ep % 5 == 0:
                serializers.save_hdf5(args.out + str(ep), model)
                serializers.save_hdf5(args.outstate + str(ep), optimizer)
            ep=ep+1
            model.train = False
            continue

        volatile = 'off' if model.train else 'on'
        x = chainer.Variable(xp.asarray(inp[0]), volatile=volatile)
        t = chainer.Variable(xp.asarray(inp[1]), volatile=volatile)

        if model.train:
            optimizer.update(model, x, t)
            if not graph_generated:
                with open(outdir + '/graph_' + args.arch + '.dot', 'w') as o:
                    o.write(computational_graph.build_computational_graph(
                        (model.loss,)).dump())
                print('generated graph', file=sys.stderr)
                graph_generated = True
                pickle.dump(args, open(outdir + '/args', 'wb'), -1)
                with open(outdir + '/args.txt', 'w') as o:
                    print(args, file=o)
        else:
            model(x, t)

        res_q.put((float(model.loss.data), float(model.accuracy.data)))
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

# Save final model
serializers.save_hdf5(args.out, model)
serializers.save_hdf5(args.outstate, optimizer)
