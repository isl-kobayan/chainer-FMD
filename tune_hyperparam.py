import subprocess
import json
import argparse

import train_fmd

parser = argparse.ArgumentParser(
    description='Learning convnet from Flickr Material Database')
parser.add_argument('--train', '-t', default='train.txt',
                    help='Path to training image-label list file')
parser.add_argument('--val', '-v', default='val.txt',
                    help='Path to validation image-label list file')
#parser.add_argument('--train', '-t', default='fmd_odd.txt',
#                    help='Path to training image-label list file')
#parser.add_argument('--val', '-v', default='fmd_even.txt',
#                    help='Path to validation image-label list file')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alex, alexbn, vgg16, googlenet, googlenet2, googlenetbn, caffealex, caffegooglenet)')
parser.add_argument('--opt', '-p', default='momentumsgd',
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
parser.add_argument('--finetune', '-f', default=True, action='store_true',
                    help='do fine-tuning if this flag is set (default: False)')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
args = parser.parse_args()

#batchsizes = [1,2,4,8,10,16,20,25,40,50,100]
batchsizes = [50]
#batchsizes = [10, 20, 50]
#initlrs = [0.0005, 0.001, 0.005]
initlrs = [0.0005]
#initlrs = [0.00005, 0.0002]
#lrsteps = [0.97, 0.8]
lrsteps = [0.8, 0.8, 0.8]
#batchsizes = [10]
#initlrs = [0.005]
#lrsteps = [0.97]

#batchsizes = [40]
#archs = ['caffealex', 'caffegooglenet', 'caffevgg']
#archs = ['caffegooglenet', 'caffealex']
#archs = ['googlenet']
archs = ['vgg16']
#archs = ['squeezenet10', 'squeezenet11']
#archs = ['caffealex']
#archs = ['googlenet']
results="dir\tarch\tbatchsize\tinitlr\tlrstep\terror(train)\terror(val)\tloss(train)\tloss(val)\n"
with open('results/TuneResults.txt', 'a') as f:
    f.write(results)

for initlr in initlrs:
    args.initlr = initlr
    for lrstep in lrsteps:
        args.lrstep = lrstep
        for arch in archs:
            args.arch = arch
            for bs in batchsizes:
                args.batchsize = bs
                args.out = 'model'
                args.outstate = 'state'
                ret = train_fmd.train(args)
                result = ret.outdir + '\t' + arch + '\t' + str(bs) + '\t' + str(initlr) + '\t' + str(lrstep) + '\t' \
                    + str(ret.train_error) + '\t' + str(ret.val_error) + '\t' \
                    + str(ret.train_loss) + '\t' + str(ret.val_loss)
                with open('results/TuneResults.txt', 'a') as f:
                            f.write(result + '\n')
                results += result + '\n'

print(results)
