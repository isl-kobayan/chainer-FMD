import subprocess
import json
import argparse

parser = argparse.ArgumentParser(
    description='Learning convnet from Flickr Material Database')
parser.add_argument('--arch', '-a', default='nin',
                    help='Convnet architecture \
                    (nin, alex, alexbn, googlenet, googlenet2, googlenetbn, caffealex, caffegooglenet, vggnet, caffevgg, chainervggnet)')
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
parser.add_argument('--epoch', '-E', default=40, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--iteration', '-I', default=0, type=int,
                    help='Number of iterations to learn')
args = parser.parse_args()

#batchsizes = [1,2,4,8,10,16,20,25,40,50,100]
batchsizes = [10,16,20,25,40,50]
initlrs = [0.0005, 0.001, 0.005]
lrsteps = [0.97, 0.94]
#batchsizes = [40]
#archs = ['caffealex', 'caffegooglenet', 'caffevgg']
archs = ['caffegooglenet', 'caffealex']
#archs = ['caffevgg']
#archs = ['caffealex']
results="arch\tbatchsize\tinitlr\tlrstep\terror(train)\terror(val)\tloss(train)\tloss(val)\n"
with open('results/GAResults.txt', 'a') as f:
    f.write(results)

for initlr in initlrs:
    for lrstep in lrsteps:
        for arch in archs:
            for bs in batchsizes:
                output = subprocess.check_output("python train_fmd.py -p momentumsgd -a " + arch + " -B " + str(bs)
	        + " -b 20 -E " + str(args.epoch) + " --initlr " + str(initlr) + " --lrstep " + str(lrstep), shell=True)
                #output = subprocess.check_output("python train_fmd.py -p adam -a " + arch + " -B " + str(bs)
	        #+ " -b 20 -E " + str(args.epoch) + " --initlr " + str(initlr) + " --lrstep " + str(lrstep), shell=True)
                #print(output)
                outputs = output.split('\n')
                final_train = None
                final_val = None
                for line in outputs:
                    if line.find('train') >= 0:
                        final_train = line
                    elif line.find('val') >= 0:
                        final_val = line
                train_info = json.loads(final_train)
                val_info = json.loads(final_val)
                result = arch + '\t' + str(bs) + '\t' + str(initlr) + '\t' + str(lrstep) + '\t' \
                    + str(train_info['error']) + '\t' + str(val_info['error']) + '\t' \
                    + str(train_info['loss']) + '\t' + str(val_info['loss'])
                with open('results/GAResults.txt', 'a') as f:
                            f.write(result + '\n')
                results += result + '\n'

print(results)
