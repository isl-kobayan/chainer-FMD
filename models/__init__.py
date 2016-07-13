import nin
import alex
import alexbn
import googlenet
import googlenet2
import googlenetbn
import vgg16
import caffealex
import caffegooglenet
import numpy as np

NIN = nin.NIN
Alex = alex.Alex
AlexBN = alexbn.AlexBN
GoogLeNet = googlenet.GoogLeNet
GoogLeNet2 = googlenet2.GoogLeNet2
GoogLeNetBN = googlenetbn.GoogLeNetBN
VGG16 = vgg16.VGG16
CaffeAlex = caffealex.CaffeAlex
CaffeGoogLeNet = caffegooglenet.CaffeGoogLeNet

def getModel(arch):
    if arch == 'nin':
        return NIN()
    elif arch == 'alex':
        return Alex()
    elif arch == 'alexbn':
        return AlexBN()
    elif arch == 'vgg16':
            return VGG16()
    elif arch == 'googlenet':
        return GoogLeNet()
    elif arch == 'googlenet2':
        return GoogLeNet2()
    elif arch == 'googlenetbn':
        return GoogLeNetBN()
    elif arch == 'caffealex':
        return CaffeAlex()
    elif arch == 'caffegooglenet':
        return CaffeGoogLeNet()
    else:
        raise ValueError('Invalid architecture name')
        return None

def getGoogLeNetMean():
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123
    return mean_image

def getVGGMean():
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 103.939
    mean_image[1] = 116.779
    mean_image[2] = 123.68
    return mean_image

def getSqueezeNetMean():
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123
    return mean_image
