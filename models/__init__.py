import nin
import alex
import alexbn
import googlenet
import googlenet2
import googlenetbn
import caffealex
import caffegooglenet
import VGGNet
import caffevgg
import chainerVGGNet
import googlenetfmd
import vgg16

NIN = nin.NIN
Alex = alex.Alex
AlexBN = alexbn.AlexBN
GoogLeNet = googlenet.GoogLeNet
GoogLeNet2 = googlenet2.GoogLeNet2
GoogLeNetBN = googlenetbn.GoogLeNetBN
CaffeAlex = caffealex.CaffeAlex
CaffeGoogLeNet = caffegooglenet.CaffeGoogLeNet
VGGNet = VGGNet.VGGNet
CaffeVGG = caffevgg.CaffeVGG
ChainerVGGNet = chainerVGGNet.ChainerVGGNet
GoogLeNetFMD = googlenetfmd.GoogLeNet
VGG16 = vgg16.VGG16

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
    elif arch == 'googlenetfmd':
        return GoogLeNetFMD()
    elif arch == 'googlenetbn':
        return GoogLeNetBN()
    elif arch == 'caffealex':
        return CaffeAlex()
    elif arch == 'caffegooglenet':
        return CaffeGoogLeNet()
    elif arch == 'vggnet':
        return VGGNet()
    elif arch == 'caffevgg':
        return CaffeVGG()
    elif arch == 'chainervggnet':
        return ChainerVGGNet()
    else:
        raise ValueError('Invalid architecture name')
        return None
