import chainer
import chainer.functions as F
import chainer.links as L
import finetune
import numpy as np

class SqueezeNet11(chainer.Chain):

    """SqueezeNet v1.0"""

    insize = 227

    def my_fire(self, x, name):
        s1 = F.relu(self[name + '/squeeze1x1'](x))
        e1 = self[name + '/expand1x1'](s1)
        e3 = self[name + '/expand3x3'](s1)
        y = F.relu(F.concat((e1, e3), axis=1))
        return y

    def add_fire(self, name, in_channels, s1, e1, e3):
        super(SqueezeNet11, self).add_link(name + '/squeeze1x1', F.Convolution2D(in_channels, s1, 1))
        super(SqueezeNet11, self).add_link(name + '/expand1x1', F.Convolution2D(s1, e1, 1))
        super(SqueezeNet11, self).add_link(name + '/expand3x3', F.Convolution2D(s1, e3, 3, pad=1))

    def getMean(self):
        mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
        mean_image[0] = 104
        mean_image[1] = 117
        mean_image[2] = 123
        return mean_image

    def set_finetune(self):
        finetune.load_param('./models/squeezenet_v1.1.pkl', self)

    def __init__(self):
        self.labelsize = 10
        super(SqueezeNet11, self).__init__()
        super(SqueezeNet11, self).add_link('conv1', F.Convolution2D(3,  64, 3, stride=2))
        self.add_fire('fire2', 64, 16, 64, 64)
        self.add_fire('fire3', 128, 16, 64, 64)
        self.add_fire('fire4', 128, 32, 128, 128)
        self.add_fire('fire5', 256, 32, 128, 128)
        self.add_fire('fire6', 256, 48, 192, 192)
        self.add_fire('fire7', 384, 48, 192, 192)
        self.add_fire('fire8', 384, 64, 256, 256)
        self.add_fire('fire9', 512, 64, 256, 256)
        super(SqueezeNet11, self).add_link('conv10_fmd', F.Convolution2D(
            512, self.labelsize, 1, pad=1,
            initialW=np.random.normal(0, 0.01, (self.labelsize, 512, 1, 1))))

        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def clear_test(self):
        self.loss = None
        self.accuracy = None
        self.acts = None
        self.lastacts = None

    def __call__(self, x, t):
        self.clear()
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.my_fire(h, 'fire2')
        h = self.my_fire(h, 'fire3')

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.my_fire(h, 'fire4')
        h = self.my_fire(h, 'fire5')

        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.my_fire(h, 'fire6')
        h = self.my_fire(h, 'fire7')
        h = self.my_fire(h, 'fire8')
        h = self.my_fire(h, 'fire9')
        h = F.dropout(h, ratio=0.5, train=self.train)
        h = F.relu(self.conv10_fmd(h))
        h = F.reshape(F.average_pooling_2d(h, h.data.shape[2]), (x.data.shape[0], self.labelsize))

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

    def test(self, x, t):
        self.clear_test()

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.my_fire(h, 'fire2')
        h = self.my_fire(h, 'fire3')
        h = self.my_fire(h, 'fire4')
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.my_fire(h, 'fire5')
        h = self.my_fire(h, 'fire6')
        h = self.my_fire(h, 'fire7')
        h = self.my_fire(h, 'fire8')
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.my_fire(h, 'fire9')
        h = F.dropout(h, ratio=0.5, train=self.train)

        h = F.relu(self.conv10_fmd(h))
        h = F.reshape(F.average_pooling_2d(h, 13), (x.data.shape[0], self.labelsize))

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss
