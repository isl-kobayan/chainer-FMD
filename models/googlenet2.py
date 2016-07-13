import chainer
import chainer.functions as F
import chainer.links as L
import finetune
import numpy as np

class GoogLeNet2(chainer.Chain):

    insize = 224

    def my_inception(self, x, name):
        out1 = self[name + '/1x1'](x)
        out3 = self[name + '/3x3'](F.relu(self[name + '/3x3_reduce'](x)))
        out5 = self[name + '/5x5'](F.relu(self[name + '/5x5_reduce'](x)))
        pool = self[name + '/pool_proj'](F.max_pooling_2d(x, 3, stride=1, pad=1))
        y = F.relu(F.concat((out1, out3, out5, pool), axis=1))
        return y

    def add_inception(self, name, in_channels, out1, proj3, out3, proj5, out5, proj_pool):
        super(GoogLeNet2, self).add_link(name + '/1x1', F.Convolution2D(in_channels, out1, 1))
        super(GoogLeNet2, self).add_link(name + '/3x3_reduce', F.Convolution2D(in_channels, proj3, 1))
        super(GoogLeNet2, self).add_link(name + '/3x3', F.Convolution2D(proj3, out3, 3, pad=1))
        super(GoogLeNet2, self).add_link(name + '/5x5_reduce', F.Convolution2D(in_channels, proj5, 1))
        super(GoogLeNet2, self).add_link(name + '/5x5', F.Convolution2D(proj5, out5, 5, pad=2))
        super(GoogLeNet2, self).add_link(name + '/pool_proj', F.Convolution2D(in_channels, proj_pool, 1))

    def getMean(self):
        mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
        mean_image[0] = 104
        mean_image[1] = 117
        mean_image[2] = 123
        return mean_image

    def set_finetune(self):
        finetune.load_param('./models/bvlc_googlenet.pkl', self)

    def __init__(self):
        self.labelsize = 10
        super(GoogLeNet2, self).__init__()
        super(GoogLeNet2, self).add_link('conv1/7x7_s2', F.Convolution2D(3,  64, 7, stride=2, pad=3))
        super(GoogLeNet2, self).add_link('conv2/3x3_reduce', F.Convolution2D(64,  64, 1))
        super(GoogLeNet2, self).add_link('conv2/3x3', F.Convolution2D(64, 192, 3, stride=1, pad=1))
        self.add_inception('inception_3a', 192,  64,  96, 128, 16,  32,  32)
        self.add_inception('inception_3b', 256, 128, 128, 192, 32,  96,  64)
        self.add_inception('inception_4a', 480, 192,  96, 208, 16,  48,  64)
        self.add_inception('inception_4b', 512, 160, 112, 224, 24,  64,  64)
        self.add_inception('inception_4c', 512, 128, 128, 256, 24,  64,  64)
        self.add_inception('inception_4d', 512, 112, 144, 288, 32,  64,  64)
        self.add_inception('inception_4e', 528, 256, 160, 320, 32, 128, 128)
        self.add_inception('inception_5a', 832, 256, 160, 320, 32, 128, 128)
        self.add_inception('inception_5b', 832, 384, 192, 384, 48, 128, 128)

        super(GoogLeNet2, self).add_link('loss3/classifier_fmd', F.Linear(1024, self.labelsize))

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
        h = F.relu(self['conv1/7x7_s2'](x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self['conv2/3x3_reduce'](h))
        h = F.relu(self['conv2/3x3'](h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.my_inception(h, 'inception_3a')
        h = self.my_inception(h, 'inception_3b')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.my_inception(h, 'inception_4a')

        h = self.my_inception(h, 'inception_4b')
        h = self.my_inception(h, 'inception_4c')
        h = self.my_inception(h, 'inception_4d')

        h = self.my_inception(h, 'inception_4e')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.my_inception(h, 'inception_5a')
        h = self.my_inception(h, 'inception_5b')

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self['loss3/classifier_fmd'](F.dropout(h, 0.4, train=self.train))
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = self.loss3
        self.accuracy = F.accuracy(h, t)
        return self.loss

    def test(self, x, t):
        self.clear_test()
        h = F.relu(self['conv1/7x7_s2'](x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self['conv2/3x3_reduce'](h))
        h = F.relu(self['conv2/3x3'](h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.my_inception(h, 'inception_3a')
        h = self.my_inception(h, 'inception_3b')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.my_inception(h, 'inception_4a')

        h = self.my_inception(h, 'inception_4b')
        h = self.my_inception(h, 'inception_4c')
        h = self.my_inception(h, 'inception_4d')

        h = self.my_inception(h, 'inception_4e')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.my_inception(h, 'inception_5a')
        h = self.my_inception(h, 'inception_5b')

        h = F.average_pooling_2d(h, 7, stride=1)
        self.acts = h
        h = self['loss3/classifier_fmd'](F.dropout(h, 0.4, train=self.train))
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = self.loss3
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss
