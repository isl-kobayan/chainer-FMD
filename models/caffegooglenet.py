import chainer
import chainer.functions as F
from chainer.functions import caffe
import sys

class CaffeGoogLeNet(caffe.CaffeFunction):

    insize = 224

    def __init__(self):
        self.labelsize = 10
        print ('loading caffemodel ...')
        #self = caffe.CaffeFunction('bvlc_googlenet.caffemodel')
        super(CaffeGoogLeNet, self).__init__('./models/bvlc_googlenet.caffemodel')
        print ('Done.')
        super(CaffeGoogLeNet, self).add_link('loss3_fc', F.Linear(1024, self.labelsize))
        super(CaffeGoogLeNet, self).add_link('loss1_fc2', F.Linear(1024, self.labelsize))
        super(CaffeGoogLeNet, self).add_link('loss2_fc2', F.Linear(1024, self.labelsize))
        self.train = True
        self.forward = self.__call__

    def caffe_inception(self, x, name):
        out1 = self[name + '/1x1'](x)
        out3 = self[name + '/3x3'](F.relu(self[name + '/3x3_reduce'](x)))
        out5 = self[name + '/5x5'](F.relu(self[name + '/5x5_reduce'](x)))
        pool = self[name + '/pool_proj'](F.max_pooling_2d(x, 3, stride=1, pad=1))
        y = F.relu(F.concat((out1, out3, out5, pool), axis=1))
        return y

    def __call__(self, x, t):
        '''
        ret = self.forward(inputs={'data': x},
            outputs=['loss1/fc', 'loss2/fc', 'pool5/7x7_s1'],
            disable=['loss1/classifier', 'loss1/loss', 'loss2/classifier', 'loss2/loss', 'loss3/classifier', 'loss3/loss3'],
            train=self.train)

        loss1, loss2, h = ret
        loss1 = self.loss1_fc2(loss1)
        self.loss1 = F.softmax_cross_entropy(loss1, t)
        loss2 = self.loss2_fc2(loss2)
        self.loss2 = F.softmax_cross_entropy(loss2, t)

        h = self.loss3_fc(h)
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = 0.3 * (loss1 + loss2) + loss3
        self.accuracy = F.accuracy(h, t)
        return self.loss
        '''
        h = F.relu(self['conv1/7x7_s2'](x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self['conv2/3x3_reduce'](h))
        h = F.relu(self['conv2/3x3'](h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.caffe_inception(h, 'inception_3a')
        h = self.caffe_inception(h, 'inception_3b')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.caffe_inception(h, 'inception_4a')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss1/conv'](l))
        l = F.dropout(F.relu(self['loss1/fc'](l)), 0.7, train=self.train)
        l = self.loss1_fc2(l)
        self.loss1 = F.softmax_cross_entropy(l, t)

        h = self.caffe_inception(h, 'inception_4b')
        h = self.caffe_inception(h, 'inception_4c')
        h = self.caffe_inception(h, 'inception_4d')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss2/conv'](l))
        l = F.dropout(F.relu(self['loss2/fc'](l)), 0.7, train=self.train)
        l = self.loss2_fc2(l)
        self.loss2 = F.softmax_cross_entropy(l, t)

        h = self.caffe_inception(h, 'inception_4e')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.caffe_inception(h, 'inception_5a')
        h = self.caffe_inception(h, 'inception_5b')

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4, train=self.train))
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = 0.3 * (self.loss1 + self.loss2) + self.loss3
        self.accuracy = F.accuracy(h, t)
        return self.loss
        #'''

    def test(self, x, t):
        '''
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)
        ret = self.forward(inputs={'data': x},
            outputs=['loss1/fc', 'loss2/fc', 'pool5/7x7_s1'],
            disable=['loss1/classifier', 'loss1/loss', 'loss2/classifier', 'loss2/loss', 'loss3/classifier', 'loss3/loss3'],
            train=self.train)

        loss1, loss2, h = ret
        loss1 = self.loss1_fc2(loss1)
        self.loss1 = F.softmax_cross_entropy(loss1, t)
        loss2 = self.loss2_fc2(loss2)
        self.loss2 = F.softmax_cross_entropy(loss2, t)

        h = self.loss3_fc(h)
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = 0.3 * (loss1 + loss2) + loss3
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss
        '''
        h = F.relu(self['conv1/7x7_s2'](x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self['conv2/3x3_reduce'](h))
        h = F.relu(self['conv2/3x3'](h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.caffe_inception(h, 'inception_3a')
        h = self.caffe_inception(h, 'inception_3b')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.caffe_inception(h, 'inception_4a')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss1/conv'](l))
        l = F.dropout(F.relu(self['loss1/fc'](l)), 0.7, train=self.train)
        l = self.loss1_fc2(l)
        self.loss1 = F.softmax_cross_entropy(l, t)

        h = self.caffe_inception(h, 'inception_4b')
        h = self.caffe_inception(h, 'inception_4c')
        h = self.caffe_inception(h, 'inception_4d')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss2/conv'](l))
        l = F.dropout(F.relu(self['loss2/fc'](l)), 0.7, train=self.train)
        l = self.loss2_fc2(l)
        self.loss2 = F.softmax_cross_entropy(l, t)

        h = self.caffe_inception(h, 'inception_4e')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.caffe_inception(h, 'inception_5a')
        h = self.caffe_inception(h, 'inception_5b')

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4, train=self.train))
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = 0.3 * (self.loss1 + self.loss2) + self.loss3
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss
        #'''

    def getacts(self, x, t):
        h = F.relu(self['conv1/7x7_s2'](x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self['conv2/3x3_reduce'](h))
        h = F.relu(self['conv2/3x3'](h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.caffe_inception(h, 'inception_3a')
        h = self.caffe_inception(h, 'inception_3b')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.caffe_inception(h, 'inception_4a')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss1/conv'](l))
        l = F.dropout(F.relu(self['loss1/fc'](l)), 0.7, train=self.train)
        l = self.loss1_fc2(l)
        self.loss1 = F.softmax_cross_entropy(l, t)

        h = self.caffe_inception(h, 'inception_4b')
        h = self.caffe_inception(h, 'inception_4c')
        h = self.caffe_inception(h, 'inception_4d')

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self['loss2/conv'](l))
        l = F.dropout(F.relu(self['loss2/fc'](l)), 0.7, train=self.train)
        l = self.loss2_fc2(l)
        self.loss2 = F.softmax_cross_entropy(l, t)

        h = self.caffe_inception(h, 'inception_4e')
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.caffe_inception(h, 'inception_5a')
        h = self.caffe_inception(h, 'inception_5b')

        h = F.average_pooling_2d(h, 7, stride=1)
        self.acts = h
        h = self.loss3_fc(F.dropout(h, 0.4, train=self.train))
        self.loss3 = F.softmax_cross_entropy(h, t)

        self.loss = 0.3 * (self.loss1 + self.loss2) + self.loss3
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss

