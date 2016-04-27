import chainer
import chainer.functions as F
from chainer.functions import caffe
import sys


class CaffeVGG(caffe.CaffeFunction):

    insize = 224

    def __init__(self):
        self.labelsize = 10
        sys.stderr.write ('loading caffemodel ...\n')
        super(CaffeVGG, self).__init__('./models/VGG_ILSVRC_16_layers.caffemodel')
        sys.stderr.write ('Done.\n')
        super(CaffeVGG, self).add_link('fc8fmd', F.Linear(4096, self.labelsize))
        self.train = True

    def __call__(self, x, t):
        h = self.conv1_2(F.relu(self.conv1_1(x)))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv2_2(F.relu(self.conv2_1(h)))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8fmd(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

    def test(self, x, t):
        h = self.conv1_2(F.relu(self.conv1_1(x)))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv2_2(F.relu(self.conv2_1(h)))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(h)))))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train, ratio=0.5)
        h = self.fc8fmd(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss
