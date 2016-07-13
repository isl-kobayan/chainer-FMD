import chainer
import chainer.functions as F
from chainer.functions import caffe
import sys

class CaffeAlex(caffe.CaffeFunction):

    insize = 224

    def __init__(self):
        self.labelsize = 10
        sys.stderr.write ('loading caffemodel ...\n')
        super(CaffeAlex, self).__init__('./models/bvlc_alexnet.caffemodel')
        sys.stderr.write ('Done.\n')
        super(CaffeAlex, self).add_link('fc8fmd', F.Linear(4096, self.labelsize))
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x)), n=5), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h)), n=5), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8fmd(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

    def test(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x)), n=5), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h)), n=5), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8fmd(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss


