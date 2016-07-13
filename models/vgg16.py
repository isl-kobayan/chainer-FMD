import chainer
import chainer.functions as F
import chainer.links as L
import finetune

class VGG16(chainer.Chain):

    insize = 224

    def set_finetune(self):
        finetune.load_param('./models/VGG_ILSVRC_16_layers.pkl', self)

    def __init__(self):
        self.labelsize = 10
        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8_fmd=L.Linear(4096, self.labelsize)
        )

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
        h = self.fc8_fmd(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss

    def test(self, x, t):
        self.clear_test()
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
        self.acts = h
        h = self.fc8_fmd(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        self.lastacts = h
        return self.loss
