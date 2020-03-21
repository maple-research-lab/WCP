import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import sys
from functools import partial

sys.path.append('../../../')
from source.chainer_functions.misc import call_bn

def add_noise(h, sigma=0.15):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h


class CNN(chainer.Chain):
    def __init__(self, n_outputs=10, dropout_rate=0.5, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        with self.init_scope():
#            self.c1=L.Convolution2D(3, 128, ksize=3, stride=1, pad=1, nobias=True)
#            
#            self.rest = chainer.Sequential(
#                L.BatchNormalization(128),
#                partial(F.leaky_relu, slope=0.1),
#                L.Convolution2D(128, 128, ksize=3, stride=1, pad=1, nobias=True),
#                L.BatchNormalization(128),
#                partial(F.leaky_relu, slope=0.1),
#                L.Convolution2D(128, 128, ksize=3, stride=1, pad=1, nobias=True),
#                L.BatchNormalization(128),
#                partial(F.leaky_relu, slope=0.1),
#                partial(F.max_pooling_2d, ksize=2, stride=2),
#                partial(F.dropout, ratio=self.dropout_rate),
#                
#                L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
#                L.BatchNormalization(256),
#                partial(F.leaky_relu, slope=0.1),
#                L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, nobias=True),
#                L.BatchNormalization(256),
#                partial(F.leaky_relu, slope=0.1),
#                L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, nobias=True),
#                L.BatchNormalization(256),
#                partial(F.leaky_relu, slope=0.1),
#                partial(F.max_pooling_2d, ksize=2, stride=2),
#                partial(F.dropout, ratio=self.dropout_rate),
#                
#                L.Convolution2D(256, 512, ksize=3, stride=1, pad=0, nobias=True),
#                L.BatchNormalization(512),
#                partial(F.leaky_relu, slope=0.1),
#                L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
#                L.BatchNormalization(256),
#                partial(F.leaky_relu, slope=0.1),
#                L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True),
#                L.BatchNormalization(128),
#                partial(F.leaky_relu, slope=0.1),
#                partial(F.average_pooling_2d, ksize=6),
#                
#                L.Linear(128, n_outputs),
#            )
#            if top_bn:
#                self.rest.append(L.BatchNormalization(n_outputs))
                    
            self.c1=L.Convolution2D(3, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.c2=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.c3=L.Convolution2D(128, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.c4=L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.c5=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.c6=L.Convolution2D(256, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.c7=L.Convolution2D(256, 512, ksize=3, stride=1, pad=0, nobias=True)
            self.c8=L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True)
            self.c9=L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True)
            self.l_cl=L.Linear(128, n_outputs)
            self.bn1=L.BatchNormalization(128)
            self.bn2=L.BatchNormalization(128)
            self.bn3=L.BatchNormalization(128)
            self.bn4=L.BatchNormalization(256)
            self.bn5=L.BatchNormalization(256)
            self.bn6=L.BatchNormalization(256)
            self.bn7=L.BatchNormalization(512)
            self.bn8=L.BatchNormalization(256)
            self.bn9=L.BatchNormalization(128)
            if top_bn:
                self.bn_cl = L.BatchNormalization(n_outputs)
            
            self.layer_list = chainer.ChainList(*[self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8, self.c9, self.l_cl])

    def __call__(self, x):

#        x = add_noise(x)
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.bn1(h), slope=0.1)
        h = self.c2(h)
        h = F.leaky_relu(self.bn2(h), slope=0.1)
        h = self.c3(h)
        h = F.leaky_relu(self.bn3(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, ratio=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.bn4(h), slope=0.1)
        h = self.c5(h)
        h = F.leaky_relu(self.bn5(h), slope=0.1)
        h = self.c6(h)
        h = F.leaky_relu(self.bn6(h), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.dropout(h, ratio=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.bn7(h), slope=0.1)
        h = self.c8(h)
        h = F.leaky_relu(self.bn8(h), slope=0.1)
        h = self.c9(h)
        h = F.leaky_relu(self.bn9(h), slope=0.1)
        h = F.average_pooling_2d(h, ksize=h.data.shape[2])
        logit = self.l_cl(h)
        if self.top_bn:
            logit = self.bn_cl(logit)
        return logit
