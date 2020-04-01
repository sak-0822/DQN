import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.link1 = L.Linear(2,6)
            self.link2 = L.Linear(6,3)
            self.link3 = L.Linear(3,5)
            self.link4 = L.Linear(5,2)
    def __call__(self, x):
            h1 = F.relu(self.link1(x))
            h2 = F.relu(self.link2(h1))
            h3 = F.relu(self.link3(h2))
            y = self.link4(h3)
            return y

epoch = 100
batchsize = 4

trainx = np.array(([0,0], [0,1], [1,0], [1,1]),dtype=np.float32)
trainy = np.array([0, 1, 1, 1], dtype=np.int32)
train = chainer.datasets.TupleDataset(trainx, trainy)
test = chainer.datasets.TupleDataset(trainx, trainy)

model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
#chainer.serializers.load_npz('result/out.model', model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer)

trainer = training.Trainer(updater, (epoch, 'epoch'))

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.run()
