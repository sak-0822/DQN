import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.link1 = L.Linear(64,100)
            self.link2 = L.Linear(100,100)
            self.link3 = L.Linear(100, 10)
    def __call__(self, x):
            h1 = F.relu(self.link1(x))
            h2 = F.relu(self.link2(h1))
            y = self.link3(h2)
            return y

epoch = 20
batchsize = 100

digits = load_digits()
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, test_size=0.2)
data_train = (data_train).astype(np.float32)
data_test = (data_test).astype(np.float32)

train = chainer.datasets.TupleDataset(data_train, label_train)
test = chainer.datasets.TupleDataset(data_test, label_test)

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
