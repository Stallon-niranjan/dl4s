import numpy as np
from mnist import load_mnist
from optimizer import SGD
from optimizer import AdaGrad
from optimizer import Adam
from conv_net import ConvNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)
train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 10000
batch_size = 100
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

net = ConvNet()
optim = SGD(net.params, lr=0.1, momentum=0.9)
# optim = AdaGrad(net.params)
# optim = Adam(net.params)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.gradient(x_batch, t_batch)
    net.params = optim.update(net.params, grad)

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
