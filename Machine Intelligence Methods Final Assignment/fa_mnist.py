import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis as FA
import torch
from torch import nn
from torch import optim
import random
import matplotlib.pyplot as plt
from torchvision import datasets

train_raw = datasets.MNIST(root='./MNIST', train=True, transform=None)
test_raw = datasets.MNIST(root='./MNIST', train=False, transform=None)
train_imgs = np.asarray(train_raw.data)
train_imgs = train_imgs.reshape(train_imgs.shape[0], -1)
train_labels = np.asarray(train_raw.targets)
test_imgs = np.asarray(test_raw.data)
test_imgs = test_imgs.reshape(test_imgs.shape[0], -1)
test_labels = np.asarray(test_raw.targets)

# train_raw = pd.read_csv('fashion-mnist_train.csv')
# test_raw = pd.read_csv('fashion-mnist_test.csv')
# train_raw = pd.read_csv('train.csv')
# test_raw = pd.read_csv('test.csv')

# train_labels = train_raw['label']
# # train_labels = np.asarray(train_labels)
# train_imgs = train_raw.drop("label", axis=1)
#
# test_labels = test_raw['label']
# test_imgs = test_raw.drop("label", axis=1)

n_components = 128
fa = FA(n_components=n_components).fit(train_imgs)
# Invoking the t-SNE method
# tsne = TSNE(n_components=2)
# tsne_results = tsne.fit_transform(X_std)

fa_train_imgs = fa.transform(train_imgs)
# print(pca_train_imgs)

# pca_train_imgs = torch.as_tensor(pca_train_imgs)
# print(pca_train_imgs.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


batch_size = 128
lr = 1e-3
num_epoch = 100

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=lr)

# train_data = []
# for i in range(len(pca_train_imgs)):
#     train_data.append((pca_train_imgs[i], train_labels[i]))

# random.shuffle(train_data)
# train_data = torch.as_tensor(train_data)
idxs = []
for i in range(len(train_labels)):
    # print(i)
    idxs.append(i)

random.shuffle(idxs)
# print(idxs[1])
# idxs = np.asarray(idxs)

pca_train = []
train_tar = []
# pca_train_imgs = np.asarray(pca_train_imgs)
# train_labels = np.asarray(train_labels)
# for i in range(len(idxs)):
#     print(i)
#     print(idxs[i].shape)
#     print(pca_train_imgs[idxs[i]].shape)
#     pca_train.append(pca_train_imgs[idxs[i]])
#     train_tar.append(train_labels[idxs[i]])
#
# pca_train_imgs = pca_train
# train_labels = train_tar

fa_train_imgs = torch.as_tensor(fa_train_imgs)
train_labels = torch.as_tensor(train_labels)
test_imgs = torch.as_tensor(np.asarray(test_imgs))
test_labels = torch.as_tensor(test_labels)
fa_train_imgs = fa_train_imgs.float()
train_labels = train_labels.long()
test_imgs = test_imgs.float()
test_labels = test_labels.long()
test_labels = test_labels.reshape(-1, 1)

# 训练
train_loss = []
train_acc = []
test_loss = []
test_acc = []
print("训练")
for epoch in range(num_epoch):
    net.train(True)
    num_acc = 0
    all_loss = 0
    num_batch = len(idxs) // batch_size

    for i in range(num_batch):
        idx = idxs[i*batch_size: min((i+1)*batch_size, len(idxs))]
        # print(type(idx))
        # print(idx.shape)
        imgs = fa_train_imgs[idx]
        labels = train_labels[idx]

        optimizer.zero_grad()
        pred = net(imgs)
        # print(pred.shape)
        # print(labels.shape)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()
        num_acc += torch.argmax(pred, dim=1).eq(labels).sum()
    print('Train Epoch: {}  acc: {:.2f}%  loss:{:.5f}'.format(epoch, 100.0 * num_acc / len(train_labels),
                                                              all_loss / (len(train_labels) / batch_size)))
    train_loss.append(all_loss / (len(train_labels) / batch_size))
    train_acc.append(num_acc / len(train_labels))

    # 测试
    net.train(False)
    num_acc = 0
    all_loss = 0
    for i in range(len(test_labels)):
        img = test_imgs[i]
        label = test_labels[i]
        # label = label.reshape(1, -1)
        # img = np.asarray(img)
        img = img.reshape(1, -1)
        # print(img.shape)
        img = fa.transform(img)
        img = torch.as_tensor(img)
        img = img.float()

        pred = net(img)
        # print(pred.shape)
        # print(label.shape)
        loss = criterion(pred, label)
        all_loss += loss.item()
        num_acc += torch.argmax(pred, dim=1).eq(label).sum()

    print('Test Epoch: {}  acc: {:.2f}%  loss:{:.5f}'.format(epoch, 100.0 * num_acc / len(test_labels),
                                                             all_loss / (len(test_labels) / batch_size)))
    test_loss.append(all_loss / (len(test_labels) / batch_size))
    test_acc.append(num_acc / len(test_labels))

plt.figure()
plt.plot(train_loss, 'b', label='train_loss')
plt.plot(test_loss, 'r', label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('fa_loss.png')

plt.figure()
plt.plot(train_acc, 'b', label='train_acc')
plt.plot(test_acc, 'r', label='test_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('fa_acc.png')
