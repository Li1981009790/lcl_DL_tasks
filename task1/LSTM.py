# 1. 导入需要的库
import numpy as np
import random
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

random.seed(10)

# 2. 加载数据
train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=transforms.ToTensor())

# 3. 定义参数
batch_size = 128
num_epoch = 30
# lr = 0.01
lr = 1e-3
# 4. 数据预处理
dataloader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# 5. 定义模型

class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(28, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        out, (h_n, c_n) = self.lstm(x)

        x = h_n[-1, :, :]
        x = self.fc(x)

        return x

# 6. 设置损失函数和优化器
model = Lstm()
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
# optimzier = optim.SGD(model.parameters(), lr=lr)
optimzier = optim.Adam(model.parameters(), lr=lr)

# 7. 开始训练
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for epoch in range(num_epoch):
    # 训练
    model.train(True)
    num_acc = 0
    all_loss = 0
    for idx, data in enumerate(dataloader_train, 1):
        img, label = data
        # print(img.shape)
        # label = F.one_hot(label)
        # label = label.to(torch.float)
        optimzier.zero_grad()
        pred = model(img)
        # print(type(pred[0][0].item()))
        loss = criterion(pred, label)
        loss.backward()
        optimzier.step()
        all_loss += loss.item()
        num_acc += torch.argmax(pred, dim=1).eq(label).sum()
    print('Train Epoch: {}  acc: {:.2f}%  loss:{:.5f}'.format(epoch, 100.0 * num_acc / len(train_dataset), all_loss / (len(train_dataset)/batch_size)))
    train_loss.append(all_loss / (len(train_dataset)/batch_size))
    train_acc.append(num_acc / len(train_dataset))

    # 测试
    model.train(False)
    num_acc = 0
    all_loss = 0
    for idx, data in enumerate(dataloader_test, 1):
        img, label = data
        label = F.one_hot(label)
        label = label.to(torch.float)
        pred = model(img)
        loss = criterion(pred, label)
        all_loss += loss.item()
        num_acc += torch.argmax(pred, dim=1).eq((torch.argmax(label, dim=1))).sum()
    print('Test Epoch: {}  acc: {:.2f}%  loss:{:.5f}'.format(epoch, 100.0 * num_acc / len(test_dataset), all_loss / (len(test_dataset)/batch_size)))
    test_loss.append(all_loss / (len(test_dataset)/batch_size))
    test_acc.append(num_acc / len(test_dataset))

plt.figure()
plt.plot(train_loss, 'b', label='train_loss')
plt.plot(test_loss, 'r', label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('lstm_loss.png')

plt.figure()
plt.plot(train_acc, 'b', label='train_acc')
plt.plot(test_acc, 'r', label='test_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('lstm_acc.png')
