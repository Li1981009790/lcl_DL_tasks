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
import math

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

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # i_t

        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # f_t
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # c_t
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_params()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        batch_size, num_seq, _ = x.shape
        hidden_seq = []

        h_t = torch.zeros(batch_size, self.hidden_size)
        c_t = torch.zeros(batch_size, self.hidden_size)

        for t in range(num_seq):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.sigmoid(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = MyLSTM(28, 64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        out, (h_n, c_n) = self.lstm(x)

        # x = h_n[-1, :, :]
        x = h_n
        # print('out:', out.shape)
        # print('h_n:', h_n.shape)
        # print('c_n:', c_n.shape)
        x = self.fc(x)

        return x


# 6. 设置损失函数和优化器
model = Lstm()
# model
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




# class MyLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):


