# 导入需要的库
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
import torchvision

# 读取数据 Fashion MNIST数据集
batch_size = 256
num_inputs = 784
num_outputs = 10
num_hiddens = 256
num_epochs = 10
lr = 0.1


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 设置超参数和初始化
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


# 定义ReLU激活函数
def relu(X):
    a = torch.zeros_like((X))
    return torch.max(X, a)


# 记住我们平输入变量，因此不需要太复杂的模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')


# 训练的过程

# 作为训练过程中记录数字的操作
def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_loss = []
test_loss = []
updater = torch.optim.SGD(params, lr=lr)
for epoch in range(num_epochs):
    metric = Accumulator(3)
    for X, y in train_iter:
        l = loss(net(X), y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(net(X), y), y.numel())
    print(metric[0] / metric[2], metric[1] / metric[2])

# 后续可以做的训练
# 1 在所有其他的参数保持不变的，更改超参数num_hiddens的值，查看超参数的变化度结果有何影响
# 2 改变学习率的变化
# 3 对learning rate、epoches、hidden layers进行联合优化

#%%
# 简洁实现
net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)

train_loss = []
test_loss = []
updater = torch.optim.SGD(params, lr=lr)
for epoch in range(num_epochs):
    metric = Accumulator(3)
    for X, y in train_iter:
        l = loss(net(X), y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(net(X), y), y.numel())
    print(metric[0] / metric[2], metric[1] / metric[2])