import  torch
import torchvision
from torch.utils import data
from torchvision import transforms

# 更改数据的类型
# 同时归一化数据
trans=transforms.ToTensor()
mnist_train=torchvision.datasets.FashionMNIST(root='../data',train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root='../data',train=False,transform=trans,download=True)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

batch_size = 128

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 0

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

import time

t1=time.time()
for X, y in train_iter:
    continue
t2=time.time()
print(t2-t1)

def load_data_fashion_mnist(batch_size, resize=None):  #@save
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
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))