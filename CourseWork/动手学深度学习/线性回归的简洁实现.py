import torch
import numpy as np
from torch.utils import data

true_w = torch.tensor([2, -3.4])
true_b = 4.2


# 生成数据集
def gen_data(w, b, num_examples):
    '''

	:param w: shape:num_examples,len(w)
	:param b: shape:num_examples,1
	:param num_examples:
	:return:
	'''
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


features, labels = gen_data(true_w, true_b, 1000)

# 调用现有框架的API来读取数据，将features和labels作为参数传递，通过数据迭代器来指定batch-size
def load_array(data_arrays,batch_size,is_Train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_Train)

batch_size=10
data_iter=load_array((features,labels),batch_size)

from torch import nn
net=nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epoches=3

for epoch in range(num_epoches):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
