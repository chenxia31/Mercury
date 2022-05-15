import random
import torch


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


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = gen_data(true_w, true_b, 1000)


# 可有绘图来看一下情况

# 读取数据集，我们在梯度下降的时候需要mini batch来更新我们的模型，由于这个过程是训练机器学习
# 算法的基础，所以有必要定义一个函数，能够打乱样本的数据集中的样本并按照mini batch的方式
# 获取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# yield 函数将一个函数改写称为一个generator函数
# 每次运行一次，然后下一次会下一次循环

# 上面实现的迭代对于教学来说很好，但是执行效率很低，可能会在实际问题上陷入麻烦，例如它需要我们
# 将所有的数据都家在在内存中，并执行大量的随机内存访问，在深度学习的框架中实现的内置迭代器的效率
# 会高很多，它提供来处理存储在文件中数据和数据流中提供的数据

# 初始化参数
def init_par():
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


# w shape:2,1
# b shape:1,1
w, b = init_par()


# 定义模型
def linreg(X, w, b):
	# X:shape:n,len(w)
	# w:len(w),1
	# b:1+1
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 随机批量下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 整个过程
lr = 0.03
num_epoches = 100
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epoches):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1},loss {float(train_l.mean()):f}')
