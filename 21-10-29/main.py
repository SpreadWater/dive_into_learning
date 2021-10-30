# 线性回归的从零开始实现
import random
import torch
from d2l import torch as d2l


# 根据带有噪声的线性模型，构造一个人造数据集。
# 我们使用线性模型参数w=[2,-3,4]T、b=4.2和噪声项e生成数据集及其标签

# 生成y=wx+b+噪声",生成训练样本
def synthetic_data(w, b, num_example):
    # x是均值为0，方差为1的随机数，n个样本，列数是w的长度
    x = torch.normal(0.0, 1.0, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    # 添加随机噪音
    y += torch.normal(0, 0.01, y.shape)
    # 把x，y当做列返回
    return x, y.reshape((-1, 1))


# 实现一个函数来读取小批量
# batch_size:批量大小
# features: 所有的特征
# labels： 标号
# 该函数接受批量大小，特征矩阵和标签向量，生成大小为batch_size的小批量
def data_iter(batch_size, features, labels):
    num_example = len(features)
    # 生成每个样本的标号
    indices = list(range(num_example))
    # 这样的样本是随机读取的，没有特定的顺序,打乱样本标号
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_example)])
        # yield 是return 返回一个值，并且记住这个返回的位置,下次迭代就从这个位置后开始
        yield features[batch_indices], labels[batch_indices]


# 定义模型(线性回归模型)
def linreg(x, w, b):
    return torch.matmul((x, w)) + b


# 定义损失函数
# y_hat:预测值，y:实际值
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
# params：参数(w,b),lr: 学习度。batch_size
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 求均值
            param.grad.zero()  # 将梯度设置成0，防止下一次梯度与上次累加


# 定义初始化参数
# lr:学习度
lr = 0.03
num_epochs = 3
batch_size = 10
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
true_w = torch.tensor([2.0, -3.0, 4.0])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        # 预测的Y和真实的y来做损失
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
    with torch.no_grad():
        train_1 = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_1.mean()):f}')
for x, y in data_iter(batch_size, features, labels):
    print(x, '\n', y)
    break
# features 中的每一行都包含一个二维数据样本
# labels中的每一行都包含一维标签值
# print('features:', features[0], '\label:', labels[0])

# d2l.set_figsize()
# 第一列和label画一个图像
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()
