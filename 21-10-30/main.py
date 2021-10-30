# 'nn'是神经网络的缩写
import random
from torch import nn
import torch


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


batch_size = 10
true_w = torch.tensor([2.0, -3.0, 4.0])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# 使用框架的预定义好的层 sequential 容器
net = nn.Sequential(nn.Linear(2, 1))
# 使用正太分布替换掉data的值
# weight是w权重，bias是偏差b
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 计算均方误差使用的是MSELoss类，也称为平法L范数
loss = nn.MSELoss()
# 实例化SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练过程代码
num_epochs = 3
for epoch in range(num_epochs):
    for x, y in data_iter:
        l = loss(net(x), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch{epoch + 1}, loss {l : f}')
