# 矩阵计算
# 自动求导计算一个函数在指定值上的导数
import torch

x = torch.arange(4.0)
# 用于存储x的梯度
x.requires_grad = True


# print(x)
# y = 2 * torch.dot(x, x)
# print(y)

# 通过调用反向传播函数，来自动计算y关于x每个分量的梯度
# y.backward()
# print(x.grad)

# 在默认情况下，PyTorch会累计梯度，我们需要清除之前的值
# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(x.grad)

# 我们的目的是批量中每个样本单独计算的偏导数之和
# y = x * x
# 等价于y.backward(torch.ones(len(X))
# y.sum().backward()
# print(y.sum())
# print(x.grad)
# 清零梯度
# x.grad.zero_()

# 将某些计算移动到记录的计算图之外
# y = x * x
# print(y)

# 将y变成一个常数
# u = y.detach()
# z = u * x
# z.sum().backward()
# print(x.grad == u)
# print(u)

# x.grad.zero_()
# y.sum().backward()
# print(x.grad == 2 * x)

# 当构建函数的计算图需要通过Python的一些控制流的时候，我们仍然可以计算得到的变量的梯度

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.rand(size=(),requires_grad=True)
d = f(a)
d.backward()
# print(a.grad)
# print(a.grad == d / a)