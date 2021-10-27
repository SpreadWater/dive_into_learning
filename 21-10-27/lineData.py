# 线性代数相关的实现
import torch

# 标量由只有一个元素的张量表示
# x = torch.tensor([3.0])
# y = torch.tensor([2.0])
# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x ** y)

# 通过制定两个分量m和n来创建一个形状为m x n 的矩阵
# A = torch.arange(20).reshape(5, 4)
# B = A.clone()

# print(A)
# 矩阵的转置
# print(A.T)

# 两个矩阵的按元素的乘法称为 哈达玛积
# print(A * B)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A.shape)
# print(A.sum())

# 制定求和汇总张量的轴
# print(A)
# A_sum_axis0 = A.sum(axis=0)
# print(A_sum_axis0)
# print(A_sum_axis0.shape)
# #
# A_sum_axis1 = A.sum(axis=1)
# print(A_sum_axis1)
# print(A_sum_axis1.shape)
#
# A_sum_axis2 = A.sum(axis=2)
# print(A_sum_axis2)
# print(A_sum_axis2.shape)

# print(A.sum(axis=[0, 1]).shape)
# print(A.sum(axis=[0, 1]))

# 计算总和或均值时保持轴数不变
# 这样的好处是可以通过广播机制，让A除以sum_A,广播机制的维度需要相同

# sum_A = A.sum(axis=1, keepdims=True)
# print(sum_A)
# print(sum_A.shape)

# 某个轴计算A元素的累计总和

# 点积是相同位置的按元素乘积的和
# x = torch.tensor([0.0, 1.0, 2.0, 3.0])
# y = torch.ones(4)
# print(x)
# print(y)
# print(torch.dot(x, y))

B = torch.ones(4, 3)
# m次矩阵向量积，形成一个n*m的矩阵
# print(torch.mm(A, B))

# 范数是向量或者矩阵的长度.向量元素的平方和的平方根

u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数，它表示为向量元素的绝对值之和
print(torch.abs(u).sum())

# 矩阵的F范数，是矩阵元素的平方和的平方根
print(torch.norm(torch.ones((4, 9))))
