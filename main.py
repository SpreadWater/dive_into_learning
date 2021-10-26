import torch

# 创建一个张量
# x = torch.arange(12)
# print(x)
# # 改变一个张量的形状
# x = x.reshape(3, 4)
# print(x)
#
# # 使用全0 和 全1
# y = torch.zeros((2, 3, 4))  # 2个三行四列的二维数组
# print(y)
#
# y = torch.ones((2, 3, 4))
# print(y)
#
# z = torch.tensor([[2, 1, 4, 3], [2, 3, 4, 1], [2, 1, 3, 4]]).shape
# print(z)

# 常见的标准算术运算符
# x = torch.tensor([1.0, 2, 4, 8])
# y = torch.tensor([2, 2, 2, 2])
# print('x+y=', x + y)
# print('x-y=', x - y)
# print('x*y=', x * y)
# print('x/y=', x / y)
# print('x**y=', x ** y)  # **运算符是求幂运算

# print(torch.exp(x))

#
# x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
# y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# # 连接多个张量
# # 合并两个元素，在第0维合并(行),在第1维合并(列)
# print(torch.cat((x, y), dim=0))
# print(torch.cat((x, y), dim=1))
#
# print(x == y)
# print(torch.sum(x))

# 即使形状不同，我们可以通过广播机制，来执行元素操作
# a = torch.arange(12).reshape((3, 4))
# b = torch.arange(2).reshape((1, 2))
# print(a)
# print(b)
# print(a + b)

#  a[-1]可以选择最后一个元素,最后一行
# print(a[-1, :])
# print(a[1:3])
# 按照索引修改值
# a[1, 2] = 9
# 批量修改二维张量的值
# a[0:2, :] = 12
# print(a)

# 内存的新分配
x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# before = id(y)
# y = y+x
# print(before)
# print(id(y))
# print(id(y) == before)

# 执行原地操作
# z = torch.zeros_like(y)
# print('id(z):', id(z))
# z[:] = x+y
# print('id(z):', id(z))

# 如果在计算中没有重复使用X，我们可以使用X[:] = x+y 或 x +=y 来减少操作的内存开销
# before = id(x)
# x +=y
# print(id(x) == before)

# 转化为NumPy张量
# A = x.numpy()
# B = torch.tensor(A)
# print(type(A))
# print(type(B))
