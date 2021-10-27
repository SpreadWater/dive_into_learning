# 数据预处理
# 创建一个人工数据集，并存储在csv（逗号分隔值）分件
import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2.0,NA,106000\n')  # 每行表示一个数据样本
    f.write('4.0,NA,178100\n')  # 每行表示一个数据样本
    f.write('NA,NA,140000\n')  # 每行表示一个数据样本

# 用pandas读取csv文件。
data = pd.read_csv(data_file)
# print(data)

# 为了处理缺失的数据，典型的方法包括 插值和删除，这里我们考虑插值
# 将第一列和第二列的所有行拿如input，最后一列的所有行放入outputs
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# print(inputs)

# # 将是NA的数据，用剩下不是NA的均值填充
inputs = inputs.fillna(inputs.mean())
# print(inputs)

# 对于inputs中的类别值和离散值，我们将‘NAN’视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 将inputs和outputs转换为张量格式
x, y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x)
print(y)