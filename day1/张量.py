# -*- coding: utf-8 -*-
# @Time    : 2022/9/22 9:50
# @Author  : ZAOXG
# @File    : 张量.py
# https://pytorch.apachecn.org/#/docs/1.7/03

import torch
import numpy as np


# 直接生成张量
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# 使用numpy数组生成
np_array = np.array(data)
torch.from_numpy(np_array)

# 通过已有的张量来生成新的张量
x_ones = torch.ones_like(x_data, dtype=torch.float)  # dtype属性指定类型
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float)  # 随机
print(f"Random Tensor: \n {x_rand} \n")

# 通过指定数据维度来生成张量
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)  # (可以通过dtype指定类型)
zeros_tensor = torch.zeros(shape)  # 生成0的

print(f"Random Tensor: \n {rand_tensor} \n")  # torch.Size([3, 4])
print(f"Ones Tensor: \n {ones_tensor} \n")  # torch.float32
print(f"Zeros Tensor: \n {zeros_tensor}")  # device(type='cpu')


# 张量属性
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")  # 张量的维度
print(f"Datatype of tensor: {tensor.dtype}")  # 数据类型
print(f"Device tensor is stored on: {tensor.device}")  # 存储的设备


# 张量运算
# 判断当前环境GPU是否可用, 然后将tensor导入GPU内运行
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# 张量的索引和切片
tensor = torch.ones(4, 4)
tensor[:,1] = 0  # 将第1列(从0开始)的数据全部赋值为0
# 张量的拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # dim=1指定横向拼接


# 张量乘积运算(自己乘自己)
# 逐个元素相乘结果
"""
tensor([[1, 2, 3],
        [3, 2, 1],
        [4, 5, 6]])
tensor([[ 1,  4,  9],
        [ 9,  4,  1],
        [16, 25, 36]])
"""
print(f"tensor.mul(tensor): \n {tensor.mul(tensor)} \n")
# 等价写法:
print(f"tensor * tensor: \n {tensor * tensor}")

# 自动赋值运算, 有_后缀的方法会改版原始值
t1.add_(5)  #


# Tensor 与 Numpy的转化
t = torch.ones(5)  # tensor([1., 1., 1., 1., 1.])
n = t.numpy()  # array([1., 1., 1., 1., 1.], dtype=float32)
# 修改tensor的值, numpy的值也会改变, 猜测是n 指向t
t.add_(2)  # tensor([3., 3., 3., 3., 3.]), n: array([3., 3., 3., 3., 3.], dtype=float32)

