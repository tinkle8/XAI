import torch
from torch.utils.tensorboard import SummaryWriter

e = torch.randn(2, 3)
print("e:",e)
f = torch.randn(2, 3)
print("f:",f)
print(e + f)

# 逐元素乘法
print(e * f)
