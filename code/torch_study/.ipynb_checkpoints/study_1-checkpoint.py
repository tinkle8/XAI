import torch
x = torch.rand( 5, 3)
# print(x)
print(torch.cpu.current_device())
print(torch.cuda.device_count())