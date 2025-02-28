import torch
from einops import rearrange

x = [[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],[[13,14,15],[17,18,19],[20,21,22],[23,24,25]]]
tensor = torch.tensor(x)
print(tensor.shape)
c, h, w = tensor.shape
tensor = rearrange(tensor, 'c h w -> w (c h)')
print(tensor.shape)
print(tensor)
tensor = rearrange(tensor, 'w (c h) -> c h w', c=c,h=h)
print(tensor.shape)
print(tensor)