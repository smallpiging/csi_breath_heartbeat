import torch
import torch.nn as nn
import torch.optim as optim
from unet import UNet1D

model = UNet1D(num_encoding_blocks=3, out_channels_first_layer=16)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)