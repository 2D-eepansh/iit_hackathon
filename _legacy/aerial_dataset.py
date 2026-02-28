import torch
from torch.cuda.amp import autocast

x = torch.randn(1024,1024, device="cuda")

with autocast():
    y = x @ x

print("AMP working")