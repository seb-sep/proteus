import mlx.core as mx
import torch

from proteus.utils import coerce_mx_to_torch, coerce_torch_to_mx

a = mx.random.normal((4, 4))
b = torch.randn((16, 16), pin_memory=True)
b = b.to("mps")

torch.mps.set_memory_mode("shared")

mx_b = coerce_torch_to_mx(b)
print("made mx_b")
print(mx_b)
