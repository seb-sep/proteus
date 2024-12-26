import mlx.core as mx
import torch

from proteus.utils import coerce_mx_to_torch, coerce_torch_to_mx
from c_extensions.build import ptr_to_mlx

mx.set_default_device(mx.gpu)
b = torch.randn((16, 16))
b = b.to("mps")

mb = ptr_to_mlx(b.data_ptr(), b.shape, mx.float32)
print(mb)
