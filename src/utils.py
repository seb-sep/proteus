import mlx.core as mx

import torch
import torch.fx as fx
from torch._functorch.aot_autograd import aot_module_simplified, aot_export_module

import numpy as np


def coerce_torch_to_mx(val) -> mx.array:
    """
    Convert some PyTorch value into an MLX array.
    Note that this currently COPIES the tensor, so use sparingly.
    """
    # print("coercing to mlx")
    if isinstance(val, mx.array):
        return val
    elif isinstance(
        val, (torch.Tensor, torch.nn.Parameter, torch.nn.parameter.Parameter)
    ):
        # print("copying tensor")
        return mx.array(val.detach().numpy())
    else:
        return mx.array(val)


def coerce_mx_to_torch(val: mx.array) -> torch.Tensor:
    """
    Convert an MLX array into a PyTorch tensor.
    Note that this currently COPIES the tensor, so use sparingly.
    """
    # print(type(val))
    return torch.tensor(np.array(val))


def aten_opset_compiler(gm: fx.GraphModule, sample_inputs):
    """Trace the aten graph and get a set of all ops used in the graph."""

    def foo(gm: fx.GraphModule, sample_inputs):
        s = set()
        for node in gm.graph.nodes:
            if node.op in ["call_function", "call_module", "call_method"]:
                s.add(node.target)

        print(gm.graph)

        print("Aten operators used in this graph:")
        for op in s:
            print(f"  - {op}")
        print(len(s))

        return gm.forward

    return aot_module_simplified(gm, sample_inputs, fw_compiler=foo)
