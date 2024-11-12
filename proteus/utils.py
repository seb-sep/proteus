from pprint import pprint

import mlx.core as mx

import torch
import torch.fx as fx
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    aot_export_module,
    aot_function,
)

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

        # print(gm.graph)

        print("Aten operators used in this graph:")
        for op in s:
            print(f"  - {op}")
        print(len(s))

        return gm.forward

    fn = aot_function(gm, fw_compiler=foo)
    _ = fn(*sample_inputs)
    return fn


class MLXCodeGenInterpreter(fx.Interpreter):
    """probably won't be using this but hold onto just in case"""

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)

    def call_function(self, target, args, kwargs):
        """
        Find the MLX function corresponding to the aten op, and call it on the given argument
        (which should have already been registered as a function input from placeholder)
        that is, construct an ast.Call on the right mlx function with the inputs of the original node
        """
        print(f"Function target: {target}")
        pprint(args)
        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        print(f"Method target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        print(f"Module target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().call_module(target, args, kwargs)

    def get_attr(self, target, args, kwargs):
        print(f"Attr target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().get_attr(target, args, kwargs)

    def placeholder(self, target, args, kwargs):
        """
        top level graph inputs like so:
        %primals_1 : [num_users=1] = placeholder[target=primals_1]
        these will just be top level inputs in the ast
        so, these should be the inputs in our mlx function signature
        """
        print(f"Placeholder target: {target, type(target)}")
        # print(f"args {args}, kwargs {kwargs}")
        self.in_args.append(target)
        return super().placeholder(target, args, kwargs)

    def output(self, target, args, kwargs):
        print(f"Output target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().output(target, args, kwargs)


class DefaultInterpreter(fx.Interpreter):
    """Interpreter that uses default behavior for all overrides"""

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)

    def call_function(self, target, args, kwargs):
        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        return super().call_module(target, args, kwargs)

    def get_attr(self, target, args, kwargs):
        # print(target, args, kwargs)
        # print(args, kwargs)
        return super().get_attr(target, args, kwargs)

    def placeholder(self, target, args, kwargs):
        return super().placeholder(target, args, kwargs)

    def output(self, target, args, kwargs):
        return super().output(target, args, kwargs)
