from typing import List
from pprint import pprint
import ast

import torch.nn as nn
import torch.fx as fx
import torch
from torch._functorch.aot_autograd import aot_module_simplified, aot_export_module
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

import mlx.core as mx

import numpy as np

from mlx_builder import MLXASTBuilder

MLX_DEVICE = mx.default_device()


def mlx_compiler(gm: fx.GraphModule, sample_inputs):
    gm.graph.print_tabular()
    builder = MLXASTBuilder()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            pprint(vars(node))
            builder.addArgument(node.target)
        elif node.op == "call_function":
            builder.addFunctionCall(node.target, node.name, node.args, node.kwargs)
        elif node.op == "output":
            # the first value in args for output is what to actually return from the graph i think,
            # we might actually only care about the first value in that tuple
            # https://pytorch.org/docs/stable/fx.html#torch.fx.Node
            builder.addReturn(node.args[0][0])
        else:
            raise ValueError(f"unhandled node type: node {node}")

    mlx_fn = builder.export()
    mx.set_default_device(mx.gpu)

    def torch_wrapper(*args):
        mlx_args = [coerce_torch_to_mx(tensor) for tensor in args]
        outs = mlx_fn(*mlx_args)
        return tuple(coerce_mx_to_torch(out) for out in outs)

    print("compiled a new callable")
    return torch_wrapper


def coerce_torch_to_mx(val) -> mx.array:
    if isinstance(val, (torch.Tensor, torch.nn.Parameter)):
        return mx.array(val.detach().numpy())
    else:
        return mx.array(val)


def coerce_mx_to_torch(val: mx.array) -> torch.Tensor:
    return torch.tensor(np.array(val))


def proteus_simplified(gm: fx.GraphModule, sample_inputs):
    def print_compiler(gm: fx.GraphModule, sample_inputs):
        # <implement your compiler here>
        print("Decomposed fx Graph in Aten IR:")
        print(gm.graph)

        for node in gm.graph.nodes:
            pprint(vars(node))

        return gm

    # Invoke AOTAutograd
    return aot_module_simplified(gm, sample_inputs, fw_compiler=mlx_compiler)


def proteus(gm: fx.GraphModule, sample_inputs):
    aten_graph = aot_export_module(
        gm,
        sample_inputs,
        trace_joint=False,
    )


def aten_opset_compiler(gm: fx.GraphModule, sample_inputs):
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
