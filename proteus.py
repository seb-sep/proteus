from typing import List, Dict
from pprint import pprint
import ast

import torch.nn as nn
import torch.fx as fx
import torch
from torch._functorch.aot_autograd import aot_module_simplified, aot_export_module
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func, aot_module, aot_function
import torch.utils._pytree as pytree

import mlx.core as mx

import numpy as np

from src.mlx_builder import MLXASTBuilder

MLX_DEVICE = mx.default_device()


class MLXCoercer:
    def __init__(self):
        self.disabled = False

    def __call__(self, args):
        if self.disabled:
            return args
        return [coerce_torch_to_mx(tensor) for tensor in args]

    def disable_coercion(self):
        self.disabled = True


# Globally scoped so that coercion on all args can be activated for
# the first run through the compiled fn, and then disabled for all future inputs
_coerce_args_to_mlx = MLXCoercer()


class MLXCompiledModule:
    """
    Wraps over the AOTAutograd-compiled MLX function to pass in the MLXified parameters and buffers.
    TODO: Perhaps an optimization could be to track the # of named params and buffers, and instead of looping over each arg,
    only loop over user passed args
    """

    def __init__(
        self,
        named_params: Dict[str, nn.Parameter],
        named_buffers: Dict[str, torch.Tensor],
        compiled_fn,
    ):
        self.named_params = named_params
        self.named_buffers = named_buffers
        self.mlx_mode = False
        self.compiled_fn = compiled_fn

    def __call__(self, *args, **kwargs):
        # if you've already compiled to an mlx fn, then you need to pass the input tensors to MLX as mlx arrays
        if self.mlx_mode:
            args = [coerce_torch_to_mx(tensor) for tensor in args]
        return self.compiled_fn(self.named_params, self.named_buffers, *args, **kwargs)

    def mlx_update(self, sample_inputs):
        """
        Compile the function on the given sample inputs and prepare the module
        for inference with the MLX function.
        """
        # prompt compilation on sample inputs
        _ = self.compiled_fn(self.named_params, self.named_buffers, sample_inputs)

        # After the pytorch tensors were used for compilation, convert them
        # to MLX arrays for use in the final fn
        self.named_params = {
            k: coerce_torch_to_mx(v) for k, v in self.named_params.items()
        }
        self.named_buffers = {
            k: coerce_torch_to_mx(v) for k, v in self.named_buffers.items()
        }
        self.mlx_mode = True
        _coerce_args_to_mlx.disable_coercion()


def params_to_mlx(mod: nn.Module) -> List[mx.array]:
    """
    Pull the named parameters and buffers from a module
    and turn them into a list of MLX arrays, in the SAME
    way that aot_export_module does it. These will be passed first
    to each invocation of the generated MLX fn.
    """

    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    return tuple(coerce_torch_to_mx(param) for param in params_and_buffers_flat)


first_compile = False


# Close over the mlx_params to pass in the final compiled fn
def mlx_compiler(gm: fx.GraphModule, _):
    """
    Compile the given FX graph of aten ops into a Python function
    calling MLX operations. Second argument is for matching the signature expected by AOTAutograd.
    """

    builder = MLXASTBuilder()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
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

    # TODO: when to compile vs not? could autotune lmao
    mlx_fn = mx.compile(builder.export())

    # Wrap the MLX function to convert the appropriate inputs and outputs to MLX arrays
    # TODO: is there any way to avoid unpacking and repacking the args tuple on each forward call?
    def torch_wrapper(*args):
        outs = mlx_fn(*_coerce_args_to_mlx(args))
        return tuple(coerce_mx_to_torch(out) for out in outs)

    return torch_wrapper


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
    return torch.tensor(np.array(val))


def proteus(mod: nn.Module, sample_inputs):

    # HUGE DISCOVERY: to raise the inputs, you literally just get mod.named_parameters() and mod.named_buffers(), turn into a single dict,
    # and pytree flatten
    # This is guaranteed to produce the same ordering each time
    # the following is shamelessly pulled from aot autograd source code
    # https://github.com/pytorch/pytorch/blob/3d3551506d4acccdd06d6f98eaf05e5288d254b3/torch/_functorch/aot_autograd.py#L1190

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    # Hold onto these for the FIRST tracing call, then swap out with the MLXified args
    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)

    # I want to use aot_function so I have control over how the inputs are raised and passed
    compiled_fn = aot_function(
        functional_call,
        fw_compiler=mlx_compiler,
        num_params_buffers=num_params_buffers,
        dynamic=True,
    )
    compiled_mod = MLXCompiledModule(named_params, named_buffers, compiled_fn)
    compiled_mod.mlx_update(sample_inputs)
    return compiled_mod


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
