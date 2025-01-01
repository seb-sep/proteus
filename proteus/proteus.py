from typing import List, Dict, Union
import logging

import torch.nn as nn
import torch.fx as fx
import torch
from functorch.compile import aot_module_simplified
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx

import mlx.core as mx

from proteus.mlx_builder import MLXASTBuilder
from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch

MLX_DEVICE = mx.default_device()
logger = logging.getLogger(__file__)


# Close over the mlx_params to pass in the final compiled fn
def mlx_compiler(
    static_mlx_params_buffers: Dict[Union[nn.Parameter, torch.Tensor], mx.array]
):
    """
    Compile the given FX graph of aten ops into a Python function calling MLX operations.
    """

    def _mlx_compiler(gm: fx.GraphModule, sample_inputs):

        aten_graph = make_fx(gm)(*sample_inputs)

        builder = MLXASTBuilder()
        builder.ingest_graph(aten_graph.graph)

        mlx_fn = mx.compile(builder.export())
        # mlx_fn = builder.export()

        # Wrap the MLX function to convert the appropriate inputs and outputs to MLX arrays
        # TODO: is there any way to avoid unpacking and repacking the args tuple on each forward call?
        # YES could inline in the codegenned MLX fn, but this is an optimization for later
        # alternatively, you could figure out which of the args are actually fresh tensors and not params,
        # reorder the fx graph to have all params and buffers at the back, and pass in the subset of args
        # plus the unpacked tuple of params used to prevent re-checking this at every forward pass
        def torch_wrapper(*args):
            mlx_args = tuple(
                (
                    mlx_param
                    if ((mlx_param := static_mlx_params_buffers.get(arg)) is not None)
                    else (
                        coerce_torch_to_mx(arg)
                        if isinstance(arg, torch.Tensor)
                        else arg
                    )
                )
                for arg in args
            )

            outs = mlx_fn(*mlx_args)
            return tuple(
                coerce_mx_to_torch(out) if isinstance(out, mx.array) else out
                for out in outs
            )

        print("proc'd proteus compiler")
        return torch_wrapper

    return _mlx_compiler


# this needs to be a one at a time function to prevent high memory utilization
# from having all tensors copied at once


def replace_param_with_mlx(
    mod: nn.Module,
    static_mlx_parameters: Dict[nn.Parameter, mx.array],
    name: str,
    param: nn.Parameter,
):
    """
    Populate static_mlx_parameters with the name and MLX array-ified parameter from the nn.Module,
    unset the parameter on the module, and reset it with the 'digital twin' of the MLX array
    (a torch tensor initialized with the same data pointer as the MLX array)
    """

    logger.info("replacing model params with MLX-backed tensors...")
    # I guess this should just work with params instead of tensors??? if not we can cast to the superclass
    mlx_param = coerce_torch_to_mx(param)
    torch_digital_twin = nn.Parameter(coerce_mx_to_torch(mlx_param))
    static_mlx_parameters[torch_digital_twin] = mlx_param

    # find and unset the parameter, then re-set it with the digital twin
    if "." not in name:
        delattr(mod, name)
        mod.register_parameter(name, torch_digital_twin)
    else:
        submod_path, param_subname = name.rsplit(".", 1)
        submod = mod.get_submodule(submod_path)
        delattr(submod, param_subname)
        submod.register_parameter(param_subname, torch_digital_twin)


def replace_buffer_with_mlx(
    mod: nn.Module,
    static_mlx_buffers: Dict[torch.Tensor, mx.array],
    name: str,
    buffer: torch.Tensor,
):
    """
    Copy of replace_param_with_mlx that works on buffers
    """
    mlx_buf = coerce_torch_to_mx(buffer)
    torch_digital_twin = coerce_mx_to_torch(mlx_buf)
    static_mlx_buffers[torch_digital_twin] = mlx_buf

    # find and unset the buffer
    if "." not in name:
        delattr(mod, name)
        mod.register_buffer(name, torch_digital_twin)
    else:
        submod_path, buffer_subname = name.rsplit(".", 1)
        submod = mod.get_submodule(submod_path)
        delattr(submod, buffer_subname)
        submod.register_buffer(buffer_subname, torch_digital_twin)


def proteus(mod: nn.Module):

    # inference only for now
    mod.eval()

    # generate static mlx array 'twins' of all the params in the module
    # no real overhead between making keys tensors or id() ints, since
    # torch.Tensor's __hash__ just returns the ID
    static_mlx_parameters_buffers: Dict[Union[nn.Parameter, torch.Tensor], mx.array] = (
        {}
    )

    # collect into a list aot to avoid issues w/mutation
    for name, param in list(mod.named_parameters()):
        replace_param_with_mlx(mod, static_mlx_parameters_buffers, name, param)
    for name, buf in list(mod.named_buffers()):
        replace_buffer_with_mlx(mod, static_mlx_parameters_buffers, name, buf)
    # at this point the module SHOULD work just as before

    setattr(mod, "old_forward", mod.forward)

    # should I only compile the forward method?
    mod.forward = torch.compile(
        mod.forward, backend=mlx_compiler(static_mlx_parameters_buffers), dynamic=False
    )
    return mod
