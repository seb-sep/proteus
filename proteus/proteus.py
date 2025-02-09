from typing import Dict, Union, TypeVar, List
import logging
from functools import partial

import torch.nn as nn
import torch.fx as fx
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import mlx.core as mx

from proteus.mlx_builder import MLXASTBuilder
from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch
from c_extensions import contiguous

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

        aten_graph = make_fx(gm, tracing_mode="fake")(*sample_inputs)

        builder = MLXASTBuilder()
        builder.ingest_graph(aten_graph.graph)

        # mlx_fn = mx.compile(builder.export())
        mlx_fn = builder.export()

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


T = TypeVar("T", bound=nn.Module)


def proteus(mod: T) -> T:

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

    # setattr(mod, "old_forward", mod.forward)

    # should I only compile the forward method?
    # dynamic should probably be None by defaut, only be True when you know you need it
    # for very specific models like LLMs

    # if i have a huggingface llm, how to overwrite the cache used? I can still only compile
    # forward for now as long as I intercept the cache used in generate
    # therefore, the generate method must still be wrapped

    try:
        from transformers import GenerationMixin

        if isinstance(mod, GenerationMixin):
            # create wrapper over generate method
            old_generate = mod.generate

            def wrapped_generate(*args, **kwargs):
                print("calling wrapped generate!")
                if "past_key_values" in kwargs:
                    cache = kwargs["past_key_values"]
                    assert isinstance(cache, StaticCache)
                    mlxify_static_cache(cache, static_mlx_parameters_buffers)
                return old_generate(*args, **kwargs)

            mod.generate = wrapped_generate
    except ImportError:
        logger.debug("transformers lib not available")

    mod.forward = torch.compile(
        mod.forward, backend=mlx_compiler(static_mlx_parameters_buffers)
    )
    return mod


from transformers import StaticCache


def mlxify_static_cache(
    cache: StaticCache, static_mlx_buffers: Dict[torch.Tensor, mx.array]
) -> None:
    """
    Replace the tensors in the HF static KV cache with tensors backed by MLX
    and populate the tensor-array mapping with the tensors in the cache.

    Mutate the cache and buffer dict in place.
    """

    print("mlxifying static cache")

    # cache values might also be registered under buf
    for name, buf in list(cache.named_buffers()):
        delattr(cache, name)
        # replace_buffer_with_mlx(cache, static_mlx_buffers, name, buf)

    # pop and create one at a time to minimize peak memory usage
    mlx_key_cache: List[torch.Tensor] = []
    while cache.key_cache:
        key = cache.key_cache.pop(0)
        mlx_key = coerce_torch_to_mx(key)
        new_torch_key = coerce_mx_to_torch(mlx_key)
        static_mlx_buffers[new_torch_key] = mlx_key

        # remember that creating a torch tensor from mlx is just a view over the mlx array
        mlx_key_cache.append(new_torch_key)

    cache.key_cache = mlx_key_cache

    mlx_value_cache: List[torch.Tensor] = []
    while cache.value_cache:
        value = cache.value_cache.pop(0)
        mlx_value = coerce_torch_to_mx(value)
        new_torch_value = coerce_mx_to_torch(mlx_value)
        static_mlx_buffers[new_torch_value] = mlx_value

        # remember that creating a torch tensor from mlx is just a view over the mlx array
        mlx_value_cache.append(new_torch_value)

    cache.value_cache = mlx_value_cache
