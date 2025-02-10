from typing import Dict, Union, TypeVar, List
import logging
import time

import torch
import torch.nn as nn
import mlx.core as mx

logger = logging.getLogger(__file__)
try:
    from transformers import GenerationMixin, StaticCache

    hf_available = True
except ImportError:
    hf_available = False


from proteus.utils import coerce_mx_to_torch, coerce_torch_to_mx


T = TypeVar("T", bound=nn.Module)


def maybe_wrap_hf_generate(
    mod: T, static_mlx_parameters_buffers: Dict[nn.Parameter, mx.array]
) -> T:
    """
    If HF Transformers is a dependency in the environment and the model
    to compile is an LLM, wrap the `generate()` method in one which
    creates an MLX-backed static KV cache.
    """
    if hf_available:
        if isinstance(mod, GenerationMixin):
            # create wrapper over generate method
            old_generate = mod.generate

            def wrapped_generate(*args, **kwargs):
                logger.debug("calling wrapped hf generate")
                if "past_key_values" in kwargs:
                    cache = kwargs["past_key_values"]
                    assert isinstance(cache, StaticCache)
                    mlxify_static_cache(cache, static_mlx_parameters_buffers)
                start = time.time()
                outs = old_generate(*args, **kwargs)
                end = time.time()
                # print(f"generated compiled output in {end-start:.2f} seconds")
                return outs

            mod.generate = wrapped_generate
    else:
        logger.debug("transformers lib not available")
    return mod


def mlxify_static_cache(
    cache: "StaticCache", static_mlx_buffers: Dict[torch.Tensor, mx.array]
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
