from typing import List, Callable, Optional
from pathlib import Path
import logging
import importlib.util

import torch
from torch._inductor.codecache import compiled_fx_graph_hash
from torch.fx import GraphModule


COMPILED_FN_NAME = "cool_mlx_fn"

CACHE_DIR = Path.home() / ".cache" / "proteus"

logger = logging.getLogger(__file__)


def cache_load(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> Optional[Callable]:

    # graph kwargs are just for inductor shit, we can keep empty or use however we like
    # I THINK input_idxs_to_check is for cudagraph static ptrs
    # drop the debug lines which is the second return val
    # this is REALLY bad but this prevents the hashing from inlining all tensors for the hash calculation
    # depends on the following behavior for tracking freezing and the check in inductor codecache
    # https://github.com/pytorch/pytorch/blob/36c461af9535f7a3739bc0f4d1d178845afe3664/torch/_inductor/freezing_utils.py#L29
    # https://github.com/pytorch/pytorch/blob/7ce4974e50fa68e7be3caafb35208505328dedc9/torch/_inductor/codecache.py#L541
    # note that behavior and usage is slightly different on pytorch mainline than on macos version, but demonstrates usage
    setattr(gm, "_has_frozen_params", True)
    hash, _ = compiled_fx_graph_hash(gm, example_inputs, {}, ())
    cache_to_load = CACHE_DIR / hash

    if cache_to_load.is_dir():
        compiled_mlx_file = cache_to_load / "mlx_fn.py"
        if compiled_mlx_file.exists():

            spec = importlib.util.spec_from_file_location(
                "compiled_mlx_fn", compiled_mlx_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.debug(f"loading compiled code from {compiled_mlx_file}")
            return getattr(module, COMPILED_FN_NAME)
        else:
            logger.debug(
                f"cache folder for hash {hash} found, but no mlx_fn.py file found"
            )
    else:
        logger.debug(f"cache miss for hash {hash}")


def cache_store(
    gm: GraphModule, example_inputs: List[torch.Tensor], compiled_code: str
) -> str:

    hash, _ = compiled_fx_graph_hash(gm, example_inputs, {}, ())
    cache_path = CACHE_DIR / hash
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = cache_path / "mlx_fn.py"
    logger.debug(f"caching compiled code at {filename}")
    with open(filename, "w") as f:
        f.write(compiled_code)

    return filename
