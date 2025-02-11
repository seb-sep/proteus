from typing import List, Callable, Optional
from pathlib import Path
import logging
import importlib.util

import torch
from torch._inductor.codecache import compiled_fx_graph_hash
from torch.fx import GraphModule

from proteus.mlx_ast.mlx_builder import COMPILED_FN_NAME

CACHE_DIR = Path.home() / ".cache" / "proteus"

logger = logging.getLogger(__file__)


def cache_lookup(
    gm: GraphModule, example_inputs: List[torch.Tensor]
) -> Optional[Callable]:

    # graph kwargs are just for inductor shit, we can keep empty or use however we like
    # I THINK input_idxs_to_check is for cudagraph static ptrs
    # drop the debug lines which is the second return val
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
            return getattr(module, COMPILED_FN_NAME)
        else:
            logger.debug(
                f"cache folder for hash {hash} found, but no mlx_fn.py file found"
            )
    else:
        logger.debug(f"cache miss for hash {hash}")
