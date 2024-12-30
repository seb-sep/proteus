from typing import Dict, Callable, List, Any
import ast

import torch

from .ast_lowering import copy_to_ast


aten = torch.ops.aten

custom_lowerings_map: Dict[
    Callable,
    Callable[
        [str, List, Dict[str, Any]], List[ast.AST]
    ],  # we might want to add more than one expression at a time
] = {
    # this is not so simple as to be replaced with a single function call; easiest way to simulate
    # in-place copying is to assign the existing tensor to this one in the constructed AST
    # if this is a perf bottleneck, I could always implement a copy kernel myself
    aten.copy.default: copy_to_ast
}