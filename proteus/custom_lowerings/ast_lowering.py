import ast
from typing import List, Dict, Any

from torch.fx import Node

# Note that custom AST lowerings take in generic args and kwargs but must still
# produce a List[AST] and List[keyword]


def copy_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate the proper AST for a call to `aten_copy_lowering`.

    for copy, args are two tensors: self (dest tensor) and src (what to copy from),
    and the new copied self is returned
    https://github.com/pytorch/pytorch/blob/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native/Copy.cpp#L353
    so the generated code should look something like this:

    `aten.copy.default(_self, src)` (torch) -> `_self = src` (mlx)
    """

    assert len(args) == 2
    _self, src = args
    assert isinstance(_self, Node) and isinstance(src, Node)

    return [
        ast.Assign(
            targets=[ast.Name(id=_self.name, ctx=ast.Store())],
            value=ast.Name(id=src.name, ctx=ast.Load()),
        ),
        ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Name(id=_self.name, ctx=ast.Load()),
        ),
    ]


a = 5
b = a
c = b
