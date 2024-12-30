import ast
from typing import List, Dict, Any

from torch.fx import Node

from proteus.arg_marshalers import module_strs_to_ast

# Note that custom AST lowerings take in generic args and kwargs but must still
# produce a List[AST] and List[keyword]


def copy_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate the proper AST for aten.copy.default.

    for copy, args are two tensors: self (dest tensor) and src (unused)
    and the new copied src is returned
    https://github.com/pytorch/pytorch/blob/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native/Copy.cpp#L353
    so the generated code should look something like this:

    `aten.copy.default(_self, src)` (torch) -> `_self = mx.array(src)` (mlx)
    Note that the mx.array constructor is essential, a simple assignment will share mutations between them
    """

    assert len(args) == 2
    _, src = args
    assert isinstance(src, Node)

    mx_array_cons = module_strs_to_ast(["mlx", "core", "array"])

    return [
        ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Call(
                func=mx_array_cons,
                args=[ast.Name(id=src.name, ctx=ast.Load())],
                keywords=[],
            ),
        )
    ]


def copy_inplace_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate the proper AST for aten.copy_.default (inplace version of aten.copy).

    should look like
    `aten.copy_.default(_self, src)` (torch) -> `_self = mx.array(src); var_name = _self` (mlx)
    Note that the mx.array constructor is essential, a simple assignment will share mutations between
    """

    assert len(args) == 2
    self, src = args
    assert isinstance(self, Node) and isinstance(src, Node)

    mx_array_cons = module_strs_to_ast(["mlx", "core", "array"])

    return (
        ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())],
            value=ast.Call(
                func=mx_array_cons,
                args=[ast.Name(id=src.name, ctx=ast.Load())],
                keywords=[],
            ),
        ),
        ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Name(id=self.name, ctx=ast.Load()),
        ),
    )
