import ast
from typing import List, Dict, Any
from itertools import zip_longest
import sys

import torch
from torch.fx import Node

from proteus.arg_marshalers import module_strs_to_ast, convert_arg_to_ast

# Note that custom AST lowerings take in generic args and kwargs but must still
# produce a List[AST] and List[keyword]


def size_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate the proper AST for aten.size.default:
    `aten.size.default(a)` -> `a.shape`
    """

    assert len(args) == 1 and len(kwargs) == 0
    return [
        ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Attribute(
                value=ast.Name(id=args[0].name, ctx=ast.Load()),
                attr="shape",
                ctx=ast.Load(),
            ),
        )
    ]


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


def _to_copy_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate the proper AST for aten._to_copy.default.

    _to_copy handles device, datatype, and layout copying; we only care about datatype since
    the notion of devices doesn't exist for MLX, so we simply invoke the MLX .astype() method
    in the case of a passed dtype and a plain copy in any other case
    """

    assert len(args) == 1
    self = args[0]
    assert isinstance(self, Node)

    if dtype := kwargs.get("dtype"):
        assert isinstance(dtype, torch.dtype)
        # call .astype() method on self name
        assign_ast = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self.name, ctx=ast.Load()),
                attr="astype",
                ctx=ast.Load(),
            ),
            args=[convert_arg_to_ast(dtype)],
            keywords=[],
        )
    else:
        # simply copy the tensor w array constructor
        mx_array_cons = module_strs_to_ast(["mlx", "core", "array"])
        assign_ast = ast.Call(
            func=mx_array_cons,
            args=[ast.Name(id=self.name, ctx=ast.Load())],
            keywords=[],
        )

    return (
        ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=assign_ast,
        ),
    )


def index_copy_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate an AST for an aten.index_copy_.default op:
    `aten.index_copy.default(_self, 0, index, tensor)` -> a[index] = tensor
    """

    # for now, assume all are args and not kwargs
    assert len(args) == 4
    _self, dim, index, tensor = args

    assert isinstance(_self, Node)
    key = "val" if "val" in _self.meta else "example_value"
    ndims = _self.meta[key].ndim

    # the index needs to be at the dim'th index of a tuple used to slice
    full_slice = ast.Slice(lower=None, upper=None, step=None)
    index_slice = ast.Name(id=index.name, ctx=ast.Load())
    slice_ast = ast.Tuple(
        elts=[full_slice if i != dim else index_slice for i in range(ndims)]
    )
    index_ast = ast.Subscript(
        value=ast.Name(id=_self.name, ctx=ast.Load()),
        slice=slice_ast,
        ctx=ast.Store(),
    )

    assign_ast = ast.Assign(
        targets=[index_ast], value=ast.Name(id=tensor.name, ctx=ast.Load())
    )
    return [
        assign_ast,
        ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Name(id=_self.name, ctx=ast.Load()),
        ),
    ]


def slice_to_ast(var_name, args: List, kwargs: Dict[str, Any]) -> List[ast.AST]:
    """
    Generate an AST for an aten.slice.Tensor op which should produce
    the proper Python slice on the array, for example:

    `aten.slice.Tensor(a, 0, 0, 4, 2)` -> a[0:4:2, :, :]
    """

    tensor, *rest = args
    assert isinstance(tensor, Node)

    lrest = len(rest)
    dim = kwargs.get("dim", 0) if lrest < 1 else rest[0]
    start = kwargs.get("start") if lrest < 2 else rest[1]
    end = kwargs.get("end") if lrest < 3 else rest[2]
    step = kwargs.get("step", 1) if lrest < 4 else rest[3]

    # aten.slice.Tensor will use the maxsize for going to the end of a slice,
    # we want this to be none to get the proper full slice
    end = end if end != sys.maxsize else None

    # for a slice on a single dim of the tensor, I need to know
    # how many dims the tensor has, aten slice only slices a single dim at a time
    key = "val" if "val" in tensor.meta else "example_value"
    ndims = tensor.meta[key].ndim

    # for the : parts of the multidim slice
    full_slice = ast.Slice(lower=None, upper=None, step=None)
    dim_slice = ast.Slice(
        lower=convert_arg_to_ast(start),
        upper=convert_arg_to_ast(end),
        step=convert_arg_to_ast(step),
    )

    slice_elts = [full_slice if i != dim else dim_slice for i in range(ndims)]

    slice_value = ast.Subscript(
        value=ast.Name(id=tensor.name, ctx=ast.Load()),
        slice=ast.Tuple(elts=slice_elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )

    return [
        ast.Assign(targets=[ast.Name(id=var_name, ctx=ast.Store())], value=slice_value)
    ]
