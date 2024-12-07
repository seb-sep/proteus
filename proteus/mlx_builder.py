from typing import Callable, List, Dict, Any, Tuple, Union
import operator
import ast
from pprint import pprint
from inspect import getmodule

import torch
from torch import fx
import mlx.core as mx
import mlx.nn as nn

from proteus.arg_marshalers import (
    passthrough_arg_marshaler,
    take_arg_marshaler,
    clone_arg_marshaler,
    arange_arg_marshaler,
)
import proteus.custom_ops.custom_ops as custom_ops

aten = torch.ops.aten

_fn_mapping = {
    aten.mm.default: (mx.matmul, passthrough_arg_marshaler),
    aten.t.default: (mx.transpose, passthrough_arg_marshaler),
    aten.transpose.int: (mx.transpose, passthrough_arg_marshaler),
    aten.expand.default: (mx.broadcast_to, passthrough_arg_marshaler),
    aten.relu.default: (nn.relu, passthrough_arg_marshaler),
    aten.silu.default: (nn.silu, passthrough_arg_marshaler),
    aten.gelu.default: (nn.gelu, passthrough_arg_marshaler),
    aten.triu.default: (mx.triu, passthrough_arg_marshaler),
    aten.tril.default: (mx.tril, passthrough_arg_marshaler),
    aten.mul.Tensor: (mx.multiply, passthrough_arg_marshaler),
    aten.div.Tensor: (mx.divide, passthrough_arg_marshaler),
    aten.add.Tensor: (mx.add, passthrough_arg_marshaler),
    aten.exp.default: (mx.exp, passthrough_arg_marshaler),
    aten.gt.Tensor: (mx.greater, passthrough_arg_marshaler),
    aten.neg.default: (mx.negative, passthrough_arg_marshaler),
    aten.cos.default: (mx.cos, passthrough_arg_marshaler),
    aten.sin.default: (mx.sin, passthrough_arg_marshaler),
    aten.rsqrt.default: (mx.rsqrt, passthrough_arg_marshaler),
    aten.cat.default: (mx.concatenate, passthrough_arg_marshaler),
    aten.select.int: (mx.take, take_arg_marshaler),
    aten.eq.Scalar: (mx.equal, passthrough_arg_marshaler),
    aten.embedding.default: (mx.array.__getitem__, passthrough_arg_marshaler),
    # aten.linear.default: (custom_ops.linear, passthrough_arg_marshaler),
    # is it ok to have multiple aten ops map to the same mlx fn like this?
    # TODO: device removal in arange marshaling sohuld be universal
    aten.arange.start: (mx.arange, arange_arg_marshaler),
    aten.arange.default: (mx.arange, arange_arg_marshaler),
    aten.unsqueeze.default: (mx.expand_dims, passthrough_arg_marshaler),
    aten.full.default: (mx.full, passthrough_arg_marshaler),
    aten.view.default: (mx.view, passthrough_arg_marshaler),
    # aten.slice.Tensor: (custom_ops.slice, passthrough_arg_marshaler),
    # TODO: dtype removal in clone marshaling should replace with mlx dtype, maybe use mx.view
    aten.clone.default: (mx.array.__copy__, clone_arg_marshaler),
    aten.copy.default: (mx.array.__copy__, passthrough_arg_marshaler),
    aten._to_copy.default: (mx.array.__copy__, clone_arg_marshaler),
    # aten.masked_fill.Scalar: (custom_ops.masked_fill, passthrough_arg_marshaler),
    # aten.slice_scatter.default: (custom_ops.slice_scatter, passthrough_arg_marshaler),
    aten.conv2d.default: (mx.conv2d, passthrough_arg_marshaler),
    aten._unsafe_view.default: (mx.reshape, passthrough_arg_marshaler),
    aten.split.Tensor: (mx.split, passthrough_arg_marshaler),
    # aten.dropout.default: (custom_ops.passthrough, passthrough_arg_marshaler),
    aten.scaled_dot_product_attention.default: (
        mx.fast.scaled_dot_product_attention,
        passthrough_arg_marshaler,
    ),
    operator.getitem: (mx.array.__getitem__, passthrough_arg_marshaler),
    aten.layer_norm.default: (mx.fast.layer_norm, passthrough_arg_marshaler),
    aten.pow.Tensor_Scalar: (mx.power, passthrough_arg_marshaler),
    aten.mean.dim: (mx.mean, passthrough_arg_marshaler),
    aten.mean.default: (mx.mean, passthrough_arg_marshaler),
    aten.einsum.default: (mx.einsum, passthrough_arg_marshaler),
}


def aten_to_mlx(
    aten_op, device: Union[mx.Device, mx.Stream]
) -> Tuple[Callable, Callable[[List, Dict], Tuple[List[ast.AST], List[ast.keyword]]]]:
    """
    Map an aten op to a tuple of the corresponding MLX function,
    and a marshaling function taking in the aten args and kwargs to the op
    (which should just be strings representing SSA values from the fx graph)
    and returns AST values to be passed as args and kwargs at the AST MLX call sites.
    """

    if aten_op in _fn_mapping:
        return _fn_mapping[aten_op]
    else:
        raise KeyError(aten_op)


def fn_to_manual_module(fn: Callable) -> Union[List[str], None]:
    return {
        nn.silu: ["mlx", "nn", "silu"],
        nn.relu: ["mlx", "nn", "relu"],
        nn.gelu: ["mlx", "nn", "gelu"],
        mx.array.__getitem__: ["mlx", "core", "array", "__getitem__"],
        mx.array.__copy__: ["mlx", "core", "array", "__copy__"],
    }.get(fn, None)


def fn_to_attr(fn: Callable) -> Union[ast.Name, ast.Attribute]:
    """
    Convert a Python callable, likely a function from MLX, to an attribute or name AST
    to be used at an AST call site.
    """
    # use inspect.getmodule().__name__ to map these callables to the right AST attr
    # for example, module_strs might be ['mlx', 'linalg', 'matmul']
    module_strs = fn_to_manual_module(fn)
    if not module_strs:
        module_strs = getmodule(fn).__name__.split(".") + [fn.__name__]

    attr_ast = ast.Name(id=module_strs.pop(0), ctx=ast.Load())
    while module_strs:
        attr_ast = ast.Attribute(
            value=attr_ast, attr=module_strs.pop(0), ctx=ast.Load()
        )
    return attr_ast


class MLXASTBuilder:
    def __init__(self):
        # input names to top level mlx function to generate
        self.in_args: List[ast.arg] = []
        # import mlx.core as mx
        self.imports = [
            ast.Import(names=[ast.alias(name="mlx", asname=None)]),
            # ast.Import(names=[ast.alias(name="src", asname=None)]),
        ]
        # mlx function calls in order of execution
        self.calls: List[ast.Call] = []
        self.device = mx.default_device()
        self.ret: ast.Return = None

    def addFunctionCall(self, aten_op, var_name, args: List, kwargs: Dict[str, Any]):
        """Ingest an aten operation and append an assign expression of the matching MLX fn to the AST body."""

        # ast.Load() means you're reading the value of the name, not setting or deleting it
        mlx_fn, arg_marshaler = aten_to_mlx(aten_op, self.device)

        # Convert args to ast.Name nodes
        ast_args, ast_kwargs = arg_marshaler(args, kwargs)

        ast_func = fn_to_attr(mlx_fn)
        ast_assign = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast_func,
                args=ast_args,
                keywords=ast_kwargs,
            ),
        )
        # print(f"calling fn {getmodule(mlx_fn).__name__}.{mlx_fn.__name__}")
        self.calls.append(ast_assign)

    def addArgument(self, arg_name: str):
        """
        Add an argument of the given name to the args
        of the top level function to return.
        """
        self.in_args.append(ast.arg(arg=arg_name))

    # def addReturn(self, arg: str):
    def addReturn(self, args: tuple):
        """
        Add a return to the end of the AST with the given variable name.
        NOTE: currently return only one value, even though you do it as a tuple
        """
        names = []
        for arg in args:
            if isinstance(arg, fx.Node):
                # If arg is a torch.fx.Node, use its name attribute
                names.append(arg.name)
            else:
                # If arg is not a torch.fx.Node, assume it's a string as before
                names.append(arg)
        self.ret = ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id=arg, ctx=ast.Load()) for arg in names],
                ctx=ast.Load(),
            )
        )

    def export(self) -> Callable:
        """Turn the accumulated AST into a Python callable."""
        self.calls.append(self.ret)

        mlx_func = ast.FunctionDef(
            name="cool_mlx_fn",
            args=ast.arguments(
                posonlyargs=[],
                args=self.in_args,
                kwonlyargs=[],
                defaults=[],
                kw_defaults=[],
            ),
            body=self.calls,
            decorator_list=[],
        )

        module = ast.Module(body=self.imports + [mlx_func], type_ignores=[])
        ast.fix_missing_locations(module)
        print(f"Generated Python code:\n{ast.unparse(module)}")
        code = compile(module, "<mlx_ast>", "exec")
        namespace = {}
        exec(code, namespace)

        return namespace[mlx_func.name]
