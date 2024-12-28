from typing import Callable, List, Dict, Any, Tuple, Union
import operator
import ast
from pprint import pprint
from inspect import getmodule

import torch
from torch.fx import Graph, Node
import mlx.core as mx
import mlx.nn as nn

from proteus.arg_marshalers import (
    passthrough_arg_marshaler,
    take_arg_marshaler,
    t_arg_marshaler,
    clone_arg_marshaler,
    transpose_int_arg_marshaler,
    expand_arg_marshaler,
    module_strs_to_ast,
)

from proteus.custom_ops import expand

from proteus.custom_lowerings import custom_lowerings_map

aten = torch.ops.aten

# implementations of aten ops are in pytorch/aten/src/ATen/native
# https://github.com/pytorch/pytorch/tree/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native

_aten_mlx_mapping: Dict[
    Callable,
    Tuple[Callable, Callable[[List, Dict], Tuple[List[ast.AST], List[ast.keyword]]]],
] = {
    aten.mm.default: (mx.matmul, passthrough_arg_marshaler),
    aten.t.default: (mx.transpose, t_arg_marshaler),
    aten.transpose.int: (mx.swapaxes, transpose_int_arg_marshaler),
    aten.expand.default: (expand, expand_arg_marshaler),
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
    # NOTE: remember that you ran into issues with what to do with device kwarg for arange
    aten.arange.start: (mx.arange, passthrough_arg_marshaler),
    aten.arange.default: (mx.arange, passthrough_arg_marshaler),
    aten.unsqueeze.default: (mx.expand_dims, passthrough_arg_marshaler),
    aten.full.default: (mx.full, passthrough_arg_marshaler),
    # NOTE: aten view only meant to work on contiguous tensors and is ALWAYS zero-copy,
    # presumably mx.reshape copies in the non-contiguous case
    aten.view.default: (mx.reshape, passthrough_arg_marshaler),
    # aten.slice.Tensor: (custom_ops.slice, passthrough_arg_marshaler),
    # TODO: dtype removal in clone marshaling should replace with mlx dtype, maybe use mx.view
    # looks like aten.clone creates a whole new tensor with data (mlx does this with modifications,
    # so using __copy__ which only actaully copies when there is mutation) should preserve semantics and be faster
    # however, aten.copy copies contents of one tensor into another
    aten.clone.default: (mx.array.__copy__, clone_arg_marshaler),
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


def fn_to_manual_module(fn: Callable) -> Union[List[str], None]:
    """
    Manually map MLX functions to their fully qualified attribute path.
    This function is unfortunately necessary because MLX's `mx.compile`d functions being JIT compiled
    plain nanobind functions don't have attributes like `__qualname__` or `__module__`.
    TODO: see if some clever use of mx.disable_compile() gets around this lack of attrs

    Instantiate the dict on each call for now for cleanness but move it out if it gets too big.
    """
    return {
        nn.silu: ["mlx", "nn", "silu"],
        nn.relu: ["mlx", "nn", "relu"],
        nn.gelu: ["mlx", "nn", "gelu"],
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
        module_strs = (fn.__module__ + "." + fn.__qualname__).split(".")

    return module_strs_to_ast(module_strs)


class MLXASTBuilder:
    def __init__(self):
        # input names to top level mlx function to generate
        self.in_args: List[ast.arg] = []
        # import mlx.core as mx
        self.imports = [
            ast.Import(names=[ast.alias(name="mlx", asname=None)]),
            ast.Import(names=[ast.alias(name="proteus", asname=None)]),
        ]
        # mlx function calls in order of execution
        # generates lines of python in the AST function body
        self.calls: List[ast.AST] = []
        self.device = mx.default_device()
        self.ret: ast.Return = None

    def ingest_graph(self, graph: Graph):
        for node in graph.nodes:
            if node.op == "placeholder":
                self.addArgument(node.name)
            elif node.op == "call_function":
                self.addFunctionCall(node.target, node.name, node.args, node.kwargs)
            elif node.op == "output":
                # the first value in args for output is what to actually return from the graph i think,
                # we might actually only care about the first value in that tuple
                # https://pytorch.org/docs/stable/fx.html#torch.fx.Node
                self.addReturn(node.args[0])
            elif node.op == "get_attr":
                # attr = getattr(gm._graph_module, node.target)
                # print(attr)
                # pprint(vars(node))
                raise ValueError(f"unhandled getattr")

            else:
                raise ValueError(f"unhandled node type: node {node, node.op}")

    def addFunctionCall(self, aten_op, var_name, args: List, kwargs: Dict[str, Any]):
        """Ingest an aten operation and append an assign expression of the matching MLX fn to the AST body."""

        if aten_op in _aten_mlx_mapping:
            # ast.Load() means you're reading the value of the name, not setting or deleting it
            mlx_fn, arg_marshaler = _aten_mlx_mapping[aten_op]

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
        elif custom_lowering := custom_lowerings_map[aten_op]:
            custom_ast = custom_lowering(var_name, args, kwargs)
            self.calls.extend(custom_ast)
        else:
            raise ValueError(f"aten op {aten_op} not supported")

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
            if isinstance(arg, Node):
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
