from typing import Callable, List, Dict, Any, Tuple, Union, Optional
import operator
import ast
import logging
import os

import torch
from torch.fx import Graph, Node
import mlx.core as mx
import mlx.nn as nn

from proteus.mlx_ast.arg_marshalers import (
    module_strs_to_ast,
    passthrough_arg_marshaler,
    take_arg_marshaler,
    t_arg_marshaler,
    clone_arg_marshaler,
    transpose_int_arg_marshaler,
    expand_arg_marshaler,
    layernorm_arg_marshaler,
    sdpa_arg_marshaler,
    sdpa_cpu_arg_marshaler,
    triangle_arg_marshaler,
    mean_arg_marshaler,
    einsum_arg_marshaler,
    arange_arg_marshaler,
    full_arg_marshaler,
    any_arg_marshaler,
    sum_arg_marshaler,
    _softmax_arg_marshaler,
    zeros_arg_marshaler,
    cumsum_arg_marshaler,
    oneslike_arg_marshaler,
)

from proteus.custom_ops import (
    expand,
    custom_split,
    custom_sdpa,
    masked_fill_scalar,
    custom_take,
    custom_new_ones,
)

from proteus.mlx_ast.custom_lowerings import custom_lowerings_map
from proteus.utils.compile_cache import COMPILED_FN_NAME


aten = torch.ops.aten
logger = logging.getLogger(__file__)

# implementations of aten ops are in pytorch/aten/src/ATen/native
# https://github.com/pytorch/pytorch/tree/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native

_aten_mlx_mapping: Dict[
    Callable,
    Tuple[Callable, Callable[[List, Dict], Tuple[List[ast.AST], List[ast.keyword]]]],
] = {
    aten.mm.default: (mx.matmul, passthrough_arg_marshaler),
    aten.bmm.default: (mx.matmul, passthrough_arg_marshaler),
    aten.addmm.default: (mx.addmm, passthrough_arg_marshaler),
    aten.t.default: (mx.transpose, t_arg_marshaler),
    aten.transpose.int: (mx.swapaxes, transpose_int_arg_marshaler),
    aten.expand.default: (expand, expand_arg_marshaler),
    aten.relu.default: (nn.relu, passthrough_arg_marshaler),
    aten.silu.default: (nn.silu, passthrough_arg_marshaler),
    aten.gelu.default: (nn.gelu, passthrough_arg_marshaler),
    aten.triu.default: (mx.triu, triangle_arg_marshaler),
    aten.tril.default: (mx.tril, triangle_arg_marshaler),
    aten.mul.Tensor: (mx.multiply, passthrough_arg_marshaler),
    # mul_ is an inplace variant of mul, make the same for now
    aten.mul_.Tensor: (mx.multiply, passthrough_arg_marshaler),
    aten.div.Tensor: (mx.divide, passthrough_arg_marshaler),
    aten.add.Tensor: (mx.add, passthrough_arg_marshaler),
    aten.sub.Tensor: (mx.subtract, passthrough_arg_marshaler),
    aten.exp.default: (mx.exp, passthrough_arg_marshaler),
    aten.gt.Tensor: (mx.greater, passthrough_arg_marshaler),
    aten.ge.Scalar: (mx.greater_equal, passthrough_arg_marshaler),
    aten.neg.default: (mx.negative, passthrough_arg_marshaler),
    aten.cos.default: (mx.cos, passthrough_arg_marshaler),
    aten.sin.default: (mx.sin, passthrough_arg_marshaler),
    aten.rsqrt.default: (mx.rsqrt, passthrough_arg_marshaler),
    aten.cat.default: (mx.concatenate, passthrough_arg_marshaler),
    # aten.select.int: (mx.take, take_arg_marshaler),
    aten.select.int: (custom_take, passthrough_arg_marshaler),
    aten.eq.Scalar: (mx.equal, passthrough_arg_marshaler),
    aten.zeros.default: (mx.zeros, zeros_arg_marshaler),
    aten.ones_like.default: (mx.ones_like, oneslike_arg_marshaler),
    aten.embedding.default: (mx.array.__getitem__, passthrough_arg_marshaler),
    aten.cumsum.default: (mx.cumsum, cumsum_arg_marshaler),
    # aten.linear.default: (custom_ops.linear, passthrough_arg_marshaler),
    # is it ok to have multiple aten ops map to the same mlx fn like this?
    # NOTE: remember that you ran into issues with what to do with device kwarg for arange
    aten.arange.start: (mx.arange, arange_arg_marshaler),
    aten.arange.default: (mx.arange, arange_arg_marshaler),
    aten.unsqueeze.default: (mx.expand_dims, passthrough_arg_marshaler),
    aten.full.default: (mx.full, full_arg_marshaler),
    # NOTE: aten view only meant to work on contiguous tensors and is ALWAYS zero-copy,
    # presumably mx.reshape copies in the non-contiguous case
    aten.view.default: (mx.reshape, passthrough_arg_marshaler),
    # looks like _unsafe_view is just a hack which is equal to view() but is treated differently by autodiff
    # for the purposes of an inference compiler we can treat it as the same
    # https://github.com/pytorch/pytorch/blob/e1abbe155ec4fb4fd94281f86282bed22d38c5ae/aten/src/ATen/native/TensorShape.cpp#L4020
    aten._unsafe_view.default: (mx.reshape, passthrough_arg_marshaler),
    # TODO: dtype removal in clone marshaling should replace with mlx dtype, maybe use mx.view
    # looks like aten.clone creates a whole new tensor with data (mlx does this with modifications,
    # so using __copy__ which only actaully copies when there is mutation) should preserve semantics and be faster
    # however, aten.copy copies contents of one tensor into another
    aten.clone.default: (mx.array.__copy__, clone_arg_marshaler),
    # implement _to_copy later when we have a better idea of what the heck it does
    # aten._to_copy.default: (mx.array.__copy__, clone_arg_marshaler),
    aten.masked_fill.Scalar: (masked_fill_scalar, passthrough_arg_marshaler),
    aten.masked_fill_.Scalar: (masked_fill_scalar, passthrough_arg_marshaler),
    # aten.slice_scatter.default: (custom_ops.slice_scatter, passthrough_arg_marshaler),
    # TODO: SHOOT conv2d for torch has shapes input (batch, Cin, H, W), weight (Cout, Cin, H, W),
    # but MLX conv2d has shapes input (batch, H, W, Cin), weight (Cout, H, W, Cin)
    # I'm not sure if I can naively add a call to swap dimensions into the model graph because
    # I don't know what kind of tensors I will be given, though the likely bet is that the dimensions
    # will be in the shape torch expects. In any case, skip this op for now and come back to it when you
    # aten.conv2d.default: (conv2d_bias, passthrough_arg_marshaler),
    aten.split.Tensor: (custom_split, passthrough_arg_marshaler),
    # aten.dropout.default: (custom_ops.passthrough, passthrough_arg_marshaler),
    aten.scaled_dot_product_attention.default: (custom_sdpa, sdpa_arg_marshaler),
    aten._scaled_dot_product_flash_attention_for_cpu.default: (
        custom_sdpa,
        sdpa_cpu_arg_marshaler,
    ),
    # this neeeds to be handled custom to dispatch properly on different types
    operator.getitem: (operator.getitem, passthrough_arg_marshaler),
    operator.add: (operator.add, passthrough_arg_marshaler),
    operator.neg: (operator.neg, passthrough_arg_marshaler),
    operator.lt: (operator.lt, passthrough_arg_marshaler),
    operator.ge: (operator.ge, passthrough_arg_marshaler),
    aten.layer_norm.default: (mx.fast.layer_norm, layernorm_arg_marshaler),
    aten.pow.Tensor_Scalar: (mx.power, passthrough_arg_marshaler),
    aten.mean.dim: (mx.mean, mean_arg_marshaler),
    aten.mean.default: (mx.mean, passthrough_arg_marshaler),
    aten.einsum.default: (mx.einsum, einsum_arg_marshaler),
    aten.detach.default: (mx.stop_gradient, passthrough_arg_marshaler),
    aten.any.dim: (mx.any, any_arg_marshaler),
    aten.sum.default: (mx.sum, sum_arg_marshaler),
    aten._softmax.default: (mx.softmax, _softmax_arg_marshaler),
    aten.new_ones.default: (custom_new_ones, passthrough_arg_marshaler),
    aten.bitwise_or.Tensor: (mx.bitwise_or, passthrough_arg_marshaler),
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


nn.LayerNorm


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
            ast.Import(names=[ast.alias(name="_operator", asname=None)]),
        ]
        # mlx function calls in order of execution
        # generates lines of python in the AST function body
        self.calls: List[ast.AST] = []
        self.device = mx.default_device()
        self.ret: ast.Return = None
        self.compiled_code: Optional[str] = None

    def lower_to_python(
        self, graph: Graph
    ) -> Callable[[List[mx.array]], List[mx.array]]:
        """
        Lowers a TorchFX aten graph into a Python AST calling MLX functions.
        The resultant compiled function receives and returns tuples of MLX arrays.
        """
        self.ingest_graph(graph)
        return self.export()

    def ingest_graph(self, graph: Graph):
        with open(os.path.expanduser("~/.cache/proteus/fx_graph.py"), "w") as f:
            f.write(graph.python_code({}).src)
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
            self.calls.append(ast_assign)
        elif custom_lowering := custom_lowerings_map.get(aten_op):
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
            name=COMPILED_FN_NAME,
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
        self.compiled_code = ast.unparse(module)
        logger.debug(f"Generated Python code:\n{self.compiled_code[:100]}...")
        namespace = {}
        exec(self.compiled_code, namespace)

        return namespace[mlx_func.name]
