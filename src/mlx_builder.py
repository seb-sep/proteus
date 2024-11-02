from typing import Callable, List, Dict, Any, Tuple, Union
import ast
from pprint import pprint
from inspect import getmodule

import torch
from torch import fx
import mlx.core as mx
import mlx.nn as nn

from src.arg_marshalers import passthrough_arg_marshaler


_fn_mapping = {
    torch.ops.aten.mm.default: (mx.matmul, passthrough_arg_marshaler),
    torch.ops.aten.t.default: (mx.transpose, passthrough_arg_marshaler),
    torch.ops.aten.transpose.int: (mx.transpose, passthrough_arg_marshaler),
    torch.ops.aten.expand.default: (mx.broadcast_to, passthrough_arg_marshaler),
    torch.ops.aten.silu.default: (nn.silu, passthrough_arg_marshaler),
    torch.ops.aten.triu.default: (mx.triu, passthrough_arg_marshaler),
    torch.ops.aten.mul.Tensor: (mx.multiply, passthrough_arg_marshaler),
    torch.ops.aten.add.Tensor: (mx.add, passthrough_arg_marshaler),
    torch.ops.aten.gt.Tensor: (mx.greater, passthrough_arg_marshaler),
    torch.ops.aten.neg.default: (mx.negative, passthrough_arg_marshaler),
    torch.ops.aten.cos.default: (mx.cos, passthrough_arg_marshaler),
    torch.ops.aten.sin.default: (mx.sin, passthrough_arg_marshaler),
    torch.ops.aten.rsqrt.default: (mx.rsqrt, passthrough_arg_marshaler),
    torch.ops.aten.cat.default: (mx.concatenate, passthrough_arg_marshaler),
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


def fn_to_manual_module(fn: Callable) -> List[str]:
    return {nn.silu: ["mlx", "nn", "silu"]}[fn]


def fn_to_attr(fn: Callable) -> Union[ast.Name, ast.Attribute]:
    """
    Convert a Python callable, likely a function from MLX, to an attribute or name AST
    to be used at an AST call site.
    """
    # use inspect.getmodule().__name__ to map these callables to the right AST attr
    # for example, module_strs might be ['mlx', 'linalg', 'matmul']
    if getmodule(fn) is None:
        module_strs = fn_to_manual_module(fn)
    else:
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
        self.mlx_import = ast.Import(names=[ast.alias(name="mlx", asname=None)])
        # mlx function calls in order of execution
        self.calls: List[ast.Call] = []
        self.device = mx.default_device()
        self.ret: ast.Return = None

    def addFunctionCall(self, aten_op, var_name, args: List, kwargs: Dict[str, Any]):
        """Ingest an aten operation and append an assign expression of the matching MLX fn to the AST body."""

        # ast.Load() means you're reading the value of the name, not setting or deleting it
        mlx_fn, arg_marshaler = aten_to_mlx(aten_op, self.device)
        print(mlx_fn)

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
        print(f"adding arg {arg_name}")
        self.in_args.append(ast.arg(arg=arg_name))

    def addReturn(self, arg: str):
        """
        Add a return to the end of the AST with the given variable name.
        NOTE: currently return only one value, even though you do it as a tuple
        """
        if isinstance(arg, fx.Node):
            # If arg is a torch.fx.Node, use its name attribute
            name = arg.name
        else:
            # If arg is not a torch.fx.Node, assume it's a string as before
            name = arg
        self.ret = ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id=arg, ctx=ast.Load()) for arg in [name]],
                ctx=ast.Load(),
            )
        )
        print(f"returning id {name}")

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

        module = ast.Module(body=[self.mlx_import, mlx_func], type_ignores=[])
        ast.fix_missing_locations(module)
        print(f"Generated Python code:\n{ast.unparse(module)}")
        code = compile(module, "<mlx_ast>", "exec")
        namespace = {}
        exec(code, namespace)

        return namespace[mlx_func.name]


class MLXCodeGenInterpreter(fx.Interpreter):
    """probably won't be using this but hold onto just in case"""

    def __init__(self, gm: fx.GraphModule):
        super().__init__(gm)

    def call_function(self, target, args, kwargs):
        """
        Find the MLX function corresponding to the aten op, and call it on the given argument
        (which should have already been registered as a function input from placeholder)
        that is, construct an ast.Call on the right mlx function with the inputs of the original node
        """
        print(f"Function target: {target}")
        pprint(args)
        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        print(f"Method target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        print(f"Module target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().call_module(target, args, kwargs)

    def get_attr(self, target, args, kwargs):
        print(f"Attr target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().get_attr(target, args, kwargs)

    def placeholder(self, target, args, kwargs):
        """
        top level graph inputs like so:
        %primals_1 : [num_users=1] = placeholder[target=primals_1]
        these will just be top level inputs in the ast
        so, these should be the inputs in our mlx function signature
        """
        print(f"Placeholder target: {target, type(target)}")
        # print(f"args {args}, kwargs {kwargs}")
        self.in_args.append(target)
        return super().placeholder(target, args, kwargs)

    def output(self, target, args, kwargs):
        print(f"Output target: {target}")
        print(f"args {args}, kwargs {kwargs}")
        return super().output(target, args, kwargs)
