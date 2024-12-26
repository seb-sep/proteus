import ast
from typing import Dict, List, Tuple
from pprint import pprint

from torch import fx
import torch


def passthrough_arg_marshaler(
    args: List[fx.Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs directly to an AST arg list and keyword list respectively."""

    # Convert args to ast.Name nodes
    # remember that your arg names are just SSA values from the fx graph so just take the name
    # and assume its already been registered as a variable somewhere in the AST
    def convert_arg_to_ast(arg):
        if isinstance(arg, fx.Node):
            return ast.Name(id=arg.name, ctx=ast.Load())
        elif isinstance(arg, torch.dtype):
            return ast.Attribute(
                value=ast.Name(id="mlx", ctx=ast.Load()), attr="float16", ctx=ast.Load()
            )
        # elif isinstance(arg, torch.device):
        #     return ast.Attribute(
        #         value=ast.Name(id="mx", ctx=ast.Load()), attr="gpu", ctx=ast.Load()
        #     )
        elif isinstance(
            arg,
            (
                fx.immutable_collections.immutable_list,
                tuple,
                str,
                int,
                float,
                type(None),
            ),
        ):
            # TODO: This is as SUPER hacky way to get the list
            parsed = ast.parse(repr(arg))
            if isinstance(parsed.body[0], ast.Expr):
                return parsed.body[0].value
            else:
                raise ValueError(f"Failed to parse fx immutable list {parsed}")
        else:
            print(f"Unexpected arg: {arg}")
            raise ValueError(
                f"Unexpected AST structure for argument: {arg} of type {type(arg)}"
            )

    # Convert args
    ast_args = []
    for arg in args:
        ast_arg = convert_arg_to_ast(arg)
        if ast_arg is not None:
            ast_args.append(ast_arg)
    # Convert kwargs to ast.keyword nodes
    ast_kwargs = [
        ast.keyword(arg=k, value=convert_arg_to_ast(v)) for k, v in kwargs.items()
    ]

    return ast_args, ast_kwargs


def t_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """For aten.t.default, which we're assuming is only for 2D tensors."""
    return passthrough_arg_marshaler(args, {"axes": (1, 0)})


def take_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs from torch.select to mx.take."""

    # Keep the tensor arg passed first, but swap the subsequent dim and index args

    return passthrough_arg_marshaler((args[0], args[2], args[1]), kwargs)


def clone_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs from torch.clone to __clone__."""

    # Keep the tensor arg passed first, but swap the subsequent dim and index args

    return passthrough_arg_marshaler(args, {})


def arange_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs from torch.clone to __clone__."""

    # Copy kwargs but remove 'device' key if present
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k != "device" and not isinstance(v, (torch.device, torch.dtype))
    }
    print(filtered_kwargs)

    return passthrough_arg_marshaler(args, filtered_kwargs)


# def cat_marshaler(args: List, kwargs: Dict) -> Tuple[List[ast.AST], List[ast.keyword]]:
#     """Map input args and kwargs from torch.cat to mlx.concatenate."""
#     for k, v in kwargs.items():
#         print(k, v)
#     return passthrough_arg_marshaler(args, {"axis": kwargs["dim"]})
