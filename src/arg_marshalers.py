import ast
from typing import Dict, List, Tuple
from pprint import pprint

from torch import fx


def passthrough_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs directly to an AST arg list and keyword list respectively."""

    # Convert args to ast.Name nodes
    # remember that your arg names are just SSA values from the fx graph so just take the name
    # and assume its already been registered as a variable somewhere in the AST
    ast_args = []
    print(args)
    for arg in args:
        if isinstance(arg, fx.Node):
            ast_args.append(ast.Name(id=arg.name, ctx=ast.Load()))
        elif isinstance(arg, (fx.immutable_collections.immutable_list, tuple, int)):
            # TODO: This is as SUPER hacky way to get the list
            parsed = ast.parse(repr(arg))
            print(parsed)
            if isinstance(parsed.body[0], ast.Expr):
                ast_args.append(parsed.body[0].value)
            else:
                raise ValueError(f"Failed to parse fx immutable list {parsed}")
        else:
            raise ValueError(
                f"Unexpected AST structure for argument: {arg} of type {type(arg)}"
            )

    # Convert kwargs to ast.keyword nodes
    ast_kwargs = [
        ast.keyword(arg=k, value=ast.Name(id=v, ctx=ast.Load()))
        for k, v in kwargs.items()
    ]

    return ast_args, ast_kwargs


def take_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs from torch.select to mx.take."""

    # Keep the tensor arg passed first, but swap the subsequent dim and index args

    return passthrough_arg_marshaler((args[0], args[2], args[1]), kwargs)


# def cat_marshaler(args: List, kwargs: Dict) -> Tuple[List[ast.AST], List[ast.keyword]]:
#     """Map input args and kwargs from torch.cat to mlx.concatenate."""
#     for k, v in kwargs.items():
#         print(k, v)
#     return passthrough_arg_marshaler(args, {"axis": kwargs["dim"]})
