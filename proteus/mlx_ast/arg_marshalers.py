import ast
from typing import Dict, List, Tuple, Union
from math import sqrt
import sys

from torch.fx import Node, Graph, immutable_collections
import torch

from proteus.utils import torch_dtype_map


def module_strs_to_ast(module_strs: List[str]) -> Union[ast.Name, ast.Attribute]:
    """
    Convert some Python module attribute string to its respective AST, where the
    attribute string is a list of strings representing attr accesses.
    For example: `mlx.core.float16` would be passed as `['mlx', 'core', 'float16']`.
    """

    attr_ast = ast.Name(id=module_strs.pop(0), ctx=ast.Load())
    while module_strs:
        attr_ast = ast.Attribute(
            value=attr_ast, attr=module_strs.pop(0), ctx=ast.Load()
        )
    return attr_ast


def convert_arg_to_ast(arg) -> ast.AST:
    if isinstance(arg, Node):
        return ast.Name(id=arg.name, ctx=ast.Load())
    elif isinstance(arg, torch.dtype):
        mlx_dtype = torch_dtype_map[arg]
        return module_strs_to_ast(str(mlx_dtype).split("."))
    # for now, force devices to be gpu (after all, we don't know why some ops might have been)
    # on cpu for macos in the first place, probably crap about unsupported ops on mac gpu
    elif isinstance(arg, torch.device):
        return module_strs_to_ast(["mlx", "core", "gpu"])
    elif isinstance(arg, (immutable_collections.immutable_list, tuple)):
        return ast.Tuple(elts=[convert_arg_to_ast(x) for x in arg], ctx=ast.Load())
    elif isinstance(arg, list):
        return ast.List(elts=[convert_arg_to_ast(x) for x in arg], ctx=ast.Load())
    elif isinstance(
        arg,
        (
            str,
            int,
            float,
            type(None),
        ),
    ):
        return ast.Constant(value=arg)
    else:
        raise ValueError(
            f"Unexpected AST structure for argument: {arg} of type {type(arg)}"
        )


def sum_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.sum.default."""
    # only thing to do is pluck out dtype kwarg, which is only kwarg for default
    return passthrough_arg_marshaler(args)


def any_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.any.dim."""
    # happens to be the same as mean_arg_marshaler
    return mean_arg_marshaler(args, kwargs)


def slice_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.slice.Tensor."""

    # if a passed value (typically end) is the system max value, this should
    # be interpreted as no end and replaced with None
    # also need to pack dim args into a single tuple to pass to __getitem__?
    tensor, *args = (arg if arg != sys.maxsize else None for arg in args)
    tensor_slice = slice(*args)
    return passthrough_arg_marshaler((tensor, tensor_slice), kwargs)


def full_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.full.default."""

    # pluck out pin_memory param
    # cannot just pop because kwargs is immutable dict
    kwargs = {k: v for k, v in kwargs.items() if k not in ("pin_memory", "device")}
    return passthrough_arg_marshaler(args, kwargs)


def _softmax_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten._softmax.default."""

    tensor = args[0]
    axis = kwargs.get("dim", args[1])

    return passthrough_arg_marshaler((tensor, axis))


def arange_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.arange.{start, default}."""
    # pluck out any kwargs that might be passed, namely device and pin_memory
    # because we (probably) don't care about them
    # NOTE: are there certain situations in mlx where it makes sense to offload some work
    # to cpu? conventionally, you keep small cpu-used tensors on cpu because overhead is bandwidth
    # but I wonder how much this applies with a unified memory arch, esp when there might be overhead
    # switching between streams in mlx. this will require further investigation

    return passthrough_arg_marshaler(
        args, {k: v for k, v in kwargs.items() if k not in ("device", "pin_memory")}
    )


def mean_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.mean.dim."""

    tensor, *rest = args

    lrest = len(rest)
    dim = kwargs.get("dim") if lrest < 1 else rest[0]
    keepdim = kwargs.get("keepdim", False) if lrest < 2 else rest[1]

    # TODO: for now, only support no dtype passed until we need it
    dtype = kwargs.get("dtype")
    assert dtype is None

    return passthrough_arg_marshaler((tensor, dim, keepdim))


def einsum_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs for aten.einsum.default."""
    equation, tensors = args
    return passthrough_arg_marshaler((equation, *tensors), kwargs)


def passthrough_arg_marshaler(
    args: List[Node] = [], kwargs: Dict = {}
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs directly to an AST arg list and keyword list respectively."""

    # Convert args to ast.Name nodes
    # remember that your arg names are just SSA values from the fx graph so just take the name
    # and assume its already been registered as a variable somewhere in the AST
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


def layernorm_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """
    Map input args and kwargs for aten.layernorm.default.

    NOTE: pytorch's layer norm normalizes over the last D dimensions,
    while MLX only normalizes over the last dim,
    so in the case where more than one dim is passed, we're cooked
    and will have to implement one ourselves
    """

    _input, normalized_shape, weight, bias = args
    eps = kwargs.get("eps", 1e-5)

    if isinstance(normalized_shape, (list, tuple)):
        # mlx only supports layer norm over one dim
        assert len(normalized_shape) == 1

    return passthrough_arg_marshaler((_input, weight, bias, eps))


def sdpa_cpu_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    return sdpa_arg_marshaler(args, {**kwargs, "is_cpu": True})


def sdpa_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """
    Transform args for aten.scaled_dot_product_attention.default.

    NOTE that torch sdpa can take in a boolean attention mask, while mlx seems to
    only work with float masks.
    """
    q, k, v = args
    attn_mask = kwargs.get("attn_mask")
    is_causal = kwargs.get("is_causal", False)
    dropout_p = kwargs.get("dropout_p", 0)
    is_cpu = kwargs.get("is_cpu", False)

    tensor_key = "val" if "val" in q.meta else "example_value"
    assert isinstance(q, Node) and (
        q.type == torch.Tensor or isinstance(q.meta.get(tensor_key), torch.Tensor)
    )
    q_shape = q.meta[tensor_key].shape
    scale = kwargs.get("scale", sqrt(q_shape[-1]) ** -1)

    # torch sdpa takes in a dropout p but mlx does not, so only support no dropout
    # in future, this could be simply unfused by injecting a dropout later afterwards
    # (though likely not very efficient)
    assert dropout_p == 0

    return passthrough_arg_marshaler(
        (q, k, v),
        {
            "scale": scale,
            "attn_mask": attn_mask,
            "is_causal": is_causal,
            "is_cpu": is_cpu,
        },
    )


def triangle_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """For aten.{triu, tril}.default"""
    if len(args) == 1:
        if "diagonal" in kwargs:
            args = (args[0], kwargs["diagonal"])

    return passthrough_arg_marshaler(args)


def t_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """For aten.t.default, which we're assuming is only for 2D tensors."""
    return passthrough_arg_marshaler(args, {"axes": (1, 0)})


def take_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs from torch.select to mx.take."""

    # Keep the tensor arg passed first, but swap the subsequent dim and index args

    return passthrough_arg_marshaler((args[0], args[2], args[1]), kwargs)


def clone_arg_marshaler(
    args: List, kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Map input args and kwargs from torch.clone to __clone__."""

    # Keep the tensor arg passed first, but swap the subsequent dim and index args

    return passthrough_arg_marshaler(args)


def transpose_int_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Convert args of aten.transpose.int op."""
    assert len(kwargs) == 0 and len(args) == 3
    tensor, a1, a2 = args
    tensor_key = "val" if "val" in tensor.meta else "example_value"
    assert (
        (
            tensor.type == torch.Tensor
            or isinstance(tensor.meta.get(tensor_key), torch.Tensor)
        )
        and (isinstance(a1, int) or a1.type == torch.SymInt)
        and (isinstance(a2, int) or a2.type == torch.SymInt)
    )
    return passthrough_arg_marshaler(args)


def expand_arg_marshaler(
    args: List[Node], kwargs: Dict
) -> Tuple[List[ast.AST], List[ast.keyword]]:
    """Convert args of aten.expand.default op."""

    assert len(kwargs) == 0
    tensor, sizes = args
    assert isinstance(sizes, (tuple, list))
    return passthrough_arg_marshaler(args)


# def cat_marshaler(args: List, kwargs: Dict) -> Tuple[List[ast.AST], List[ast.keyword]]:
#     """Map input args and kwargs from torch.cat to mlx.concatenate."""
#     for k, v in kwargs.items():
#         print(k, v)
#     return passthrough_arg_marshaler(args, {"axis": kwargs["dim"]})
