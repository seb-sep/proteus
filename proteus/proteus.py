from typing import List, Dict
from pprint import pprint

import torch.nn as nn
import torch.fx as fx
import torch
from functorch.compile import aot_function, aot_module_simplified, aot_module
from functorch.experimental.control_flow import cond
import torch.utils._pytree as pytree
import torch._dynamo as dynamo

import mlx.core as mx

from proteus.mlx_builder import MLXASTBuilder
from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch

MLX_DEVICE = mx.default_device()


class MLXCoercer:
    def __init__(self):
        self.disabled = False

    def __call__(self, args):
        if self.disabled:
            return args
        return [coerce_torch_to_mx(tensor) for tensor in args]

    def disable_coercion(self):
        self.disabled = True


# Globally scoped so that coercion on all args can be activated for
# the first run through the compiled fn, and then disabled for all future inputs
_coerce_args_to_mlx = MLXCoercer()


class MLXCompiledModule:
    """
    Wraps over the AOTAutograd-compiled MLX function to pass in the MLXified parameters and buffers.
    TODO: Perhaps an optimization could be to track the # of named params and buffers, and instead of looping over each arg,
    only loop over user passed args
    """

    def __init__(
        self,
        named_params: Dict[str, nn.Parameter],
        named_buffers: Dict[str, torch.Tensor],
        compiled_fn,
    ):
        self.named_params = named_params
        self.named_buffers = named_buffers
        self.mlx_mode = False
        self.compiled_fn = compiled_fn
        # self.config = compiled_fn.config

    def __call__(self, *args, **kwargs):
        # if you've already compiled to an mlx fn, then you need to pass the input tensors to MLX as mlx arrays
        if self.mlx_mode:
            args = tuple(coerce_torch_to_mx(tensor) for tensor in args)
            kwargs = {k: coerce_torch_to_mx(v) for k, v in kwargs.items()}
            print(f"coerced args to mlx, about to call compiled fn for real")
            return self.compiled_fn(
                *(self.named_params + self.named_buffers + args)  # , **kwargs
            )
            # return self.compiled_fn(*args, **kwargs)
        else:
            self.mlx_update(*args, **kwargs)
            return self(*args, **kwargs)

    def mlx_update(self, *args, **kwargs):
        """
        Compile the function on the given sample inputs and prepare the module
        for inference with the MLX function.
        """
        # prompt compilation on sample inputs
        # print(args, kwargs)
        print("invoking fn on pytorch args to compile: ")
        # print(self.compiled_fn)
        # target=torch.ops.higher_order.wrap_with_set_grad_enabled]
        exported_graph = torch.export.export(self.compiled_fn, args, kwargs)
        self.compiled_fn = mlx_compiler(exported_graph, None)
        # _ = self.compiled_fn(self.named_params, self.named_buffers, *args, **kwargs)
        # print("after compilation: ")
        # print(self.compiled_fn)

        # After the pytorch tensors were used for compilation, convert them
        # to MLX arrays for use in the final fn
        print("coercing params to mlx")
        self.named_params = tuple(
            coerce_torch_to_mx(v) for v in self.named_params.values()
        )
        self.named_buffers = tuple(
            coerce_torch_to_mx(v) for v in self.named_buffers.values()
        )
        self.mlx_mode = True
        print("went into mlx mode")
        _coerce_args_to_mlx.disable_coercion()


def params_to_mlx(mod: nn.Module) -> List[mx.array]:
    """
    Pull the named parameters and buffers from a module
    and turn them into a list of MLX arrays, in the SAME
    way that aot_export_module does it. These will be passed first
    to each invocation of the generated MLX fn.
    """

    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    return tuple(coerce_torch_to_mx(param) for param in params_and_buffers_flat)


first_compile = False


def to_aten_compiler(gm: fx.GraphModule, sample_inputs):
    print("to aten compiler")
    fn = aot_module_simplified(gm, sample_inputs, mlx_compiler)
    return fn


# Close over the mlx_params to pass in the final compiled fn
def mlx_compiler(gm: fx.GraphModule, _):
    """
    Compile the given FX graph of aten ops into a Python function
    calling MLX operations. Second argument is for matching the signature expected by AOTAutograd.
    """
    with open("graph.txt", "w") as f:
        f.write(str(gm.graph))
    print("Graph module variables:")
    for var_name in vars(gm):
        print(f"  {var_name}")
    print("end of vars")
    builder = MLXASTBuilder()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            builder.addArgument(node.target)
        elif node.op == "call_function":
            builder.addFunctionCall(node.target, node.name, node.args, node.kwargs)
        elif node.op == "output":
            # the first value in args for output is what to actually return from the graph i think,
            # we might actually only care about the first value in that tuple
            # https://pytorch.org/docs/stable/fx.html#torch.fx.Node
            # builder.addReturn(node.args[0][0])
            builder.addReturn(node.args[0])
        elif node.op == "get_attr":
            attr = getattr(gm._graph_module, node.target)
            print(attr)
            pprint(vars(node))
            raise ValueError(f"unhandled getattr")

        else:
            raise ValueError(f"unhandled node type: node {node, node.op}")

    # TODO: when to compile vs not? could autotune lmao
    mlx_fn = mx.compile(builder.export())

    # Wrap the MLX function to convert the appropriate inputs and outputs to MLX arrays
    # TODO: is there any way to avoid unpacking and repacking the args tuple on each forward call?
    def torch_wrapper(*args):
        mlx_args = _coerce_args_to_mlx(args)
        # print(mlx_args)
        outs = mlx_fn(*mlx_args)
        # pprint(outs)
        return tuple(coerce_mx_to_torch(out) for out in outs)

    return torch_wrapper


def proteus(mod: nn.Module):
    """
    Compile an eagerly computing PyTorch module to a function which computes in MLX under the hood.
    """

    # HUGE DISCOVERY: to raise the inputs, you literally just get mod.named_parameters() and mod.named_buffers(), turn into a single dict,
    # and pytree flatten
    # This is guaranteed to produce the same ordering each time
    # the following is shamelessly pulled from aot autograd source code
    # https://github.com/pytorch/pytorch/blob/3d3551506d4acccdd06d6f98eaf05e5288d254b3/torch/_functorch/aot_autograd.py#L1190

    # print(f"{len(sample_inputs)} sample inputs:")
    # pprint(sample_inputs)
    # mod.print_readable()
    # exit()

    # of COURSE this will proc when using torch compile: it can handle graph breaks
    # so it will preserve the full Python semantics instead of just swapping out
    # for a full graph, which IS the behavior you want
    mod.eval()

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        print("running functional call")
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    # Hold onto these for the FIRST tracing call, then swap out with the MLXified args
    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)

    # I want to use aot_function so I have control over how the inputs are raised and passed
    # compiled_fn = aot_function(
    #     functional_call,
    #     fw_compiler=mlx_compiler,
    #     num_params_buffers=num_params_buffers,
    #     dynamic=True,
    # )
    # dynamo.optimize
    compiled_fn = torch.compile(functional_call, backend=to_aten_compiler)
    # compiled_fn = to_aten_compiler()
    compiled_mod = MLXCompiledModule(named_params, named_buffers, compiled_fn)
    return compiled_fn


def proteus_no_compile(mod: nn.Module):
    """
    Compile an eagerly computing PyTorch module to a function which computes in MLX under the hood.
    """

    # HUGE DISCOVERY: to raise the inputs, you literally just get mod.named_parameters() and mod.named_buffers(), turn into a single dict,
    # and pytree flatten
    # This is guaranteed to produce the same ordering each time
    # the following is shamelessly pulled from aot autograd source code
    # https://github.com/pytorch/pytorch/blob/3d3551506d4acccdd06d6f98eaf05e5288d254b3/torch/_functorch/aot_autograd.py#L1190

    # print(f"{len(sample_inputs)} sample inputs:")
    # pprint(sample_inputs)
    # mod.print_readable()
    # exit()

    # of COURSE this will proc when using torch compile: it can handle graph breaks
    # so it will preserve the full Python semantics instead of just swapping out
    # for a full graph, which IS the behavior you want
    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        print("running functional call")
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    # Hold onto these for the FIRST tracing call, then swap out with the MLXified args
    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)

    # I want to use aot_function so I have control over how the inputs are raised and passed
    # compiled_fn = torch.export.export(
    #     functional_call,
    #     fw_compiler=mlx_compiler,
    #     num_params_buffers=num_params_buffers,
    #     dynamic=True,
    # )
    # dynamo.optimize
    # compiled_fn = to_aten_compiler()
    compiled_mod = MLXCompiledModule(named_params, named_buffers, mod)
    return compiled_mod
