from typing import List, Dict, Union
from pprint import pprint
import time

import torch.nn as nn
import torch.fx as fx
import torch
from functorch.compile import aot_function, aot_module_simplified, aot_module
from functorch.experimental.control_flow import cond
import torch.utils._pytree as pytree
import torch._dynamo as dynamo
from torch.fx.experimental.proxy_tensor import make_fx

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

    def __call__(self, args):
        # if you've already compiled to an mlx fn, then you need to pass the input tensors to MLX as mlx arrays
        # return self.compiled_fn(
        #     self.named_params, self.named_buffers, args
        # )  # , **kwargs
        if self.mlx_mode:
            # args = tuple(coerce_torch_to_mx(tensor) for tensor in args)
            # kwargs = {k: coerce_torch_to_mx(v) for k, v in kwargs.items()}
            print(f"coerced args to mlx, about to call compiled fn for real")
            # print(type(self.compiled_fn))
            out = self.compiled_fn(self.named_params, self.named_buffers, args)
            print(out)
            return out
            # return self.compiled_fn(*args)
        else:
            self.mlx_update(args)
            return self(
                args,
            )

    def mlx_update(self, args, **kwargs):
        """
        Compile the function on the given sample inputs and prepare the module
        for inference with the MLX function.
        """
        # prompt compilation on sample inputs
        # print(args, kwargs)
        print("invoking fn on pytorch args to compile: ")
        params_and_buffers = list(self.named_params.values()) + list(
            self.named_buffers.values()
        )
        a = self.compiled_fn(self.named_params, self.named_buffers, args)  # , **kwargs
        b = self.compiled_fn(self.named_params, self.named_buffers, args)  # , **kwargs
        # print(self.compiled_fn)
        # target=torch.ops.higher_order.wrap_with_set_grad_enabled]
        # _ = self.compiled_fn(self.named_params, self.named_buffers, *args, **kwargs)
        # print("after compilation: ")
        # print(self.compiled_fn)

        # After the pytorch tensors were used for compilation, convert them
        # to MLX arrays for use in the final fn
        # print("coercing params to mlx")
        # self.named_params = tuple(
        #     coerce_torch_to_mx(v) for v in self.named_params.values()
        # )
        # self.named_buffers = tuple(
        #     coerce_torch_to_mx(v) for v in self.named_buffers.values()
        # )
        self.mlx_mode = True
        # print("went into mlx mode")
        # _coerce_args_to_mlx.disable_coercion()


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
def mlx_compiler(
    static_mlx_params_buffers: Dict[Union[nn.Parameter, torch.Tensor], mx.array]
):
    """
    Compile the given FX graph of aten ops into a Python function calling MLX operations.
    """

    def _mlx_compiler(gm: fx.GraphModule, sample_inputs):

        # print(gm.graph)
        aten_graph = make_fx(gm)(*sample_inputs)
        # print(aten_graph.graph)

        builder = MLXASTBuilder()
        builder.ingest_graph(aten_graph.graph)

        # TODO: when to compile vs not? could autotune lmao
        mlx_fn = mx.compile(builder.export())

        # Wrap the MLX function to convert the appropriate inputs and outputs to MLX arrays
        # TODO: is there any way to avoid unpacking and repacking the args tuple on each forward call?
        # YES could inline in the codegenned MLX fn, but this is an optimization for later
        def torch_wrapper(*args):
            mlx_args = tuple(
                (
                    mlx_param
                    if ((mlx_param := static_mlx_params_buffers.get(arg)) is not None)
                    else coerce_torch_to_mx(arg)
                )
                for arg in args
            )

            outs = mlx_fn(*mlx_args)
            return tuple(coerce_mx_to_torch(out) for out in outs)

        return torch_wrapper

    return _mlx_compiler


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
    return compiled_mod


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


def proteus_v3(mod: nn.Module):

    mod.eval()

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    compiled_fn = torch._dynamo.optimize(
        backend=mlx_compiler,
    )(functional_call)

    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    compiled_mod = MLXCompiledModule(named_params, named_buffers, compiled_fn)
    return compiled_mod


# this needs to be a one at a time function to prevent high memory utilization
# from having all tensors copied at once


class ParameterWithName(nn.Parameter):
    def __new__(cls, data=None, name="", *, requires_grad=True):
        # First create the parameter instance using parent's __new__
        instance = super().__new__(cls, data, requires_grad)

        # For both standard tensors and custom tensors, we need to add our custom attribute
        # Note: We add it in __new__ because custom tensor path returns a different instance
        # that won't go through our __init__
        instance.name_in_module = name
        return instance

    def __init__(self, data=None, name="", *, requires_grad=True):
        # No need to call super().__init__ since Parameter doesn't define it
        # The name_in_module attribute is already set in __new__
        pass


_global_param_lookup_table: Dict[int, mx.array] = {}


def replace_param_with_mlx(
    mod: nn.Module,
    static_mlx_parameters: Dict[nn.Parameter, mx.array],
    name: str,
    param: nn.Parameter,
):
    """
    Populate static_mlx_parameters with the name and MLX array-ified parameter from the nn.Module,
    unset the parameter on the module, and reset it with the 'digital twin' of the MLX array
    (a torch tensor initialized with the same data pointer as the MLX array)
    """

    # I guess this should just work with params instead of tensors??? if not we can cast to the superclass
    mlx_param = coerce_torch_to_mx(param)
    torch_digital_twin = nn.Parameter(coerce_mx_to_torch(mlx_param))
    static_mlx_parameters[torch_digital_twin] = mlx_param

    # find and unset the parameter, then re-set it with the digital twin
    if "." not in name:
        delattr(mod, name)
        mod.register_parameter(name, torch_digital_twin)
    else:
        submod_path, param_subname = name.rsplit(".", 1)
        submod = mod.get_submodule(submod_path)
        delattr(submod, param_subname)
        submod.register_parameter(param_subname, torch_digital_twin)


def replace_buffer_with_mlx(
    mod: nn.Module,
    static_mlx_buffers: Dict[torch.Tensor, mx.array],
    name: str,
    buffer: torch.Tensor,
):
    """
    Copy of replace_param_with_mlx that works on buffers
    """
    mlx_buf = coerce_torch_to_mx(buffer)
    torch_digital_twin = coerce_mx_to_torch(mlx_buf)
    static_mlx_buffers[torch_digital_twin] = mlx_buf

    # find and unset the buffer
    if "." not in name:
        delattr(mod, name)
        mod.register_buffer(name, torch_digital_twin)
    else:
        submod_path, buffer_subname = name.rsplit(".", 1)
        submod = mod.get_submodule(submod_path)
        delattr(submod, buffer_subname)
        submod.register_buffer(buffer_subname, torch_digital_twin)


def proteus_v4(mod: nn.Module):

    mod.eval()

    # generate static mlx array 'twins' of all the params in the module
    # no real overhead between making keys tensors or id() ints, since
    # torch.Tensor's __hash__ just returns the ID
    static_mlx_parameters_buffers: Dict[Union[nn.Parameter, torch.Tensor], mx.array] = (
        {}
    )

    # collect into a list aot to avoid issues w/mutation
    # let's ignore buffers for now
    for name, param in list(mod.named_parameters()):
        replace_param_with_mlx(mod, static_mlx_parameters_buffers, name, param)
    for name, buf in list(mod.named_buffers()):
        replace_buffer_with_mlx(mod, static_mlx_parameters_buffers, name, buf)
    # at this point the module SHOULD work just as before

    compiled_fn = torch.compile(
        mod, backend=mlx_compiler(static_mlx_parameters_buffers)
    )
    return compiled_fn
