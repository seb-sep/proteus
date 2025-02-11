import time
import cProfile
import pstats

import torch
from torch.fx import GraphModule
from torch.export import export
from functorch.compile import aot_function, aot_module_simplified, aot_module

import mlx.core as mx
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusion3Pipeline

from tqdm import tqdm

from proteus.proteus import proteus, proteus_no_compile
from proteus.utils.utils import (
    aten_opset_compiler,
    coerce_mx_to_torch,
    coerce_torch_to_mx,
)
from proteus.mlx_ast.mlx_builder import DefaultInterpreter
from tests.test_modules import (
    TestModule,
    cool_mlx_fn,
    SimpleTransformer,
    SimpleModule,
    EmbeddingModule,
)


def test_embed():
    vocab_size, embed_dim, h_dim = 1024, 2048, 256
    model = EmbeddingModule(vocab_size, embed_dim, h_dim)
    test_input = torch.randint(0, vocab_size, (16,))
    test_out = model(test_input)
    compiled_model = proteus_no_compile(model)
    compiled_out = compiled_model(test_input)
    print(compiled_out, test_out)


def test():
    in_dim, h_dim, out_dim = 4092, 2048, 256
    model = EmbeddingModule(in_dim, h_dim, out_dim)
    # model = SimpleModule(in_dim, h_dim, out_dim)
    test_input = torch.rand((16, in_dim))
    mlx_input = coerce_torch_to_mx(test_input)
    # for k, v in model.named_parameters():
    #     print(k, v)

    m = proteus_no_compile(model)
    out = m(test_input)
    print(out)
    exit()

    # print(f"MLX compiled model output: {coerce_mx_to_torch(m(test_input))}")
    print(f"Original PyTorch model output: {model(test_input)}")

    n_iters = 100
    start = time.time()
    for _ in tqdm(range(n_iters)):
        _ = model(test_input)
    stop = time.time()
    print(f"uncompiled model: {n_iters} iters in {stop-start} s")

    mlx_p1, mlx_p2 = tuple(
        coerce_torch_to_mx(tensor) for _, tensor in model.named_parameters()
    )
    start = time.time()
    for _ in tqdm(range(n_iters)):
        _ = m(test_input)
        # _ = cool_mlx_fn(mlx_input, mlx_p1, mlx_p2)
    stop = time.time()
    print(f"compiled model: {n_iters} iters in {stop-start} s")


def prof_compiled():
    in_dim, h_dim, out_dim = 4, 4, 4
    model = TestModule(in_dim, h_dim, out_dim)
    test_input = torch.rand((4, in_dim))
    m = proteus(model, test_input)
    mlx_input = coerce_torch_to_mx(test_input)

    for _ in range(100000):
        _ = m(mlx_input)


def prof_mlx():
    in_dim, h_dim, out_dim = 4, 4, 4
    model = TestModule(in_dim, h_dim, out_dim)
    test_input = torch.rand((4, in_dim))
    mlx_input = coerce_torch_to_mx(test_input)
    mlx_p1, mlx_p2 = tuple(
        coerce_torch_to_mx(tensor) for _, tensor in model.named_parameters()
    )
    for _ in range(100000):
        _ = cool_mlx_fn(mlx_input, mlx_p1, mlx_p2)


def cprof_compiled():
    cProfile.run("prof_compiled()", "compiled_res")
    print()
    cProfile.run("prof_mlx()", "mlx_res")
    print_profile_stats("compiled_res")
    print_profile_stats("mlx_res")


def print_profile_stats(profile_path: str):
    print(f"\nTop 10 slowest functions for {profile_path}:")
    p = pstats.Stats(profile_path)
    p.strip_dirs().sort_stats("cumulative").print_stats(50)


def simple_speed_test():
    a = torch.randn((4092, 4092))
    b = torch.randn((4092, 4092))

    a_m = coerce_torch_to_mx(a)
    b_m = coerce_torch_to_mx(b)

    a.to("mps")
    b.to("mps")

    mx.set_default_device(mx.gpu)
    mx.set_default_stream(mx.default_stream(mx.default_device()))
    start = time.time()
    c_m = mx.matmul(a_m, b_m)
    for _ in range(1000):
        c_m = mx.matmul(c_m, b_m)
    mx.eval(c_m)
    end = time.time()
    print(f"mlx in {end - start} s")

    start = time.time()
    c = a @ b
    for _ in range(1000):
        c = c @ b
    torch.mps.synchronize()
    end = time.time()
    print(f"pt in {end - start} s")


# cprof_compiled()
# simple_speed_test()
# test()


def transformer_opset():
    model = SimpleTransformer(512, 8, 2048)
    # compiled_graph = torch.compile(model, backend=aten_opset_compiler)
    compiled_graph = torch.compile(model, backend=aten_opset_compiler)
    _ = compiled_graph(torch.randn((16, 2, 512)))


compile_sd()
