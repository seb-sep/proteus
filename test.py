import time
import cProfile
import pstats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule
from torch.export import export

import mlx
import mlx.core as mx
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from src.proteus import proteus
from src.utils import aten_opset_compiler, coerce_mx_to_torch, coerce_torch_to_mx


class TestModule(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h_dim, bias=False)
        self.fc2 = nn.Linear(h_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = torch.triu(x)
        x = self.fc2(x)
        x = torch.sin(x)
        # x = torch.cat((x, x), dim=1)
        return x


def cool_mlx_fn(primals_1, primals_2, primals_3):
    t = mx.transpose(primals_1)
    mm = mx.matmul(primals_2, t)
    silu = mlx.nn.silu(mm)
    triu = mx.triu(silu)
    t_1 = mx.transpose(primals_3)
    mm_1 = mx.matmul(triu, t_1)
    sin = mx.sin(mm_1)
    return (mx.eval(sin),)


def test():
    in_dim, h_dim, out_dim = 4092, 2048, 256
    # in_dim, h_dim, out_dim = 4, 4, 4
    model = TestModule(in_dim, h_dim, out_dim)
    test_input = torch.rand((in_dim, in_dim))
    mlx_input = coerce_torch_to_mx(test_input)
    # for k, v in model.named_parameters():
    #     print(k, v)

    # m = torch.compile(model, backend=proteus)
    # _ = aot_export_module(model, test_input, trace_joint=False)
    m = proteus(model, test_input)
    print(f"MLX compiled model output: {coerce_mx_to_torch(m(test_input))}")
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
        _ = m(mlx_input)
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


def compile_llama():
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

    text = "Proteus is a cracked PyTorch compiler"
    test_in = tokenizer(text, return_tensors="pt")

    # Set the model as our test model
    test_out = model(test_in["input_ids"], attention_mask=test_in["attention_mask"])

    # use export().run_decompositions() to get core aten ir graph
    # without lifing model params into inputs
    graph: GraphModule = (
        export(
            model,
            (test_in["input_ids"],),
            kwargs={"attention_mask": test_in["attention_mask"]},
            strict=False,
        )
        .run_decompositions()
        .module()
    )

    compiled_graph = torch.compile(graph, backend=aten_opset_compiler)
    _ = compiled_graph(test_in["input_ids"], attention_mask=test_in["attention_mask"])


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
test()
