import time

import mlx.core as mx
import torch

from proteus.utils import coerce_mx_to_torch, coerce_torch_to_mx


def bench_torch_to_mx(m, n, n_iters=1000):
    a = torch.randn((m, n))

    start = time.time()
    for _ in range(n_iters):
        b = coerce_torch_to_mx(a)
        mx.eval(b)
    stop = time.time()
    print(f"coercion in {stop - start} s")

    # Benchmark copying a to b
    start = time.time()
    for _ in range(n_iters):
        b = a.clone()
    stop = time.time()
    print(f"torch copy in {stop - start} s")

    # Benchmark assigning b to a
    start = time.time()
    for _ in range(n_iters):
        a = b
    stop = time.time()
    print(f"torch assign in {stop - start} s")


def bench_mx_to_torch(m, n, n_iters=1000):
    a = mx.random.normal((m, n))

    start = time.time()
    for _ in range(n_iters):
        b = coerce_mx_to_torch(a)
    stop = time.time()
    print(f"coercion in {stop - start} s")

    # Benchmark copying a to b
    start = time.time()
    for _ in range(n_iters):
        # okay this might not copy
        b = mx.array(a)
        mx.eval(b)
    stop = time.time()
    print(f"mlx copy in {stop - start} s")

    # Benchmark assigning b to a
    start = time.time()
    for _ in range(n_iters):
        b = a
        mx.eval(b)
    stop = time.time()
    print(f"mlx assign in {stop - start} s")


if __name__ == "__main__":
    # bench_torch_to_mx(1024, 2048, 1000)
    bench_mx_to_torch(1024, 2048, 1000)