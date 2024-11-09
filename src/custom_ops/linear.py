import mlx.core as mx


def linear(x: mx.array, w: mx.array, b: mx.array):
    if b is not None:
        out = mx.addmm(b, x, w.T)
    else:
        out = x @ w.T
    return out
