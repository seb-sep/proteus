from typing import List, Union

import mlx.core as mx


def expand(x: mx.array, shape: List[int]) -> mx.array:
    shape = [x.shape[i] if val == -1 else val for i, val in enumerate(shape)]
    return mx.broadcast_to(x, shape)


def custom_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    attn_mask: Union[mx.array, None],
    is_causal: bool,
    scale: bool,
    is_cpu: bool,
):
    if attn_mask is not None and attn_mask.dtype == mx.bool_:
        attn_mask = mx.where(attn_mask, 0, -float("inf"))
    if is_causal:
        attn_mask = mx.triu(
            mx.full((q.shape[-2], k.shape[-2]), -float("inf"), dtype=q.dtype), k=1
        )

    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_mask)
    return (attn, mx.array([])) if is_cpu else attn


def custom_split(
    a: mx.array, split_size_or_sections: Union[int, List[int]], dim: int = 0
):
    """
    Like torch.split() on MLX arrays.

    TODO: if the dimensions of a are sufficiently statically known at compile time
    this can be inlined in instead of doing runtime arg conversions to what MLX expects
    """

    if isinstance(split_size_or_sections, int):
        sections = a.shape[dim] // split_size_or_sections
        return mx.split(a, sections, dim)
    indices = tuple(
        sum(split_size_or_sections[:i]) - 1
        for i in range(1, len(split_size_or_sections) + 1)
    )
    return mx.split(a, indices, dim)


def custom_take(a: mx.array, dim: int, index: int):
    index = a.shape[dim] - 1 if index == -1 else index
    return mx.take(a, index, dim)


def custom_new_ones(
    a: mx.array, size, dtype=None, layout=None, device=None, pin_memory=None
):
    # ignore rest of args for now
    return mx.ones(
        size,
        dtype=(dtype if dtype is not None else a.dtype),
    )


# NOTE: whether or not you will have a bias is technically statically known,
# so whether to use this function or plain mx.conv2d should be knowable aot
# preventing the runtime check from being necessary
def conv2d_bias(_input, weight, bias, **kwargs):
    conv = mx.conv2d(_input, weight, **kwargs)
    return conv + bias if bias is not None else conv


def linear(x: mx.array, w: mx.array, b: Union[mx.array, None]):
    if b is not None:
        out = mx.addmm(b, x, w.T)
    else:
        out = x @ w.T
    return out


def slice(
    x: mx.array,
    dim: int = 0,
    start: Union[int, None] = None,
    end: Union[int, None] = None,
    step: int = 1,
) -> mx.array:
    """
    Slices the input array along a given dimension using mx.take.

    Args:
        x: Input array
        dim: Dimension to slice along
        start: Starting index
        end: Ending index (exclusive)
        step: Step size (default: 1)
    """
    indices = mx.arange(start if start is not None else 0, end, step)
    return mx.take(x, indices, axis=dim)


def masked_fill_scalar(_self: mx.array, mask: mx.array, value):
    """Implementation of aten.masked_fill.Scalar."""
    return mx.where(mask, value, _self)
