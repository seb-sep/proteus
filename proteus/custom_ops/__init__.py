from typing import List, Union

import mlx.core as mx


def expand(x: mx.array, shape: List[int]) -> mx.array:
    shape = [x.shape[i] if val == -1 else val for i, val in enumerate(shape)]
    return mx.broadcast_to(x, shape)


def linear(x: mx.array, w: mx.array, b: Union[mx.array, None]):
    if b is not None:
        out = mx.addmm(b, x, w.T)
    else:
        out = x @ w.T
    return out


def slice(x: mx.array, dim: int, start: int, end: int, step: int = 1) -> mx.array:
    """
    Slices the input array along a given dimension using mx.take.

    Args:
        x: Input array
        dim: Dimension to slice along
        start: Starting index
        end: Ending index (exclusive)
        step: Step size (default: 1)
    """
    indices = mx.arange(start, end, step)
    return mx.take(x, indices, axis=dim)


def masked_fill(x: mx.array, mask: mx.array, value: float) -> mx.array:
    """
    Fills elements of input tensor x with value where mask is True.

    Args:
        x: Input array
        mask: Boolean mask array (must be broadcastable to x)
        value: Value to fill with

    Returns:
        Array with masked values filled
    """
    return mx.where(mask, value, x)


def slice_scatter(
    input: mx.array, src: mx.array, dim: int, start: int, end: int = None, step: int = 1
) -> mx.array:
    """
    Copies elements from src into input at the specified slice indices along dim.

    Args:
        input: Destination array to be modified
        src: Source array with values to copy
        dim: Dimension along which to scatter
        start: Starting index
        end: Ending index (exclusive). If None, inferred from src size
        step: Step size (default: 1)

    Returns:
        Modified copy of input array with scattered values
    """
    # Handle negative dimension
    if dim < 0:
        dim += input.ndim

    # Infer end if not provided
    if end is None:
        end = start + src.shape[dim]

    # Create indices for the scatter
    indices = mx.arange(start, end, step)

    # Create output array
    output = mx.array(input)  # Create a copy

    # Create slice objects for all dimensions
    slices = [slice(None)] * input.ndim
    slices[dim] = indices

    # Update values
    output = output.at[tuple(slices)].set(src)

    return output


def passthrough(*args):
    return args[0]
