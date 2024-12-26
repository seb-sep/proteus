from typing import Union, Optional, List

import mlx.core as mx

def axpby(
    x: mx.array,
    y: mx.array,
    alpha: float,
    beta: float,
    stream: Optional[Union[mx.Stream, mx.Device]] = None,
) -> mx.array: ...
def ptr_to_mlx(data_ptr: int, shape: list[int], dtype: mx.Dtype) -> mx.array: ...
def get_strides(arr: mx.array) -> List[int]: ...
