from typing import Union, Optional

import mlx.core

def axpby(
    x: mlx.core.array,
    y: mlx.core.array,
    alpha: float,
    beta: float,
    stream: Optional[Union[mlx.core.Stream, mlx.core.Device]] = None,
) -> mlx.core.array: ...
def ptr_to_mlx(
    data_ptr: int, shape: list[int], dtype: mlx.core.Dtype
) -> mlx.core.array: ...
