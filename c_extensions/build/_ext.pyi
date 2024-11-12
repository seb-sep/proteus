import mlx.core
from typing import Union, Optional

def axpby(
    x: mlx.core.array,
    y: mlx.core.array,
    alpha: float,
    beta: float,
    stream: Optional[mlx.core.StreamOrDevice] = None,
) -> mlx.core.array: ...
