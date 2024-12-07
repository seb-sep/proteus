#pragma once

#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/dtype.h>
#include <vector>

mlx::core::array ptr_to_mlx(
    uintptr_t data_ptr,
    std::vector<int> shape,
    mlx::core::Dtype dtype = mlx::core::float32
);