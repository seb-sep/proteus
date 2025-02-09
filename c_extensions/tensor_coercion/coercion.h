#pragma once

#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/dtype.h>
#include <vector>

using mlx::core::array;

array ptr_to_mlx(
    uintptr_t data_ptr,
    std::vector<int> shape,
    mlx::core::Dtype dtype = mlx::core::float32
);

std::vector<size_t> get_strides(array arr);

array mlx_contiguous(array arr);

uint64_t get_data_ptr(array a);
