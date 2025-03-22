# C Extensions

MLX uses [`nanobind`](https://github.com/wjakob/nanobind) to bind the MLX C++ library and custom user made ops to Python. proteus uses C++ for custom operator support and other utilities leveraging the MLX C++ API.

## Building

Standard CMake fare to build the extension:

```bash
mkdir build
cmake .. <-G Ninja>
<ninja/make>
```

Aftterwards, a `cpp_ext.<cpython_version>.so` is produced in `build` which is imported and re-exported by `__init__.py`. Each logical set of C extensions gets its own folder in `c_extensions`; go off `axpby` as a template for implementing custom MLX ops. Refer to [this page in the MLX docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) for writing Metal kernels.

