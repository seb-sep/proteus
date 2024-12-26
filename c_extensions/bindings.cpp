#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/array.h>

#include "axpby/axpby.h"
#include "tensor_coercion/coercion.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

NB_MODULE(cpp_ext, m) {
  m.doc() = "Sample extension for MLX";

  /*
  m.def(
      "axpby",
      &my_ext::axpby,
      "x"_a,
      "y"_a,
      "alpha"_a,
      "beta"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )");

    m.def("ptr_to_mlx", 
        &ptr_to_mlx,
        "data_ptr"_a,
        "shape"_a,
        "dtype"_a = mlx::core::float32,
        "Create an MLX array from a pointer and shape");
    */

    m.def("get_strides", 
      &get_strides,
      "arr"_a,
      "get the strides of the mlx array as a list of ints");
}
