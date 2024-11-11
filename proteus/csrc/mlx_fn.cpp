#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx {

array axpby(
    const array& x,
    const array& y,
    const float alpha,
    const float beta,
    StreamOrDevice s = {}) {
    // Handle dtype promotion and broadcasting
    auto promoted_dtype = promote_types(x.dtype(), y.dtype());
    auto out_dtype = is_floating_point(promoted_dtype)
        ? promoted_dtype
        : promote_types(promoted_dtype, float32);
    
    auto broadcasted_inputs = broadcast_arrays({x, y});
    auto [bx, by] = broadcasted_inputs;

    return array(
        broadcasted_inputs[0].shape(),
        out_dtype,
        std::make_shared<Axpby>(to_stream(s), alpha, beta),
        broadcasted_inputs);
}

} // namespace mlx