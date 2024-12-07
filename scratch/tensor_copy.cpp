#include <torch/types.h>

int main() {

    auto options = torch::TensorOptions().device(torch::kMPS);
    auto a = torch::randn({1, 1}, options);

    auto foo = a.storage();

}

