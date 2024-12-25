#include <mlx/ops.h>
#include <mlx/primitives.h>
#include <mlx/dtype.h>
#include <mlx/mlx.h>
#include <iostream>
#include <Metal/Metal.hpp>


#include "coercion.h"

// use this to create a torch tensor from the buffer
int mlx_to_ptr(mlx::core::array a) {
    return reinterpret_cast<std::uintptr_t>(a.data<int>());
}

std::vector<size_t> get_strides(mlx::core::array arr) { return arr.strides(); }


// void* offset_ptr = static_cast<void*>(static_cast<char*>(static_cast<void*>(your_data)) - sizeof(size_t));
// auto arr = mlx::core::array(
//     mlx::core::allocator::Buffer(offset_ptr),
//     std::vector<int>{shape...},
//     mlx::core::float32,
//     your_deleter  // Optional
// );

// mlx::core::array ptr_to_mlx(uintptr_t data_ptr, std::vector<int> shape, mlx::core::Dtype type) {
//     // Cast integer to pointer
//     void* ptr = reinterpret_cast<void*>(static_cast<std::uintptr_t>(data_ptr));
    
//     // Offset the pointer backwards by sizeof(size_t)
//     void* offset_ptr = static_cast<void*>(static_cast<char*>(ptr) - sizeof(size_t));
//     // void* offset_ptr = static_cast<void*>(static_cast<char*>(ptr));
    
//     // Print the raw pointer value for debugging
//     std::cout << "Raw pointer value: " << offset_ptr << std::endl;

//     // auto arr = mlx::core::array({1.0});
//     // std::cout << "arr data pointer: " << arr.data<void*>() << std::endl;
    
//     // Create a buffer view of the existing data
//     mlx::core::allocator::Buffer buffer{offset_ptr};

//     std::cout << "created buffer\n";

    
//     // Use a no-op deleter to prevent MLX from freeing the memory
//     // auto no_op_deleter = [](mlx::core::allocator::Buffer buf) {};
//     // std::cout << "allocated no op deleter" << std::endl;

//     std::cout << "Buffer raw pointer: " << buffer.ptr() << std::endl;
    
//     // Create array that views the existing data
//     auto arr = mlx::core::array(buffer, shape, type, [](mlx::core::allocator::Buffer buf){});
//     // mlx::core::eval(arr);
//     std::cout << "eval'd arr to " << arr << std::endl;
//     return arr;
// }

// mlx::core::array ptr_to_mlx(uintptr_t data_ptr, std::vector<int> shape, mlx::core::Dtype type) {
//     // Cast integer to pointer and then to MTL::Buffer*
//     MTL::Buffer* metal_buffer = reinterpret_cast<MTL::Buffer*>(reinterpret_cast<void*>(data_ptr));
//     metal_buffer->retain();
    
//     // Add debug check
//     if (!metal_buffer) {
//         throw std::runtime_error("Invalid Metal buffer pointer");
//     }
    
//     std::cout << "Metal buffer address: " << metal_buffer << std::endl;
//     std::cout << "Metal buffer length: " << metal_buffer->length() << std::endl;
//     std::cout << "Metal buffer contents: " << metal_buffer->contents() << std::endl;
    
//     // Try using the contents pointer directly
//     mlx::core::allocator::Buffer buffer{metal_buffer->contents()};
//     std::cout << "Created MLX buffer" << std::endl;
    
//     // Create array that views the existing data
//     auto arr = mlx::core::array(buffer, shape, type, [metal_buffer](mlx::core::allocator::Buffer buf){
//         std::cout << "Deleter called" << std::endl;
//         metal_buffer->release();
//     });
    
//     std::cout << "Created MLX array" << std::endl;
//     std::cout << "Array shape: ";
//     for (auto dim : arr.shape()) {
//         std::cout << dim << " ";
//     }
//     std::cout << std::endl;
    
//     return arr;
// }

mlx::core::array ptr_to_mlx(uintptr_t data_ptr, std::vector<int> shape, mlx::core::Dtype type) {
    MTL::Buffer* metal_buffer = reinterpret_cast<MTL::Buffer*>(reinterpret_cast<void*>(data_ptr));
    metal_buffer->retain();

    // Debug storage mode
    MTL::StorageMode mode = metal_buffer->storageMode();
    std::cout << "Storage mode: ";
    switch(mode) {
        case MTL::StorageModeShared:
            std::cout << "Shared" << std::endl;
            break;
        case MTL::StorageModePrivate:
            std::cout << "Private" << std::endl;
            break;
        case MTL::StorageModeManaged:
            std::cout << "Managed" << std::endl;
            break;
        default:
            std::cout << "Unknown (" << static_cast<int>(mode) << ")" << std::endl;
    }
    

    // Create MLX buffer with the Metal buffer directly, not its contents
    mlx::core::allocator::Buffer buffer{metal_buffer};
    
    auto arr = mlx::core::array(buffer, shape, type, [metal_buffer](mlx::core::allocator::Buffer buf){
        metal_buffer->release();
    });
    
    return arr;
}