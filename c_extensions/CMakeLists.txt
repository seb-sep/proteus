cmake_minimum_required(VERSION 3.27)

project(cpp_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")


# ----------------------------- Dependencies -----------------------------
find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)

# Get MLX installation path from Python
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx; print(mlx.__path__[0])"
  OUTPUT_VARIABLE MLX_PYTHON_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
get_filename_component(MLX_ROOT "${MLX_PYTHON_PATH}/.." ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH "${MLX_ROOT}")


find_package(MLX CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)


# ----------------------------- Extensions -----------------------------

# Add library
add_library(mlx_ext)

# Add sources
target_sources(mlx_ext PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensor_coercion/coercion.cpp
)

# Add include headers
target_include_directories(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Link to mlx
target_link_libraries(mlx_ext PUBLIC mlx)

get_target_property(TARGET_LINK_LIBRARIES mlx_ext LINK_LIBRARIES)
message(STATUS "mlx_ext LINK_LIBRARIES: ${TARGET_LINK_LIBRARIES}")

get_target_property(TARGET_INCLUDE_DIRS mlx_ext INCLUDE_DIRECTORIES)
message(STATUS "mlx_ext INCLUDE_DIRECTORIES: ${TARGET_INCLUDE_DIRS}")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ----------------------------- Metal -----------------------------

# Build metallib
if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mlx_ext_metallib
    TITLE
    mlx_ext
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_ext mlx_ext_metallib)

endif()

# ----------------------------- Python Bindings -----------------------------

nanobind_add_module(
  cpp_ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(cpp_ext PRIVATE mlx_ext)

message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

set_target_properties(cpp_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

get_target_property(MODULE_LIBRARY_OUTPUT ${PROJECT_NAME} LIBRARY_OUTPUT_DIRECTORY)
message(STATUS "Module library output directory: ${MODULE_LIBRARY_OUTPUT}")

# Copy .pyi file to build directory
add_custom_command(
    TARGET cpp_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_LIST_DIR}/cpp_ext.pyi
    ${CMAKE_CURRENT_BINARY_DIR}/cpp_ext.pyi
)

# Create empty __init__.py in build directory
add_custom_command(
    TARGET cpp_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
)


if(BUILD_SHARED_LIBS)
  target_link_options(cpp_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
