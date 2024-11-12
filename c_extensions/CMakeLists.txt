cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
# Print SDK path
message(STATUS "CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT}")

# Print linker flags
message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")

# Print library search paths
message(STATUS "CMAKE_LIBRARY_PATH: ${CMAKE_LIBRARY_PATH}")
message(STATUS "CMAKE_FRAMEWORK_PATH: ${CMAKE_FRAMEWORK_PATH}")

# Print output directories
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# Print MLX-specific variables
message(STATUS "MLX_INCLUDE_DIRS: ${MLX_INCLUDE_DIRS}")
message(STATUS "MLX_LIBRARIES: ${MLX_LIBRARIES}")


find_package(MLX CONFIG REQUIRED)
find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)
# if(DEFINED MLX_CONFIG)
#   message(STATUS "Contents of MLX_CONFIG file:")
#   file(READ "${MLX_CONFIG}" MLX_CONFIG_CONTENTS)
#   message(STATUS "${MLX_CONFIG_CONTENTS}")
# endif()
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
target_sources(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/axpby/axpby.cpp)

# Add include headers
target_include_directories(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Link to mlx
target_link_libraries(mlx_ext PUBLIC mlx)

get_target_property(TARGET_LINK_LIBRARIES mlx_ext LINK_LIBRARIES)
message(STATUS "mlx_ext LINK_LIBRARIES: ${TARGET_LINK_LIBRARIES}")

get_target_property(TARGET_INCLUDE_DIRS mlx_ext INCLUDE_DIRECTORIES)
message(STATUS "mlx_ext INCLUDE_DIRECTORIES: ${TARGET_INCLUDE_DIRS}")

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
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
