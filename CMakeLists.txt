cmake_minimum_required(VERSION 3.18)
project(CudaProject CUDA CXX)

# Find CUDA or NVHPC
find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
    find_package(NVHPC REQUIRED)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add include directory
include_directories(include)

# Define executables
add_executable(executable_batch src/main_batch.cu)
add_executable(executable_sort src/main_sortBatch.cu)
add_executable(executable_window src/main_window.cu)

# Set compile options
foreach(target executable_batch executable_sort executable_window)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

# Set output directory
set_target_properties(executable_batch executable_sort executable_window
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build"
)