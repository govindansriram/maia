cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 100a)
endif()

project(kl LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Keep -I flags inline so editors reading compile_commands.json can see them.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

find_package(CUDAToolkit 12.8 REQUIRED)

set(KL_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass"
    CACHE PATH "CUTLASS checkout; kl only uses an allow-listed subset of its headers")
if(NOT EXISTS "${KL_CUTLASS_DIR}/include/cute/layout.hpp")
  message(FATAL_ERROR
    "CUTLASS not found at '${KL_CUTLASS_DIR}'. "
    "Initialize the submodule (git submodule update --init) or pass -DKL_CUTLASS_DIR=<path>.")
endif()

add_library(kl INTERFACE)
add_library(kl::kl ALIAS kl)
target_include_directories(kl INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${KL_CUTLASS_DIR}/include>)
target_compile_features(kl INTERFACE cxx_std_20 cuda_std_20)
target_compile_options(kl INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(kl INTERFACE CUDA::cudart CUDA::cuda_driver)

option(KL_BUILD_TESTS "Build kl's tests" ${PROJECT_IS_TOP_LEVEL})
if(KL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
