include(FetchContent)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# 89 lets the arch-neutral tests run on a local Ada GPU; 100a is the real target.
set(KL_TEST_CUDA_ARCHITECTURES "89;100a" CACHE STRING "CUDA architectures for kl's tests")

add_executable(kl_unit_tests
  unit/test_smem_tile.cu
  unit/test_umma_desc.cu
  unit/test_tma_desc.cu)
target_link_libraries(kl_unit_tests PRIVATE kl GTest::gtest_main)
target_include_directories(kl_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(kl_unit_tests PROPERTIES CUDA_ARCHITECTURES "${KL_TEST_CUDA_ARCHITECTURES}")
gtest_discover_tests(kl_unit_tests DISCOVERY_MODE PRE_TEST)

# Device tests issue Blackwell-only instructions (TMA, mbarrier, later tcgen05), so they are
# built for sm_100a alone and skip on other GPUs.
set(KL_DEVICE_TEST_CUDA_ARCHITECTURES "100a" CACHE STRING "CUDA architectures for kl's device tests")

add_executable(kl_device_tests
  device/test_tma_copy.cu
  device/test_barrier_patterns.cu)
target_link_libraries(kl_device_tests PRIVATE kl GTest::gtest_main)
target_include_directories(kl_device_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(kl_device_tests PROPERTIES CUDA_ARCHITECTURES "${KL_DEVICE_TEST_CUDA_ARCHITECTURES}")
gtest_discover_tests(kl_device_tests DISCOVERY_MODE PRE_TEST)

# The device tests must really use TMA, not CuTe's fallback paths. Needs no GPU, but does need
# cuobjdump, which slim CUDA installs leave out.
find_program(KL_CUOBJDUMP cuobjdump HINTS ${CUDAToolkit_BIN_DIR})
if(KL_CUOBJDUMP)
  add_test(NAME kl_device_tests_use_tma
    COMMAND ${CMAKE_COMMAND} -DBINARY=$<TARGET_FILE:kl_device_tests> -DCUOBJDUMP=${KL_CUOBJDUMP}
            -DOPCODES=UTMALDG$<SEMICOLON>UTMASTG -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sass.cmake)
else()
  message(STATUS "cuobjdump not found: skipping the SASS check of kl_device_tests")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME kl_include_allowlist
  COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/tools/check_includes.py
          --nvcc ${CMAKE_CUDA_COMPILER}
          --include ${PROJECT_SOURCE_DIR}/include
          --include ${KL_CUTLASS_DIR}/include)
