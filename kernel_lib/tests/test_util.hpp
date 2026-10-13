#pragma once

// Helpers shared by kl's tests.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

namespace kl_test {

inline bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// The driver only encodes tensor maps on GPUs that have TMA (sm_90+).
inline bool have_tma_gpu() {
  int major = 0;
  if (!have_gpu()) return false;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  return major >= 9;
}

// Datacenter Blackwell (sm_100 / sm_103), the only GPUs kl's device code targets.
inline bool have_blackwell_gpu() {
  int major = 0;
  if (!have_gpu()) return false;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  return major == 10;
}

}  // namespace kl_test

// Skip the current test when the hardware is missing. These are macros because GTEST_SKIP()
// returns from the function it's written in: inside a helper it would end the helper, not the test.
#define KL_REQUIRE_GPU() \
  if (!kl_test::have_gpu()) GTEST_SKIP() << "no CUDA device"
#define KL_REQUIRE_TMA() \
  if (!kl_test::have_tma_gpu()) GTEST_SKIP() << "no TMA-capable device (sm_90+)"
#define KL_REQUIRE_BLACKWELL() \
  if (!kl_test::have_blackwell_gpu()) GTEST_SKIP() << "needs a datacenter Blackwell GPU (sm_100 / sm_103)"
