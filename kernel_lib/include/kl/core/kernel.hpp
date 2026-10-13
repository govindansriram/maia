#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include <kl/config.hpp>
#include <kl/cute.hpp>

namespace kl {

// Dynamic shared memory is only guaranteed a small alignment, but swizzled tiles need up to
// 1024 bytes (and a misaligned tile fails silently: TMA and the MMA disagree on the swizzle).
// shared_storage() rounds the base up to alignof(Smem), so launches reserve this many bytes.
template <typename Smem>
inline constexpr size_t smem_bytes = sizeof(Smem) + alignof(Smem);

// The kernel's shared-memory struct, placed in dynamic shared memory at an aligned address.
template <typename Smem>
KL_DEVICE Smem& shared_storage() {
  extern __shared__ unsigned char kl_dynamic_smem[];
  uint32_t base = cute::cast_smem_ptr_to_uint(kl_dynamic_smem);
  uint32_t pad = (alignof(Smem) - base % alignof(Smem)) % alignof(Smem);
  return *reinterpret_cast<Smem*>(kl_dynamic_smem + pad);
}

// Host: allow `kernel` to use smem_bytes<Smem> of dynamic shared memory (required above 48 KB).
template <typename Smem, typename Kernel>
cudaError_t set_smem_size(Kernel* kernel) {
  return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(smem_bytes<Smem>));
}

// Host: a plain launch config with dynamic shared memory sized for Smem. Edit it freely.
template <typename Smem>
cudaLaunchConfig_t launch_config(dim3 grid, dim3 block, cudaStream_t stream = nullptr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem_bytes<Smem>;
  cfg.stream = stream;
  return cfg;
}

}  // namespace kl
