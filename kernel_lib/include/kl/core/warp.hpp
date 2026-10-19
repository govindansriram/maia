#pragma once

#include <kl/config.hpp>
#include <kl/ptx/elect.hpp>

// kl kernels use 1-D thread blocks; warps and warpgroups are numbered along threadIdx.x.
namespace kl {

// warp_id() and warpgroup_id() broadcast lane 0's value with __shfl_sync. Every lane computes the
// same number anyway, but the compiler cannot prove that from threadIdx.x, which differs per lane,
// so it would treat the result as per-lane. After the broadcast it knows the value is warp-uniform,
// so code that branches on it (role dispatch: `if (warp_id() == 0)`) compiles as a uniform branch
// with no divergence handling.
KL_DEVICE int warp_id() { return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x) / 32, 0); }
KL_DEVICE int warpgroup_id() { return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x) / 128, 0); }
KL_DEVICE int lane_id() { return static_cast<int>(threadIdx.x) % 32; }

}  // namespace kl
