#pragma once

#include <kl/config.hpp>
#include <kl/cute.hpp>

// kl kernels use 1-D thread blocks; warps and warpgroups are numbered along threadIdx.x.
namespace kl {

// Broadcast from lane 0 so the compiler knows the value is the same on every lane of the warp.
KL_DEVICE int warp_id() { return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x) / 32, 0); }
KL_DEVICE int warpgroup_id() { return __shfl_sync(0xffffffff, static_cast<int>(threadIdx.x) / 128, 0); }
KL_DEVICE int lane_id() { return static_cast<int>(threadIdx.x) % 32; }

// True on exactly one lane of the calling warp. Every lane must call it.
KL_DEVICE bool elect_one() { return cute::elect_one_sync(); }

}  // namespace kl
