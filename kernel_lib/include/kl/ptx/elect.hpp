#pragma once

#include <cstdint>

#include <kl/config.hpp>

namespace kl {

// True on exactly one lane of the calling warp, the same lane on every call (election is
// deterministic for a given member mask). Every lane of the warp must call it.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync
KL_DEVICE bool elect_one() {
  uint32_t elected;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "elect.sync _|p, 0xffffffff;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}"
      : "=r"(elected));
  return elected != 0;
}

}  // namespace kl
