#pragma once

#include <cstdint>

#include <kl/config.hpp>
#include <kl/cute.hpp>

// Thin wrappers over CUTLASS's mbarrier PTX. A barrier is a plain uint64_t in shared memory.
// Each phase completes when its expected arrivals and transaction bytes have all come in; waits
// name the phase by its parity (0, 1, 0, ...).
namespace kl::barrier {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// One thread initializes; call fence_init() afterwards and sync before anyone else uses it.
KL_DEVICE void init(uint64_t* bar, uint32_t arrivals) { Barrier::init(bar, arrivals); }
KL_DEVICE void fence_init() { cutlass::arch::fence_barrier_init(); }

KL_DEVICE void arrive(uint64_t* bar) { Barrier::arrive(bar); }
// Arrive and add `bytes` to what this phase waits for (TMA copies count them down as they land).
KL_DEVICE void arrive_expect_tx(uint64_t* bar, uint32_t bytes) { Barrier::arrive_and_expect_tx(bar, bytes); }

// Block until the phase with parity `parity` has completed.
KL_DEVICE void wait(uint64_t* bar, uint32_t parity) { Barrier::wait(bar, parity); }
// Like wait, but gives up after a hardware-defined time and returns whether the phase completed.
KL_DEVICE bool try_wait(uint64_t* bar, uint32_t parity) { return Barrier::try_wait(bar, parity); }

}  // namespace kl::barrier
