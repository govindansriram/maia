#pragma once

#include <kl/config.hpp>

// Bulk async-groups: each thread's cp.async.bulk operations (TMA stores included) are tracked per
// thread. commit closes the thread's pending ones into a group; wait blocks until at most N of
// its groups are still pending. Commit and wait must come from the same thread.
namespace kl {

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
KL_DEVICE void bulk_commit_group() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

// .read: wait only until the groups have read their shared-memory sources (so those may be
// overwritten), not until their global writes land.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
template <int N>
KL_DEVICE void bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

}  // namespace kl
