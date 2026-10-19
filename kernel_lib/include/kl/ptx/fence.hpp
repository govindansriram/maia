#pragma once

#include <kl/config.hpp>

namespace kl {

// Makes this thread's earlier shared-memory writes visible to the async proxy (TMA, tensor core).
// Each writing thread fences, then syncs with the thread that issues the async operation.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
KL_DEVICE void fence_view_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// One fence to publish this thread's earlier writes to the whole cluster, so the peer arrives
// after it can all be Sem::relaxed instead of each Sem::release paying for its own fence:
//
//   *kl::peer_ptr(&s.data, r) = v;                 // for each peer r
//   kl::fence_release_cluster();                   // once
//   s.ready.peer(r).arrive<kl::Sem::relaxed>();    // for each peer r: now publishes the writes
//
// Writes by other lanes are covered too if they __syncwarp() with this lane before the fence.
// The peers wait with Scope::cluster. It is fence.release rather than fence.acq_rel because the
// acquire half would also invalidate this SM's L1 cache, which only a thread about to read others'
// writes needs. Requires PTX ISA 8.6. Pattern: ptx/examples.md, section 8.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
KL_DEVICE void fence_release_cluster() { asm volatile("fence.release.cluster;" ::: "memory"); }

// Makes initialized mbarriers visible to the other threads, the async proxy (TMA) and the rest of
// the cluster. It exists only as .release.cluster; a CTA without a cluster is a cluster of one.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
KL_DEVICE void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

}  // namespace kl
