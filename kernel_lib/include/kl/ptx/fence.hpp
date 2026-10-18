#pragma once

#include <kl/config.hpp>

namespace kl {

// Makes this thread's earlier shared-memory writes visible to the async proxy (TMA, tensor core).
// Each writing thread fences, then syncs with the thread that issues the async operation.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
KL_DEVICE void fence_view_async_shared() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// Followed by Sem::relaxed peer arrives, acts as a release.cluster on each of them (PTX memory
// model, "release pattern"): one fence publishes this thread's earlier writes to several peers,
// where each Sem::release arrive would pay for its own. Through causality order it also covers
// writes by other threads that synced with this one first (e.g. every lane writes, __syncwarp,
// then one elected lane fences and arrives). The peers wait with Scope::cluster.
// Release only: fence.acq_rel.cluster would also invalidate L1 (CCTL.IVALL), which publishing
// does not need. Requires PTX ISA 8.6.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
KL_DEVICE void fence_release_cluster() { asm volatile("fence.release.cluster;" ::: "memory"); }

// Makes initialized mbarriers visible to the other threads, the async proxy (TMA) and the rest of
// the cluster. It exists only as .release.cluster; a CTA without a cluster is a cluster of one.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar
KL_DEVICE void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

}  // namespace kl
