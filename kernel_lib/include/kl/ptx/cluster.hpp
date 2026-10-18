#pragma once

#include <cstdint>

#include <kl/config.hpp>
#include <kl/ptx/order.hpp>

// The cluster barrier: a hardware barrier across every thread of every CTA in the cluster, like
// __syncthreads() for the whole cluster. There is no shared-memory object and no phase to track:
// every thread arrives, and cluster_wait() returns once all non-exited threads of the cluster
// have. The calls are .aligned, so a warp makes them together: never under a branch that splits
// the warp.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster
namespace kl {

// This CTA's rank in its cluster, as peer() takes it (0 in a launch without clusters).
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctarank
KL_DEVICE uint32_t cluster_rank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

// The same variable as `local` (a pointer into this CTA's shared memory), but in the CTA with rank
// `rank` of this cluster: an ordinary pointer, so plain C++ loads and stores of any width work
// through it (a uint4 store is a single 16-byte ST.E.128). The accesses are plain, unordered
// ones: publish stores with a Sem::release peer arrive or fence_release_cluster() (examples.md).
// This is the generic form, mapa.u64. Mbarrier::peer() uses the .shared::cluster form,
// mapa.shared::cluster.u32, whose 32-bit result only .shared::cluster PTX instructions (the
// mbarrier ones) can take and C++ cannot dereference.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa
template <typename T>
KL_DEVICE T* peer_ptr(T* local, uint32_t rank) {
  uint64_t peer;
  asm volatile("mapa.u64 %0, %1, %2;" : "=l"(peer) : "l"(reinterpret_cast<uint64_t>(local)), "r"(rank));
  return reinterpret_cast<T*>(peer);
}

// Sem::release publishes this thread's earlier writes to every thread of the cluster once its
// cluster_wait() returns; it costs a GPU-wide fence. Sem::relaxed publishes nothing: use it when a
// fence already did, e.g. right after fence_mbarrier_init() (ptx/examples.md, section 5).
template <Sem S = Sem::release>
KL_DEVICE void cluster_arrive() {
  if constexpr (S == Sem::release)
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  else
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

// Block until every thread of the cluster has arrived. An acquire: what they published is
// visible afterwards.
KL_DEVICE void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

// Every thread's earlier writes are visible to every thread of the cluster afterwards. Also the
// end-of-kernel sync for clusters: no CTA exits while another may still use its shared memory.
KL_DEVICE void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

}  // namespace kl
