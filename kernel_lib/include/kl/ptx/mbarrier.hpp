#pragma once

#include <cstdint>
#include <type_traits>

#include <kl/config.hpp>
#include <kl/cute.hpp>
#include <kl/ptx/fence.hpp>
#include <kl/ptx/order.hpp>

// mbarriers, written directly in PTX. An Mbarrier is 8 bytes in this CTA's shared memory. Each
// phase completes when its expected arrivals and transaction bytes have all come in; waits name
// the phase by its parity (0, 1, 0, ...).
//
// A barrier lives in the CTA that waits on it: PTX can only init and wait on a barrier in the
// caller's own CTA. Other CTAs of the cluster can only arrive on it, through peer(rank), which
// returns a PeerMbarrier that has no wait. So the type says which address window an operation
// uses (.shared::cta for Mbarrier, .shared::cluster for PeerMbarrier); callers choose only the
// memory ordering, where it is a real choice.
//
// Memory ordering: arrive is a release and a successful wait is an acquire, so whatever a thread
// wrote before arriving is visible to the threads that waited on that phase. Scope is who that
// covers: .cta is this CTA; .cluster adds the other CTAs of the cluster and costs a GPU-wide
// fence on each release. That covers ordinary loads and stores. The TMA engine and the tensor
// core read shared memory through the async proxy, which needs fence_view_async_shared()
// (ptx/fence.hpp) instead; data they write is tracked by transaction bytes, not by a release.
//
// PTX ISA reference for mbarriers (phases, arrival and transaction counts):
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier
namespace kl {

// This CTA's barrier, seen from another CTA of the cluster. Arrive only: a CTA cannot wait on
// another CTA's barrier. Arrivals come from outside the waiting CTA, so the scope is always
// .cluster, and the caller states the semantics:
//   Sem::relaxed  a signal, e.g. "my copy of this stage is free"; no fences.
//   Sem::release  also publishes this thread's earlier writes (e.g. st.shared::cluster into the
//                 peer's smem); the peer must wait<Scope::cluster>(). Costs a GPU-wide fence.
class PeerMbarrier {
 public:
  // From a .shared::cluster address (what mapa returns); Mbarrier::peer() is the usual way in.
  KL_DEVICE explicit PeerMbarrier(uint32_t cluster_addr) : addr_(cluster_addr) {}

  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
  template <Sem S>
  KL_DEVICE void arrive() const {
    if constexpr (S == Sem::release)
      asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(addr_) : "memory");
    else
      asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(addr_) : "memory");
  }

  // Arrive and add `bytes` to what the peer's phase waits for.
  template <Sem S>
  KL_DEVICE void arrive_expect_tx(uint32_t bytes) const {
    if constexpr (S == Sem::release)
      asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;" ::"r"(addr_),
                   "r"(bytes)
                   : "memory");
    else
      asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;" ::"r"(addr_),
                   "r"(bytes)
                   : "memory");
  }

  // The .shared::cluster address, for hand-written PTX.
  KL_DEVICE uint32_t cluster_u32() const { return addr_; }

 private:
  uint32_t addr_;
};

// A barrier in this CTA's shared memory; declare it in the kernel's Smem struct. It cannot be
// copied: a copy would live in registers, and arriving on it would be meaningless.
class Mbarrier {
 public:
  Mbarrier() = default;
  Mbarrier(Mbarrier const&) = delete;
  Mbarrier& operator=(Mbarrier const&) = delete;

  // One thread initializes; then fence_mbarrier_init() and a CTA (or cluster) sync before anyone
  // else, TMA and peer CTAs included, uses it.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
  KL_DEVICE void init(uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32()), "r"(arrivals) : "memory");
  }

  // Arrivals from this CTA are seen by this CTA's waiters, so CTA scope is enough.
  // Always release (here and in arrive_expect_tx): at CTA scope it compiles to the same SASS as
  // relaxed, so there is no cheaper option.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
  KL_DEVICE void arrive() {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32()) : "memory");
  }

  // Arrive and add `bytes` to what this phase waits for (TMA copies count them down as they land).
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
  KL_DEVICE void arrive_expect_tx(uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32()), "r"(bytes)
                 : "memory");
  }

  // Returns whether the phase with parity `parity` has completed, after waiting for it a
  // hardware-defined while (the thread is suspended meanwhile, so retrying in a loop is cheap).
  // Use Scope::cluster when a peer arrived with Sem::release and you read what it wrote.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
  template <Scope Sc = Scope::cta>
  KL_DEVICE bool try_wait(uint32_t parity) {
    uint32_t done;
    if constexpr (Sc == Scope::cta)
      asm volatile(
          "{\n"
          ".reg .pred p;\n"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
          "selp.u32 %0, 1, 0, p;\n"
          "}"
          : "=r"(done)
          : "r"(smem_u32()), "r"(parity)
          : "memory");
    else
      asm volatile(
          "{\n"
          ".reg .pred p;\n"
          "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
          "selp.u32 %0, 1, 0, p;\n"
          "}"
          : "=r"(done)
          : "r"(smem_u32()), "r"(parity)
          : "memory");
    return done != 0;
  }

  // Block until the phase with parity `parity` has completed. Each try_wait suspends the thread
  // until the phase completes or a time limit passes, whichever comes first; on a timeout it
  // returns false and the loop tries again. `suspend_ns` replaces the hardware's default limit
  // with 10 ms (CUTLASS's value). It is not a delay, since the thread wakes as soon as the phase
  // completes, and it does not bound the total wait. The loop is in PTX rather than C++ so ptxas
  // emits a quick non-blocking phase check before the suspending loop.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
  template <Scope Sc = Scope::cta>
  KL_DEVICE void wait(uint32_t parity) {
    constexpr uint32_t suspend_ns = 10'000'000;
    if constexpr (Sc == Scope::cta)
      asm volatile(
          "{\n"
          ".reg .pred p;\n"
          "WAIT:\n"
          "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n"
          "@p bra DONE;\n"
          "bra WAIT;\n"
          "DONE:\n"
          "}" ::"r"(smem_u32()),
          "r"(parity), "r"(suspend_ns)
          : "memory");
    else
      asm volatile(
          "{\n"
          ".reg .pred p;\n"
          "WAIT:\n"
          "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1, %2;\n"
          "@p bra DONE;\n"
          "bra WAIT;\n"
          "DONE:\n"
          "}" ::"r"(smem_u32()),
          "r"(parity), "r"(suspend_ns)
          : "memory");
  }

  // The same barrier in the CTA with rank `cta_rank` in this cluster (your own rank is allowed).
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa
  KL_DEVICE PeerMbarrier peer(uint32_t cta_rank) const {
    uint32_t addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(addr) : "r"(smem_u32()), "r"(cta_rank));
    return PeerMbarrier(addr);
  }

  // Escape hatches for TMA, tcgen05.commit and hand-written PTX.
  KL_DEVICE uint64_t* ptr() { return &raw_; }
  KL_DEVICE uint32_t smem_u32() const { return cute::cast_smem_ptr_to_uint(&raw_); }

 private:
  uint64_t raw_;
};

static_assert(sizeof(Mbarrier) == 8 && alignof(Mbarrier) == 8, "an Mbarrier is exactly one mbarrier object");
static_assert(std::is_standard_layout_v<Mbarrier> && std::is_trivially_default_constructible_v<Mbarrier>,
              "an Mbarrier must sit in shared memory like a plain uint64_t");

}  // namespace kl
