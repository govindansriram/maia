#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <kl/kl.hpp>

#include "test_util.hpp"

// One test per pattern in include/kl/ptx/examples.md (BarrierPatterns.<section name>). The code
// each section shows is the code between the "examples.md" markers below.

// The address window is in the type: a peer barrier can be arrived on, never waited on or
// initialized, and an Mbarrier cannot be copied out of shared memory.
template <typename B>
concept CanWait = requires(B b) {
  b.wait(0u);
  b.try_wait(0u);
};
template <typename B>
concept CanInit = requires(B b) { b.init(1u); };
static_assert(CanWait<kl::Mbarrier> && CanInit<kl::Mbarrier>);
static_assert(!CanWait<kl::PeerMbarrier> && !CanInit<kl::PeerMbarrier>);
static_assert(!std::is_copy_constructible_v<kl::Mbarrier> && !std::is_copy_assignable_v<kl::Mbarrier>);

namespace {

using kl::bf16;

// What CTA `src` writes for `lane` in the peer tests.
KL_HOST_DEVICE uint32_t peer_value(uint32_t src, uint32_t lane) { return 1000 * (src + 1) + lane; }

// ---- Local handoff, try_wait --------------------------------------------------------------------

constexpr int kRounds = 8;

struct HandoffSmem {
  kl::Mbarrier full;   // the producer filled buf
  kl::Mbarrier empty;  // both consumers are done with buf
  uint32_t buf[32];
};

// Warp 0 fills one buffer kRounds times; warps 1 and 2 read every round, warp 1 with wait and
// warp 2 by polling try_wait.
__global__ void handoff_kernel(uint32_t* out) {
  auto& s = kl::shared_storage<HandoffSmem>();
  int warp = kl::warp_id(), lane = kl::lane_id();
  if (warp == 0 && kl::elect_one()) {
    s.full.init(1);   // the producer's arrive
    s.empty.init(2);  // one arrive per consumer warp
    kl::fence_mbarrier_init();
  }
  __syncthreads();

  // examples.md: local handoff
  if (warp == 0) {
    for (int i = 0; i < kRounds; ++i) {
      if (i > 0) s.empty.wait((i - 1) % 2);  // consumers are done with round i - 1
      s.buf[lane] = 100 * i + lane;
      __syncwarp();
      if (kl::elect_one()) s.full.arrive();
    }
  } else {
    for (int i = 0; i < kRounds; ++i) {
      if (warp == 1) {
        s.full.wait(i % 2);
      } else {
        while (!s.full.try_wait(i % 2)) {
          // examples.md: try_wait (free to do other work here)
        }
      }
      out[((warp - 1) * kRounds + i) * 32 + lane] = s.buf[lane];
      __syncwarp();
      if (kl::elect_one()) s.empty.arrive();
    }
  }
  // examples.md: end
}

// ---- Threads write a tile, then TMA stores it ---------------------------------------------------

using StoreTile = kl::SmemTile<bf16, 128, 64>;

struct StoreSmem {
  StoreTile tile;
};

KL_HOST_DEVICE float store_value(int r, int k) { return float((r + k) % 256); }  // exact in bf16

// All four warps write the tile; warp 0 stores it.
__global__ void store_kernel(const __grid_constant__ kl::TmaStore<StoreTile> dst) {
  auto& s = kl::shared_storage<StoreSmem>();
  auto view = s.tile.tensor();

  // examples.md: tma store
  for (int i = threadIdx.x; i < StoreTile::size; i += blockDim.x) {
    int r = i / StoreTile::cols, k = i % StoreTile::cols;
    view(r, k) = bf16(store_value(r, k));
  }
  kl::fence_view_async_shared();  // every thread that wrote
  __syncthreads();
  if (kl::warp_id() == 0) {
    kl::tma_store(dst, kl::TileCoord<2>{0, 0}, s.tile);
    kl::tma_store_wait<0>();
  }
  // examples.md: end
}

// ---- Peer signal ----------------------------------------------------------------------------------

struct SignalSmem {
  kl::Mbarrier from_peer;
  kl::Mbarrier from_self;
};

__global__ void __cluster_dims__(2, 1, 1) signal_kernel(uint32_t* done) {
  auto& s = kl::shared_storage<SignalSmem>();
  uint32_t rank = kl::cluster_rank(), other = rank ^ 1;

  // examples.md: cluster setup
  if (kl::elect_one()) {
    s.from_peer.init(1);
    s.from_self.init(1);
    kl::fence_mbarrier_init();
  }
  kl::cluster_arrive<kl::Sem::relaxed>();  // fence_mbarrier_init() already published the init
  kl::cluster_wait();                      // every CTA's barriers exist before anyone arrives remotely
  // examples.md: end

  // examples.md: peer signal
  if (kl::elect_one()) {
    s.from_peer.peer(other).arrive<kl::Sem::relaxed>();
    s.from_self.peer(rank).arrive<kl::Sem::relaxed>();  // your own rank is allowed
  }
  s.from_peer.wait(0);
  // examples.md: end
  s.from_self.wait(0);
  if (kl::lane_id() == 0) done[rank] = 1;

  // examples.md: cluster teardown
  kl::cluster_sync();  // no CTA exits while another may still arrive on its barriers
  // examples.md: end
}

// ---- Publish to one peer --------------------------------------------------------------------------

struct PublishSmem {
  kl::Mbarrier ready;  // the peer has written data
  uint32_t data[32];
};

// Each CTA writes 32 values into the other CTA's data, every lane arriving with Sem::release.
// CTA 0 waits with wait, CTA 1 polls try_wait, both at cluster scope.
__global__ void __cluster_dims__(2, 1, 1) publish_kernel(uint32_t* out) {
  auto& s = kl::shared_storage<PublishSmem>();
  uint32_t rank = kl::cluster_rank(), other = rank ^ 1, lane = kl::lane_id();
  if (kl::elect_one()) {
    s.ready.init(32);  // one arrive per lane of the peer
    kl::fence_mbarrier_init();
  }
  kl::cluster_arrive<kl::Sem::relaxed>();  // fence_mbarrier_init() already published the init
  kl::cluster_wait();                      // every CTA's barriers exist before anyone arrives remotely

  // examples.md: publish to one peer
  *kl::peer_ptr(&s.data[lane], other) = peer_value(rank, lane);
  s.ready.peer(other).arrive<kl::Sem::release>();

  if (rank == 0) {
    s.ready.wait<kl::Scope::cluster>(0);
  } else {
    while (!s.ready.try_wait<kl::Scope::cluster>(0)) {
    }
  }
  out[rank * 32 + lane] = s.data[lane];
  // examples.md: end

  kl::cluster_sync();
}

// ---- Publish to several peers with one fence ------------------------------------------------------

constexpr uint32_t kFanOut = 4;

struct FanOutSmem {
  kl::Mbarrier ready;          // every other CTA has written its row of data
  uint32_t data[kFanOut][32];  // data[src]: written by CTA src
};

// Every CTA writes its row into the 3 other CTAs, then one lane fences once and signals all 3.
__global__ void __cluster_dims__(kFanOut, 1, 1) fan_out_kernel(uint32_t* out) {
  auto& s = kl::shared_storage<FanOutSmem>();
  uint32_t rank = kl::cluster_rank(), lane = kl::lane_id();
  if (kl::elect_one()) {
    s.ready.init(kFanOut - 1);  // one arrive per other CTA
    kl::fence_mbarrier_init();
  }
  kl::cluster_arrive<kl::Sem::relaxed>();  // fence_mbarrier_init() already published the init
  kl::cluster_wait();                      // every CTA's barriers exist before anyone arrives remotely

  // examples.md: publish to several peers
  for (uint32_t p = 0; p < kFanOut; ++p)
    if (p != rank) *kl::peer_ptr(&s.data[rank][lane], p) = peer_value(rank, lane);
  __syncwarp();  // the elected lane's fence now covers every lane's stores
  if (kl::elect_one()) {
    kl::fence_release_cluster();
    for (uint32_t p = 0; p < kFanOut; ++p)
      if (p != rank) s.ready.peer(p).arrive<kl::Sem::relaxed>();
  }
  s.ready.wait<kl::Scope::cluster>(0);
  // examples.md: end

  for (uint32_t src = 0; src < kFanOut; ++src)
    if (src != rank) out[(rank * kFanOut + src) * 32 + lane] = s.data[src][lane];
  kl::cluster_sync();
}

// ---- Arm a peer's barrier for TMA bytes -----------------------------------------------------------

using McTile = kl::SmemTile<bf16, 64, 64>;
static_assert(kl::TmaBox<McTile>::count == 1, "one multicast copy loads the whole tile");

struct MulticastSmem {
  McTile tile;
  kl::Mbarrier full;
  uint32_t note;  // written by CTA 0 before arming (WithNote)
};

KL_HOST_DEVICE float multicast_value(int r, int k) { return float((3 * r + k) % 256); }
KL_HOST_DEVICE uint32_t note_value(uint32_t rank) { return 7 + rank; }

// CTA 0 arms both CTAs' barriers for the tile's bytes, then one multicast TMA load fills the
// tile in both. WithNote: CTA 0 also stores a note into each CTA first and publishes it through
// the same arrive.
template <bool WithNote>
__global__ void __cluster_dims__(2, 1, 1)
    multicast_kernel(const __grid_constant__ kl::TmaLoad<McTile> src, bf16* out, uint32_t* notes) {
  auto& s = kl::shared_storage<MulticastSmem>();
  uint32_t rank = kl::cluster_rank();
  if (kl::elect_one()) {
    s.full.init(1);  // CTA 0's arm
    kl::fence_mbarrier_init();
  }
  kl::cluster_arrive<kl::Sem::relaxed>();  // fence_mbarrier_init() already published the init
  kl::cluster_wait();                      // every CTA's barriers exist before anyone arrives remotely

  if (rank == 0 && kl::elect_one()) {
    for (uint32_t r = 0; r < 2; ++r) {
      if constexpr (WithNote) {
        // examples.md: arm and publish
        *kl::peer_ptr(&s.note, r) = note_value(r);
        s.full.peer(r).arrive_expect_tx<kl::Sem::release>(McTile::bytes);
        // examples.md: end
      } else {
        // examples.md: arm a peer
        s.full.peer(r).arrive_expect_tx<kl::Sem::relaxed>(McTile::bytes);
        // examples.md: end
      }
    }
    cute::SM90_TMA_LOAD_MULTICAST::copy(src.get(), s.full.ptr(), uint16_t(0b11),
                                        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                        s.tile.storage, 0, 0);
  }
  if constexpr (WithNote)
    s.full.wait<kl::Scope::cluster>(0);  // acquire what CTA 0 stored
  else
    s.full.wait(0);  // TMA's bytes need no cluster acquire

  auto view = s.tile.tensor();
  for (int i = threadIdx.x; i < McTile::size; i += 32)
    out[rank * McTile::size + i] = view(i / McTile::cols, i % McTile::cols);
  if (WithNote && kl::lane_id() == 0) notes[rank] = s.note;
  kl::cluster_sync();
}

template <bool WithNote>
void expect_multicast() {
  thrust::host_vector<bf16> h_src(McTile::size);
  for (int i = 0; i < McTile::size; ++i) h_src[i] = bf16(multicast_value(i / McTile::cols, i % McTile::cols));
  thrust::device_vector<bf16> src = h_src, out(2 * McTile::size);
  thrust::device_vector<uint32_t> notes(2, 0);
  auto load = kl::make_tma_load<McTile>(
      kl::row_major(thrust::raw_pointer_cast(src.data()), McTile::rows, McTile::cols));

  multicast_kernel<WithNote><<<2, 32, (kl::smem_bytes<MulticastSmem>)>>>(
      load, thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(notes.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<bf16> got = out;
  for (int rank = 0; rank < 2; ++rank)
    for (int i = 0; i < McTile::size; ++i)
      ASSERT_EQ(float(got[rank * McTile::size + i]), float(h_src[i])) << "CTA " << rank << ", element " << i;
  if (WithNote) {
    thrust::host_vector<uint32_t> got_notes = notes;
    for (uint32_t rank = 0; rank < 2; ++rank) EXPECT_EQ(got_notes[rank], note_value(rank)) << "CTA " << rank;
  }
}

}  // namespace

TEST(BarrierPatterns, LocalHandoff) {
  KL_REQUIRE_BLACKWELL();
  thrust::device_vector<uint32_t> out(2 * kRounds * 32, 0);
  handoff_kernel<<<1, 96, (kl::smem_bytes<HandoffSmem>)>>>(thrust::raw_pointer_cast(out.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<uint32_t> got = out;
  for (int consumer = 0; consumer < 2; ++consumer)
    for (int i = 0; i < kRounds; ++i)
      for (int lane = 0; lane < 32; ++lane)
        ASSERT_EQ(got[(consumer * kRounds + i) * 32 + lane], uint32_t(100 * i + lane))
            << (consumer == 0 ? "wait" : "try_wait") << " consumer, round " << i << ", lane " << lane;
}

TEST(BarrierPatterns, TmaStoreAfterThreadWrites) {
  KL_REQUIRE_BLACKWELL();
  thrust::device_vector<bf16> out(StoreTile::size);
  auto store = kl::make_tma_store<StoreTile>(
      kl::row_major(thrust::raw_pointer_cast(out.data()), StoreTile::rows, StoreTile::cols));
  store_kernel<<<1, 128, (kl::smem_bytes<StoreSmem>)>>>(store);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<bf16> got = out;
  for (int r = 0; r < StoreTile::rows; ++r)
    for (int k = 0; k < StoreTile::cols; ++k)
      ASSERT_EQ(float(got[r * StoreTile::cols + k]), store_value(r, k)) << "row " << r << ", col " << k;
}

TEST(BarrierPatterns, PeerSignal) {
  KL_REQUIRE_BLACKWELL();
  thrust::device_vector<uint32_t> done(2, 0);
  signal_kernel<<<2, 32, (kl::smem_bytes<SignalSmem>)>>>(thrust::raw_pointer_cast(done.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  thrust::host_vector<uint32_t> got = done;
  EXPECT_EQ(got[0], 1u);
  EXPECT_EQ(got[1], 1u);
}

TEST(BarrierPatterns, PublishToOnePeer) {
  KL_REQUIRE_BLACKWELL();
  thrust::device_vector<uint32_t> out(2 * 32, 0);
  publish_kernel<<<2, 32, (kl::smem_bytes<PublishSmem>)>>>(thrust::raw_pointer_cast(out.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<uint32_t> got = out;
  for (uint32_t rank = 0; rank < 2; ++rank)
    for (uint32_t lane = 0; lane < 32; ++lane)
      EXPECT_EQ(got[rank * 32 + lane], peer_value(rank ^ 1, lane))
          << "CTA " << rank << (rank == 0 ? " (wait)" : " (try_wait)") << ", lane " << lane;
}

TEST(BarrierPatterns, PublishToSeveralPeers) {
  KL_REQUIRE_BLACKWELL();
  thrust::device_vector<uint32_t> out(kFanOut * kFanOut * 32, 0);
  fan_out_kernel<<<kFanOut, 32, (kl::smem_bytes<FanOutSmem>)>>>(thrust::raw_pointer_cast(out.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  thrust::host_vector<uint32_t> got = out;
  for (uint32_t rank = 0; rank < kFanOut; ++rank)
    for (uint32_t src = 0; src < kFanOut; ++src)
      for (uint32_t lane = 0; lane < 32; ++lane)
        if (src != rank)
          EXPECT_EQ(got[(rank * kFanOut + src) * 32 + lane], peer_value(src, lane))
              << "CTA " << rank << ", from CTA " << src << ", lane " << lane;
}

TEST(BarrierPatterns, ArmPeerForTmaBytes) {
  KL_REQUIRE_BLACKWELL();
  expect_multicast</*WithNote=*/false>();
}

TEST(BarrierPatterns, ArmPeerAndPublish) {
  KL_REQUIRE_BLACKWELL();
  expect_multicast</*WithNote=*/true>();
}
