#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <utility>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <kl/kl.hpp>

#include "test_util.hpp"

// Copies a global tensor tile by tile through shared memory and checks the result bit for bit.
// This is the first place the TMA descriptors, box offsets and SmemTile layouts meet hardware.

namespace {

template <typename Tile>
struct CopySmem {
  Tile tile;
  uint64_t bar;
};

// One single-warp CTA per tile (blockIdx = tile coordinate). The tile is TMA-loaded, then
// written back twice:
//  - to out_tma with a TMA store: the round trip checks coordinates, boxes and edge handling;
//  - to out_threads with plain stores read through tile.tensor(), kl's own swizzled view: this
//    checks that TMA put every element where kl's layout says it is.
// MisalignSmem puts a static __shared__ array in front of dynamic shared memory, so its base is
// no longer 1024-aligned and shared_storage() has to realign it.
template <typename Tile, int Rank, bool MisalignSmem>
__global__ void copy_kernel(const __grid_constant__ kl::TmaLoad<Tile, Rank> src,
                            const __grid_constant__ kl::TmaStore<Tile, Rank> out_tma,
                            typename Tile::value_type* out_threads, int64_t rows, int64_t cols,
                            uint32_t* dynamic_smem_misalignment) {
  if constexpr (MisalignSmem) {
    __shared__ volatile uint32_t bump[4];  // volatile: the compiler may not drop it
    bump[threadIdx.x % 4] = 0;
  }
  extern __shared__ unsigned char dynamic_smem[];
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    *dynamic_smem_misalignment = cute::cast_smem_ptr_to_uint(dynamic_smem) % 1024;

  auto& s = kl::shared_storage<CopySmem<Tile>>();
  kl::TileCoord<Rank> c;
  c.v[Rank - 1] = blockIdx.x;
  c.v[Rank - 2] = blockIdx.y;
  if constexpr (Rank == 3) c.v[0] = blockIdx.z;

  if (kl::elect_one()) {
    kl::barrier::init(&s.bar, 1);
    kl::barrier::fence_init();
  }
  __syncwarp();
  if (kl::elect_one()) kl::barrier::arrive_expect_tx(&s.bar, Tile::bytes);
  kl::tma_load(s.tile, src, c, &s.bar);
  kl::barrier::wait(&s.bar, 0);

  kl::tma_store(out_tma, c, s.tile);

  auto view = s.tile.tensor();
  int64_t batch = Rank == 3 ? blockIdx.z : 0;
  for (int i = threadIdx.x; i < Tile::size; i += 32) {
    int r = i / Tile::cols, k = i % Tile::cols;
    int64_t gr = int64_t(blockIdx.y) * Tile::rows + r, gk = int64_t(blockIdx.x) * Tile::cols + k;
    if (gr < rows && gk < cols) out_threads[(batch * rows + gr) * cols + gk] = view(r, k);
  }
  kl::tma_store_wait<0>();
}

// The source is `batch` (rows x cols) matrices. With row_stride > cols it is a column slice
// (starting at col_offset) of a wider matrix; batched sources are compact.
struct Problem {
  int64_t batch = 1, rows, cols;
  int64_t row_stride = 0;  // 0: compact
  int64_t col_offset = 0;
};

template <typename Tile, int Rank, bool MisalignSmem = false>
void expect_copy(Problem p) {
  using T = typename Tile::value_type;
  int64_t ld = p.row_stride ? p.row_stride : p.cols;
  int64_t out_elems = p.batch * p.rows * p.cols;

  // Every byte different, so a misplaced element shows up.
  thrust::host_vector<T> h_src(p.col_offset + p.batch * p.rows * ld);
  auto* bytes = reinterpret_cast<uint8_t*>(h_src.data());
  for (size_t j = 0; j < h_src.size() * sizeof(T); ++j) bytes[j] = uint8_t((j * 2654435761u) >> 13);

  thrust::device_vector<T> src = h_src;
  thrust::device_vector<T> out_tma(out_elems), out_threads(out_elems);
  thrust::device_vector<uint32_t> misalignment(1);
  T* src_ptr = thrust::raw_pointer_cast(src.data()) + p.col_offset;
  T* tma_ptr = thrust::raw_pointer_cast(out_tma.data());

  auto [load, store] = [&] {
    if constexpr (Rank == 3)
      return std::pair{kl::make_tma_load<Tile>(kl::row_major(src_ptr, p.batch, p.rows, p.cols)),
                       kl::make_tma_store<Tile>(kl::row_major(tma_ptr, p.batch, p.rows, p.cols))};
    else
      return std::pair{kl::make_tma_load<Tile>(kl::row_major_strided(src_ptr, p.rows, p.cols, ld)),
                       kl::make_tma_store<Tile>(kl::row_major(tma_ptr, p.rows, p.cols))};
  }();

  auto kernel = copy_kernel<Tile, Rank, MisalignSmem>;
  ASSERT_EQ(kl::set_smem_size<CopySmem<Tile>>(kernel), cudaSuccess);
  dim3 grid((p.cols + Tile::cols - 1) / Tile::cols, (p.rows + Tile::rows - 1) / Tile::rows, p.batch);
  kernel<<<grid, 32, kl::smem_bytes<CopySmem<Tile>>>>>(load, store, thrust::raw_pointer_cast(out_threads.data()),
                                                        p.rows, p.cols, thrust::raw_pointer_cast(misalignment.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  if (MisalignSmem) {
    cudaFuncAttributes attrs{};
    ASSERT_EQ(cudaFuncGetAttributes(&attrs, kernel), cudaSuccess);
    ASSERT_GT(attrs.sharedSizeBytes, 0u) << "the static __shared__ array was optimized away";
    ASSERT_NE(uint32_t(misalignment[0]), 0u) << "dynamic smem is still 1024-aligned after "
                                             << attrs.sharedSizeBytes << " bytes of static smem";
  }

  // The compact copy of the source view that both outputs should equal.
  thrust::host_vector<T> expected(out_elems);
  for (int64_t b = 0; b < p.batch; ++b)
    for (int64_t r = 0; r < p.rows; ++r)
      std::memcpy(&expected[(b * p.rows + r) * p.cols], &h_src[p.col_offset + (b * p.rows + r) * ld], p.cols * sizeof(T));

  thrust::host_vector<T> got_tma = out_tma, got_threads = out_threads;
  auto first_mismatch = [&](thrust::host_vector<T> const& got) {
    for (int64_t e = 0; e < out_elems; ++e)
      if (std::memcmp(&got[e], &expected[e], sizeof(T)) != 0) return e;
    return int64_t(-1);
  };
  auto where = [&](int64_t e) {
    return testing::Message() << "first mismatch at (batch " << e / (p.rows * p.cols) << ", row "
                              << e / p.cols % p.rows << ", col " << e % p.cols << ")";
  };
  int64_t e = first_mismatch(got_tma);
  EXPECT_EQ(e, -1) << "TMA store round trip: " << where(e);
  e = first_mismatch(got_threads);
  EXPECT_EQ(e, -1) << "reading the tile through tile.tensor(): " << where(e);
}

using kl::bf16;
using kl::SmemTile;

}  // namespace

TEST(TmaCopy, RaggedEdges) {
  KL_REQUIRE_BLACKWELL();
  expect_copy<SmemTile<bf16, 128, 64>, 2>({.rows = 1000, .cols = 1000});
}

TEST(TmaCopy, TwoColumnBoxes) {
  KL_REQUIRE_BLACKWELL();
  expect_copy<SmemTile<bf16, 128, 128>, 2>({.rows = 520, .cols = 776});
}

TEST(TmaCopy, TwoRowBoxes) {
  KL_REQUIRE_BLACKWELL();
  expect_copy<SmemTile<bf16, 512, 64>, 2>({.rows = 1100, .cols = 200});
}

TEST(TmaCopy, EverySwizzle) {
  KL_REQUIRE_BLACKWELL();
  {
    SCOPED_TRACE("64B swizzle");
    expect_copy<SmemTile<bf16, 128, 32, kl::Swizzle64B>, 2>({.rows = 300, .cols = 200});
  }
  {
    SCOPED_TRACE("32B swizzle");
    expect_copy<SmemTile<bf16, 128, 16, kl::Swizzle32B>, 2>({.rows = 300, .cols = 200});
  }
  {
    SCOPED_TRACE("no swizzle");
    expect_copy<SmemTile<bf16, 128, 8, kl::SwizzleNone>, 2>({.rows = 300, .cols = 200});
  }
}

TEST(TmaCopy, Int8) {
  KL_REQUIRE_BLACKWELL();
  expect_copy<SmemTile<int8_t, 128, 128>, 2>({.rows = 300, .cols = 400});
}

TEST(TmaCopy, StridedView) {
  KL_REQUIRE_BLACKWELL();
  // The middle third of a [600, 768] matrix, like K out of a fused QKV buffer.
  expect_copy<SmemTile<bf16, 128, 64>, 2>({.rows = 600, .cols = 256, .row_stride = 768, .col_offset = 256});
}

TEST(TmaCopy, Batched) {
  KL_REQUIRE_BLACKWELL();
  expect_copy<SmemTile<bf16, 128, 64>, 3>({.batch = 3, .rows = 300, .cols = 200});
}

TEST(TmaCopy, MisalignedSmemBase) {
  KL_REQUIRE_BLACKWELL();
  expect_copy<SmemTile<bf16, 128, 128>, 2, /*MisalignSmem=*/true>({.rows = 520, .cols = 776});
}
