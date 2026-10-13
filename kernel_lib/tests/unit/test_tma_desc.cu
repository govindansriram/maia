#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>

#include <kl/kl.hpp>

#include "test_util.hpp"

// Reference only: kl itself never includes atom headers.
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>

namespace {

// Encode the same global tensor with kl and with CuTe's make_tma_copy.
// `ref` is the same memory in CuTe's mode order (the tile's modes first).
template <typename Tile, typename G, typename Ref>
void expect_tma_matches_cute(G const& g, Ref const& ref) {
  auto ours = kl::make_tma_load<Tile>(g);
  auto theirs = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, ref, typename Tile::layout_type{});
  EXPECT_EQ(std::memcmp(&ours.map, theirs.get_tma_descriptor(), sizeof(CUtensorMap)), 0);
}

template <typename Tile>
void expect_tma_matches_cute(int rows, int cols) {
  using T = typename Tile::value_type;
  thrust::device_vector<T> buf(size_t(rows) * cols);
  auto g = kl::row_major(thrust::raw_pointer_cast(buf.data()), rows, cols);
  expect_tma_matches_cute<Tile>(g, g);
}

}  // namespace

TEST(TmaDesc, MatchesCute) {
  KL_REQUIRE_TMA();
  {
    SCOPED_TRACE("bf16 128x64 SW128");
    expect_tma_matches_cute<kl::SmemTile<kl::bf16, 128, 64>>(4096, 4096);
  }
  {
    SCOPED_TRACE("bf16 256x64 SW128");
    expect_tma_matches_cute<kl::SmemTile<kl::bf16, 256, 64>>(8192, 4096);
  }
  {
    SCOPED_TRACE("int8 128x128 SW128");
    expect_tma_matches_cute<kl::SmemTile<int8_t, 128, 128>>(4096, 4096);
  }
  {
    SCOPED_TRACE("bf16 128x128 SW128 (two boxes)");
    expect_tma_matches_cute<kl::SmemTile<kl::bf16, 128, 128>>(4096, 4096);
  }
  {
    SCOPED_TRACE("bf16 128x32 SW64");
    expect_tma_matches_cute<kl::SmemTile<kl::bf16, 128, 32, kl::Swizzle64B>>(4096, 8192);
  }
  {
    SCOPED_TRACE("bf16 128x16 SW32");
    expect_tma_matches_cute<kl::SmemTile<kl::bf16, 128, 16, kl::Swizzle32B>>(4096, 4096);
  }
  {
    SCOPED_TRACE("bf16 128x8 no swizzle");
    expect_tma_matches_cute<kl::SmemTile<kl::bf16, 128, 8, kl::SwizzleNone>>(4096, 4096);
  }
  {
    SCOPED_TRACE("bf16 128x64 SW128, batched (B, M, K)");
    using Tile = kl::SmemTile<kl::bf16, 128, 64>;
    constexpr int B = 4, M = 1024, K = 512;
    thrust::device_vector<kl::bf16> buf(size_t(B) * M * K);
    auto* p = thrust::raw_pointer_cast(buf.data());
    auto g = kl::row_major(p, B, M, K);
    // CuTe tiles the leading modes, so it wants the batch mode last: (M, K, B).
    auto ref = cute::make_tensor(cute::make_gmem_ptr(p), cute::make_shape(M, K, B),
                                 cute::make_stride(K, cute::_1{}, M * K));
    static_assert(decltype(kl::make_tma_load<Tile>(g))::rank == 3);
    expect_tma_matches_cute<Tile>(g, ref);
  }
}

// A box is at most 256 rows, so a 512-row tile is loaded as two boxes and its descriptor must be
// exactly the 256-row tile's. (CuTe has no reference here: it can't build a box over 256.)
TEST(TmaDesc, TallTilesUse256RowBoxes) {
  KL_REQUIRE_TMA();
  thrust::device_vector<kl::bf16> buf(4096 * 4096);
  auto g = kl::row_major(thrust::raw_pointer_cast(buf.data()), 4096, 4096);
  auto tall = kl::make_tma_load<kl::SmemTile<kl::bf16, 512, 64>>(g);
  auto one_box = kl::make_tma_load<kl::SmemTile<kl::bf16, 256, 64>>(g);
  EXPECT_EQ(std::memcmp(&tall.map, &one_box.map, sizeof(CUtensorMap)), 0);
}

TEST(TmaDesc, RejectsInvalidGlobalLayouts) {
  KL_REQUIRE_GPU();
  using A = kl::SmemTile<kl::bf16, 128, 64>;
  thrust::device_vector<kl::bf16> buf(4096 * 4096);
  auto* base = thrust::raw_pointer_cast(buf.data());
  EXPECT_THROW(kl::make_tma_load<A>(kl::row_major(base + 1, 4096, 4096)), std::invalid_argument);  // misaligned
  EXPECT_THROW(kl::make_tma_load<A>(kl::row_major_strided(base, 4096, 4096, 1024)), std::invalid_argument);  // row_stride < cols
  // Pitch is the distance in bytes from the start of one row to the start of the next; TMA needs
  // it to be a multiple of 16. Here it's 4100 * 2 = 8200 bytes.
  EXPECT_THROW(kl::make_tma_load<A>(kl::row_major(base, 4096, 4100)), std::invalid_argument);  // pitch % 16 != 0
  EXPECT_THROW(kl::make_tma_load<A>(kl::row_major(base, 0, 4096)), std::invalid_argument);  // empty
  if (kl_test::have_tma_gpu()) {
    EXPECT_NO_THROW(kl::make_tma_load<A>(kl::row_major_strided(base, 4096, 4096, 4096 + 64)));  // padded rows are fine
  } else {
    try {
      kl::make_tma_load<A>(kl::row_major(base, 4096, 4096));
      ADD_FAILURE() << "expected a 'no TMA' error on a pre-sm_90 device";
    } catch (std::invalid_argument const& e) {
      EXPECT_NE(std::string(e.what()).find("no TMA"), std::string::npos) << e.what();
    }
  }
}
