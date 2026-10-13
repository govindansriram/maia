#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <kl/kl.hpp>

#include "test_util.hpp"

namespace {

// The CuTe way, as CUTLASS kernels do it: a descriptor tensor over the (row block, K-slice)
// partition of the tile.
template <int BlockRows, typename Tile>
__device__ uint64_t cute_reference_desc(Tile const& tile, int m, int k) {
  using T = typename Tile::value_type;
  auto part = cute::zipped_divide(tile.tensor(), cute::Shape<cute::Int<BlockRows>, cute::Int<kl::mma_k<T>>>{});
  auto descs = cute::make_tensor<cute::UMMA::smem_desc<cute::UMMA::Major::K>>(part);
  return descs(0, cute::make_coord(m, k));
}

// Descriptors for every (row block, K-slice); output index is m * num_k_blocks + k.
template <typename Tile, int BlockRows>
__global__ void desc_kernel(uint64_t* kl_out, uint64_t* ref_out, uint32_t* base_out) {
  extern __shared__ __align__(1024) unsigned char smem[];
  auto& tile = *reinterpret_cast<Tile*>(smem);
  constexpr int nk = kl::num_k_blocks<Tile>;
  for (int m = 0; m < Tile::rows / BlockRows; ++m) {
    for (int k = 0; k < nk; ++k) {
      kl_out[m * nk + k] = kl::umma_desc(tile, k, m * BlockRows);
      ref_out[m * nk + k] = cute_reference_desc<BlockRows>(tile, m, k);
    }
  }
  *base_out = tile.smem_u32();
}

struct Descs {
  thrust::host_vector<uint64_t> kl, ref;
  uint32_t base = 0;
};

// Computes kl's and CuTe's descriptors for a Tile on the device.
template <typename Tile, int BlockRows = Tile::rows>
void run_desc_kernel(Descs& out) {
  constexpr int n = (Tile::rows / BlockRows) * kl::num_k_blocks<Tile>;
  thrust::device_vector<uint64_t> d_kl(n), d_ref(n);
  thrust::device_vector<uint32_t> d_base(1);
  constexpr int smem_bytes = sizeof(Tile);
  if constexpr (smem_bytes > 48 * 1024) {
    ASSERT_EQ(cudaFuncSetAttribute(desc_kernel<Tile, BlockRows>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   smem_bytes),
              cudaSuccess);
  }
  desc_kernel<Tile, BlockRows><<<1, 1, smem_bytes>>>(thrust::raw_pointer_cast(d_kl.data()),
                                                     thrust::raw_pointer_cast(d_ref.data()),
                                                     thrust::raw_pointer_cast(d_base.data()));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  out.kl = d_kl;
  out.ref = d_ref;
  out.base = d_base[0];
}

template <typename Tile, int BlockRows>
void expect_row_blocks_match_cute() {
  Descs d;
  run_desc_kernel<Tile, BlockRows>(d);
  for (size_t i = 0; i < d.kl.size(); ++i)
    EXPECT_EQ(d.kl[i], d.ref[i]) << "row block " << i / kl::num_k_blocks<Tile> << ", k_block "
                                 << i % kl::num_k_blocks<Tile>;
}

}  // namespace

TEST(InstrDesc, Bf16M128N256BothKMajor) {
  // The instruction descriptor is the 32-bit operand of tcgen05.mma that says *what* to compute
  // (the two smem descriptors say where A and B are). Layout, from cute::UMMA::InstrDescriptor:
  //   bits  0-3   sparsity, saturate          0: dense, no saturate
  //   bits  4-5   c_format (accumulator)      1: F32
  //   bits  7-9   a_format                    1: BF16 (0 = F16, 2 = TF32)
  //   bits 10-12  b_format                    1: BF16
  //   bits 13-14  negate A / B                0: no
  //   bits 15-16  a_major / b_major           0: K-major (1 = MN-major)
  //   bits 17-22  N >> 3                      256 >> 3 = 32
  //   bits 24-28  M >> 4                      128 >> 4 = 8
  //   bits 29-31  k_size, max_shift           0: only used by other MMA kinds
  // Built by hand here, independently of CuTe, so a swapped M/N or type argument shows up.
  constexpr uint32_t expected = (1u << 4) | (1u << 7) | (1u << 10) | (32u << 17) | (8u << 24);
  static_assert(expected == 0x08400490u);
  EXPECT_EQ((kl::mma_instr_desc<kl::bf16, kl::bf16, float, 128, 256>()), expected);
}

TEST(UmmaDesc, KnownFieldsBf16K128x64Sw128) {
  KL_REQUIRE_GPU();
  using A = kl::SmemTile<kl::bf16, 128, 64>;
  Descs d;
  run_desc_kernel<A>(d);
  ASSERT_EQ(d.kl.size(), 4u);
  for (int k = 0; k < 4; ++k) {
    SCOPED_TRACE(testing::Message() << "k_block " << k);
    kl::SmemDesc desc;
    desc.desc_ = d.kl[k];
    EXPECT_EQ(int(desc.layout_type_), int(cute::UMMA::LayoutType::SWIZZLE_128B));
    EXPECT_EQ(int(desc.version_), 1);
    EXPECT_EQ(int(desc.stride_byte_offset_), 1024 / 16);  // 8 rows x 128 B between core-matrix groups
    EXPECT_EQ(int(desc.leading_byte_offset_), 1);
    EXPECT_EQ(int(desc.base_offset_), 0);
    // each MMA K-slice is 16 bf16 = 32 B further along the (un-swizzled) row
    EXPECT_EQ(int(desc.start_address_), int((d.base >> 4) + 2 * k));
  }
}

template <typename Tile>
class UmmaDescMatchesCute : public testing::Test {};

using DescTiles = testing::Types<
    kl::SmemTile<kl::bf16, 128, 64>,                              // K, 128B swizzle
    kl::SmemTile<kl::bf16, 256, 64>,                              // K, N = 256
    kl::SmemTile<kl::bf16, 128, 128>,                             // K, two atom columns
    kl::SmemTile<kl::bf16, 128, 64, kl::Swizzle64B>,              // K, 64B swizzle
    kl::SmemTile<kl::bf16, 128, 32, kl::Swizzle32B>,              // K, 32B swizzle
    kl::SmemTile<int8_t, 128, 128>,                               // int8
    kl::SmemTile<kl::e4m3, 128, 128>,                             // fp8
    kl::SmemTile<float, 128, 32>>;                                // tf32
TYPED_TEST_SUITE(UmmaDescMatchesCute, DescTiles);

TYPED_TEST(UmmaDescMatchesCute, EveryKBlock) {
  KL_REQUIRE_GPU();
  Descs d;
  run_desc_kernel<TypeParam>(d);
  for (size_t k = 0; k < d.kl.size(); ++k) EXPECT_EQ(d.kl[k], d.ref[k]) << "k_block " << k;
}

// One tile feeding several MMAs along M or N.
TEST(UmmaDesc, RowBlocksMatchCute) {
  KL_REQUIRE_GPU();
  {
    SCOPED_TRACE("bf16 256x64, two M = 128 blocks");
    expect_row_blocks_match_cute<kl::SmemTile<kl::bf16, 256, 64>, 128>();
  }
  {
    SCOPED_TRACE("bf16 512x64, two N = 256 blocks");
    expect_row_blocks_match_cute<kl::SmemTile<kl::bf16, 512, 64>, 256>();
  }
  {
    SCOPED_TRACE("bf16 128x128, two M = 64 blocks, two atom columns");
    expect_row_blocks_match_cute<kl::SmemTile<kl::bf16, 128, 128>, 64>();
  }
  {
    SCOPED_TRACE("bf16 256x64 SW64, N = 128 blocks");
    expect_row_blocks_match_cute<kl::SmemTile<kl::bf16, 256, 64, kl::Swizzle64B>, 128>();
  }
  {
    SCOPED_TRACE("int8 256x128, M = 128 blocks");
    expect_row_blocks_match_cute<kl::SmemTile<int8_t, 256, 128>, 128>();
  }
}

TEST(UmmaDesc, RowBlockStartsOnPatternBoundary) {
  KL_REQUIRE_GPU();
  // bf16 512x64 SW128: row 256 is 256 x 128 B = 32 KB in, so base_offset stays 0.
  using B = kl::SmemTile<kl::bf16, 512, 64>;
  Descs d;
  run_desc_kernel<B, 256>(d);
  kl::SmemDesc second;
  second.desc_ = d.kl[1 * kl::num_k_blocks<B>];
  EXPECT_EQ(int(second.start_address_), int((d.base + 32 * 1024) >> 4));
  EXPECT_EQ(int(second.base_offset_), 0);
}
