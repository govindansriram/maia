#pragma once

#include <cassert>
#include <cstdint>

#include <kl/config.hpp>
#include <kl/cute.hpp>
#include <kl/layout/smem_tile.hpp>
#include <kl/types.hpp>

namespace kl {

using SmemDesc = cute::UMMA::SmemDescriptor;

// K covered by one dense tcgen05.mma (kind::f16, tf32, f8f6f4, i8): always 32 bytes.
template <typename T>
inline constexpr int mma_k = 256 / bits_of<T>;

template <SmemTileType Tile>
inline constexpr int num_k_blocks = Tile::cols / mma_k<typename Tile::value_type>;

// Offset of the k-th MMA K-slice, starting at `row`, from the start of the tile, in 16-byte units.
// Measured on the un-swizzled layout: the hardware applies the swizzle itself.
template <SmemTileType Tile>
KL_HOST_DEVICE constexpr uint32_t block_offset_16B(int k_block, int row = 0) {
  using T = typename Tile::value_type;
  auto inner = typename Tile::layout_type{}.layout_b();
  return static_cast<uint32_t>(inner(row, k_block * mma_k<T>) * bits_of<T> / 128);
}

// Shared-memory descriptor for the k-th MMA K-slice of `tile`, for rows [row, row + M or N).
// `row` lets one tile feed several MMAs along M or N (e.g. a 256-row B tile as two N = 128
// MMAs); it must be a multiple of 8 so the slice starts on a swizzle-pattern boundary.
// Field encoding comes from CuTe.
template <SmemTileType Tile>
KL_DEVICE SmemDesc umma_desc(Tile const& tile, int k_block = 0, int row = 0) {
  using T = typename Tile::value_type;
  static_assert(Tile::cols % mma_k<T> == 0, "tile K extent must be a multiple of the MMA K");
  assert(row % 8 == 0 && row < Tile::rows);
  auto first = cute::local_tile(tile.tensor(), cute::Shape<cute::Int<Tile::rows>, cute::Int<mma_k<T>>>{},
                                cute::make_coord(0, 0));
  SmemDesc desc = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(first);
  // Same 32-bit add CuTe's DescriptorIterator does: start address is in the low bits, and the
  // shared-memory window is small enough that it never carries into the LBO field.
  desc.lo += block_offset_16B<Tile>(k_block, row);
  return desc;
}

// 32-bit instruction descriptor for a dense, cta_group::1 tcgen05.mma with K-major A and B.
// Not usable in constant expressions: CuTe builds it through a bitfield union.
template <typename A, typename B, typename Acc, int M, int N>
KL_HOST_DEVICE uint32_t mma_instr_desc() {
  static_assert(M == 64 || M == 128, "cta_group::1 supports M = 64 or 128");
  static_assert(N <= 256 && N >= (M == 64 ? 8 : 16) && N % (M == 64 ? 8 : 16) == 0,
                "N must be a multiple of 8 (M=64) or 16 (M=128), at most 256");
  return static_cast<uint32_t>(
      cute::UMMA::make_instr_desc<A, B, Acc, M, N, cute::UMMA::Major::K, cute::UMMA::Major::K>());
}

}  // namespace kl
