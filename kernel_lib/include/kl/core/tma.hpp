#pragma once

#include <cassert>
#include <cstdint>
#include <utility>

#include <kl/config.hpp>
#include <kl/core/warp.hpp>
#include <kl/cute.hpp>
#include <kl/layout/smem_tile.hpp>
#include <kl/layout/tma_desc.hpp>
#include <kl/ptx/bulk_group.hpp>
#include <kl/ptx/fence.hpp>
#include <kl/ptx/mbarrier.hpp>

// Device side of TMA: copy whole SmemTiles between shared memory and a global tensor described
// by a TmaLoad / TmaStore (see layout/tma_desc.hpp). Each call is made by one whole warp; one
// lane issues the copies, one per TMA box of the tile.
namespace kl {

// Index of a tile in its global tensor, counted in tiles, outermost mode first like the tensor's
// shape. The last two modes are split into tiles; the outer modes (batch, heads, ...) are taken
// one index at a time. For example, a [B, 1024, 4096] tensor in 256 x 64 tiles is [B, 4, 64]
// tiles, and {0, 1, 63} is batch 0, rows 256..511, columns 4032..4095.
template <int Rank>
struct TileCoord {
  int32_t v[Rank];
};

namespace detail {

// Element coordinates of one TMA box, innermost mode first as TMA wants them.
template <int Rank>
struct BoxCoord {
  int32_t v[Rank];
};

// Coordinates of the box numbered `linear_box_idx` (0 .. TmaBox<Tile>::count - 1) of the tile at `c`.
template <SmemTileType Tile, int Rank>
KL_DEVICE BoxCoord<Rank> box_coords(TileCoord<Rank> const& c, int linear_box_idx) {
  BoxCoord<Rank> crd;
  crd.v[0] = c.v[Rank - 1] * Tile::cols + TmaBox<Tile>::col(linear_box_idx);
  crd.v[1] = c.v[Rank - 2] * Tile::rows + TmaBox<Tile>::row(linear_box_idx);
  for (int m = 2; m < Rank; ++m) crd.v[m] = c.v[Rank - 1 - m];
  return crd;
}

// Calls f(crd.v[0], ..., crd.v[Rank - 1]): unpacks the box's coordinates into separate arguments.
// CuTe's SM90_TMA_LOAD / SM90_TMA_STORE take one argument per coordinate and pick the 1-D..5-D
// instruction from how many there are, so the unpacking happens at compile time.
template <int Rank, typename F>
KL_DEVICE void unpack_coords(BoxCoord<Rank> const& crd, F&& f) {
  // Make the index sequence 0, 1, ..., Rank - 1, then expand crd.v[I]... into f's argument list.
  [&]<int... I>(std::integer_sequence<int, I...>) { f(crd.v[I]...); }(std::make_integer_sequence<int, Rank>{});
}

}  // namespace detail

// Pull the descriptor into the TMA unit's cache ahead of first use. Optional. Called by one
// whole warp, like tma_load; one lane issues the prefetch.
template <SmemTileType Tile, typename Direction, int Rank>
KL_DEVICE void prefetch(TmaDesc<Tile, Direction, Rank> const& desc) {
  if (elect_one()) cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(desc.get()));
  __syncwarp();
}

// Load the tile at `c` into `dst`. Completion is signalled on `bar`, which must already expect
// Tile::bytes for this phase (bar.arrive_expect_tx); out-of-bounds elements are zero-filled and
// still count toward those bytes.
template <SmemTileType Tile, int Rank>
KL_DEVICE void tma_load(Tile& dst, TmaLoad<Tile, Rank> const& desc, TileCoord<Rank> const& c, Mbarrier& bar) {
  assert(dst.smem_u32() % Tile::alignment == 0);
  if (elect_one()) {
    for (int i = 0; i < TmaBox<Tile>::count; ++i) {
      auto const crd = detail::box_coords<Tile>(c, i);
      detail::unpack_coords(crd, [&](auto... x) {
        // EVICT_NORMAL: no L2 eviction preference, the same as no hint. It is CuTe's default, as in
        // CUTLASS's GEMMs; per-load hints (EVICT_FIRST for streamed data) would come with TmaOptions.
        cute::SM90_TMA_LOAD::copy(desc.get(), bar.ptr(), static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                  dst.storage + TmaBox<Tile>::smem_offset(i), x...);
      });
    }
  }
  __syncwarp();
}

// Store `src` to the tile at `c`; out-of-bounds elements are skipped. Call it once the tile is
// complete in shared memory: this warp's own writes are fenced here, but if other warps wrote
// to the tile they must fence (fence_view_async_shared) and sync with this warp first.
// Lane 0 issues the stores and owns their completion, so this same warp calls tma_store_wait.
template <SmemTileType Tile, int Rank>
KL_DEVICE void tma_store(TmaStore<Tile, Rank> const& desc, TileCoord<Rank> const& c, Tile const& src) {
  assert(src.smem_u32() % Tile::alignment == 0);
  fence_view_async_shared();  // make this warp's smem writes visible to TMA
  __syncwarp();
  if (lane_id() == 0) {
    for (int i = 0; i < TmaBox<Tile>::count; ++i) {
      auto const crd = detail::box_coords<Tile>(c, i);
      detail::unpack_coords(crd, [&](auto... x) {
        cute::SM90_TMA_STORE::copy(desc.get(), src.storage + TmaBox<Tile>::smem_offset(i), x...);
      });
    }
    bulk_commit_group();  // this tile's stores become one bulk async-group
  }
  __syncwarp();
}

// Wait until at most `Pending` of this warp's tma_store calls are still reading shared memory.
// After that, their source tiles may be overwritten. Call with 0 before the kernel exits.
template <int Pending = 0>
KL_DEVICE void tma_store_wait() {
  if (lane_id() == 0) bulk_wait_group_read<Pending>();
  __syncwarp();
}

}  // namespace kl
