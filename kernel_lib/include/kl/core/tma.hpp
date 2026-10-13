#pragma once

#include <cassert>
#include <cstdint>
#include <utility>

#include <kl/config.hpp>
#include <kl/core/warp.hpp>
#include <kl/cute.hpp>
#include <kl/layout/smem_tile.hpp>
#include <kl/layout/tma_desc.hpp>

// Device side of TMA: copy whole SmemTiles between shared memory and a global tensor described
// by a TmaLoad / TmaStore (see layout/tma_desc.hpp). Each call is made by one whole warp; one
// lane issues the copies, one per TMA box of the tile.
namespace kl {

// Where a tile sits in its global tensor: one entry per mode, outermost first like the tensor's
// shape, all counted in tiles. The tile is 1 deep in the outer modes (batch, heads, ...), so
// those entries are plain indices: for row_major(q, B, S, D), {b, s_tile, d_tile}.
template <int Rank>
struct TileCoord {
  int32_t v[Rank];
};

namespace detail {

// Element coordinates of box i of the tile at `c`, innermost mode first as TMA wants them.
template <SmemTileType Tile, int Rank>
KL_DEVICE void box_coords(TileCoord<Rank> const& c, int i, int32_t (&crd)[Rank]) {
  crd[0] = c.v[Rank - 1] * Tile::cols + TmaBox<Tile>::col(i);
  crd[1] = c.v[Rank - 2] * Tile::rows + TmaBox<Tile>::row(i);
  for (int m = 2; m < Rank; ++m) crd[m] = c.v[Rank - 1 - m];
}

// CuTe's SM90_TMA_LOAD / SM90_TMA_STORE pick the 1-D..5-D instruction from the argument count.
template <int Rank, typename F>
KL_DEVICE void with_coords(int32_t const (&crd)[Rank], F&& f) {
  [&]<int... I>(std::integer_sequence<int, I...>) { f(crd[I]...); }(std::make_integer_sequence<int, Rank>{});
}

}  // namespace detail

// Pull the descriptor into the TMA unit's cache ahead of first use. Optional.
template <SmemTileType Tile, typename Direction, int Rank>
KL_DEVICE void prefetch(TmaDesc<Tile, Direction, Rank> const& desc) {
  cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(desc.get()));
}

// Load the tile at `c` into `dst`. Completion is signalled on `bar`, which must already expect
// Tile::bytes for this phase (barrier::arrive_expect_tx); out-of-bounds elements are zero-filled
// and still count toward those bytes.
template <SmemTileType Tile, int Rank>
KL_DEVICE void tma_load(Tile& dst, TmaLoad<Tile, Rank> const& desc, TileCoord<Rank> const& c, uint64_t* bar) {
  assert(dst.smem_u32() % Tile::alignment == 0);
  if (elect_one()) {
    for (int i = 0; i < TmaBox<Tile>::count; ++i) {
      int32_t crd[Rank];
      detail::box_coords<Tile>(c, i, crd);
      detail::with_coords(crd, [&](auto... x) {
        cute::SM90_TMA_LOAD::copy(desc.get(), bar, static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
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
  cutlass::arch::fence_view_async_shared();  // make generic-proxy smem writes visible to TMA
  __syncwarp();
  if (lane_id() == 0) {
    for (int i = 0; i < TmaBox<Tile>::count; ++i) {
      int32_t crd[Rank];
      detail::box_coords<Tile>(c, i, crd);
      detail::with_coords(crd, [&](auto... x) {
        cute::SM90_TMA_STORE::copy(desc.get(), src.storage + TmaBox<Tile>::smem_offset(i), x...);
      });
    }
    cute::tma_store_arrive();  // commit this tile's stores as one bulk async-group
  }
  __syncwarp();
}

// Wait until at most `Pending` of this warp's tma_store calls are still reading shared memory.
// After that, their source tiles may be overwritten. Call with 0 before the kernel exits.
template <int Pending = 0>
KL_DEVICE void tma_store_wait() {
  if (lane_id() == 0) cute::tma_store_wait<Pending>();
  __syncwarp();
}

}  // namespace kl
