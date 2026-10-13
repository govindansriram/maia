#pragma once

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <kl/layout/smem_tile.hpp>
#include <kl/types.hpp>

namespace kl {

// Plain TMA loads/stores only need the element size (describing everything as UINT8 would work).
// Real types matter for reducing stores (add/min/max) and Blackwell's packed fp4/fp6 unpacking,
// and keep our descriptors byte-identical to CuTe's.
template <typename T>
constexpr CUtensorMapDataType tma_dtype() {
  if constexpr (std::is_same_v<T, bf16>) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr (std::is_same_v<T, f16>) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same_v<T, float>) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if constexpr (bits_of<T> == 8) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;  // int8 / uint8 / fp8 move as raw bytes
  } else {
    static_assert(sizeof(T) == 0, "no TMA data type for this element type");
  }
}

template <typename Sw>
constexpr CUtensorMapSwizzle tma_swizzle() {
  if constexpr (Sw::bytes == 128) return CU_TENSOR_MAP_SWIZZLE_128B;
  else if constexpr (Sw::bytes == 64) return CU_TENSOR_MAP_SWIZZLE_64B;
  else if constexpr (Sw::bytes == 32) return CU_TENSOR_MAP_SWIZZLE_32B;
  else return CU_TENSOR_MAP_SWIZZLE_NONE;
}

// How a tile is covered by TMA boxes. A box is one swizzle atom wide and at most 256 rows tall,
// so a wider or taller tile takes several TMA instructions. Box i is column strip i / row_boxes,
// row block i % row_boxes; that order walks the tile's storage front to back.
template <SmemTileType Tile>
struct TmaBox {
  using T = typename Tile::value_type;
  static constexpr int inner = Tile::atom_elems;
  static constexpr int outer = Tile::rows < 256 ? Tile::rows : 256;
  static constexpr int col_boxes = Tile::cols / inner;
  static constexpr int row_boxes = Tile::rows / outer;
  static constexpr int count = col_boxes * row_boxes;
  static constexpr int bytes = inner * outer * bits_of<T> / 8;

  static_assert(Tile::rows % outer == 0, "tiles taller than 256 rows must be a multiple of 256 rows");
  // cuTensorMapEncodeTiled's box rules. They hold by construction (a box row is one swizzle
  // span); the asserts keep them true if the box shape ever changes.
  static_assert(inner <= 256 && outer <= 256, "TMA boxes are at most 256 along each dimension");
  static_assert(inner * bits_of<T> / 8 % 16 == 0, "a TMA box row must be a multiple of 16 bytes");
  static_assert(inner * bits_of<T> / 8 <= Tile::swizzle::bytes,
                "a swizzled TMA box row can't be wider than the swizzle span");

  // Position of box i's first element inside the tile, in elements.
  KL_HOST_DEVICE static constexpr int row(int i) { return i % row_boxes * outer; }
  KL_HOST_DEVICE static constexpr int col(int i) { return i / row_boxes * inner; }

  // Byte offset of box i inside the tile.
  KL_HOST_DEVICE static constexpr uint32_t smem_offset(int i) {
    auto inner_layout = typename Tile::layout_type{}.layout_b();
    return static_cast<uint32_t>(inner_layout(row(i), col(i)) * bits_of<T> / 8);
  }
};

struct TmaLoadTag {};
struct TmaStoreTag {};

// A TMA descriptor for moving `Tile`s between shared memory and a rank-`Rank` global tensor.
// Pass it to kernels as `const __grid_constant__`.
template <SmemTileType Tile, typename Direction, int Rank = 2>
struct TmaDesc {
  using tile_type = Tile;
  using direction = Direction;
  static constexpr int rank = Rank;  // picks the TMA instruction (one coordinate per mode)
  CUtensorMap map;

  KL_HOST_DEVICE CUtensorMap const* get() const { return &map; }
};

template <SmemTileType Tile, int Rank = 2>
using TmaLoad = TmaDesc<Tile, TmaLoadTag, Rank>;
template <SmemTileType Tile, int Rank = 2>
using TmaStore = TmaDesc<Tile, TmaStoreTag, Rank>;

// Describe a global tensor for TMA (build it with kl::row_major, or pass any flat cute::Tensor).
// The last mode must be contiguous: TMA can't transpose, so global memory is row-major like the
// tile. The last two modes are the tile's (rows, cols); outer modes (batch, heads, ...) get a
// box of 1.
template <SmemTileType Tile, typename Direction, typename Engine, typename Layout>
auto make_tma_desc(cute::Tensor<Engine, Layout> const& g) {
  using T = typename Tile::value_type;
  using Box = TmaBox<Tile>;
  constexpr int R = Layout::rank;
  static_assert(std::is_same_v<std::remove_cv_t<typename Engine::value_type>, T>,
                "the global tensor's element type must match the tile's");
  static_assert(R >= 2 && R <= 5, "TMA tensors have 2 to 5 modes");
  static_assert(decltype(cute::depth(g.layout()))::value == 1, "global tensor modes must be flat");
  static_assert(cute::is_constant<1, decltype(cute::stride<R - 1>(g))>::value,
                "the last mode must be contiguous (static stride _1): kl supports row-major only");
  static_assert(!(std::is_same_v<Direction, TmaStoreTag> && std::is_const_v<typename Engine::element_type>),
                "a TMA store needs a non-const tensor");

  int64_t shape[R], stride[R];  // outermost mode first, as in g
  [&]<int... I>(std::integer_sequence<int, I...>) {
    ((shape[I] = cute::size<I>(g), stride[I] = cute::stride<I>(g)), ...);
  }(std::make_integer_sequence<int, R>{});

  // TMA lists dimensions innermost first; the innermost stride is implicit.
  // dims is the shape of the global memory tensor
  // box is the shape of the copy tile, this function always ensures the 
  // box only applies to the last 2 dimensions making the copy dim of others
  // all 1

  uint64_t dims[R], strides[R - 1];
  uint32_t box[R], elem_strides[R];
  for (int i = 0; i < R; ++i) {
    dims[i] = static_cast<uint64_t>(shape[R - 1 - i]);
    box[i] = i == 0 ? Box::inner : i == 1 ? Box::outer : 1;
    elem_strides[i] = 1;
  }
  for (int i = 1; i < R; ++i) strides[i - 1] = static_cast<uint64_t>(stride[R - 1 - i]) * bits_of<T> / 8;

  auto fail = [](std::string const& what) { throw std::invalid_argument("kl::make_tma_desc: " + what); };
  void const* gmem = cute::raw_pointer_cast(g.data());
  if (reinterpret_cast<uintptr_t>(gmem) % 16 != 0) fail("global address must be 16-byte aligned");
  for (int i = 0; i < R; ++i)
    if (shape[i] < 1 || shape[i] > (int64_t(1) << 32)) fail("extents must be in [1, 2^32]");
  for (int i = 0; i < R - 1; ++i)
    if (stride[i] <= 0 || strides[R - 2 - i] % 16 != 0 || strides[R - 2 - i] >= (uint64_t(1) << 40))
      fail("outer strides must be positive multiples of 16 bytes, below 2^40 bytes");
  if (stride[R - 2] < shape[R - 1]) fail("rows overlap: the row stride is smaller than cols");

  // NOTE: benchmark CU_TENSOR_MAP_L2_PROMOTION_L2_128B to see if this is the performant choice

  TmaDesc<Tile, Direction, R> desc{};
  CUresult res = cuTensorMapEncodeTiled(
      &desc.map, tma_dtype<T>(), R, const_cast<void*>(gmem), dims, strides, box, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, tma_swizzle<typename Tile::swizzle>(), CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (res == CUDA_ERROR_NOT_SUPPORTED) fail("the current device has no TMA (needs sm_90 or newer)");
  if (res != CUDA_SUCCESS) fail("cuTensorMapEncodeTiled failed with CUresult " + std::to_string(int(res)));
  return desc;
}

template <SmemTileType Tile, typename Engine, typename Layout>
auto make_tma_load(cute::Tensor<Engine, Layout> const& g) {
  return make_tma_desc<Tile, TmaLoadTag>(g);
}

template <SmemTileType Tile, typename Engine, typename Layout>
auto make_tma_store(cute::Tensor<Engine, Layout> const& g) {
  return make_tma_desc<Tile, TmaStoreTag>(g);
}

}  // namespace kl
