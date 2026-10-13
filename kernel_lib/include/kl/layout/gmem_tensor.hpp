#pragma once

#include <cstdint>

#include <kl/config.hpp>
#include <kl/cute.hpp>

namespace kl {

// A compact row-major view of global memory as a CuTe tensor: the last mode is contiguous
// (stride _1). Extents are listed outermost first, the way shapes are usually written:
// row_major(a, M, K), row_major(q, B, H, S, D).
template <typename T, typename... Extents>
KL_HOST_DEVICE auto row_major(T* ptr, Extents... extents) {
  static_assert(sizeof...(Extents) >= 1, "row_major needs at least one extent");
  auto shape = cute::make_shape(static_cast<int64_t>(extents)...);
  return cute::make_tensor(cute::make_gmem_ptr(ptr), cute::make_layout(shape, cute::LayoutRight{}));
}

// A (rows x cols) row-major matrix whose rows are `row_stride` elements apart (row_stride >= cols).
// For any other layout, build the cute::Tensor directly.
//
// Example: a fused QKV projection writes qkv = X * W_qkv^T as one [tokens, 3*d] buffer, each row
// being [q | k | v]. Q, K and V are column slices of it: d elements per row, rows 3*d apart.
//   auto q = row_major_strided(qkv,         tokens, d, 3 * d);
//   auto k = row_major_strided(qkv + d,     tokens, d, 3 * d);
//   auto v = row_major_strided(qkv + 2 * d, tokens, d, 3 * d);
// (For TMA each slice must start 16-byte aligned: d a multiple of 8 for bf16.)
template <typename T>
KL_HOST_DEVICE auto row_major_strided(T* ptr, int64_t rows, int64_t cols, int64_t row_stride) {
  return cute::make_tensor(cute::make_gmem_ptr(ptr), cute::make_shape(rows, cols), cute::make_stride(row_stride, cute::_1{}));
}

}  // namespace kl
