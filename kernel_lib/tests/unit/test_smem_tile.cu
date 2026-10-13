#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <vector>

#include <kl/kl.hpp>

// Reference only: kl itself never includes atom headers.
#include <cute/atom/mma_traits_sm90_gmma.hpp>

namespace GMMA = cute::SM90::GMMA;

template <typename T>
constexpr bool atoms_match_cute() {
  return std::is_same_v<kl::smem_atom<T, kl::SwizzleNone>, GMMA::Layout_K_INTER_Atom<T>> &&
         std::is_same_v<kl::smem_atom<T, kl::Swizzle32B>, GMMA::Layout_K_SW32_Atom<T>> &&
         std::is_same_v<kl::smem_atom<T, kl::Swizzle64B>, GMMA::Layout_K_SW64_Atom<T>> &&
         std::is_same_v<kl::smem_atom<T, kl::Swizzle128B>, GMMA::Layout_K_SW128_Atom<T>>;
}

static_assert(atoms_match_cute<kl::bf16>());
static_assert(atoms_match_cute<kl::f16>());
static_assert(atoms_match_cute<float>());
static_assert(atoms_match_cute<int8_t>());
static_assert(atoms_match_cute<kl::e4m3>());

TEST(SmemTile, SizesAndAlignment) {
  using A = kl::SmemTile<kl::bf16, 128, 64>;
  static_assert(A::bytes == 128 * 64 * 2);
  static_assert(A::alignment == 1024);
  static_assert(sizeof(A) == A::bytes);
  static_assert(alignof(A) == 1024);
  static_assert(A::atom_elems == 64);
  EXPECT_EQ(int(cute::size(A::layout_type{})), 128 * 64);
  EXPECT_EQ(int(cute::cosize(A::layout_type{})), 128 * 64);

  using W = kl::SmemTile<int8_t, 128, 128>;
  static_assert(W::atom_elems == 128);
  static_assert(W::bytes == 128 * 128);

  using C = kl::SmemTile<kl::bf16, 128, 32, kl::Swizzle64B>;
  static_assert(C::alignment == 512);
  static_assert(C::atom_elems == 32);
}


// Every (row, col) maps to its own storage offset and every offset in [0, size) is used exactly
// once: no two elements overlap, and nothing lands past the end of `storage`, which is sized as
// rows * cols elements. TMA box offsets, UMMA descriptors and tile sizes all rely on this.
TEST(SmemTile, LayoutIsBijectionOntoStorage) {
  using A = kl::SmemTile<kl::bf16, 128, 128>;
  auto L = A::layout_type{};
  std::vector<int> hits(A::size, 0);
  for (int r = 0; r < A::rows; ++r)
    for (int c = 0; c < A::cols; ++c) ++hits[L(r, c)];
  for (int i = 0; i < A::size; ++i) ASSERT_EQ(hits[i], 1) << "offset " << i;
}

TEST(SmemTile, StructOfTilesKeepsAlignment) {
  struct Stage {
    kl::SmemTile<kl::bf16, 128, 64> a;
    kl::SmemTile<kl::bf16, 256, 64> b;
  };
  static_assert(offsetof(Stage, b) % 1024 == 0);
  static_assert(sizeof(Stage) == 128 * 64 * 2 + 256 * 64 * 2);
}

TEST(TmaBox, CoversTileInAtomWideStrips) {
  using A = kl::SmemTile<kl::bf16, 128, 64>;
  static_assert(kl::TmaBox<A>::inner == 64 && kl::TmaBox<A>::outer == 128 && kl::TmaBox<A>::count == 1);
  static_assert(kl::TmaBox<A>::bytes == A::bytes);

  // 128 bf16 of K with 128B swizzle: two atom columns, each a contiguous 128x64 block.
  using X = kl::SmemTile<kl::bf16, 128, 128>;
  static_assert(kl::TmaBox<X>::inner == 64 && kl::TmaBox<X>::count == 2);
  EXPECT_EQ(kl::TmaBox<X>::smem_offset(0), 0u);
  EXPECT_EQ(kl::TmaBox<X>::smem_offset(1), 128u * 64 * 2);

  // 512 rows: boxes are capped at 256 rows, so 2 row blocks x 2 column strips, in storage order.
  using T = kl::TmaBox<kl::SmemTile<kl::bf16, 512, 128>>;
  static_assert(T::outer == 256 && T::row_boxes == 2 && T::col_boxes == 2 && T::count == 4);
  constexpr int expected[4][3] = {{0, 0, 0}, {256, 0, 32768}, {0, 64, 65536}, {256, 64, 98304}};  // row, col, bytes
  for (int i = 0; i < T::count; ++i) {
    SCOPED_TRACE(testing::Message() << "box " << i);
    EXPECT_EQ(T::row(i), expected[i][0]);
    EXPECT_EQ(T::col(i), expected[i][1]);
    EXPECT_EQ(T::smem_offset(i), uint32_t(expected[i][2]));
  }
}
