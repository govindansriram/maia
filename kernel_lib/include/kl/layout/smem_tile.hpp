#pragma once

#include <cstdint>
#include <type_traits>

#include <kl/config.hpp>
#include <kl/cute.hpp>
#include <kl/types.hpp>

namespace kl {

// Shared-memory swizzle modes, named by their span in bytes. `bits` is the B in cute::Swizzle<B,4,3>.
struct SwizzleNone {
  static constexpr int bytes = 16;
  static constexpr int bits = 0;
};
struct Swizzle32B {
  static constexpr int bytes = 32;
  static constexpr int bits = 1;
};
struct Swizzle64B {
  static constexpr int bytes = 64;
  static constexpr int bits = 2;
};
struct Swizzle128B {
  static constexpr int bytes = 128;
  static constexpr int bits = 3;
};

// 8 rows x `Sw::bytes` swizzle atom, written in bits so upcast<bits_of<T>> fits any element width
// (incl. sub-byte). smem_ptr_flag defers the swizzle until a smem pointer exists, since it XORs the
// pointer's byte address. Identical to CuTe's Layout_K_*_Atom (checked in tests).
template <typename T, typename Sw>
using smem_atom = decltype(cute::upcast<bits_of<T>>(
    cute::ComposedLayout<cute::Swizzle<Sw::bits, 4, 3>, cute::smem_ptr_flag,
                         cute::Layout<cute::Shape<cute::_8, cute::Int<Sw::bytes * 8>>,
                                      cute::Stride<cute::Int<Sw::bytes * 8>, cute::_1>>>{}));

// Atoms are stacked down the rows first, so each atom-wide strip of columns is one contiguous
// block: exactly one TMA box.
template <typename T, int Rows, int Cols, typename Sw>
using smem_layout = decltype(cute::tile_to_shape(smem_atom<T, Sw>{}, cute::Shape<cute::Int<Rows>, cute::Int<Cols>>{}));

// A swizzled, row-major (cols contiguous) tile in shared memory. MMA operands are (M or N, K),
// so they are K-major; The tile owns its storage, so
// it can be a plain member of a kernel's shared-memory struct. The layout type is the single source of truth for the
// TMA box, the UMMA descriptor and address math.
template <typename T, int Rows, int Cols, typename Sw = Swizzle128B>
struct SmemTile {
  using value_type = T;
  using swizzle = Sw;
  using layout_type = smem_layout<T, Rows, Cols, Sw>;

  static constexpr int rows = Rows;
  static constexpr int cols = Cols;
  static constexpr int size = Rows * Cols;
  static constexpr int bytes = size * bits_of<T> / 8;
  // Swizzle patterns repeat every 8 rows of `Sw::bytes`; TMA needs at least 128 B.
  static constexpr int alignment = Sw::bytes * 8 < 128 ? 128 : Sw::bytes * 8;

  static constexpr int atom_elems = Sw::bytes * 8 / bits_of<T>;  // columns per swizzle atom

  static_assert(bits_of<T> >= 8, "sub-byte element types are not supported yet");
  static_assert(Cols % atom_elems == 0, "cols must be a multiple of the swizzle atom width");
  static_assert(Rows % 8 == 0, "rows must be a multiple of 8");

  alignas(alignment) unsigned char storage[bytes];

  KL_HOST_DEVICE T* data() { return reinterpret_cast<T*>(storage); }
  KL_HOST_DEVICE T const* data() const { return reinterpret_cast<T const*>(storage); }

  KL_HOST_DEVICE auto tensor() { return cute::make_tensor(cute::make_smem_ptr(data()), layout_type{}); }
  KL_HOST_DEVICE auto tensor() const { return cute::make_tensor(cute::make_smem_ptr(data()), layout_type{}); }

  KL_DEVICE uint32_t smem_u32() const { return cute::cast_smem_ptr_to_uint(storage); }
};

template <typename T>
inline constexpr bool is_smem_tile_v = false;
template <typename T, int R, int C, typename S>
inline constexpr bool is_smem_tile_v<SmemTile<T, R, C, S>> = true;

template <typename T>
concept SmemTileType = is_smem_tile_v<std::remove_cvref_t<T>>;

}  // namespace kl
