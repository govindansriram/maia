#pragma once

#include <cstdint>

#include <kl/cute.hpp>

namespace kl {

using bf16 = cutlass::bfloat16_t;
using f16 = cutlass::half_t;
using e4m3 = cutlass::float_e4m3_t;
using e5m2 = cutlass::float_e5m2_t;

template <typename T>
inline constexpr int bits_of = cute::sizeof_bits_v<T>;

}  // namespace kl
