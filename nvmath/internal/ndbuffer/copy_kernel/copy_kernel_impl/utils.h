// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_IMPL_UTILS_H_
#define NVMATH_COPY_KERNEL_IMPL_UTILS_H_

#include "copy_kernel_impl/type_utils.h"

#if defined(__CUDACC__)
#define HOST_DEV __host__ __device__
#else
#define HOST_DEV
#endif

namespace nvmath {

// Some of stl type traits are not available with nvrtc
namespace type_traits {
template <bool B, class T, class F> struct conditional {};

template <class T, class F> struct conditional<true, T, F> {
  using type = T;
};

template <class T, class F> struct conditional<false, T, F> {
  using type = F;
};

template <bool B, class T = void> struct enable_if {};
template <class T> struct enable_if<true, T> {
  typedef T type;
};

template <typename T> struct unsign {};

template <> struct unsign<int64_t> {
  using type = uint64_t;
};

template <> struct unsign<int32_t> {
  using type = uint32_t;
};

template <typename T> struct is_32_or_64_int {
  static constexpr bool value = false;
};

template <> struct is_32_or_64_int<int32_t> {
  static constexpr bool value = true;
};

template <> struct is_32_or_64_int<int64_t> {
  static constexpr bool value = true;
};

template <typename T> struct min_val {};

template <> struct min_val<int32_t> {
  static constexpr int32_t value = -2147483648;
};

template <> struct min_val<int64_t> {
  static constexpr int64_t value = -9223372036854775808LL;
};
} // namespace type_traits

template <typename T, T val> struct const_val {
  using type = T;
  static constexpr T value = val;
};

template <int N> using iconst = const_val<int, N>;

template <bool B> using bconst = const_val<bool, B>;

template <typename true_val_t, typename false_val_t>
auto HOST_DEV __forceinline__ constexpr cond_val(bconst<false>, true_val_t &&true_val, false_val_t &&false_val) {
  return false_val;
}

template <typename true_val_t, typename false_val_t>
auto HOST_DEV __forceinline__ constexpr cond_val(bconst<true>, true_val_t &&true_val, false_val_t &&false_val) {
  return true_val;
}

#if defined(__CUDACC__)

__device__ __forceinline__ int ffs(uint32_t x) { return __ffs(x); }

__device__ __forceinline__ int ffs(int32_t x) { return __ffs(x); }

__device__ __forceinline__ int ffs(uint64_t x) { return __ffsll(x); }

__device__ __forceinline__ int ffs(int64_t x) { return __ffsll(x); }

#endif

constexpr int log2_floor(const int k) {
  return k == 1 ? 0 : 1 + log2_floor(k >> 1);
}

template <int k>
struct mod_div {
  static_assert(k > 0, "k must be positive");
  static_assert((k & (k - 1)) == 0, "k must be a power of 2");
  static constexpr int value = k;
  static constexpr int log2 = log2_floor(k);
  static constexpr int mask = k - 1;
  constexpr int operator()() const {
    return k;
  }
};

template <typename T, int k>
HOST_DEV __forceinline__ constexpr T operator/(const T a, const mod_div<k>) {
  return a >> mod_div<k>::log2;
}

template <typename T, int k>
HOST_DEV __forceinline__ constexpr T operator%(const T a, const mod_div<k>) {
  return a & mod_div<k>::mask;
}

template <typename T, int k>
HOST_DEV __forceinline__ constexpr T operator*(const T a, const mod_div<k>) {
  return a << mod_div<k>::log2;
}

} // namespace nvmath

#endif // NVMATH_COPY_KERNEL_IMPL_UTILS_H_
