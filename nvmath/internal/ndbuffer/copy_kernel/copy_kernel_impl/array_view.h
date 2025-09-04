// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_IMPL_ARRAY_VIEW_H_
#define NVMATH_COPY_KERNEL_IMPL_ARRAY_VIEW_H_

#include "copy_kernel_impl/utils.h"
#include "copy_kernel_impl/vec.h"

#if defined(_MSC_VER)
    // For Visual Studio, use __restrict
    #define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    // For GCC and Clang, use __restrict__
    #define RESTRICT __restrict__
#else
    // Fallback for other compilers, or if restrict is not supported
    #define RESTRICT
#endif

namespace nvmath {

template <typename T, typename _coords_t> struct array_view {
  // While indices cannot be negative (only strides can),
  // we're using the same 32- or 64-bit signed type to represent both
  // indices and strides for simplicity. In the end we need to convert
  // both to the same signed type when computing the offset.
  using coords_t = _coords_t;
  using stride_t = typename coords_t::type;
  using dtype_t = T;
  static constexpr int ndim = coords_t::ndim;

  HOST_DEV constexpr array_view(T *__restrict__ data, coords_t shape, coords_t strides)
      : shape_(shape), strides_(strides), data_(data) {}

  HOST_DEV __forceinline__ T &operator[](const coords_t idx) const { return data_[offset(idx)]; }
  HOST_DEV __forceinline__ T &operator[](const stride_t offset) const { return data_[offset]; }
  HOST_DEV __forceinline__ stride_t offset(const coords_t idx) const { return dot(idx, strides()); }
  HOST_DEV __forceinline__ coords_t shape() const { return shape_; }
  HOST_DEV __forceinline__ coords_t strides() const { return strides_; }
  HOST_DEV __forceinline__ T *data() const { return data_; }

protected:
  coords_t shape_;
  coords_t strides_;
  T *RESTRICT data_;
};

} // namespace nvmath

#endif // NVMATH_COPY_KERNEL_IMPL_ARRAY_VIEW_H_
