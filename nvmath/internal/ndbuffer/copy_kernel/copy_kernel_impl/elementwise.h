// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_IMPL_ELEMENTWISE_H
#define NVMATH_COPY_KERNEL_IMPL_ELEMENTWISE_H

#include "copy_kernel_impl/array_view.h"
#include "copy_kernel_impl/grid_indexer.h"
#include "copy_kernel_impl/type_utils.h"
#include "copy_kernel_impl/utils.h"
#include "copy_kernel_impl/vec.h"

namespace nvmath {

namespace detail {

template <typename coords_t, typename stride_t = typename coords_t::type>
__device__ __forceinline__ coords_t unravel_idx(const stride_t flat_idx, const coords_t shape) {
  constexpr int ndim = coords_t::ndim;
  if constexpr (ndim <= 0) {
    return {};
  } else if constexpr (ndim == 1) {
    return {flat_idx};
  } else if constexpr (ndim > 1) {

    // the extents cannot be negative and the arithmetic on unsigned integer
    // is noticeably faster
    using u_stride_t = typename type_traits::unsign<stride_t>::type;
    u_stride_t u_flat_idx = flat_idx;
    coords_t unraveled_coords;
#pragma unroll
    for (int i = ndim - 1; i >= 1; i--) {
      u_stride_t extent = shape[i];
      if (extent & (extent - 1)) {
        u_stride_t next_flat_idx = u_flat_idx / extent;
        unraveled_coords[i] = u_flat_idx - next_flat_idx * extent;
        u_flat_idx = next_flat_idx;
      } else {
        unraveled_coords[i] = u_flat_idx & (extent - 1);
        u_flat_idx >>= ffs(extent) - 1;
      }
    }
    unraveled_coords[0] = u_flat_idx;
    return unraveled_coords;
  }
}

} // namespace detail

template <bool has_equal_shapes, typename dst_array_view_t, typename src_array_view_t, typename grid_indexer_t>
struct elementwise_copy_impl {
  using stride_t = typename dst_array_view_t::stride_t;

  void __forceinline__ __device__ operator()(const dst_array_view_t &&dst_view, const src_array_view_t &&src_view,
                                             const grid_indexer_t &&grid_helper) {
    grid_helper.with_grid_stride_loop([=](const stride_t flat_element_idx) {
      const auto dst_coords = detail::unravel_idx(flat_element_idx, dst_view.shape());
      const auto src_coords =
          cond_val(bconst<has_equal_shapes>(), dst_coords, detail::unravel_idx(flat_element_idx, src_view.shape()));
      dst_view[dst_coords] = src_view[src_coords];
    });
  }
};

} // namespace nvmath

#endif // NVMATH_COPY_KERNEL_IMPL_ELEMENTWISE_H
