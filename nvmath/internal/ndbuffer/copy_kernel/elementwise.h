// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_ELEMENTWISE_H
#define NVMATH_COPY_KERNEL_ELEMENTWISE_H

#include "args.h"
#include "copy_kernel_impl/array_view.h"
#include "copy_kernel_impl/elementwise.h"
#include "copy_kernel_impl/grid_indexer.h"
#include "copy_kernel_impl/type_utils.h"
#include "copy_kernel_impl/utils.h"
#include "copy_kernel_impl/vec.h"

#define ELEMENTWISE_KERNEL(stride_t, dst_ndim, src_ndim, itemsize, needs_grid_stride_loop)                             \
  extern "C" {                                                                                                         \
  constexpr int N = dst_ndim > src_ndim ? dst_ndim : src_ndim;                                                         \
  void __global__ elementwise_copy(const nvmath::KernelArgs<N> args) {                                                 \
    nvmath::elementwise_copy<nvmath::##stride_t, dst_ndim, src_ndim, itemsize, needs_grid_stride_loop> kernel;         \
    kernel(args);                                                                                                      \
  }                                                                                                                    \
  }

namespace nvmath {

template <typename stride_t, int dst_ndim, int src_ndim, int itemsize, int needs_grid_stride_loop>
struct elementwise_copy {
  using dtype_t = opaque_t<itemsize>;
  using dst_coords_t = vec<dst_ndim, stride_t>;
  using src_coords_t = vec<src_ndim, stride_t>;
  using dst_array_view_t = array_view<dtype_t, dst_coords_t>;
  using src_array_view_t = array_view<const dtype_t, src_coords_t>;
  using grid_indexer_t = element_indexer<stride_t, needs_grid_stride_loop>;
  constexpr static bool has_equal_shapes = dst_ndim == src_ndim;
  constexpr static int ndim = dst_ndim > src_ndim ? dst_ndim : src_ndim;

  void __forceinline__ __device__ operator()(const KernelArgs<ndim> args) {
    dst_coords_t dst_shape{args.dst_shape};
    src_coords_t src_shape{args.src_shape};
    dst_coords_t dst_strides{args.dst_strides};
    src_coords_t src_strides{args.src_strides};
    dst_array_view_t dst_array_view{static_cast<dtype_t *>(args.dst_ptr), std::move(dst_shape), std::move(dst_strides)};
    src_array_view_t src_array_view{static_cast<const dtype_t *>(args.src_ptr), std::move(src_shape),
                                    std::move(src_strides)};
    auto kernel = elementwise_copy_impl<has_equal_shapes, dst_array_view_t, src_array_view_t, grid_indexer_t>{};
    kernel(std::move(dst_array_view), std::move(src_array_view), grid_indexer_t{static_cast<stride_t>(args.grid_arg)});
  }
};

} // namespace nvmath

#endif // NVMATH_COPY_KERNEL_ELEMENTWISE_H
