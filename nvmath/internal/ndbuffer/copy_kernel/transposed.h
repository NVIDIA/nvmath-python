// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_TRANSPOSED_H
#define NVMATH_COPY_KERNEL_TRANSPOSED_H

#include "args.h"
#include "copy_kernel_impl/array_view.h"
#include "copy_kernel_impl/grid_indexer.h"
#include "copy_kernel_impl/transposed.h"
#include "copy_kernel_impl/type_utils.h"
#include "copy_kernel_impl/utils.h"
#include "copy_kernel_impl/vec.h"

#define TRANSPOSE_KERNEL(stride_t, ndim, itemsize, needs_grid_stride_loop, transposed_dim, tile_y, tile_x,             \
                         reading_order)                                                                                \
  extern "C" {                                                                                                         \
  void __global__ transpose_copy(const nvmath::KernelArgs<ndim> args) {                                                \
    nvmath::transpose_copy<nvmath::##stride_t, ndim, itemsize, needs_grid_stride_loop, transposed_dim, tile_y, tile_x, \
                           reading_order>                                                                              \
        kernel;                                                                                                        \
    kernel(args);                                                                                                      \
  }                                                                                                                    \
  }

namespace nvmath {

template <typename stride_t, int ndim, int itemsize, int needs_grid_stride_loop, int transposed_dim, int tile_y,
          int tile_x, char reading_order>
struct transpose_copy {
  using dtype_t = opaque_t<itemsize>;
  using coords_t = vec<ndim, stride_t>;
  using dst_array_view_t = array_view<dtype_t, coords_t>;
  using src_array_view_t = array_view<const dtype_t, coords_t>;
  using grid_indexer_t = block_indexer<stride_t, needs_grid_stride_loop>;
  static_assert(tile_y > 0 && tile_x > 0, "tile_y and tile_x must be positive");
  using copy_helper_t = detail::transpose2d<dtype_t, stride_t, transposed_dim, tile_y, tile_x, reading_order>;

  void __forceinline__ __device__ operator()(const KernelArgs<ndim> args) {
    __shared__ stride_t dst_offsets[copy_helper_t::tile_num_elements];
    __shared__ dtype_t shared_data[copy_helper_t::tile_num_elements];
    coords_t dst_shape{args.dst_shape};
    coords_t src_shape{args.src_shape};
    coords_t dst_strides{args.dst_strides};
    coords_t src_strides{args.src_strides};
    dst_array_view_t dst_array_view{static_cast<dtype_t *>(args.dst_ptr), std::move(dst_shape), std::move(dst_strides)};
    src_array_view_t src_array_view{static_cast<const dtype_t *>(args.src_ptr), std::move(src_shape),
                                    std::move(src_strides)};
    auto kernel = transpose_copy_impl<dst_array_view_t, src_array_view_t, grid_indexer_t, copy_helper_t>{};
    kernel(std::move(dst_array_view), std::move(src_array_view), grid_indexer_t{static_cast<stride_t>(args.grid_arg)},
           copy_helper_t{shared_data, dst_offsets});
  }
};

} // namespace nvmath

#endif // NVMATH_COPY_KERNEL_TRANSPOSED_H
