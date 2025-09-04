// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_IMPL_GRID_INDEXER_H_
#define NVMATH_COPY_KERNEL_IMPL_GRID_INDEXER_H_

#include "copy_kernel_impl/utils.h"
#include "copy_kernel_impl/vec.h"

namespace nvmath {

template <typename stride_t, bool _needs_grid_stride_loop> struct element_indexer {
  // stride_t can be 32-bit integer for tensor_volume and gridDim * blockDim up to INT_MAX,
  // this way unsigned x < INT_MAX; x += INT_MAX cannot overflow
  using ustride_t = typename type_traits::unsign<stride_t>::type;
  static constexpr bool needs_grid_stride_loop = _needs_grid_stride_loop;

  constexpr HOST_DEV __forceinline__ element_indexer(const stride_t tensor_volume) : tensor_volume(tensor_volume) {}

  template <typename Cb> __device__ __forceinline__ void with_grid_stride_loop(Cb &&cb) const {
    // early cast the special indexing variables to the desired integer width type
    // to avoid arithmetic on 32-bit integers when 64-bit stride_t is used
    const ustride_t thread_idx = threadIdx.x;
    const ustride_t block_idx = blockIdx.x;
    const ustride_t block_dim = blockDim.x;
    if constexpr (!needs_grid_stride_loop) {
      const ustride_t x = block_idx * block_dim + thread_idx;
      if (x < tensor_volume) {
        cb(x);
      }
    } else if constexpr (needs_grid_stride_loop) {
      const ustride_t grid_dim = gridDim.x;
      const ustride_t grid_size = grid_dim * block_dim;
      for (ustride_t x = block_idx * block_dim + thread_idx; x < tensor_volume; x += grid_size) {
        cb(x);
      }
    }
  }

  ustride_t tensor_volume;
};

template <typename stride_t, bool _needs_grid_stride_loop> struct block_indexer {
  using ustride_t = typename type_traits::unsign<stride_t>::type;
  static constexpr bool needs_grid_stride_loop = _needs_grid_stride_loop;

  constexpr HOST_DEV __forceinline__ block_indexer(const stride_t n_blocks) : n_blocks(n_blocks) {}

  template <typename Cb> __device__ __forceinline__ void with_grid_stride_loop(Cb &&cb) const {
    // early cast the special indexing variables to the desired integer width type
    // to avoid arithmetic on 32-bit integers when 64-bit stride_t is used
    const ustride_t thread_idx = threadIdx.x;
    const ustride_t block_idx = blockIdx.x;
    if constexpr (!needs_grid_stride_loop) {
      cb(block_idx, thread_idx);
    } else if constexpr (needs_grid_stride_loop) {
      const ustride_t grid_dim = gridDim.x;
      for (ustride_t x = block_idx; x < n_blocks; x += grid_dim) {
        cb(x, thread_idx);
      }
    }
  }

  ustride_t n_blocks;
};

} // namespace nvmath
#endif // NVMATH_COPY_KERNEL_IMPL_GRID_INDEXER_H_
