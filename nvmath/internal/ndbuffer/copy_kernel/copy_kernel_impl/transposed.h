// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVMATH_COPY_KERNEL_IMPL_TRANSPOSED_H
#define NVMATH_COPY_KERNEL_IMPL_TRANSPOSED_H

#include "copy_kernel_impl/array_view.h"
#include "copy_kernel_impl/grid_indexer.h"
#include "copy_kernel_impl/type_utils.h"
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

namespace detail {

template <typename _T, typename _stride_t, int _transposed_dim, int _block_height, int _block_width,
          char _reading_order>
struct transpose2d {
  using T = _T;
  using stride_t = _stride_t;

  // the thread position in the tile is represented as a 2d index
  using thread_coords_t = vec<2, stride_t>;

  // Consecutive flat thread index can be unraveled into 2d coordinates
  // so that either the left or right coordinate changes faster.
  // Reading order tells which of the two ways to pick for global memory reads.
  static constexpr char reading_order = _reading_order;
  static_assert(reading_order == 'C' || reading_order == 'F', "reading_order must be 'C' or 'F'");

  // Describes splitting of the src/dst shape into two parts
  // [0, transposed_dim] and [transposed_dim + 1, ndim - 1] to be traversed
  // with the thread_coords.
  static constexpr int transposed_dim = _transposed_dim;

  // to unravel flat cuda thread index to 2d index in the tile fast,
  // we require the tile extents (_block_height, _block_height)
  // to be powers of 2, and use compile-time helpers that turn
  // the division/modulo into a bit shift/mask.
  using warp_size_t = mod_div<32>;
  using block_height_t = mod_div<_block_height>;
  using block_width_t = mod_div<_block_width>;
  using swizzle_stride_t = mod_div<(_block_height > 32 ? 1 : 32 / _block_height)>;
  using warps_in_width_t = mod_div<(_block_width < 32 ? 1 : _block_width / 32)>;
  static constexpr block_height_t block_height = block_height_t{};
  static constexpr block_width_t block_width = block_width_t{};
  static constexpr swizzle_stride_t swizzle_stride = swizzle_stride_t{};
  static constexpr warps_in_width_t warps_in_width = warps_in_width_t{};
  static constexpr warp_size_t warp_size = warp_size_t{};
  static constexpr int tile_num_elements = _block_height * _block_width;

  HOST_DEV __forceinline__ transpose2d(T *RESTRICT data, stride_t *RESTRICT dst_offset)
      : data(data), dst_offset(dst_offset) {}

  // 2d index of the thread in the tile in C-like order (i.e. second index changing faster)
  HOST_DEV __forceinline__ constexpr thread_coords_t unravel_thread_idx_c(const stride_t thread_idx) const {
    return {thread_idx / block_width, thread_idx % block_width};
  }

  // 2d index of the thread in the tile in Fortran-like order (i.e. first index changing faster)
  HOST_DEV __forceinline__ constexpr thread_coords_t unravel_thread_idx_f(const stride_t thread_idx) const {
    return {thread_idx % block_height, thread_idx / block_height};
  }

  HOST_DEV __forceinline__ constexpr thread_coords_t unravel_thread_idx_reading(const stride_t thread_idx) const {
    if constexpr (reading_order == 'C') {
      return unravel_thread_idx_c(thread_idx);
    } else {
      return unravel_thread_idx_f(thread_idx);
    }
  }

  HOST_DEV __forceinline__ constexpr thread_coords_t unravel_thread_idx_writing(const stride_t thread_idx) const {
    if constexpr (reading_order == 'C') {
      return unravel_thread_idx_f(thread_idx);
    } else {
      return unravel_thread_idx_c(thread_idx);
    }
  }

  HOST_DEV __forceinline__ constexpr stride_t shm_offset(const thread_coords_t idx) const {
    const stride_t y = idx[0];
    const stride_t x = idx[1];
    // Note, offset(unravel_thread_idx_c(thread_idx)) = idx
    const stride_t offset = y * block_width + x;
    const stride_t offset_y = offset / warp_size;
    const stride_t offset_x = offset % warp_size;
    // In the simplest case of 32x32 tile, we need to rotate elements
    // by one every 32 elements to make sure that accessing with
    // unravel_thread_idx_f(thread_idx)=((0, x), ..., (31, x)) does not
    // introduce bank conflicts. If block_height is smaller than 32,
    // there will be 32/block_height different xs in the
    // unravel_thread_idx_f(thread_idx) warp, so we rotate by
    // swizzle_stride to make sure that different xs do not land in the same bank.
    // If block_width is bigger than 32, we want to make sure that
    // we make one rotation per one y, hence the division by warps_in_width.
    const stride_t swizzle = (offset_y / warps_in_width) * swizzle_stride;
    return offset_y * warp_size + ((offset_x + swizzle) % warp_size);
  }

  // shared memory array to store the elements read from the src tensor
  T *RESTRICT data;
  // for data[i], the dst_offset[i] is the offset of the element data[i] in the dst tensor.
  stride_t *RESTRICT dst_offset;
};
} // namespace detail

template <typename dst_array_view_t, typename src_array_view_t, typename grid_indexer_t, typename copy_helper_t>
struct transpose_copy_impl {
  // 32 or 64 bit signed integer
  using stride_t = typename dst_array_view_t::stride_t;
  // ndim vector of stride_t integers
  using coords_t = typename dst_array_view_t::coords_t;
  static constexpr int ndim = dst_array_view_t::ndim;
  static_assert(ndim == src_array_view_t::ndim, "src and dst must have the same number of dimensions");
  static constexpr int transposed_dim = copy_helper_t::transposed_dim;
  static_assert(0 <= transposed_dim && transposed_dim < ndim - 1, "transposed_dim must be between 0 and ndim - 2");

  // use min possible offset to indicate that the element index is out of tensor bounds
  static constexpr stride_t out_of_bounds_sentinel = type_traits::min_val<stride_t>::value;

  /**
   * @brief Unravel flat block index and 2d thread index to ndim index of the element in the `shape`.
   *
   * E. g. Given tensor of shape 2x3x4 and transposed_dim = 1
   * [[[0, 1, 2, 3,],
   *   [4, 5, 6, 7,],
   *   [8, 9, 10, 11,]],

   *  [[12, 13, 14, 15,],
   *   [16, 17, 18, 19,],
   *   [20, 21, 22, 23,]]]
   *
   * and a 2x3 tile:
   * [[(0,0), (0, 1), (0, 2)],
   *  [(1,0), (1, 1), (1, 2)]]
   *
   * maps to the 3D indices of the following elements:
   * block_idx = 0        block_idx = 1        block_idx = 2        block_idx = 3
   * [[0, 1, 2],          [[3, 8, 9],          [[10, 11, 16],       [[17, 18, 19],
   *  [4, 5, 6]]           [7, 12, 13]]         [14, 15, 20]]        [21, 22, 23]]
   *
   *
   * Note, if the tiled extents are not divisible by the tile dimensions,
   * the parts of the tile are carried over to the next position. Potential
   * uncoalesced memory accesses are preferred over threads with no work to do
   * (when mapped to invalid positions in the tensors).
   *
   * @param block_idx The threadblock index in the grid
   * @param shape The shape of the tensor
   * @param thread_idx The 2d index of the thread in the tile
   * @return The ndim index of the element in the tensor
   */
  __device__ __forceinline__ coords_t unravel_tiled_idx(const stride_t block_idx, const coords_t shape,
                                                        const vec<2, stride_t> thread_idx) {

    static_assert(ndim >= 2, "ndim must be at least 2");
    // the extents cannot be negative and the arithmetic on unsigned integer
    // is noticeably faster
    using u_stride_t = typename type_traits::unsign<stride_t>::type;
    u_stride_t flat_idx;
    coords_t unraveled_idx;
    auto unravel_extent = [&flat_idx, &unraveled_idx, &shape](int i) {
      u_stride_t extent = shape[i];
      if (extent & (extent - 1)) {
        u_stride_t next_flat_idx = flat_idx / extent;
        unraveled_idx[i] = flat_idx - next_flat_idx * extent;
        flat_idx = next_flat_idx;
      } else {
        unraveled_idx[i] = flat_idx & (extent - 1);
        flat_idx >>= ffs(extent) - 1;
      }
    };
    flat_idx = block_idx * copy_helper_t::block_width + thread_idx[1];
#pragma unroll
    for (int i = ndim - 1; i > transposed_dim; i--) {
      unravel_extent(i);
    }
    flat_idx = flat_idx * copy_helper_t::block_height + thread_idx[0];
#pragma unroll
    for (int i = transposed_dim; i > 0; i--) {
      unravel_extent(i);
    }
    unraveled_idx[0] = flat_idx;
    return unraveled_idx;
  }

  void __forceinline__ __device__ operator()(const dst_array_view_t dst_view, const src_array_view_t src_view,
                                             const grid_indexer_t grid_helper, const copy_helper_t transpose_helper) {
    grid_helper.with_grid_stride_loop([=](const stride_t flat_block_idx, const stride_t flat_thread_idx) {
      {
        // 2d index of the thread
        const auto thread_coords = transpose_helper.unravel_thread_idx_reading(flat_thread_idx);
        // ndim index of the element in the source tensor
        const auto src_coords = unravel_tiled_idx(flat_block_idx, src_view.shape(), thread_coords);
        const bool is_in_bounds = all([](auto a, auto b) { return a < b; }, src_coords, src_view.shape());
        if constexpr (grid_indexer_t::needs_grid_stride_loop) {
          __syncthreads();
        }
        // flat index in the shared memory arrays where thread_coords should store its data
        const auto shm_offset = transpose_helper.shm_offset(thread_coords);
        if (is_in_bounds) {
          // The data is copied from src to dst through shared memory in two steps:
          // 1. src[unravel_tiled_idx(.., thread_coords)] -> shm[shm_offset(thread_coords)]
          // 2. shm[shm_offset(thread_coords_transposed)] -> dst[unravel_tiled_idx(.., thread_coords_transposed)]
          // To avoid computing the unravel_tiled_idx twice, we compute the dst stride here
          // and store it together with the data.
          transpose_helper.data[shm_offset] = src_view[src_coords];
          transpose_helper.dst_offset[shm_offset] = dst_view.offset(src_coords);
        } else {
          transpose_helper.dst_offset[shm_offset] = out_of_bounds_sentinel;
        }
        __syncthreads();
      }
      // 2d index of the thread with flipped order (i.e. the other index is changing faster)
      const auto thread_coords = transpose_helper.unravel_thread_idx_writing(flat_thread_idx);
      const auto shm_offset = transpose_helper.shm_offset(thread_coords);
      const auto dst_offset = transpose_helper.dst_offset[shm_offset];
      if (dst_offset != out_of_bounds_sentinel) {
        dst_view[dst_offset] = transpose_helper.data[shm_offset];
      }
    });
  }
};
} // namespace nvmath

#endif // NVMATH_COPY_KERNEL_IMPL_TRANSPOSED_H
