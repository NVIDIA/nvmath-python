# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


def calc_slab_shape(global_shape, partition_dim, rank, nranks):
    """Calculate local slab shape, according to default optimized slab distribution used by
    cuFFTMp"""
    n = nranks
    S = global_shape[partition_dim]
    partition_dim_local_size = (S // n + 1) if rank < S % n else S // n
    slab_shape = list(global_shape)
    slab_shape[partition_dim] = partition_dim_local_size
    return tuple(slab_shape)
