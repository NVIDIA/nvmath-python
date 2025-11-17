# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import Matmul
from common import random_real
from nvmath.device.common import axpby, clear, copy, copy_wait, make_tensor
from nvmath.device.cublasdx_backend import MAX_ALIGNMENT


def main():
    m, n, k = 64, 64, 64
    block_size = 128
    alpha, beta = 1, 2
    data_type = "real"
    precision = np.float16

    MM = Matmul(
        size=(m, n, k),
        precision=(precision, precision, precision),
        data_type=data_type,
        arrangement=("row_major", "col_major", "col_major"),
        execution="Block",
        block_size=block_size,
        alignment=MAX_ALIGNMENT,
    )
    grid_dim = 1

    a_cosize = MM.suggest_layout_smem_a().cosize
    b_cosize = MM.suggest_layout_smem_b().cosize
    c_cosize = MM.suggest_layout_rmem_c().cosize
    c_size = MM.suggest_layout_rmem_c().size

    @cuda.jit
    def f(a, b, c, alpha, beta, output):
        # We have same precision for all tensors
        smem = cuda.shared.array(shape=(0,), dtype=precision, alignment=16)
        smem_a_buff, smem = smem[:a_cosize], smem[a_cosize:]
        smem_b_buff, smem = smem[:b_cosize], smem[b_cosize:]
        rmem_c_buff = cuda.local.array(shape=(c_cosize,), dtype=MM.c_value_type, alignment=16)
        rmem_c_out_buff = cuda.local.array(shape=(c_cosize,), dtype=MM.c_value_type, alignment=16)

        gmem_a = make_tensor(a, MM.get_layout_gmem_a())
        gmem_b = make_tensor(b, MM.get_layout_gmem_b())
        gmem_c = make_tensor(c, MM.get_layout_gmem_c())
        gmem_output = make_tensor(output, MM.get_layout_gmem_c())

        smem_a = make_tensor(smem_a_buff, MM.suggest_layout_smem_a())
        smem_b = make_tensor(smem_b_buff, MM.suggest_layout_smem_b())
        rmem_c = make_tensor(rmem_c_buff, MM.suggest_layout_rmem_c())
        rmem_c_out = make_tensor(rmem_c_out_buff, MM.suggest_layout_rmem_c())

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)

        copy_wait()

        clear(rmem_c)

        partitioner = MM.suggest_partitioner()
        gmem_c_partition = partitioner.partition_like_C(gmem_c)
        gmem_output_partition = partitioner.partition_like_C(gmem_output)

        # Use copy_fragment(gmem_c, rmem_c_out) instead since it provides
        # better performance achieved by vectorization. This is for functional
        # demonstration purposes only.
        for i in range(c_size):
            rmem_c_out_buff[i] = gmem_c_partition[i]

        alpha = c.dtype.type(alpha)
        beta = c.dtype.type(beta)

        MM.execute(smem_a, smem_b, rmem_c)
        axpby(alpha, rmem_c, beta, rmem_c_out)

        # Use copy_fragment(rmem_c_out, gmem_output) instead since it provides
        # better performance achieved by vectorization. This is for functional
        # demonstration purposes only.
        for i in range(c_size):
            gmem_output_partition[i] = rmem_c_out_buff[i]

    a = random_real(MM.a_dim, precision, order="C")
    b = random_real(MM.b_dim, precision, order="F")
    c = random_real(MM.c_dim, precision, order="F")
    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    shared_memory_size = MM.get_shared_storage_size_ab(
        MM.suggest_layout_smem_a(),
        MM.suggest_layout_smem_b(),
    )

    f[grid_dim, MM.block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b) + beta * c
    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"Relative error: {error}")
    assert error < 1e-2


if __name__ == "__main__":
    main()
