# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from common import random
from numba import cuda

from nvmath.device import Matmul
from nvmath.device.common import axpby, clear, copy, copy_wait, make_tensor
from nvmath.device.cublasdx_backend import MAX_ALIGNMENT


def main():
    m, n, k = 64, 64, 64
    block_size = 128
    alpha, beta = 1, 2

    MM = Matmul(
        size=(m, n, k),
        precision=(np.float16, np.float16, np.float32),
        data_type="real",
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
        smem = cuda.shared.array(shape=(0,), dtype=MM.a_value_type, alignment=16)
        smem_a_buff, smem = smem[:a_cosize], smem[a_cosize:]
        smem_b_buff, smem = smem[:b_cosize], smem[b_cosize:]
        rmem_c_buff = cuda.local.array(shape=(c_cosize,), dtype=MM.c_value_type, alignment=16)
        rmem_c_out_buff = cuda.local.array(shape=(c_cosize,), dtype=MM.c_value_type, alignment=16)

        gmem_a = make_tensor(a, MM.get_layout_gmem_a())
        gmem_b = make_tensor(b, MM.get_layout_gmem_b())

        smem_a = make_tensor(smem_a_buff, MM.suggest_layout_smem_a())
        smem_b = make_tensor(smem_b_buff, MM.suggest_layout_smem_b())
        rmem_c = make_tensor(rmem_c_buff, MM.suggest_layout_rmem_c())
        rmem_c_out = make_tensor(rmem_c_out_buff, MM.suggest_layout_rmem_c())

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)
        clear(rmem_c)
        copy_wait()

        MM.execute(smem_a, smem_b, rmem_c)

        accumulator = MM.suggest_accumulator()

        # Use copy_fragment(gmem_c, rmem_c_out) instead since it provides
        # better performance achieved by vectorization. This is for functional
        # demonstration purposes only.
        if accumulator.is_thread_active():
            for i in range(c_size):
                if (not accumulator.is_predicated()) or accumulator.is_index_in_bounds(i):
                    x, y = accumulator.map_fragment_index(i)
                    rmem_c_out_buff[i] = c[x, y]

        alpha = c.dtype.type(alpha)
        beta = c.dtype.type(beta)

        axpby(alpha, rmem_c, beta, rmem_c_out)

        # Use copy_fragment(rmem_c_out, gmem_output) instead since it provides
        # better performance achieved by vectorization. This is for functional
        # demonstration purposes only.
        if accumulator.is_thread_active():
            for i in range(c_size):
                if (not accumulator.is_predicated()) or accumulator.is_index_in_bounds(i):
                    x, y = accumulator.map_fragment_index(i)
                    output[x, y] = rmem_c_out_buff[i]

    a = random(MM.a_dim, dtype=MM.a_value_type, arrangement=MM.arrangement.a)
    b = random(MM.b_dim, dtype=MM.b_value_type, arrangement=MM.arrangement.b)
    c = random(MM.c_dim, dtype=MM.c_value_type, arrangement=MM.arrangement.c)
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
    data_ref = alpha * a.astype(MM.c_value_type) @ b.astype(MM.c_value_type) + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"Relative error: {error}")
    assert error < 1e-2


if __name__ == "__main__":
    main()
