# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import Matmul
from common import random_real
from nvmath.device.common import copy, copy_wait, make_tensor, axpby


def main():
    m, n, k = 128, 128, 32
    block_size = 256

    MM = Matmul(
        size=(m, n, k),
        precision=(np.float16, np.float16, np.float32),
        data_type="real",
        arrangement=("col_major", "row_major", "row_major"),
        execution="Block",
        block_size=block_size,
    )

    a_layout = MM.get_layout_smem_a()
    b_layout = MM.get_layout_smem_b()

    @cuda.jit
    def f(alpha, a, b, beta, c, output):
        smem = cuda.shared.array(shape=(0,), dtype=np.float16, alignment=16)

        smem_a_buffer, smem = smem[: a_layout.cosize], smem[a_layout.cosize :]
        smem_b_buffer, smem = smem[: b_layout.cosize], smem[b_layout.cosize :]

        gmem_a = make_tensor(a, MM.get_layout_gmem_a())
        gmem_b = make_tensor(b, MM.get_layout_gmem_b())
        gmem_c = make_tensor(c, MM.get_layout_gmem_c())
        gmem_output = make_tensor(output, MM.get_layout_gmem_c())

        smem_a = make_tensor(smem_a_buffer, a_layout)
        smem_b = make_tensor(smem_b_buffer, b_layout)

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)
        copy_wait()

        accumulator = MM.get_accumulator()
        c_frag = accumulator.make_partition_and_copy(gmem_c)

        MM.execute(smem_a, smem_b, accumulator)

        results = accumulator.get_results()
        axpby(alpha, results, beta, c_frag)

        accumulator.partition_and_copy(c_frag, gmem_output)

    a = random_real(MM.a_dim, np.float16, order="F")
    b = random_real(MM.b_dim, np.float16, order="C")
    c = random_real(MM.c_dim, np.float32, order="C")
    output = np.empty_like(c)

    alpha = 2.0
    beta = 3.0

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    output_d = cuda.to_device(output)

    shared_memory_size = MM.get_shared_storage_size_ab(
        MM.suggest_layout_smem_a(),
        MM.suggest_layout_smem_b(),
    )

    f[1, MM.block_dim, 0, shared_memory_size](alpha, a_d, b_d, beta, c_d, output_d)
    cuda.synchronize()

    data_test = output_d.copy_to_host()
    data_ref = alpha * a.astype(np.float32) @ b.astype(np.float32) + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-5, f"Error: {error} > 1e-5"


if __name__ == "__main__":
    main()
