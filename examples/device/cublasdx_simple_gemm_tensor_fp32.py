# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from nvmath.device import matmul
from common import random_real
from nvmath.device.common import copy, copy_fragment, clear, copy_wait, make_tensor, axpby


def main():
    m, n, k = 128, 128, 32
    block_size = 256

    MM = matmul(
        size=(m, n, k),
        precision=(np.float16, np.float16, np.float32),
        data_type="real",
        arrangement=("col_major", "row_major", "row_major"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
        tensor_types=("suggested_smem_a", "suggested_smem_b", "suggested_rmem_c"),
        execute_api="tensors",
    )

    a_layout = MM.suggest_layout_smem_a()
    b_layout = MM.suggest_layout_smem_b()
    c_layout = MM.suggest_layout_rmem_c()

    @cuda.jit(link=MM.files)
    def f(alpha, a, b, beta, c, output):
        smem = cuda.shared.array(shape=(0,), dtype=np.float16, alignment=16)

        # smem_* are buffers for opaque tensors with an underlying layout
        # defined by copy and gemm functions:
        # cute::Tensor<cute::ViewEngine<cute::smem_ptr<__half*> >,
        #              cute::ComposedLayout<cute::Swizzle<2, 3, 3>, cute::C<0>,
        #                                   cute::Layout<cute::tuple<cute::C<128>,
        #                                                            cute::C<32> >,
        #                                                cute::tuple<cute::C<32>,
        #                                                            cute::C<1> > > > >
        smem_a_buffer, smem = smem[: a_layout.cosize], smem[a_layout.cosize :]
        # cute::Tensor<cute::ViewEngine<cute::smem_ptr<__half*> >,
        #              cute::ComposedLayout<cute::Swizzle<2, 3, 3>, cute::C<0>,
        #                                   cute::Layout<cute::tuple<cute::C<32>,
        #                                                            cute::C<128> >,
        #                                                cute::tuple<cute::C<1>,
        #                                                            cute::C<32> > > > >
        smem_b_buffer, smem = smem[: b_layout.cosize], smem[b_layout.cosize :]
        # rmem_c is a buffer for an opaque tensor with and underlying loyout
        # defined by copy and gemm functions:
        # cute::Tensor<cute::ArrayEngine<float, 128ul>,
        #              cute::Layout<cute::tuple<cute::tuple<cute::C<2>, cute::C<2> >,
        #                                       cute::C<4>, cute::C<8> >,
        #                           cute::tuple<cute::tuple<cute::C<1>, cute::C<2> >,
        #                                       cute::C<4>, cute::C<16> > > >
        rmem_c_compute_buffer = cuda.local.array(shape=(c_layout.cosize,), dtype=MM.c_value_type)

        gmem_a = make_tensor(a, MM.get_layout_gmem_a())
        gmem_b = make_tensor(b, MM.get_layout_gmem_b())
        gmem_c = make_tensor(c, MM.get_layout_gmem_c())
        gmem_output = make_tensor(output, MM.get_layout_gmem_c())

        smem_a = make_tensor(smem_a_buffer, a_layout)
        smem_b = make_tensor(smem_b_buffer, b_layout)

        rmem_c_compute = make_tensor(rmem_c_compute_buffer, c_layout)

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)
        clear(rmem_c_compute)
        copy_wait()

        MM.execute(smem_a, smem_b, rmem_c_compute)

        rmem_c_buffer = cuda.local.array(shape=(c_layout.cosize,), dtype=MM.c_value_type)
        rmem_c = make_tensor(rmem_c_buffer, c_layout)

        copy_fragment(gmem_c, rmem_c)
        axpby(alpha, rmem_c_compute, beta, rmem_c)
        copy_fragment(rmem_c, gmem_output)

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
