# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
from .helpers import _TOLERANCE, random_real, random_complex, show_MM_traits, set_device, time_this
import time
from nvmath.device import current_device_lto, matmul, float16x2_type, float32x2_type, float64x2_type, Dim3
from nvmath.device import TransposeMode, BlasOptions
from nvmath.device.cublasdx import BlasCompiled, BlasNumba
import pytest


def flip_if(shape, trans):
    assert len(shape) == 2
    assert trans in ["transposed", "conj_transposed", "non_transposed"]
    if trans == "transposed" or trans == "conj_transposed":
        return (shape[1], shape[0])
    elif trans == "non_transposed":
        return (shape[0], shape[1])
    else:
        raise AssertionError()


@pytest.mark.parametrize(
    "shape,block_size,block_dim,data_type,trans,precision,np_type,numba_type,explicit_ld",
    [
        # Various data_types and T/N/C
        (
            (16, 4, 2),
            32,
            None,
            "complex",
            ("conj_transposed", "conj_transposed"),
            np.float32,
            np.complex64,
            float32x2_type,
            True,
        ),
        (
            (4, 8, 2),
            32,
            None,
            "complex",
            ("conj_transposed", "conj_transposed"),
            np.float32,
            np.complex64,
            float32x2_type,
            False,
        ),
        (
            (16, 32, 2),
            32,
            None,
            "complex",
            ("non_transposed", "non_transposed"),
            np.float16,
            np.dtype([("x", np.float16), ("y", np.float16)]),
            float16x2_type,
            False,
        ),
        (
            (4, 32, 8),
            32,
            None,
            "complex",
            ("non_transposed", "non_transposed"),
            np.float16,
            np.dtype([("x", np.float16), ("y", np.float16)]),
            float16x2_type,
            False,
        ),
        (
            (4, 16, 2),
            128,
            None,
            "complex",
            ("non_transposed", "non_transposed"),
            np.float32,
            np.complex64,
            float32x2_type,
            False,
        ),
        (
            (2, 2, 32),
            256,
            None,
            "complex",
            ("non_transposed", "non_transposed"),
            np.float64,
            np.complex128,
            float64x2_type,
            True,
        ),
        (
            (16, 16, 16),
            32,
            None,
            "real",
            ("non_transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        ((8, 8, 8), 32, None, "real", ("transposed", "non_transposed"), np.float64, np.float64, np.float64, False),
        (
            (16, 16, 16),
            32,
            None,
            "real",
            ("non_transposed", "non_transposed"),
            np.float16,
            np.float16,
            np.float16,
            False,
        ),
        ((2, 2, 2), 64, None, "real", ("non_transposed", "transposed"), np.float32, np.float32, np.float32, True),
        ((2, 4, 2), 64, None, "real", ("transposed", "transposed"), np.float64, np.float64, np.float64, True),
        ((8, 16, 4), 64, None, "real", ("non_transposed", "transposed"), np.float64, np.float64, np.float64, True),
        ((2, 16, 32), 64, None, "real", ("transposed", "non_transposed"), np.float64, np.float64, np.float64, True),
        # Non powers of 2 sizes
        ((5, 8, 13), 33, None, "real", ("non_transposed", "transposed"), np.float16, np.float16, np.float16, False),
        ((3, 64, 1), 37, None, "real", ("transposed", "transposed"), np.float64, np.float64, np.float64, False),
        ((1, 1, 1), 63, None, "real", ("non_transposed", "transposed"), np.float64, np.float64, np.float64, False),
        ((7, 5, 3), 127, None, "real", ("transposed", "non_transposed"), np.float32, np.float32, np.float32, False),
        ((7, 5, 3), 1023, None, "real", ("transposed", "non_transposed"), np.float32, np.float32, np.float32, False),
        # 2D+ block_dim
        (
            (15, 17, 19),
            None,
            (32, 1, 1),
            "real",
            ("transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        (
            (1, 59, 1),
            None,
            (33, 1, 1),
            "real",
            ("transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        (
            (59, 1, 1),
            None,
            (1023, 1, 1),
            "real",
            ("transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        (
            (1, 1, 17),
            None,
            (4, 16, 1),
            "real",
            ("transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        (
            (1, 1, 17),
            None,
            (4, 4, 4),
            "real",
            ("transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        (
            (3, 3, 3),
            None,
            (5, 7, 3),
            "real",
            ("transposed", "non_transposed"),
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
    ],
)
def test_matmul(shape, block_size, block_dim, data_type, trans, precision, np_type, numba_type, explicit_ld):
    m, n, k = shape
    a_trans, b_trans = trans

    SM = set_device()
    MM = time_this(
        "matmul codegen",
        matmul,
        size=(m, n, k),
        data_type=data_type,
        precision=precision,
        transpose_mode=TransposeMode(a_trans, b_trans),
        block_size=block_size,
        block_dim=block_dim,
        execution="Block",
        compiler="numba",
    )
    show_MM_traits(MM)

    input_type = MM.input_type
    output_type = MM.output_type

    assert MM.size == (m, n, k)
    assert all(f.endswith(".ltoir") for f in MM.files)
    assert MM.transpose_mode == TransposeMode(a_trans, b_trans)
    assert MM.value_type == numba_type
    assert MM.input_type == numba_type
    assert MM.output_type == numba_type
    assert MM.a_dim == flip_if((m, k), trans[0])
    assert MM.b_dim == flip_if((k, n), trans[1])
    assert MM.c_dim == (m, n)
    assert MM.a_size == m * k
    assert MM.b_size == k * n
    assert MM.c_size == m * n
    if precision == np.float16 and data_type == "complex":
        assert MM.shared_memory_size == 2 * np.float16(1.0).itemsize * (MM.a_size + MM.b_size + MM.c_size)
    else:
        assert MM.shared_memory_size == np_type(1.0).itemsize * (MM.a_size + MM.b_size + MM.c_size)
    if block_size is not None:
        assert MM.block_dim == (block_size, 1, 1)
    elif block_dim is not None:
        assert MM.block_dim == Dim3(*block_dim)
    assert MM.max_threads_per_block <= 1024
    assert MM.code_type.kind == "lto"
    assert MM.code_type.cc.major == SM[0]
    assert MM.code_type.cc.minor == SM[1]

    a_size = MM.a_size
    b_size = MM.b_size
    c_size = MM.c_size
    block_dim = MM.block_dim

    if data_type == "real":
        alpha = 3.0
        beta = -2.0
    else:
        alpha = 1.0 + 2.0j
        beta = 3.0 + 4.0j

    lda, ldb, ldc = MM.leading_dimension.a, MM.leading_dimension.b, MM.leading_dimension.c

    A_TRANS_CONJ = a_trans == "transposed" or a_trans == "conj_transposed"
    B_TRANS_CONJ = b_trans == "transposed" or b_trans == "conj_transposed"

    @cuda.jit(link=MM.files)
    def f(a_global, b_global, c_global):
        # Input/output
        a_smem = cuda.shared.array(shape=(a_size,), dtype=input_type)
        b_smem = cuda.shared.array(shape=(b_size,), dtype=input_type)
        c_smem = cuda.shared.array(shape=(c_size,), dtype=output_type)

        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
            if A_TRANS_CONJ:  # A is k x m, col-major-lda
                for j in range(m):
                    for i in range(k):
                        a_smem[j * lda + i] = a_global[i, j]  # k x m
            else:  # A is m x k, col-major-lda
                for j in range(k):
                    for i in range(m):
                        a_smem[j * lda + i] = a_global[i, j]  # m x k
            if B_TRANS_CONJ:  # B is n x k, col-major-ldc
                for j in range(k):
                    for i in range(n):
                        b_smem[j * ldb + i] = b_global[i, j]  # n x k
            else:  # B is k x n, col-major-ldc
                for j in range(n):
                    for i in range(k):
                        b_smem[j * ldb + i] = b_global[i, j]  # k x n
            for j in range(n):
                for i in range(m):
                    c_smem[j * ldc + i] = c_global[i, j]  # m x n
        cuda.syncthreads()

        # Execute FFT
        if explicit_ld:
            MM(alpha, a_smem, lda, b_smem, ldb, beta, c_smem, ldc)
        else:
            MM(alpha, a_smem, b_smem, beta, c_smem)

        cuda.syncthreads()
        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
            for j in range(n):
                for i in range(m):
                    c_global[i, j] = c_smem[j * ldc + i]

    generate = random_real if data_type == "real" else random_complex

    if a_trans == "non_transposed":
        a = generate((m, k), precision)
    else:
        a = generate((k, m), precision)

    if b_trans == "non_transposed":
        b = generate((k, n), precision)
    else:
        b = generate((n, k), precision)

    c = generate((m, n), precision)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    print("Input (a): ", a)
    print("Input (b): ", b)
    print("Input (c): ", c)

    t0 = time.time()
    f[1, block_dim](a_d, b_d, c_d)
    t1 = time.time()
    print("Numba codegen time ", t1 - t0)
    cuda.synchronize()

    c_test = c_d.copy_to_host()

    if a_trans == "non_transposed":
        a_ref = a
    elif a_trans == "conj_transposed":
        a_ref = a.conj().T
    else:
        assert a_trans == "transposed"
        a_ref = a.T

    if b_trans == "non_transposed":
        b_ref = b
    elif b_trans == "conj_transposed":
        b_ref = b.conj().T
    else:
        assert b_trans == "transposed"
        b_ref = b.T

    c_ref = alpha * (a_ref @ b_ref) + beta * c

    print("Output test: ", c_test)
    print("Output ref: ", c_ref)

    error = np.linalg.norm(c_test - c_ref) / np.linalg.norm(c_ref)
    assert np.linalg.norm(c_test) > 0 and np.linalg.norm(c_ref) > 0
    assert error < _TOLERANCE[precision]


def test_valid():
    base_MM = BlasOptions(
        size=(8, 4, 16),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("transposed", "non_transposed"),
        execution="Block",
        code_type=current_device_lto(),
    )

    count = 0
    for (bd,) in base_MM.valid("block_dim"):
        MM0 = base_MM.create(block_dim=bd, compiler="numba")
        assert isinstance(MM0, BlasNumba)
        MM1 = base_MM.create(block_dim=bd, compiler="numba")
        assert isinstance(MM1, BlasCompiled)
        count += 1

    assert count > 0
