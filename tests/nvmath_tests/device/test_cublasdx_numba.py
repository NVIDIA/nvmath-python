# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
import numpy as np
from numba import cuda

from nvmath.bindings import mathdx
from nvmath.device.common import axpby, clear, copy, copy_fragment, copy_wait, make_tensor
from nvmath.device.cublasdx_backend import Arrangement, Precision
from .helpers import (
    _TOLERANCE,
    random_real,
    random_complex,
    random_int,
    show_MM_traits,
    set_device,
    skip_nvbug_5218000,
    time_this,
)
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
    "shape,block_size,block_dim,data_type,trans,arrangement,precision,np_type,numba_type,explicit_ld",
    [
        # Various data_types and T/N/C
        (
            (16, 4, 2),
            32,
            None,
            "complex",
            ("conj_transposed", "conj_transposed"),
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
        ((8, 8, 8), 32, None, "real", ("transposed", "non_transposed"), None, np.float64, np.float64, np.float64, False),
        (
            (16, 16, 16),
            32,
            None,
            "real",
            ("non_transposed", "non_transposed"),
            None,
            np.float16,
            np.float16,
            np.float16,
            False,
        ),
        ((2, 2, 2), 64, None, "real", ("non_transposed", "transposed"), None, np.float32, np.float32, np.float32, True),
        ((2, 4, 2), 64, None, "real", ("transposed", "transposed"), None, np.float64, np.float64, np.float64, True),
        ((8, 16, 4), 64, None, "real", ("non_transposed", "transposed"), None, np.float64, np.float64, np.float64, True),
        ((2, 16, 32), 64, None, "real", ("transposed", "non_transposed"), None, np.float64, np.float64, np.float64, True),
        # arrangement
        ((2, 4, 8), 64, None, "real", None, ("row_major", "row_major", "row_major"), np.float32, np.float32, np.float32, True),
        ((2, 4, 8), 64, None, "real", None, ("row_major", "row_major", "col_major"), np.float32, np.float32, np.float32, True),
        ((2, 4, 8), 64, None, "real", None, ("row_major", "col_major", "col_major"), np.float32, np.float32, np.float32, True),
        ((2, 4, 8), 64, None, "real", None, ("col_major", "col_major", "col_major"), np.float32, np.float32, np.float32, True),
        # precision
        (
            (2, 4, 8),
            64,
            None,
            "real",
            None,
            ("col_major", "col_major", "col_major"),
            (np.float16, np.float16, np.float32),
            (np.float16, np.float16, np.float32),
            (np.float16, np.float16, np.float32),
            True,
        ),
        (
            (2, 4, 8),
            64,
            None,
            "real",
            None,
            ("col_major", "col_major", "col_major"),
            (np.int8, np.int8, np.int32),
            (np.int8, np.int8, np.int32),
            (np.int8, np.int8, np.int32),
            True,
        ),
        (
            (2, 4, 8),
            64,
            None,
            "real",
            None,
            ("col_major", "col_major", "col_major"),
            (np.uint8, np.uint8, np.uint32),
            (np.uint8, np.uint8, np.uint32),
            (np.uint8, np.uint8, np.uint32),
            True,
        ),
        # Non powers of 2 sizes
        ((5, 8, 13), 33, None, "real", ("non_transposed", "transposed"), None, np.float16, np.float16, np.float16, False),
        ((3, 64, 1), 37, None, "real", ("transposed", "transposed"), None, np.float64, np.float64, np.float64, False),
        ((1, 1, 1), 63, None, "real", ("non_transposed", "transposed"), None, np.float64, np.float64, np.float64, False),
        ((7, 5, 3), 127, None, "real", ("transposed", "non_transposed"), None, np.float32, np.float32, np.float32, False),
        ((7, 5, 3), 1023, None, "real", ("transposed", "non_transposed"), None, np.float32, np.float32, np.float32, False),
        # 2D+ block_dim
        (
            (15, 17, 19),
            None,
            (32, 1, 1),
            "real",
            ("transposed", "non_transposed"),
            None,
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
            None,
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
            None,
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
            None,
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
            None,
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
            None,
            np.float32,
            np.float32,
            np.float32,
            False,
        ),
    ],
)
def test_matmul(shape, block_size, block_dim, data_type, trans, arrangement, precision, np_type, numba_type, explicit_ld):
    skip_nvbug_5218000(precision, size=shape, dynamic_ld=explicit_ld)

    a_precision = precision[0] if isinstance(precision, Sequence) else precision
    b_precision = precision[1] if isinstance(precision, Sequence) else precision
    c_precision = precision[2] if isinstance(precision, Sequence) else precision

    if issubclass(a_precision, np.integer) and mathdx.get_version() < 201:
        pytest.skip("Integer precision not supported in mathdx < 0.2.1")

    m, n, k = shape

    SM = set_device()
    MM = time_this(
        "matmul codegen",
        matmul,
        size=(m, n, k),
        data_type=data_type,
        precision=precision,
        transpose_mode=trans,
        arrangement=arrangement,
        block_size=block_size,
        block_dim=block_dim,
        execution="Block",
        compiler="numba",
        execute_api="static_leading_dimensions" if not explicit_ld else "dynamic_leading_dimensions",
    )
    show_MM_traits(MM)

    a_value_type = MM.a_value_type
    b_value_type = MM.b_value_type
    c_value_type = MM.c_value_type

    assert MM.size == (m, n, k)
    assert all(f.endswith(".ltoir") for f in MM.files)
    if trans:
        assert MM.transpose_mode == TransposeMode(*trans)
    else:
        assert MM.transpose_mode is None
    if arrangement:
        assert MM.arrangement == Arrangement(*arrangement)
    else:
        assert MM.arrangement is None
    assert MM.a_value_type == numba_type[0] if isinstance(numba_type, Sequence) else numba_type
    assert MM.b_value_type == numba_type[1] if isinstance(numba_type, Sequence) else numba_type
    assert MM.c_value_type == numba_type[2] if isinstance(numba_type, Sequence) else numba_type
    assert MM.a_dim == flip_if((m, k), trans[0] if trans else "non_transposed")
    assert MM.b_dim == flip_if((k, n), trans[1] if trans else "non_transposed")
    assert MM.c_dim == (m, n)
    assert MM.a_size == m * k
    assert MM.b_size == k * n
    assert MM.c_size == m * n

    # There is a dedicated test for shared memory size
    assert MM.shared_memory_size > 0

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
        if issubclass(c_precision, np.unsignedinteger):
            beta = 2.0
    else:
        alpha = 1.0 + 2.0j
        beta = 3.0 + 4.0j

    lda, ldb, ldc = MM.leading_dimension.a, MM.leading_dimension.b, MM.leading_dimension.c

    A_TRANS_CONJ = trans and trans[0] in {"transposed", "conj_transposed"}
    B_TRANS_CONJ = trans and trans[1] in {"transposed", "conj_transposed"}

    A_ROW_MAJOR = arrangement and arrangement[0] == "row_major"
    B_ROW_MAJOR = arrangement and arrangement[1] == "row_major"
    C_ROW_MAJOR = arrangement and arrangement[2] == "row_major"

    @cuda.jit(link=MM.files)
    def f(a_global, b_global, c_global):
        # Input/output
        a_smem = cuda.shared.array(shape=(a_size,), dtype=a_value_type)
        b_smem = cuda.shared.array(shape=(b_size,), dtype=b_value_type)
        c_smem = cuda.shared.array(shape=(c_size,), dtype=c_value_type)

        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
            if A_TRANS_CONJ:  # A is k x m, col-major-lda
                for j in range(m):
                    for i in range(k):
                        a_smem[j * lda + i] = a_global[i, j]  # k x m
            else:  # A is m x k, col-major-lda
                for j in range(k):
                    for i in range(m):
                        if A_ROW_MAJOR:
                            a_smem[i * lda + j] = a_global[i, j]  # m x k
                        else:
                            a_smem[j * lda + i] = a_global[i, j]  # m x k
            if B_TRANS_CONJ:  # B is n x k, col-major-ldc
                for j in range(k):
                    for i in range(n):
                        b_smem[j * ldb + i] = b_global[i, j]  # n x k
            else:  # B is k x n, col-major-ldc
                for j in range(n):
                    for i in range(k):
                        if B_ROW_MAJOR:
                            b_smem[i * ldb + j] = b_global[i, j]  # k x n
                        else:
                            b_smem[j * ldb + i] = b_global[i, j]  # k x n
            for j in range(n):
                for i in range(m):
                    if C_ROW_MAJOR:
                        c_smem[i * ldc + j] = c_global[i, j]  # m x n
                    else:
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
                    if C_ROW_MAJOR:
                        c_global[i, j] = c_smem[i * ldc + j]
                    else:
                        c_global[i, j] = c_smem[j * ldc + i]

    generate = random_real if data_type == "real" else random_complex
    if issubclass(a_precision, np.integer):
        generate = random_int

    if trans and trans[0] != "non_transposed":
        a = generate((k, m), a_precision)
    else:
        a = generate((m, k), a_precision)

    if trans and trans[1] != "non_transposed":
        b = generate((n, k), b_precision)
    else:
        b = generate((k, n), b_precision)

    c = generate((m, n), c_precision)

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

    if not trans or trans[0] == "non_transposed":
        a_ref = a
    elif trans[0] == "conj_transposed":
        a_ref = a.conj().T
    else:
        assert trans[0] == "transposed"
        a_ref = a.T

    if not trans or trans[1] == "non_transposed":
        b_ref = b
    elif trans[1] == "conj_transposed":
        b_ref = b.conj().T
    else:
        assert trans[1] == "transposed"
        b_ref = b.T

    if a_precision != c_precision or b_precision != c_precision:
        c_ref = (
            np.dtype(c.dtype).type(alpha) * (a_ref.astype(c_precision) @ b_ref.astype(c_precision))
            + np.dtype(c.dtype).type(beta) * c
        )
    else:
        c_ref = alpha * (a_ref @ b_ref) + beta * c

    print("Output test: ", c_test)
    print("Output ref: ", c_ref)

    error = np.linalg.norm(c_test - c_ref) / np.linalg.norm(c_ref)
    assert np.linalg.norm(c_test) > 0 and np.linalg.norm(c_ref) > 0
    max_error = (
        0
        if issubclass(c_precision, np.integer)
        else max(_TOLERANCE[p] for p in precision)
        if isinstance(precision, Sequence)
        else _TOLERANCE[precision]
    )
    assert error <= max_error


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


@pytest.mark.parametrize(
    "tensor_types",
    [
        ("smem_a", "smem_b", "smem_c"),
        ("smem_a", "smem_b", "suggested_rmem_c"),
        ("suggested_smem_a", "suggested_smem_b", "suggested_smem_c"),
        ("suggested_smem_a", "suggested_smem_b", "suggested_rmem_c"),
    ],
)
def test_opaque_tensor(tensor_types):
    print(tensor_types)
    m, n, k = 4, 2, 8
    block_size = 64
    precision = Precision(np.float32, np.float32, np.float64)

    assert precision.a == precision.b
    MM = matmul(
        size=(m, n, k),
        precision=precision,
        data_type="real",
        arrangement=("col_major", "row_major", "row_major"),
        execution="Block",
        block_size=block_size,
        compiler="numba",
        tensor_types=tensor_types,
        execute_api="tensors",
    )

    print("uids: ", MM._gmem_tensor_uids)
    print("tids:", MM._target_tensor_uids)

    is_suggested_a = "suggested" in tensor_types[0]
    is_suggested_b = "suggested" in tensor_types[1]
    is_suggested_c = "suggested" in tensor_types[2]

    is_rmem_c = "rmem" in tensor_types[2]

    @cuda.jit(link=MM.files)
    def f(alpha, a, b, beta, c, output):
        # Workaround to set shared memory alignment = 16 bytes (size of c64).
        smem = cuda.shared.array(shape=(0,), dtype=np.complex64).view(precision.a)
        smem_a_buffer, smem = smem[: MM.a_size], smem[MM.a_size :]
        smem_b_buffer, smem = smem[: MM.b_size], smem[MM.b_size :]

        gmem_a = make_tensor(a, MM.get_layout_gmem_a())
        gmem_b = make_tensor(b, MM.get_layout_gmem_b())
        gmem_c = make_tensor(c, MM.get_layout_gmem_c())
        gmem_output = make_tensor(output, MM.get_layout_gmem_c())

        layout_a = MM.suggest_layout_smem_a() if is_suggested_a else MM.get_layout_smem_a()
        layout_b = MM.suggest_layout_smem_b() if is_suggested_b else MM.get_layout_smem_b()

        smem_a = make_tensor(smem_a_buffer, layout_a)
        smem_b = make_tensor(smem_b_buffer, layout_b)

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)
        copy_wait()

        if not is_rmem_c:
            smem_c_buffer = smem.view(precision.c)
            layout_c = MM.suggest_layout_smem_c() if is_suggested_c else MM.get_layout_smem_c()
            smem_c = make_tensor(smem_c_buffer, layout_c)

            copy(gmem_c, smem_c)
            copy_wait()

            MM.execute(alpha, smem_a, smem_b, beta, smem_c)

            cuda.syncthreads()

            copy(smem_c, gmem_output)

            copy_wait()

            return

        rmem_c_compute_buffer = cuda.local.array(shape=(MM.c_size,), dtype=MM.c_value_type)
        rmem_c_compute = make_tensor(rmem_c_compute_buffer, MM.suggest_layout_rmem_c())

        clear(rmem_c_compute)

        MM.execute(smem_a, smem_b, rmem_c_compute)

        rmem_c_buffer = cuda.local.array(shape=(MM.c_size,), dtype=MM.c_value_type)
        rmem_c = make_tensor(rmem_c_buffer, MM.suggest_layout_rmem_c())

        copy_fragment(gmem_c, rmem_c)
        axpby(alpha, rmem_c_compute, beta, rmem_c)
        copy_fragment(rmem_c, gmem_output)

    a = random_real(MM.a_dim, precision.a, order="F")
    b = random_real(MM.b_dim, precision.b, order="C")
    c = random_real(MM.c_dim, precision.c, order="C")
    output = np.empty_like(c)

    alpha = 2.0
    beta = 3.0

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    output_d = cuda.to_device(output)

    f[1, MM.block_dim, 0, MM.shared_memory_size](alpha, a_d, b_d, beta, c_d, output_d)
    cuda.synchronize()

    data_test = output_d.copy_to_host()
    data_ref = alpha * a.astype(precision.c) @ b.astype(precision.c) + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2
