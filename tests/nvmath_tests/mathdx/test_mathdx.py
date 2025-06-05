from examples.device.common_numba import load_to_shared, store_from_shared
from nvmath.bindings import mathdx
from nvmath.device import matmul

import numpy as np
from numba import cuda

from nvmath.device.types import REAL_NP_TYPES
from nvmath.device.common_numba import NP_TYPES_TO_NUMBA_FE_TYPES


import pytest

NUMBA_FE_TYPES_TO_NP_TYPES = {v: k for (k, v) in NP_TYPES_TO_NUMBA_FE_TYPES.items()}


@pytest.mark.parametrize(
    "library",
    [
        "cublasdx",
        "cufftdx",
        "cusolverdx",
    ],
)
def test_handle(library):
    create = getattr(mathdx, library + "_create_descriptor")
    destroy = getattr(mathdx, library + "_destroy_descriptor")
    h = create()
    destroy(h)


@pytest.mark.parametrize(
    "library, operator, value",
    [
        ("cublasdx", mathdx.CublasdxOperatorType.API, mathdx.CublasdxApi.SMEM),
        ("cufftdx", mathdx.CufftdxOperatorType.API, mathdx.CufftdxApi.LMEM),
        ("cusolverdx", mathdx.CusolverdxOperatorType.API, mathdx.CusolverdxApi.SMEM),
    ],
)
def test_set_operator(library, operator, value):
    create = getattr(mathdx, library + "_create_descriptor")
    destroy = getattr(mathdx, library + "_destroy_descriptor")
    set_operator = getattr(mathdx, library + "_set_operator_int64")

    h = create()

    set_operator(h, operator, value)

    destroy(h)


@pytest.mark.parametrize(
    "library, operator, value",
    [
        ("cublasdx", mathdx.CublasdxOperatorType.BLOCK_DIM, [3, 1, 1]),
        ("cufftdx", mathdx.CufftdxOperatorType.BLOCK_DIM, [3, 1, 1]),
        ("cusolverdx", mathdx.CusolverdxOperatorType.BLOCK_DIM, [3, 1, 1]),
    ],
)
def test_set_operator_int64_array(library, operator, value):
    create = getattr(mathdx, library + "_create_descriptor")
    destroy = getattr(mathdx, library + "_destroy_descriptor")
    set_operator = getattr(mathdx, library + "_set_operator_int64s")

    h = create()

    set_operator(h, operator, len(value), value)

    destroy(h)


@pytest.mark.parametrize(
    "precision",
    [t for t in REAL_NP_TYPES],
)
@pytest.mark.parametrize(
    "data_type",
    [
        "real",
        "complex",
    ],
)
def test_cublasdx_call(precision, data_type):
    m, n, k = 2, 2, 2

    MM = matmul(
        size=(m, n, k),
        precision=precision,
        data_type=data_type,
        transpose_mode=("non_transposed", "transposed"),
        execution="Block",
        compiler="numba",
    )

    value_type = MM.value_type

    a_size = MM.a_size
    b_size = MM.b_size
    a_dim = MM.a_dim
    b_dim = MM.b_dim
    c_dim = MM.c_dim
    block_dim = MM.block_dim
    ld = MM.leading_dimension
    lda, ldb, ldc = ld.a, ld.b, ld.c
    shared_memory_size = MM.get_shared_storage_size()

    @cuda.jit(link=MM.files)
    def f(a, b, c, alpha, beta, output):
        smem = cuda.shared.array(shape=(0,), dtype=value_type)
        smem_a = smem[0:]
        smem_b = smem[a_size:]
        smem_c = smem[a_size + b_size :]

        load_to_shared(a, smem_a, a_dim, lda)
        load_to_shared(b, smem_b, b_dim, ldb)
        load_to_shared(c, smem_c, c_dim, ldc)

        cuda.syncthreads()

        MM(alpha, smem_a, smem_b, beta, smem_c)

        cuda.syncthreads()

        store_from_shared(smem_c, output, c_dim, ldc)

    a = np.ones(a_dim, dtype=precision)
    b = np.ones(b_dim, dtype=precision)
    c = np.ones(c_dim, dtype=precision)
    if data_type == "complex":
        a = a + 1.0j * np.zeros(a_dim, dtype=precision)
        b = b + 1.0j * np.zeros(b_dim, dtype=precision)
        c = c + 1.0j * np.zeros(c_dim, dtype=precision)

    o = np.zeros_like(c)

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    o_d = cuda.to_device(o)

    alpha = 2.0
    beta = 3.0

    if data_type == "complex":
        alpha += 1j
        beta += 1j

    f[1, block_dim, 0, shared_memory_size](a_d, b_d, c_d, alpha, beta, o_d)
    cuda.synchronize()

    data_test = o_d.copy_to_host()
    data_ref = alpha * (a @ b.T) + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    assert error < 1e-2
