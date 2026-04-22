# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import scipy.sparse as sps

from nvmath.sparse.generic import Matmul
from nvmath.sparse.ust import NamedFormats, Tensor
from nvmath.sparse.ust._emitter import emit_apply, emit_matmul

from ..utils.common_axes import JIT_AVAILABLE


def _modify_value_only(value):
    return value * 2


def _modify_value_with_indices(value, i, j):
    return value + i + j


def _compile_and_apply_kernel_value_only(a):
    u = Tensor.from_package(a)
    u.set_kernel(_modify_value_only, with_indices=False)
    u.run_kernel()
    return u


def _compile_and_apply_kernel_with_indices(a):
    u = Tensor.from_package(a)
    u.set_kernel(_modify_value_with_indices, with_indices=True)
    u.run_kernel()
    return u


def test_source_code():
    cp = pytest.importorskip("cupy")
    a = cp.ones((10, 20), dtype=np.int64)
    u = Tensor.from_package(a)
    src = emit_apply(u, with_indices=False)
    assert (
        src == ""
        "using VAL = long long;\n"
        "using POS = int;\n"
        "using CRD = int;\n"
        "\n"
        'extern "C" __device__ VAL apply(VAL);\n'
        "\n"
        'extern "C" __global__ void apply_kernel(\n'
        "  VAL* __restrict__ Avalues,\n"
        "  POS Anse\n"
        ") {\n"
        "  const unsigned long tidx = threadIdx.x;\n"
        "  const unsigned long bidx = blockIdx.x;\n"
        "  const unsigned long tid = bidx * blockDim.x + tidx;\n"
        "  if (tid >= Anse)\n"
        "    return;\n"
        "  Avalues[tid] = apply(Avalues[tid]);\n"
        "}\n"
    )
    src = emit_apply(u, with_indices=True)
    assert (
        src == ""
        "using VAL = long long;\n"
        "using POS = int;\n"
        "using CRD = int;\n"
        "\n"
        'extern "C" __device__ VAL apply(VAL, CRD, CRD);\n'
        "\n"
        'extern "C" __global__ void apply_kernel(\n'
        "  VAL* __restrict__ Avalues\n"
        ") {\n"
        "  const unsigned long tidx = threadIdx.x;\n"
        "  const unsigned long bidx = blockIdx.x;\n"
        "  const unsigned long tid = bidx * blockDim.x + tidx;\n"
        "  CRD l0, l1;\n"
        "  const CRD p0 = tid;\n"
        "  if (p0 >= 10)\n"
        "    return;\n"
        "  l0 = p0;\n"
        "  {\n"
        "    POS p1 = p0 * 20;\n"
        "    for (CRD i1 = 0; i1 < 20; i1++, p1++) {\n"
        "      l1 = i1;\n"
        "      Avalues[p1] = apply(Avalues[p1], l0, l1);\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    x = cp.ones((20,), dtype=np.int64)
    y = cp.ones((10,), dtype=np.int64)
    xu = Tensor.from_package(x)
    yu = Tensor.from_package(y)
    src = emit_matmul(u, xu, yu, "long long")
    assert (
        src == ""
        "using VAL = long long;\n"
        "using POS = int;\n"
        "using CRD = int;\n"
        "using ATP = long long;\n"
        "using CTP = long long;\n"
        "\n"
        'extern "C" __device__ CTP add(CTP, CTP);\n'
        'extern "C" __device__ ATP atomic_add(ATP *, ATP);\n'
        'extern "C" __device__ CTP mul(CTP, CTP);\n'
        "\n"
        'extern "C" __device__ CTP prolog_a(CTP);\n'
        'extern "C" __device__ CTP prolog_b(CTP);\n'
        'extern "C" __device__ CTP prolog_c(CTP);\n'
        'extern "C" __device__ CTP epilog(CTP);\n'
        "\n"
        'extern "C" __global__ void matmul(\n'
        "  VAL* __restrict__ Avalues,\n"
        "  unsigned long Ad0, unsigned long Ad1,\n"
        "  unsigned long Al0, unsigned long Al1,\n"
        "  VAL* __restrict__ Bvalues,\n"
        "  VAL* __restrict__ Cvalues\n"
        ") {\n"
        "  const unsigned long tidx = threadIdx.x;\n"
        "  const unsigned long bidx = blockIdx.x;\n"
        "  const unsigned long tid = bidx * blockDim.x + tidx;\n"
        "  CRD l0, l1;\n"
        "  const CRD p0 = tid;\n"
        "  if (p0 >= Al0)\n"
        "    return;\n"
        "  l0 = p0;\n"
        "  CTP acc = prolog_c(static_cast<CTP>(Cvalues[l0]));\n"
        "  {\n"
        "    POS p1 = p0 * Al1;\n"
        "    for (CRD i1 = 0; i1 < Al1; i1++, p1++) {\n"
        "      l1 = i1;\n"
        "      {\n"
        "        acc = add(acc, mul(prolog_a(static_cast<CTP>(Avalues[p1])), prolog_b(static_cast<CTP>(Bvalues[l1]))));\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "  Cvalues[l0] = static_cast<VAL>(epilog(acc));\n"
        "}\n"
    )


@pytest.mark.skipif(not JIT_AVAILABLE, reason="jitting is required for this test")
def test_apply_cupy_jit():
    cp = pytest.importorskip("cupy")
    cps = pytest.importorskip("cupyx.scipy.sparse")
    row = cp.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
    col = cp.array([0, 1, 1, 3, 2, 3], dtype=np.int32)
    val = cp.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    coo = cps.coo_matrix((val, (row, col)), shape=(4, 8))
    coo.sum_duplicates()
    csr = cps.csr_matrix(coo)
    csc = cps.csc_matrix(coo)

    e1 = cp.array([2, 4, 6, 8, 10, 12], dtype=np.float32)
    e2 = cp.array([2, 5, 8, 12, 14, 18], dtype=np.float32)
    e3 = cp.array([2, 4, 6, 10, 8, 12], dtype=np.float32)
    e4 = cp.array([2, 5, 8, 14, 12, 18], dtype=np.float32)
    e5 = cp.array([2, 6, 10, 16, 18, 24], dtype=np.float32)

    u = _compile_and_apply_kernel_value_only(coo)
    assert cp.array_equal(u.val.tensor, e1)
    u = _compile_and_apply_kernel_with_indices(coo)
    assert cp.array_equal(u.val.tensor, e2)

    u = _compile_and_apply_kernel_value_only(csr)
    assert cp.array_equal(u.val.tensor, e1)
    u = _compile_and_apply_kernel_with_indices(csr)
    assert cp.array_equal(u.val.tensor, e2)

    u = _compile_and_apply_kernel_value_only(csc)
    assert cp.array_equal(u.val.tensor, e3)
    u = _compile_and_apply_kernel_with_indices(csc)
    assert cp.array_equal(u.val.tensor, e4)

    d = u.convert(tensor_format=NamedFormats.DELTA(2))
    d.set_kernel(_modify_value_with_indices, with_indices=True)
    d.run_kernel()
    assert cp.array_equal(d.val.tensor, e5)


@pytest.mark.skipif(not JIT_AVAILABLE, reason="jitting is required for this test")
def test_apply_cupy_dia_jit():
    cps = pytest.importorskip("cupyx.scipy.sparse")
    row = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    col = np.array([0, 15, 0, 1, 7, 15], dtype=np.int32)
    val = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    coo = sps.coo_array((val, (row, col)), shape=(3, 16))
    dia = cps.dia_matrix(coo)  # builds fromp np/scipy only

    u = _compile_and_apply_kernel_value_only(dia)
    assert str(u) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, j: <LevelFormat.RANGE>)\n"
        "device   : cuda\n"
        "dim      : [3, 16]\n"
        "lvl      : [18, 16]\n"
        "nse      : 80\n"
        "pos[0]   : [0, 5] #2\n"
        "crd[0]   : [-1, 0, 6, 13, 15] #5\n"
        "values   : [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
        "2.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ..., 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0] #80\n"
        "data     : 348 bytes\n"
        "sparsity : -66.67%\n"
        "----"
    )
    u = _compile_and_apply_kernel_with_indices(dia)
    assert str(u) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, j: <LevelFormat.RANGE>)\n"
        "device   : cuda\n"
        "dim      : [3, 16]\n"
        "lvl      : [18, 16]\n"
        "nse      : 80\n"
        "pos[0]   : [0, 5] #2\n"
        "crd[0]   : [-1, 0, 6, 13, 15] #5\n"
        "values   : [7.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
        "2.0, 10.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ..., 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 15.0, 29.0, 0.0, 0.0, 0.0, "
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0] #80\n"
        "data     : 348 bytes\n"
        "sparsity : -66.67%\n"
        "----"
    )


@pytest.mark.skipif(not JIT_AVAILABLE, reason="jitting is required for this test")
def test_apply_torch_jit():
    torch = pytest.importorskip("torch")
    a = torch.ones([10, 20], dtype=torch.float64).cuda()
    e = 2 * a
    u = _compile_and_apply_kernel_value_only(a)
    b = u.to_package()
    assert torch.equal(b, e)
    assert torch.equal(a, e)  # changed underlying array!


def test_matmul_cupy_jit():
    cp = pytest.importorskip("cupy")
    # VV operation.
    x = cp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    y = cp.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    z = cp.zeros((), dtype=np.float32)
    xu = Tensor.from_package(x)
    yu = Tensor.from_package(y)
    zu = Tensor.from_package(z)
    mm = Matmul(xu, yu, zu, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()
    assert z == 120.0

    # MV operation.
    A = cp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 4, 1]], dtype=np.float32)
    ust = Tensor.from_package(A)
    x = cp.array([10, 20, 30, 40], dtype=np.float32)
    y = cp.zeros((3,), dtype=np.float32)
    u = Tensor.from_package(x)
    v = Tensor.from_package(y)
    mm = Matmul(ust, u, v, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()
    e = cp.array([300, 700, 410], dtype=np.float32)
    assert cp.array_equal(v.val.tensor, e)

    # MM operation.
    X = cp.array([[10, 1, -1], [20, 2, -2], [30, 3, -3], [40, 4, -4]], dtype=np.float32)
    Y = cp.zeros((3, 3), dtype=np.float32)
    U = Tensor.from_package(X)
    V = Tensor.from_package(Y)
    mm = Matmul(ust, U, V, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()
    e = cp.array([300.0, 30.0, -30.0, 700.0, 70.0, -70.0, 410.0, 41.0, -41.0], dtype=np.float32)
    assert cp.array_equal(V.val.tensor, e)


def test_matmul_torch_jit():
    """
    Prepare a batched MM: C(b,i,k) = A(b,i,j) B(j,k)

    [[[ 1.,  2.,  3.,  4.],
      [ 5.,  6.,  7.,  8.],     [[1., 2.],
      [ 9., 10., 11., 12.]], x   [3., 4.],
               .                 [5., 6.],
     [[13., 14., 15., 16.],      [7., 8.]],
      [17., 18., 19., 20.],
      [21., 22., 23., 24.]]]  =  [[[ 50.,  60.],
                                   [114., 140.],
                                   [178., 220.]],
                                        .
                                  [[242., 300.],
                                   [306., 380.],
                                   [370., 460.]]
    """
    torch = pytest.importorskip("torch")
    # This is dense/dense/dense batching.
    A = (1.0 + torch.arange(2 * 3 * 4)).reshape(2, 3, 4).cuda()
    B = (1.0 + torch.arange(4 * 2)).reshape(4, 2).cuda()
    C = torch.zeros((2, 3, 2), dtype=torch.float32).cuda()
    E = torch.matmul(A, B)
    A_u = Tensor.from_package(A)
    B_u = Tensor.from_package(B)
    C_u = Tensor.from_package(C)
    mm = Matmul(A_u, B_u, C_u, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()

    assert torch.equal(E, C)

    # This is batch/dense/compressed batching.
    C = torch.zeros((2, 3, 2), dtype=torch.float32).cuda()
    batched_csr = A.to_sparse_csr(dense_dim=0)
    A_u = Tensor.from_package(batched_csr)
    B_u = Tensor.from_package(B)
    C_u = Tensor.from_package(C)
    mm = Matmul(A_u, B_u, C_u, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()

    assert torch.equal(E, C)

    # This is batch/dense/compressed batching with more dimensions.
    A = (1.0 + torch.arange(2 * 3 * 4 * 5)).reshape(2, 3, 4, 5).cuda()
    B = (1.0 + torch.arange(5 * 3)).reshape(5, 3).cuda()
    C = torch.zeros((2, 3, 4, 3), dtype=torch.float32).cuda()
    E = torch.matmul(A, B)
    batched_csr = A.to_sparse_csr(dense_dim=0)
    A_u = Tensor.from_package(batched_csr)
    B_u = Tensor.from_package(B)
    C_u = Tensor.from_package(C)
    mm = Matmul(A_u, B_u, C_u, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()

    assert torch.equal(E, C)

    # This is batch/dense/compressed batching with more dimensions (csc).
    C = torch.zeros((2, 3, 4, 3), dtype=torch.float32).cuda()
    batched_csc = A.to_sparse_csc(dense_dim=0)
    A_u = Tensor.from_package(batched_csc)
    C_u = Tensor.from_package(C)
    mm = Matmul(A_u, B_u, C_u, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()

    assert torch.equal(E, C)

    # This is batch/dense/compressed batching with more dimensions (bsr).
    A = (1.0 + torch.arange(2 * 3 * 4 * 8)).reshape(2, 3, 4, 8).cuda()
    B = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    C = torch.zeros((2, 3, 4, 8), dtype=torch.float32).cuda()
    E = torch.matmul(A, B)
    batched_bsr = A.to_sparse_bsr(blocksize=(2, 2), dense_dim=0)
    A_u = Tensor.from_package(batched_bsr)
    B_u = Tensor.from_package(B)
    C_u = Tensor.from_package(C)
    mm = Matmul(A_u, B_u, C_u, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()

    assert torch.equal(E, C)

    # This is batch/dense/compressed batching with more dimensions (bsc).
    C = torch.zeros((2, 3, 4, 8), dtype=torch.float32).cuda()
    batched_bsc = A.to_sparse_bsc(blocksize=(2, 2), dense_dim=0)
    A_u = Tensor.from_package(batched_bsc)
    C_u = Tensor.from_package(C)
    mm = Matmul(A_u, B_u, C_u, beta=1.0, options={"codegen": True})
    mm.plan()
    mm.execute()

    assert torch.equal(E, C)
