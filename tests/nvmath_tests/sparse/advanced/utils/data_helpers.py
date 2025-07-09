# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


from .common_axes import (
    Framework,
    DType,
    DenseRHS,
    RHSMatrix,
    RHSVector,
    RHSBatch,
    SparseArrayType,
    OperandPlacement,
    framework2dtype,
    framework2operand_placement,
    device_id_from_array,
    framework_from_array,
    is_complex,
)
from .common_axes import sp, cp, torch, csp, np
import math


def create_random_sparse_matrix(
    framework: Framework,
    operand_placement: OperandPlacement,
    sparse_array_type: SparseArrayType,
    n: int,
    m: int,
    density: None | float,
    dtype: DType,
    seed: int,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id=None,
    batch_dims=(),
    index_dtype: DType = DType.int32,
    ones=False,
):
    if isinstance(batch_dims, int):
        batch_dims = (batch_dims,)
    a = create_np_random_matrix(n, m, dtype, seed, lo, hi, density, batch_dims, ones)
    sa = as_sparse_matrix(a, framework, operand_placement, sparse_array_type, dtype, device_id, index_dtype)
    assert sa.shape == (*batch_dims, n, m)
    return sa


def create_random_sparse_alg_matrix(
    framework: Framework,
    operand_placement: OperandPlacement,
    sparse_array_type: SparseArrayType,
    n: int,
    m: int,
    density: None | float,
    dtype: DType,
    seed: int,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id=None,
    batch_dims=(),
    index_dtype: DType = DType.int32,
    ones=False,
    alg_matrix_type=None,
    alg_matrix_view=None,
):
    if isinstance(batch_dims, int):
        batch_dims = (batch_dims,)
    a = create_np_random_matrix(n, m, dtype, seed, lo, hi, density, batch_dims, ones, alg_matrix_type)
    match alg_matrix_view:
        case None:
            a_view = a
        case "lower":
            a_view = np.tril(a)
        case "upper":
            a_view = np.triu(a)
        case _:
            raise AssertionError(f"Unsupported alg_matrix_view: {alg_matrix_view}")
    sa = as_sparse_matrix(a, framework, operand_placement, sparse_array_type, dtype, device_id, index_dtype)
    assert sa.shape == (*batch_dims, n, m)
    sa_view = as_sparse_matrix(a_view, framework, operand_placement, sparse_array_type, dtype, device_id, index_dtype)
    assert sa_view.shape == (*batch_dims, n, m)
    return sa, sa_view


def _zero_out_non_diag(a, density, rng):
    assert 0 < density <= 1
    # For the numerical stability in the tests, we keep the diagonal elements
    # and only zero out the required number of non-diagonal elements.
    n, m = a.shape[-2:]
    diag = min(n, m)
    batch_vol = math.prod(a.shape[:-2])
    nz = int((1 - density) * batch_vol * n * m)
    indices = rng.choice(batch_vol * (n * m - diag), size=nz, replace=False)
    batch_idx = indices // (n * m - diag)
    matrix_idx = indices % (n * m - diag)
    matrix_idx += np.minimum(matrix_idx // m + 1, diag)
    indices = batch_idx * (n * m) + matrix_idx
    a.flat[indices] = 0


def create_np_random_matrix(
    n: int,
    m: int,
    dtype: DType,
    seed: int,
    lo: float,
    hi: float,
    density: None | float = None,
    batch_dims=(),
    ones=False,
    alg_matrix_type=None,
):
    assert lo < hi
    if n != m and not ones:
        raise NotImplementedError("Non-square matrices are not supported when ones=False")
    if density is not None and alg_matrix_type is not None:
        raise NotImplementedError("Density and alg_matrix_type are not supported together")

    rng = np.random.default_rng(seed)
    shape = (*batch_dims, n, m)
    if ones:
        assert alg_matrix_type is None
        assert density is None
        a = np.ones(shape)
    else:
        a = rng.uniform(lo, hi, size=shape)
        if is_complex(dtype):
            b = rng.uniform(lo, hi, size=shape)
            a = a + 1j * b

        if alg_matrix_type is None:
            a += np.diag([hi * n] * min(n, m))
            if density is not None:
                _zero_out_non_diag(a, density, rng)
        else:
            a_h = a.T.conj() if is_complex(dtype) else a.T
            a = ((a + a_h) / 2) + (np.diag([hi * n] * min(n, m)))
            if alg_matrix_type == "positive":
                a_h = a.T.conj() if is_complex(dtype) else a.T
                a = a @ a_h
            else:
                assert alg_matrix_type == "symmetric"
    return a


def as_sparse_matrix(
    a,
    framework: Framework,
    operand_placement: OperandPlacement,
    sparse_array_type: SparseArrayType,
    dtype: DType,
    device_id=None,
    index_dtype: DType = DType.int32,
):
    if sparse_array_type != SparseArrayType.CSR:
        raise NotImplementedError(f"Sparse array type {sparse_array_type} is not supported")

    if operand_placement == OperandPlacement.host:
        if device_id is None:
            device_id = "cpu"
        assert device_id == "cpu"
    else:
        assert operand_placement == OperandPlacement.device
        if device_id is None:
            device_id = 0
        assert isinstance(device_id, int) and device_id >= 0

    assert operand_placement in framework2operand_placement[framework]
    framework_dtype = framework2dtype[framework][dtype]

    if framework == Framework.cupyx:
        assert index_dtype == DType.int32, "cupyx only supports int32 index dtype"
        assert operand_placement == OperandPlacement.device, "cupyx only supports GPU operand placement"
        with cp.cuda.Device(device_id):
            a = cp.asarray(a).astype(framework_dtype)
            a = csp.csr_matrix(a)
        assert device_id_from_array(a) == device_id, f"{device_id_from_array(a)} != {device_id}"
        return a
    if framework == Framework.torch:
        a = torch.from_numpy(a).type(framework_dtype)
        a = a.to_sparse_csr()
        # Note that torch uses int64 for index buffers, whereas cuDSS currently requires
        # int32. We need to convert the index buffers to int32.
        torch_index_dtype = framework2dtype[Framework.torch][index_dtype]
        a = torch.sparse_csr_tensor(
            a.crow_indices().to(dtype=torch_index_dtype),
            a.col_indices().to(dtype=torch_index_dtype),
            a.values(),
            size=a.size(),
            device=f"cuda:{device_id}" if isinstance(device_id, int) else device_id,
        )
        assert a.col_indices().dtype == torch_index_dtype
        assert a.crow_indices().dtype == torch_index_dtype
        assert a.is_cuda == (operand_placement == OperandPlacement.device)
        assert device_id_from_array(a) == device_id, f"{device_id_from_array(a)} != {device_id}"
        return a
    if framework == Framework.scipy:
        assert device_id == "cpu"
        assert index_dtype == DType.int32, "scipy only supports int32 index dtype"
        a = a.astype(framework_dtype)
        a = sp.csr_matrix(a)
        assert operand_placement == OperandPlacement.host, "scipy only supports CPU operand placement"
        return a
    raise RuntimeError(f"Framework {framework} is not supported")


def create_dense_rhs(
    framework: Framework,
    operand_placement: OperandPlacement,
    rhs: RHSVector | RHSMatrix | RHSBatch,
    dtype: DType,
    lo: float = 0,
    hi: float = 1,
    device_id=None,
    start=1,
):
    assert lo < hi
    if operand_placement == OperandPlacement.host:
        if device_id is None:
            device_id = "cpu"
        assert device_id == "cpu"
    else:
        assert operand_placement == OperandPlacement.device
        if device_id is None:
            device_id = 0
        assert isinstance(device_id, int) and device_id >= 0

    assert operand_placement in framework2operand_placement[framework]
    framework_dtype = framework2dtype[framework][dtype]

    if rhs.type == DenseRHS.vector:
        ret = np.arange(start, rhs.n + start)
    elif rhs.type == DenseRHS.matrix:
        assert isinstance(rhs, RHSMatrix)
        ret = np.arange(start, rhs.n * rhs.k + start).reshape(rhs.k, rhs.n)
    elif rhs.type == DenseRHS.batch:
        assert isinstance(rhs, RHSBatch)
        batch_dims = rhs.batch_dims
        n, k = rhs.n, rhs.k
        ret = np.arange(start, n * k * math.prod(batch_dims) + start).reshape(*batch_dims, k, n)
    else:
        raise ValueError(f"Unsupported dense RHS type: {rhs.type}")

    match framework:
        case Framework.numpy:
            ret = np.asarray(ret).astype(framework_dtype)
            if rhs.type != DenseRHS.vector:
                ret = ret.swapaxes(-2, -1)
            return ret
        case Framework.cupy:
            with cp.cuda.Device(device_id):
                ret = cp.asarray(ret).astype(framework_dtype)
                if rhs.type != DenseRHS.vector:
                    ret = ret.swapaxes(-2, -1)
            assert device_id_from_array(ret) == device_id, f"{device_id_from_array(ret)} != {device_id}"
            return ret
        case Framework.torch:
            ret = torch.from_numpy(ret).type(framework_dtype)
            if operand_placement == OperandPlacement.device:
                ret = ret.to(f"cuda:{device_id}")
            if rhs.type != DenseRHS.vector:
                ret = ret.swapaxes(-2, -1)
            assert ret.is_cuda == (operand_placement == OperandPlacement.device)
            assert device_id_from_array(ret) == device_id, f"{device_id_from_array(ret)} != {device_id}"
            return ret
    raise RuntimeError(f"Framework {framework} is not supported")


def _torch_sparse_constructor(format):
    def create(shape, dtype, placement):
        dense = torch.ones(
            size=shape,
            dtype=framework2dtype[Framework.torch][dtype],
            device="cpu" if placement == OperandPlacement.host else "cuda",
        )
        kwargs = {}
        if format in ("bsr", "bsc"):
            kwargs["blocksize"] = (1, 1)
        return getattr(dense, f"to_sparse_{format}")(**kwargs)

    return create


def _scipy_sparse_constructor(format):
    def create(shape, dtype, placement):
        m, n = shape
        return sp.eye_array(m, n, dtype=framework2dtype[Framework.scipy][dtype], format=format)

    return create


def _cupyx_sparse_constructor(format):
    def create(shape, dtype, placement):
        m, n = shape
        return csp.eye(m, n, dtype=framework2dtype[Framework.cupyx][dtype], format=format)

    return create


unsupported_sparse_formats = {
    Framework.torch: {f: _torch_sparse_constructor(f) for f in ("coo", "csc", "bsr", "bsc")},
    Framework.scipy: {f: _scipy_sparse_constructor(f) for f in ("coo", "csc", "bsr", "dia", "dok", "lil")},
    Framework.cupyx: {f: _cupyx_sparse_constructor(f) for f in ("coo", "csc", "dia")},
}


def sparse_matrix_add_const_inplace(a, c):
    framework = framework_from_array(a)
    match framework:
        case Framework.cupyx | Framework.scipy:
            a.data += c
        case Framework.torch:
            values = a.values()
            values += c
        case _:
            raise ValueError(f"Framework {framework} is not supported")
