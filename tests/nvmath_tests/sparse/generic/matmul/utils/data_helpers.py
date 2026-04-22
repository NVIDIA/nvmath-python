# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


import math

from .....linalg.utils import assert_tensors_equal
from ....utils.common_axes import (
    DType,
    Framework,
    OperandPlacement,
    SparseArrayType,
    cp,
    csp,
    device_id_from_array,
    framework2dtype,
    framework2operand_placement,
    framework2sparse_array_type_support,
    framework2tensor_framework,
    framework_from_array,
    get_values_dtype_from_array,
    is_complex,
    np,
    operand_placement_from_array,
    sp,
    sparse_supporting_frameworks,
    torch,
)
from ....utils.utils import coalesce_array, get_framework_device_ctx, to_dense_numpy

# ==========================
# Helper functions
# ==========================


def _create_np_random_matrix(
    n: int,
    m: int,
    dtype: DType,
    seed: int,
    lo: float,
    hi: float,
    density: None | float = None,
    batch_dims=(),
):
    assert lo < hi

    rng = np.random.default_rng(seed)
    shape = (*batch_dims, n, m)

    a = rng.uniform(lo, hi, size=shape)
    if is_complex(dtype):
        b = rng.uniform(lo, hi, size=shape)
        a = a + 1j * b

    if density is not None:
        _zero_out(a, density, rng)

    return a


def _zero_out(a, density, rng):
    assert 0 < density <= 1

    n, m = a.shape[-2:]
    batch_vol = math.prod(a.shape[:-2])
    size_per_batch = n * m
    zeros_per_batch = int((1 - density) * size_per_batch)

    if batch_vol <= 1:
        nz = zeros_per_batch if batch_vol == 1 else 0
        if nz > 0:
            indices = rng.choice(size_per_batch, size=nz, replace=False)
            a.flat[indices] = 0
        return

    a_flat = a.reshape(batch_vol, size_per_batch)
    indices = rng.choice(size_per_batch, size=zeros_per_batch, replace=False)
    for i in range(batch_vol):
        a_flat[i, indices] = 0


def _cupy_as_sparse(a, sparse_array_type: SparseArrayType, dtype):
    a = cp.asarray(a).astype(dtype)

    match sparse_array_type:
        case SparseArrayType.COO:
            sa = csp.coo_matrix(a)
        case SparseArrayType.CSR:
            sa = csp.csr_matrix(a)
        case SparseArrayType.CSC:
            sa = csp.csc_matrix(a)
        case _:
            raise RuntimeError(f"Sparse array type {sparse_array_type} is not supported for cupy")
    return sa


def _torch_as_sparse(a, sparse_array_type: SparseArrayType, dtype, index_dtype):
    a = torch.from_numpy(a).type(dtype)
    m, n = a.shape[-2:]
    block_size = (2 if m % 2 == 0 else 1, 2 if n % 2 == 0 else 1)

    match sparse_array_type:
        case SparseArrayType.COO:
            sa = a.to_sparse_coo()
            sa = torch.sparse_coo_tensor(
                sa.indices().to(dtype=index_dtype),
                sa.values(),
                size=sa.size(),
            ).coalesce()
            # assert sa.indices().dtype == index_dtype TODO(jlisowski):
            # check if this is torch upcast indexes to int64
            assert sa.values().dtype == dtype
        case SparseArrayType.CSR:
            sa = a.to_sparse_csr()
            sa = torch.sparse_csr_tensor(
                sa.crow_indices().to(dtype=index_dtype),
                sa.col_indices().to(dtype=index_dtype),
                sa.values(),
                size=sa.size(),
            )
            assert sa.col_indices().dtype == index_dtype
            assert sa.crow_indices().dtype == index_dtype
            assert sa.values().dtype == dtype
        case SparseArrayType.CSC:
            sa = a.to_sparse_csc()
            sa = torch.sparse_csc_tensor(
                sa.ccol_indices().to(dtype=index_dtype),
                sa.row_indices().to(dtype=index_dtype),
                sa.values(),
                size=sa.size(),
            )
            assert sa.ccol_indices().dtype == index_dtype
            assert sa.row_indices().dtype == index_dtype
            assert sa.values().dtype == dtype
        case SparseArrayType.BSR:
            sa = a.to_sparse_bsr(block_size)
            sa = torch.sparse_bsr_tensor(
                sa.crow_indices().to(dtype=index_dtype),
                sa.col_indices().to(dtype=index_dtype),
                sa.values(),
                size=sa.size(),
            )
            assert sa.crow_indices().dtype == index_dtype
            assert sa.col_indices().dtype == index_dtype
            assert sa.values().dtype == dtype
        case SparseArrayType.BSC:
            sa = a.to_sparse_bsc(block_size)
            sa = torch.sparse_bsc_tensor(
                sa.ccol_indices().to(dtype=index_dtype),
                sa.row_indices().to(dtype=index_dtype),
                sa.values(),
                size=sa.size(),
            )
            assert sa.ccol_indices().dtype == index_dtype
            assert sa.row_indices().dtype == index_dtype
            assert sa.values().dtype == dtype
        case _:
            raise RuntimeError(f"Sparse array type {sparse_array_type} is not supported for torch")

    return sa


def _scipy_as_sparse(a, sparse_array_type: SparseArrayType, dtype):
    a = a.astype(dtype)

    match sparse_array_type:
        case SparseArrayType.COO:
            return sp.coo_matrix(a)
        case SparseArrayType.CSR:
            return sp.csr_matrix(a)
        case SparseArrayType.CSC:
            return sp.csc_matrix(a)
        case SparseArrayType.BSR:
            return sp.bsr_matrix(a)
        case SparseArrayType.DIA:
            return sp.dia_matrix(a)
        case _:
            raise RuntimeError(f"Sparse array type {sparse_array_type} is not supported for scipy")


def _default_device_id(operand_placement: OperandPlacement, device_id: int | None):
    if operand_placement == OperandPlacement.host:
        if device_id is None:
            device_id = "cpu"
        assert device_id == "cpu"
    else:
        assert operand_placement == OperandPlacement.device
        if device_id is None:
            device_id = 0
        assert isinstance(device_id, int) and device_id >= 0

    return device_id


def _as_sparse_matrix(
    a,
    framework: Framework,
    operand_placement: OperandPlacement,
    sparse_array_type: SparseArrayType,
    dtype: DType,
    device_id=None,
    index_dtype: DType = DType.int32,
    batch_dims=(),
):
    device_id = _default_device_id(operand_placement, device_id)

    assert operand_placement in framework2operand_placement[framework]
    assert dtype in framework2dtype[framework]
    framework_dtype = framework2dtype[framework][dtype]
    framework_index_dtype = framework2dtype[framework][index_dtype]

    assert sparse_array_type in framework2sparse_array_type_support[framework]

    arr = None
    with get_framework_device_ctx(device_id, framework):
        match framework:
            case Framework.cupyx:
                assert index_dtype == DType.int32, "cupyx only supports int32 index dtype"
                assert operand_placement == OperandPlacement.device, "cupyx only supports GPU operand placement"
                assert batch_dims == (), "cupyx does not support batch dimensions"
                arr = _cupy_as_sparse(a, sparse_array_type, framework_dtype)
            case Framework.torch:
                arr = _torch_as_sparse(a, sparse_array_type, framework_dtype, framework_index_dtype)
                if operand_placement == OperandPlacement.device:
                    arr = arr.to("cuda")
            case Framework.scipy:
                assert device_id == "cpu"
                assert index_dtype == DType.int32, "scipy only supports int32 index dtype"
                assert operand_placement == OperandPlacement.host, "scipy only supports CPU operand placement"
                assert batch_dims == (), "scipy does not support batch dimensions"
                arr = _scipy_as_sparse(a, sparse_array_type, framework_dtype)
            case _:
                raise RuntimeError(f"Framework {framework} is not supported")
        arr = coalesce_array(arr)

    assert operand_placement_from_array(arr) == operand_placement, f"{operand_placement_from_array(arr)} != {operand_placement}"
    assert device_id_from_array(arr) == device_id, f"{device_id_from_array(arr)} != {device_id}"
    return arr


def _move_to_framework(
    a,
    dtype: DType,
    framework: Framework,
    operand_placement: OperandPlacement,
    device_id=None,
    batch_dims=(),
):
    device_id = _default_device_id(operand_placement, device_id)
    assert operand_placement in framework2operand_placement[framework]
    assert dtype in framework2dtype[framework]
    framework_dtype = framework2dtype[framework][dtype]

    arr = None
    with get_framework_device_ctx(device_id, framework):
        match framework:
            case Framework.numpy | Framework.scipy:
                assert operand_placement == OperandPlacement.host
                assert device_id == "cpu"
                assert batch_dims == (), "numpy and scipy do not support batch dimensions"
                arr = a.astype(framework_dtype)
            case Framework.cupy | Framework.cupyx:
                assert operand_placement == OperandPlacement.device
                assert device_id >= 0
                assert batch_dims == (), "cupy and cupyx do not support batch dimensions"
                arr = cp.asarray(a).astype(framework_dtype)
            case Framework.torch:
                arr = torch.from_numpy(a).type(framework_dtype)
                if operand_placement == OperandPlacement.device:
                    arr = arr.to("cuda")
            case _:
                raise RuntimeError(f"Framework {framework} is not supported")

    assert operand_placement_from_array(arr) == operand_placement, f"{operand_placement_from_array(arr)} != {operand_placement}"
    assert device_id_from_array(arr) == device_id, f"{device_id_from_array(arr)} != {device_id}"
    return arr


def _cast_dtype(a, dtype):
    framework = framework_from_array(a)
    assert dtype in framework2dtype[framework]
    framework_dtype = framework2dtype[framework][dtype]
    device_id = device_id_from_array(a)

    with get_framework_device_ctx(device_id, framework):
        match framework:
            case Framework.numpy | Framework.scipy | Framework.cupy | Framework.cupyx:
                return a.astype(framework_dtype)
            case Framework.torch:
                return a.to(framework_dtype)
            case _:
                raise RuntimeError(f"Framework {framework} is not supported")


def get_rtol(dtype):
    _COMPLEX32_RTOL = 1e-2

    match dtype:
        case DType.complex32:
            return _COMPLEX32_RTOL
        case _:
            return None


# ==========================
# Sample generation
# ==========================


def create_random_sparse_matrix(
    framework: Framework,
    operand_placement: OperandPlacement,
    sparse_array_type: SparseArrayType,
    n: int,
    m: int,
    density: None | float,
    dtype: DType,
    seed: int,
    lo: float = 0.0,
    hi: float = 1.0,
    device_id=None,
    batch_dims=(),
    index_dtype: DType = DType.int32,
):
    if isinstance(batch_dims, int):
        batch_dims = (batch_dims,)
    a = _create_np_random_matrix(n, m, dtype, seed, lo, hi, density, batch_dims)
    sa = _as_sparse_matrix(a, framework, operand_placement, sparse_array_type, dtype, device_id, index_dtype, batch_dims)
    assert sa.shape == (*batch_dims, n, m)
    return sa


def create_random_dense_matrix(
    framework: Framework,
    operand_placement: OperandPlacement,
    n: int,
    m: int,
    dtype: DType,
    seed: int,
    lo: float = 0.0,
    hi: float = 1.0,
    device_id=None,
    density: None | float = None,
    batch_dims=(),
):
    if isinstance(batch_dims, int):
        batch_dims = (batch_dims,)

    a = _create_np_random_matrix(n, m, dtype, seed, lo, hi, density, batch_dims)
    return _move_to_framework(a, dtype, framework, operand_placement, device_id, batch_dims)


def check_meta_data(a, b, c, framework, operand_placement, device_id, dtype, a_shape, b_shape, c_shape):
    assert framework in sparse_supporting_frameworks, f"{framework} is not a sparse supporting framework"

    device_id = _default_device_id(operand_placement, device_id)
    sparse_framework = framework
    dense_framework = framework2tensor_framework[framework]

    a_framework = framework_from_array(a)
    b_framework = framework_from_array(b)
    c_framework = framework_from_array(c)

    assert a_framework == sparse_framework, f"{a_framework} != {sparse_framework}"
    assert b_framework == dense_framework, f"{b_framework} != {dense_framework}"
    assert c_framework == dense_framework, f"{c_framework} != {dense_framework}"

    assert operand_placement_from_array(a) == operand_placement, f"{operand_placement_from_array(a)} != {operand_placement}"
    assert operand_placement_from_array(b) == operand_placement, f"{operand_placement_from_array(b)} != {operand_placement}"
    assert operand_placement_from_array(c) == operand_placement, f"{operand_placement_from_array(c)} != {operand_placement}"

    assert device_id_from_array(a) == device_id, f"{device_id_from_array(a)} != {device_id}"
    assert device_id_from_array(b) == device_id, f"{device_id_from_array(b)} != {device_id}"
    assert device_id_from_array(c) == device_id, f"{device_id_from_array(c)} != {device_id}"

    assert get_values_dtype_from_array(a) == dtype, f"{get_values_dtype_from_array(a)} != {dtype}"
    assert get_values_dtype_from_array(b) == dtype, f"{get_values_dtype_from_array(b)} != {dtype}"
    assert get_values_dtype_from_array(c) == dtype, f"{get_values_dtype_from_array(c)} != {dtype}"

    assert tuple(a.shape) == tuple(a_shape), f"{a.shape} != {a_shape}"
    assert tuple(b.shape) == tuple(b_shape), f"{b.shape} != {b_shape}"
    assert tuple(c.shape) == tuple(c_shape), f"{c.shape} != {c_shape}"


def calculate_reference(a, b, c, dtype, alpha=1.0, beta=1.0, qualifiers=None, device_id=None):
    c_framework = framework_from_array(c)

    framework_dtype = framework2dtype[c_framework][dtype]
    assert a.dtype == framework_dtype, f"{a.dtype} != {framework_dtype}"
    assert b.dtype == framework_dtype, f"{b.dtype} != {framework_dtype}"
    assert c.dtype == framework_dtype, f"{c.dtype} != {framework_dtype}"
    operand_placement = operand_placement_from_array(a)

    _COMPUTE_TYPES = {
        DType.float32: DType.float32,
        DType.float64: DType.float64,
        DType.complex64: DType.complex64,
        DType.complex128: DType.complex128,
        DType.bfloat16: DType.float32,
        DType.complex32: DType.complex64,
        DType.float16: DType.float32,
    }
    assert dtype in _COMPUTE_TYPES, f"{dtype} is not supported by reference calculation"
    compute_type = _COMPUTE_TYPES[dtype]

    a = _cast_dtype(a, compute_type)
    b = _cast_dtype(b, compute_type)
    c = _cast_dtype(c, compute_type)

    a_np = to_dense_numpy(a)
    b_np = to_dense_numpy(b)
    c_np = to_dense_numpy(c)

    if qualifiers is not None and qualifiers[0]["is_transpose"]:
        a_np = a_np.swapaxes(-2, -1)
    if qualifiers is not None and qualifiers[1]["is_transpose"]:
        b_np = b_np.swapaxes(-2, -1)
    if qualifiers is not None and qualifiers[2]["is_transpose"]:
        c_np = c_np.swapaxes(-2, -1)
    if qualifiers is not None and qualifiers[0]["is_conjugate"]:
        a_np = np.conj(a_np)
    if qualifiers is not None and qualifiers[1]["is_conjugate"]:
        b_np = np.conj(b_np)
    if qualifiers is not None and qualifiers[2]["is_conjugate"]:
        c_np = np.conj(c_np)

    result = (alpha * a_np) @ b_np + (beta * c_np)

    return _move_to_framework(result, dtype, c_framework, operand_placement, device_id, ())


def compare_results(original_result, reference_result, dtype, rtol=None):
    _CAST_RULES = {
        DType.complex32: DType.complex64,
        DType.bfloat16: DType.float16,
    }

    if dtype in _CAST_RULES:
        reference = _cast_dtype(reference_result, _CAST_RULES[dtype])
        result = _cast_dtype(original_result, _CAST_RULES[dtype])
    else:
        reference, result = reference_result, original_result

    result = to_dense_numpy(result)
    reference = to_dense_numpy(reference)
    assert_tensors_equal(result, reference, rtol=rtol if rtol is not None else get_rtol(dtype))
