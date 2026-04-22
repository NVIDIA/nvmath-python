# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest

import nvmath.sparse.ust as ust
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.sparse import ComputeType, MatmulOptions, matmul, matmul_matrix_qualifiers_dtype

from ....linalg.utils import assert_tensors_equal
from ...utils.common_axes import (
    JIT_AVAILABLE,
    DType,
    Framework,
    OperandPlacement,
    Param,
    SparseArrayType,
    framework2dtype,
    framework2index_dtype,
    framework2operand_placement,
    framework2sparse_array_type_support,
    sparse_supporting_frameworks,
)
from ...utils.utils import allow_cusparse_unsupported, idfn, is_known_linker_error
from .utils.data_helpers import (
    calculate_reference,
    check_meta_data,
    compare_results,
    create_random_dense_matrix,
    create_random_sparse_matrix,
)
from .utils.support_matrix import (
    batched_named_formats,
    direct_named_formats,
    supported_codegen_index_dtypes,
    supported_dtypes,
    supported_formats,
    supported_index_dtypes,
)

if Framework.torch in Framework.enabled():
    maybe_register_package("torch")
if Framework.cupy in Framework.enabled():
    maybe_register_package("cupy")

RNG_SEED = 92

# ==========================
# Helper functions
# ==========================


def dtype_to_compute_type(dtype):
    match dtype:
        case DType.float32:
            return ComputeType.CUDA_R_32F
        case DType.float64:
            return ComputeType.CUDA_R_64F
        case DType.complex64:
            return ComputeType.CUDA_C_32F
        case DType.complex128:
            return ComputeType.CUDA_C_64F
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def run_matmul(
    size,
    qualifiers=None,
    alpha=1.0,
    beta=1.0,
    batch_dims=(),
    framework=Framework.torch,
    operand_placement=OperandPlacement.device,
    sparse_array_type=SparseArrayType.CSR,
    dtype=DType.float32,
    index_dtype=DType.int32,
    options=None,
    use_ust=False,
    density=0.5,
    rtol=None,
    named_format=None,
):
    assert named_format is None or use_ust, "NamedFormats can only be used with UST"

    if isinstance(batch_dims, tuple) and len(batch_dims) == 3 and all(isinstance(b, tuple) for b in batch_dims):
        batch_a, batch_b, batch_c = batch_dims
    else:
        batch_a = batch_b = batch_c = batch_dims
    m, n, k = size

    a_ref = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        density,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
        batch_dims=batch_a,
    )
    b_ref = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED, batch_dims=batch_b)
    c_ref = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED, batch_dims=batch_c)

    reference = calculate_reference(a_ref, b_ref, c_ref, dtype, alpha=alpha, beta=beta, qualifiers=qualifiers)

    try:
        a = ust.Tensor.from_package(a_ref) if use_ust else a_ref
        b = ust.Tensor.from_package(b_ref) if use_ust else b_ref
        c = ust.Tensor.from_package(c_ref) if use_ust else c_ref
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    if named_format is not None:
        assert use_ust, "NamedFormats can only be used with UST"
        a = a.convert(tensor_format=named_format)

    try:
        result = matmul(a, b, c=c, options=options, alpha=alpha, beta=beta, qualifiers=qualifiers)
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")
    except Exception as e:
        if is_known_linker_error(str(e)):
            pytest.skip("CUDA linker error: duplicate __half/__nv_bfloat16 symbol (CTK 12.x issue).")
        raise

    check_meta_data(
        a_ref, b_ref, c_ref, framework, operand_placement, None, dtype, (*batch_a, m, n), (*batch_b, n, k), (*batch_c, m, k)
    )

    if use_ust:
        a = a.to_package() if not named_format else a_ref
        b = b.to_package()
        result = result.to_package()

    check_meta_data(
        a, b, result, framework, operand_placement, None, dtype, (*batch_a, m, n), (*batch_b, n, k), (*batch_c, m, k)
    )
    compare_results(result, reference, dtype, rtol=rtol)


def run_options_check(options, use_ust=False):
    run_matmul(
        size=(32, 32, 32),
        options=options,
        use_ust=use_ust,
        operand_placement=OperandPlacement.host,
        framework=Framework.torch,
        dtype=DType.complex64,
    )


def _torch_column_major_last_two_dims(t):
    assert t.ndim >= 2
    return t.transpose(-2, -1).contiguous().transpose(-2, -1)


def _torch_wrong_batch_strides(batch_dims, rows, cols, reference_tensor):
    import torch

    b0, b1 = batch_dims
    elem = rows * cols
    bad_strides = (elem, b0 * elem, cols, 1)
    t = torch.empty_strided((*batch_dims, rows, cols), bad_strides, device=reference_tensor.device, dtype=torch.float32)
    t.copy_(reference_tensor)
    return t


# ==========================
# Positive tests
# ==========================


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "index_dtype",
        "dtype",
        "size",
        "density",
        "use_ust",
        "enforce_codegen",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            index_dtype,
            dtype,
            Param("size", size),
            Param("density", density),
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for index_dtype in (supported_codegen_index_dtypes if use_ust else supported_index_dtypes)
        if index_dtype in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for size in [(23, 27, 29), (64, 64, 64)]
        for density in [0.5, 0.2]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_input_dtype(
    framework, operand_placement, sparse_array_type, index_dtype, dtype, size, density, use_ust, enforce_codegen
):
    if sparse_array_type == SparseArrayType.COO and framework == Framework.torch and index_dtype != DType.int64:
        pytest.skip("Torch COO currently only supports int64 index dtype")

    size = size.value
    density = density.value
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=size,
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            index_dtype=index_dtype,
            dtype=dtype,
            density=density,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen),
        )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    (
        "named_format",
        "index_dtype",
        "dtype",
        "size",
        "density",
    ),
    [
        (
            Param("named_format", named_format),
            index_dtype,
            dtype,
            Param("size", size),
            Param("density", density),
        )
        for named_format in direct_named_formats
        for index_dtype in supported_codegen_index_dtypes
        if index_dtype in framework2index_dtype[Framework.torch]
        for dtype in supported_dtypes
        if dtype in framework2dtype[Framework.torch]
        for size in [(23, 27, 29), (64, 64, 64)]
        for density in [0.5, 0.2]
    ],
    ids=idfn,
)
def test_named_formats_single_source(named_format, index_dtype, dtype, size, density):
    size = size.value
    density = density.value
    named_format = named_format.value

    try:
        run_matmul(
            size=size,
            framework=Framework.torch,
            operand_placement=OperandPlacement.device,
            sparse_array_type=SparseArrayType.CSR,
            index_dtype=index_dtype,
            dtype=dtype,
            density=density,
            use_ust=True,
            options=MatmulOptions(codegen=True),
            named_format=named_format,
            batch_dims=(4,) if named_format in batched_named_formats else (),
        )
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform matmul: {str(e)}")


@pytest.mark.parametrize(
    (
        "named_format",
        "framework",
        "operand_placement",
        "sparse_array_type",
    ),
    [
        (
            Param("named_format", named_format),
            framework,
            operand_placement,
            sparse_array_type,
        )
        for named_format in direct_named_formats
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        # only torch supports batched dimensions
        if framework == Framework.torch or named_format not in batched_named_formats
    ],
    ids=idfn,
)
def test_named_formats_format_conversions(framework, operand_placement, sparse_array_type, named_format):
    named_format = named_format.value
    size = (4, 4, 4)
    density = 0.5
    index_dtype = DType.int32
    dtype = DType.float64

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=size,
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            index_dtype=index_dtype,
            dtype=dtype,
            density=density,
            use_ust=True,
            options=MatmulOptions(codegen=True),
            named_format=named_format,
            batch_dims=(2,) if named_format in batched_named_formats else (),
        )


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "size",
        "density",
        "use_ust",
        "enforce_codegen",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("size", size),
            Param("density", density),
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for size in [
            (11, 13, 17),
            (32, 32, 8),
            (1, 1, 1),
            (100, 100, 100),
            (1, 77, 1),
            (77, 1, 1),
            (1, 1, 77),
            (64, 64, 64),
            (32, 64, 128),
        ]
        for density in [0.5, 0.2]
        for use_ust in [False, True]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_shapes(framework, operand_placement, sparse_array_type, size, density, use_ust, enforce_codegen):
    size = size.value
    density = density.value
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=size,
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            density=density,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen),
        )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    (
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "batch_dims",
        "broadcast",
        "enforce_codegen",
    ),
    [
        (
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("batch_dims", batch_dims),
            broadcast,
            Param("enforce_codegen", enforce_codegen),
        )
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[Framework.torch]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[Framework.torch]
        for batch_dims in [(7,), (2, 2, 2), (2, 2), (1,)]
        for broadcast in ["a", "b", ""]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_batching(operand_placement, sparse_array_type, use_ust, batch_dims, broadcast, enforce_codegen):
    use_ust = use_ust.value
    batch_dims = batch_dims.value
    enforce_codegen = enforce_codegen.value

    a_batch_dims = () if broadcast == "a" else batch_dims
    b_batch_dims = () if broadcast == "b" else batch_dims
    c_batch_dims = batch_dims

    run_matmul(
        size=(32, 32, 32),
        framework=Framework.torch,
        operand_placement=operand_placement,
        sparse_array_type=sparse_array_type,
        batch_dims=(a_batch_dims, b_batch_dims, c_batch_dims),
        use_ust=use_ust,
        options=MatmulOptions(codegen=enforce_codegen),
    )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    (
        "operand_placement",
        "b_column_major",
        "c_column_major",
        "use_ust",
        "enforce_codegen",
        "batch_dims",
    ),
    [
        (
            operand_placement,
            Param("b_column_major", b_column_major),
            Param("c_column_major", c_column_major),
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("batch_dims", batch_dims),
        )
        for operand_placement in framework2operand_placement[Framework.torch]
        for b_column_major in [False, True]
        for c_column_major in [False, True]
        for use_ust in [False, True]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for batch_dims in [(), (3,)]
    ],
    ids=idfn,
)
def test_dense_operand_c_and_f_layout(
    operand_placement,
    b_column_major,
    c_column_major,
    use_ust,
    enforce_codegen,
    batch_dims,
):
    b_column_major = b_column_major.value
    c_column_major = c_column_major.value
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    batch_dims = batch_dims.value

    m, n, k = 17, 19, 23
    dtype = DType.float32

    a = create_random_sparse_matrix(
        Framework.torch,
        operand_placement,
        SparseArrayType.CSR,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=DType.int32,
        batch_dims=batch_dims,
    )
    b = create_random_dense_matrix(Framework.torch, operand_placement, n, k, dtype, seed=RNG_SEED, batch_dims=batch_dims)
    c = create_random_dense_matrix(Framework.torch, operand_placement, m, k, dtype, seed=RNG_SEED, batch_dims=batch_dims)

    reference = calculate_reference(a, b, c, dtype, alpha=1.0, beta=1.0)

    if b_column_major:
        b = _torch_column_major_last_two_dims(b)
    if c_column_major:
        c = _torch_column_major_last_two_dims(c)

    try:
        if use_ust:
            a = ust.Tensor.from_package(a)
            b = ust.Tensor.from_package(b)
            c = ust.Tensor.from_package(c)
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    try:
        result = matmul(
            a,
            b,
            c=c,
            options=MatmulOptions(codegen=enforce_codegen),
            alpha=1.0,
            beta=1.0,
        )
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")

    if use_ust:
        a = a.to_package()
        b = b.to_package()
        result = result.to_package()

    check_meta_data(
        a,
        b,
        result,
        Framework.torch,
        operand_placement,
        None,
        dtype,
        (*batch_dims, m, n),
        (*batch_dims, n, k),
        (*batch_dims, m, k),
    )
    compare_results(result, reference, dtype)


@pytest.mark.skipif(not JIT_AVAILABLE, reason="Jitting is required for this test")
@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "batch_dims",
        "size",
        "transpose_a",
        "transpose_b",
        "conjugate_a",
        "conjugate_b",
        "use_ust",
        "enforce_codegen",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("batch_dims", batch_dims),
            Param("size", size),
            Param("transpose_a", transpose_a),
            Param("transpose_b", transpose_b),
            Param("conjugate_a", conjugate_a),
            Param("conjugate_b", conjugate_b),
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
        )
        for batch_dims in [(), (2, 2)]
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        and (batch_dims == () or framework == Framework.torch)  # Only torch supports batching
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for size in [(32, 32, 32), (13, 17, 23)]
        for transpose_a in [True, False]
        for transpose_b in [True, False]
        for conjugate_a in [True, False]
        for conjugate_b in [True, False]
        for use_ust in [False, True]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_qualifiers(
    framework,
    operand_placement,
    sparse_array_type,
    batch_dims,
    size,
    transpose_a,
    transpose_b,
    conjugate_a,
    conjugate_b,
    use_ust,
    enforce_codegen,
):
    size = size.value
    batch_dims = batch_dims.value
    use_ust = use_ust.value
    m, n, k = size
    transpose_a = transpose_a.value
    transpose_b = transpose_b.value
    conjugate_a = conjugate_a.value
    conjugate_b = conjugate_b.value
    enforce_codegen = enforce_codegen.value
    a_cols = m if transpose_a else n
    b_rows = k if transpose_b else n

    if a_cols != b_rows:
        pytest.skip("Incompatible matrix dimensions after transpose qualifiers")

    if not enforce_codegen and not transpose_a and conjugate_a:
        pytest.skip("cuSPARSE does not support conjugate without transpose")

    if not enforce_codegen and not transpose_b and conjugate_b:
        pytest.skip("cuSPARSE does not support conjugate without transpose")

    if not enforce_codegen and sparse_array_type == SparseArrayType.BSR and (transpose_a or conjugate_a):
        pytest.skip("cuSPARSE does not support transpose or conjugate for BSR A operand")

    qualifiers = np.zeros((3,), dtype=matmul_matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = transpose_a
    qualifiers[1]["is_transpose"] = transpose_b
    qualifiers[0]["is_conjugate"] = conjugate_a
    qualifiers[1]["is_conjugate"] = conjugate_b

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=size,
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            index_dtype=DType.int32,
            dtype=DType.complex64,
            qualifiers=qualifiers,
            use_ust=use_ust,
            batch_dims=batch_dims,
            options=MatmulOptions(codegen=enforce_codegen),
        )


@pytest.mark.skipif(not JIT_AVAILABLE, reason="Jitting is required for this test")
@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "alpha",
        "beta",
        "enforce_codegen",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            alpha,
            beta,
            Param("enforce_codegen", enforce_codegen),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for use_ust in [False, True]
        for alpha in [-1.3, 3.2, 1.0]
        for beta in [-2.4, 4.5, 1.0]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_alpha_beta(framework, operand_placement, sparse_array_type, use_ust, alpha, beta, enforce_codegen):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    m, n, k = (23, 27, 29)
    dtype = DType.float32

    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, m, n, 0.5, dtype, seed=RNG_SEED, index_dtype=DType.int32
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED)
    reference = calculate_reference(a, b, c, dtype, alpha=alpha, beta=beta)

    try:
        if use_ust:
            a = ust.Tensor.from_package(a)
            b = ust.Tensor.from_package(b)
            c = ust.Tensor.from_package(c)
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    try:
        with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
            result = matmul(a, b, c=c, alpha=alpha, beta=beta, options=MatmulOptions(codegen=enforce_codegen))
    except NotImplementedError as e:
        # Note: BSC is not supported for dispatching
        pytest.skip(f"Unable to perform matmul: {str(e)}")

    if use_ust:
        a = a.to_package()
        b = b.to_package()
        result = result.to_package()

    check_meta_data(a, b, result, framework, operand_placement, None, dtype, (m, n), (n, k), (m, k))
    compare_results(result, reference, dtype)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize("operand_placement", [OperandPlacement.host, OperandPlacement.device])
def test_memory_limit(operand_placement):
    run_matmul(
        size=(32, 32, 32),
        framework=Framework.torch,
        operand_placement=operand_placement,
        options=MatmulOptions(memory_limit=0.9),
    )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_logger():
    from io import StringIO

    log_stream = StringIO()
    logger = logging.Logger("test_logger", level=logging.DEBUG)
    logger.addHandler(logging.StreamHandler(log_stream))
    options = MatmulOptions(logger=logger)
    run_options_check(options)
    assert len(log_stream.getvalue()) > 0


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_allocator():
    from nvmath.memory import _MEMORY_MANAGER

    allocator = _MEMORY_MANAGER["torch"](0, logging.getLogger())
    run_options_check(MatmulOptions(allocator=allocator))


@pytest.mark.skipif(Framework.cupy not in Framework.enabled(), reason="Cupy is required for this test")
@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_different_allocator():
    from nvmath.memory import _MEMORY_MANAGER

    allocator = _MEMORY_MANAGER["cupy"](0, logging.getLogger())
    run_options_check(MatmulOptions(allocator=allocator))


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_custom_allocator():
    from nvmath.memory import _MEMORY_MANAGER

    class MockAllocator(_MEMORY_MANAGER["torch"]):
        def __init__(self, device_id, logger):
            super().__init__(device_id, logger)
            self.counter = 0

        def memalloc(self, size, *args, **kwargs):
            self.counter += 1
            return super().memalloc(size, *args, **kwargs)

        def memalloc_async(self, size, stream, *args, **kwargs):
            self.counter += 1
            return super().memalloc_async(size, stream, *args, **kwargs)

    allocator = MockAllocator(0, logging.getLogger())
    run_options_check(MatmulOptions(allocator=allocator))
    assert allocator.counter > 0


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "dtypes",
        "enforce_codegen",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("dtypes", dtypes),
            Param("enforce_codegen", enforce_codegen),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for dtypes in [
            (DType.float32, DType.float64),
            (DType.float64, DType.float32),
            (DType.float32, DType.float32),
            (DType.float64, DType.float64),
            (DType.complex64, DType.complex128),
            (DType.complex128, DType.complex64),
            (DType.complex64, DType.complex64),
            (DType.complex128, DType.complex128),
        ]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_compute_type(framework, operand_placement, sparse_array_type, use_ust, dtypes, enforce_codegen):
    _DTYPE_TO_WIDTH = {
        DType.float32: 32,
        DType.float64: 64,
        DType.complex64: 32,
        DType.complex128: 64,
    }

    use_ust = use_ust.value
    dtypes = dtypes.value
    enforce_codegen = enforce_codegen.value
    input_dtype, compute_dtype = dtypes
    is_lower_precision = _DTYPE_TO_WIDTH[input_dtype] > _DTYPE_TO_WIDTH[compute_dtype]
    compute_dtype = dtype_to_compute_type(compute_dtype)

    with allow_cusparse_unsupported(enabled=not enforce_codegen):
        run_matmul(
            size=(32, 32, 32),
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            dtype=input_dtype,
            use_ust=use_ust,
            options=MatmulOptions(compute_type=compute_dtype, codegen=enforce_codegen),
            rtol=1e-4 if is_lower_precision else None,
        )


# ==========================
# Negative tests
# ==========================


@pytest.mark.parametrize(
    (
        "placement",
        "framework0",
        "use_ust0",
        "framework1",
        "use_ust1",
        "framework2",
        "use_ust2",
    ),
    [
        (
            placement,
            framework0,
            Param("use_ust0", use_ust0),
            framework1,
            Param("use_ust1", use_ust1),
            framework2,
            Param("use_ust2", use_ust2),
        )
        for placement in [OperandPlacement.host, OperandPlacement.device]
        for framework0 in Framework.enabled()
        if framework0 in sparse_supporting_frameworks and placement in framework2operand_placement[framework0]
        for framework1 in Framework.enabled()
        if framework1 in sparse_supporting_frameworks and placement in framework2operand_placement[framework1]
        for framework2 in Framework.enabled()
        if framework2 in sparse_supporting_frameworks and placement in framework2operand_placement[framework2]
        for use_ust0 in [False, True]
        for use_ust1 in [False, True]
        for use_ust2 in [False, True]
        if not ((use_ust0 == use_ust1 == use_ust2) and (framework0 == framework1 == framework2))
    ],
    ids=idfn,
)
def test_mixing_frameworks(placement, framework0, use_ust0, framework1, use_ust1, framework2, use_ust2):
    use_ust0 = use_ust0.value
    use_ust1 = use_ust1.value
    use_ust2 = use_ust2.value

    a = create_random_sparse_matrix(
        framework0, placement, SparseArrayType.CSR, 32, 32, 0.5, DType.float32, seed=RNG_SEED, index_dtype=DType.int32
    )
    b = create_random_dense_matrix(framework1, placement, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(framework2, placement, 32, 32, DType.float32, seed=RNG_SEED)

    if use_ust0:
        a = ust.Tensor.from_package(a)
    if use_ust1:
        b = ust.Tensor.from_package(b)
    if use_ust2:
        c = ust.Tensor.from_package(c)

    with pytest.raises(TypeError):
        matmul(a, b, c=c)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_c_cannot_promote():
    a = create_random_sparse_matrix(
        Framework.torch,
        OperandPlacement.device,
        SparseArrayType.CSR,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
        batch_dims=(2, 2),
    )
    b = create_random_dense_matrix(
        Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED, batch_dims=(2, 2)
    )
    c = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)

    with pytest.raises(ValueError, match="must match"):
        matmul(a, b, c=c)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    ("sparse_array_type", "basic_placement", "mismatch_operand", "use_ust"),
    [
        (
            sparse_array_type,
            basic_placement,
            mismatch_operand,
            Param("use_ust", use_ust),
        )
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[Framework.torch]
        for basic_placement in [OperandPlacement.host, OperandPlacement.device]
        for mismatch_operand in ["a", "b", "c"]
        for use_ust in [False, True]
    ],
    ids=idfn,
)
def test_device_mismatch(sparse_array_type, basic_placement, mismatch_operand, use_ust):
    use_ust = use_ust.value
    mismatch_placement = OperandPlacement.device if basic_placement == OperandPlacement.host else OperandPlacement.host
    a_placement = mismatch_placement if mismatch_operand == "a" else basic_placement
    b_placement = mismatch_placement if mismatch_operand == "b" else basic_placement
    c_placement = mismatch_placement if mismatch_operand == "c" else basic_placement

    a = create_random_sparse_matrix(
        Framework.torch, a_placement, sparse_array_type, 32, 32, 0.5, DType.float32, seed=RNG_SEED, index_dtype=DType.int32
    )
    b = create_random_dense_matrix(Framework.torch, b_placement, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, c_placement, 32, 32, DType.float32, seed=RNG_SEED)

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)

    with pytest.raises(ValueError, match="not on the same device"):
        matmul(a, b, c=c)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "base_dtype",
        "mismatch_dtype",
        "mismatch_operand",
        "use_ust",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            base_dtype,
            mismatch_dtype,
            mismatch_operand,
            Param("use_ust", use_ust),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for base_dtype in supported_dtypes
        if base_dtype in framework2dtype[framework]
        for mismatch_dtype in supported_dtypes
        if mismatch_dtype in framework2dtype[framework] and mismatch_dtype != base_dtype
        for mismatch_operand in ["a", "b", "c"]
        for use_ust in [False, True]
    ],
    ids=idfn,
)
def test_dtype_mismatch(framework, operand_placement, sparse_array_type, base_dtype, mismatch_dtype, mismatch_operand, use_ust):
    use_ust = use_ust.value
    a_dtype = mismatch_dtype if mismatch_operand == "a" else base_dtype
    b_dtype = mismatch_dtype if mismatch_operand == "b" else base_dtype
    c_dtype = mismatch_dtype if mismatch_operand == "c" else base_dtype

    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, 32, 32, 0.5, a_dtype, seed=RNG_SEED, index_dtype=DType.int32
    )
    b = create_random_dense_matrix(framework, operand_placement, 32, 32, b_dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, 32, 32, c_dtype, seed=RNG_SEED)

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)

    with pytest.raises(NotImplementedError, match="not supported"):
        matmul(a, b, c=c)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "shape_mismatch",
    ),
    [
        (framework, operand_placement, sparse_array_type, Param("use_ust", use_ust), shape_mismatch)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for use_ust in [False, True]
        for shape_mismatch in ["a_cols", "b_rows", "c_rows", "c_cols"]
    ],
    ids=idfn,
)
def test_shape_mismatch(framework, operand_placement, sparse_array_type, use_ust, shape_mismatch):
    use_ust = use_ust.value
    m, n, k = (3, 5, 7)

    a_rows, a_cols = (m, n) if shape_mismatch != "a_cols" else (m, n + 1)
    b_rows, b_cols = (n, k) if shape_mismatch != "b_rows" else (n + 1, k)
    c_rows, c_cols = (m, k + 1) if shape_mismatch == "c_cols" else (m + 1, k) if shape_mismatch == "c_rows" else (m, k)

    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        a_rows,
        a_cols,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(framework, operand_placement, b_rows, b_cols, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, c_rows, c_cols, DType.float32, seed=RNG_SEED)

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)

    with pytest.raises((ValueError, NotImplementedError)):
        matmul(a, b, c=c)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
    ],
    ids=idfn,
)
def test_b_must_be_dense(framework, operand_placement, sparse_array_type):
    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    c = create_random_dense_matrix(framework, operand_placement, 32, 32, DType.float32, seed=RNG_SEED)

    with pytest.raises(ValueError, match="must be dense"):
        matmul(a, b, c=c)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
    ],
    ids=idfn,
)
def test_c_must_be_dense(framework, operand_placement, sparse_array_type):
    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(framework, operand_placement, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )

    with pytest.raises(ValueError, match="must be dense"):
        matmul(a, b, c=c)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
    ),
    [
        (framework, operand_placement)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
    ],
    ids=idfn,
)
def test_a_must_be_sparse(framework, operand_placement):
    a = create_random_dense_matrix(framework, operand_placement, 32, 32, DType.float32, seed=RNG_SEED)
    b = create_random_dense_matrix(framework, operand_placement, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, 32, 32, DType.float32, seed=RNG_SEED)

    with pytest.raises(TypeError, match="must be an N-D sparse"):
        matmul(a, b, c=c)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_qualifiers_wrong_dtype():
    a = create_random_sparse_matrix(
        Framework.torch,
        OperandPlacement.device,
        SparseArrayType.CSR,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)

    qualifiers = np.zeros((3,), dtype=np.int8)

    with allow_cusparse_unsupported(enabled=SparseArrayType.BSR), pytest.raises(ValueError, match="matrix_qualifiers_dtype"):
        matmul(a, b, c=c, qualifiers=qualifiers)


def test_invalid_allocator():
    with pytest.raises(TypeError):
        MatmulOptions(allocator="Hello, I'm a real allocator!")


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_invalid_options_type():
    with pytest.raises(TypeError, match="MatmulOptions"):
        run_options_check(-7)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_unsupported_dtype():
    a = create_random_sparse_matrix(
        Framework.torch,
        OperandPlacement.device,
        SparseArrayType.CSR,
        32,
        32,
        0.5,
        DType.int32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.int32, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.int32, seed=RNG_SEED)

    with pytest.raises(TypeError, match="is not supported"):
        matmul(a, b, c=c)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "dtypes",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            dtypes,
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for dtypes in [
            (DType.float16, DType.complex64),
            (DType.float32, DType.complex64),
            (DType.float64, DType.complex64),
            (DType.float16, DType.complex128),
            (DType.float32, DType.complex128),
            (DType.float64, DType.complex128),
            (DType.complex64, DType.float32),
            (DType.complex64, DType.float64),
            (DType.complex128, DType.float32),
            (DType.complex128, DType.float64),
        ]
        if dtypes[0] in framework2dtype[framework]
    ],
    ids=idfn,
)
def test_invalid_compute_type(framework, operand_placement, sparse_array_type, dtypes):
    input_dtype, compute_dtype = dtypes
    compute_dtype = dtype_to_compute_type(compute_dtype)

    with pytest.raises(ValueError, match="compute type"):
        run_matmul(
            size=(32, 32, 32),
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            dtype=input_dtype,
            options=MatmulOptions(compute_type=compute_dtype),
        )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize("operand_placement", framework2operand_placement[Framework.torch])
@pytest.mark.parametrize("bad_operand", [Param("bad_operand", name) for name in ("b", "c")], ids=idfn)
def test_layout_mismatch(operand_placement, bad_operand):
    import torch

    bad_operand = bad_operand.value
    m, n, k = 29, 31, 37
    dtype = DType.float32

    a = create_random_sparse_matrix(
        Framework.torch,
        operand_placement,
        SparseArrayType.CSR,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(Framework.torch, operand_placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, operand_placement, m, k, dtype, seed=RNG_SEED)

    if bad_operand == "b":
        b = torch.empty_strided((n, k), (64, 2), dtype=torch.float32, device=b.device)
    else:
        c = torch.empty_strided((m, k), (64, 2), dtype=torch.float32, device=c.device)

    with pytest.raises(ValueError, match="Unsupported layout"):
        matmul(a, b, c=c)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize("operand_placement", framework2operand_placement[Framework.torch])
@pytest.mark.parametrize("bad_operand", [Param("bad_operand", name) for name in ("b", "c")], ids=idfn)
def test_dense_invalid_batch_layout_not_c_order(operand_placement, bad_operand):
    bad_operand = bad_operand.value
    m, n, k = 11, 13, 17
    dtype = DType.float32
    batch_dims = (2, 3)

    a = create_random_sparse_matrix(
        Framework.torch,
        operand_placement,
        SparseArrayType.CSR,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=DType.int32,
        batch_dims=batch_dims,
    )
    b_ref = create_random_dense_matrix(Framework.torch, operand_placement, n, k, dtype, seed=RNG_SEED, batch_dims=batch_dims)
    c_ref = create_random_dense_matrix(Framework.torch, operand_placement, m, k, dtype, seed=RNG_SEED, batch_dims=batch_dims)

    b = _torch_wrong_batch_strides(batch_dims, n, k, b_ref) if bad_operand == "b" else b_ref
    c = _torch_wrong_batch_strides(batch_dims, m, k, c_ref) if bad_operand == "c" else c_ref

    with pytest.raises(ValueError, match="C-order"):
        matmul(a, b, c=c)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for batching")
@pytest.mark.parametrize(
    (
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "batch_mismatch",
        "batch_dims",
        "mismatch_batch_dims",
    ),
    [
        (
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            batch_mismatch,
            Param("batch_dims", batch_dims),
            Param("mismatch_batch_dims", mismatch_batch_dims),
        )
        for operand_placement in framework2operand_placement[Framework.torch]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[Framework.torch]
        for use_ust in [False, True]
        for batch_mismatch in ["a", "b", "c"]
        for batch_dims in [(2, 2, 2), (2, 2)]
        for mismatch_batch_dims in [(3, 3, 3), (3, 3, 3, 3)]
    ],
    ids=idfn,
)
def test_batch_size_mismatch(
    operand_placement,
    sparse_array_type,
    use_ust,
    batch_mismatch,
    batch_dims,
    mismatch_batch_dims,
):
    use_ust = use_ust.value
    batch_dims = batch_dims.value
    mismatch_batch_dims = mismatch_batch_dims.value
    m, n, k = 32, 32, 32

    a = create_random_sparse_matrix(
        Framework.torch,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
        batch_dims=mismatch_batch_dims if batch_mismatch == "a" else batch_dims,
    )
    b = create_random_dense_matrix(
        Framework.torch,
        operand_placement,
        n,
        k,
        DType.float32,
        seed=RNG_SEED,
        batch_dims=mismatch_batch_dims if batch_mismatch == "b" else batch_dims,
    )
    c = create_random_dense_matrix(
        Framework.torch,
        operand_placement,
        m,
        k,
        DType.float32,
        seed=RNG_SEED,
        batch_dims=mismatch_batch_dims if batch_mismatch == "c" else batch_dims,
    )

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)

    with pytest.raises(ValueError, match="must match"):
        try:
            matmul(a, b, c=c)
        except NotImplementedError as e:
            pytest.skip(f"Unable to perform matmul: {e}")


# ==========================
# Guard tests
# ==========================


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_c_required():
    a = create_random_sparse_matrix(
        Framework.torch,
        OperandPlacement.device,
        SparseArrayType.CSR,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)

    with pytest.raises(NotImplementedError, match="C is currently required"):
        matmul(a, b)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_default_alpha_beta():
    a = create_random_sparse_matrix(
        Framework.torch,
        OperandPlacement.device,
        SparseArrayType.CSR,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)
    c1 = c.clone()

    assert_tensors_equal(matmul(a, b, c=c), matmul(a, b, c=c1, alpha=1.0, beta=1.0))


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize("flags", [(1, 1), (1, 0), (0, 1)])
def test_qualifiers_not_supported_for_c(flags):
    qualifiers = np.zeros((3,), dtype=matmul_matrix_qualifiers_dtype)
    qualifiers[2]["is_transpose"] = flags[0]
    qualifiers[2]["is_conjugate"] = flags[1]

    with pytest.raises((ValueError, NotImplementedError), match="not supported for operand C"):
        run_matmul(size=(32, 32, 32), qualifiers=qualifiers)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize("index_dtype", [DType.int8, DType.int16])
def test_unsupported_index_dtype(index_dtype):
    a = create_random_sparse_matrix(
        Framework.torch,
        OperandPlacement.device,
        SparseArrayType.CSR,
        32,
        32,
        0.5,
        DType.float32,
        seed=RNG_SEED,
        index_dtype=index_dtype,
    )
    b = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, OperandPlacement.device, 32, 32, DType.float32, seed=RNG_SEED)

    with pytest.raises(TypeError, match="is not supported"):
        matmul(a, b, c=c)
