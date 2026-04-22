# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import numpy as np
import pytest

import nvmath.sparse
import nvmath.sparse.ust as ust
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.sparse import Matmul, MatmulOptions, compile_matmul_epilog, compile_matmul_prolog, matmul_matrix_qualifiers_dtype

from ...utils.common_axes import (
    JIT_AVAILABLE,
    DType,
    Framework,
    OperandPlacement,
    Param,
    SparseArrayType,
    framework2dtype,
    framework2operand_placement,
    framework2sparse_array_type_support,
    nc,
    numba,
    sparse_supporting_frameworks,
)
from ...utils.utils import (
    allow_cusparse_unsupported,
    copy_array,
    idfn,
    is_known_linker_error,
    to_dense_numpy,
    transform_sparse_array,
)
from .utils.data_helpers import (
    calculate_reference,
    check_meta_data,
    compare_results,
    create_random_dense_matrix,
    create_random_sparse_matrix,
)
from .utils.support_matrix import (
    supported_callback_dtypes,
    supported_formats,
)

if Framework.torch in Framework.enabled():
    maybe_register_package("torch")
if Framework.cupy in Framework.enabled():
    maybe_register_package("cupy")

pytestmark = pytest.mark.skipif(not JIT_AVAILABLE, reason="numba is required for callback matmul tests")

RNG_SEED = 92


def _matmul_plan(mm, /, **kwargs):
    try:
        mm.plan(**kwargs)
    except Exception as e:
        if is_known_linker_error(str(e)):
            pytest.skip("CUDA linker error: duplicate __half/__nv_bfloat16 symbol (CTK 12.x issue).")
        raise


# ==========================
# Helper functions
# ==========================

_TEST_TO_COMPILE_DTYPE = {
    DType.float32: "float32",
    DType.float64: "float64",
    DType.complex64: "complex64",
    DType.complex128: "complex128",
}

_TEST_TO_NUMBA_DTYPE = (
    {}
    if numba is None
    else {
        DType.float32: numba.float32,
        DType.float64: numba.float64,
        DType.complex64: numba.complex64,
        DType.complex128: numba.complex128,
    }
)


def run_matmul(
    size,
    qualifiers=None,
    alpha=None,
    beta=None,
    batch_dims=(),
    framework=Framework.torch,
    operand_placement=OperandPlacement.device,
    sparse_array_type=SparseArrayType.CSR,
    dtype=DType.float32,
    index_dtype=DType.int32,
    options=None,
    use_ust=False,
    density=0.5,
    prolog_a=None,
    prolog_b=None,
    prolog_c=None,
    epilog=None,
    semiring=None,
    named_format=None,
):
    assert named_format is None or use_ust, "named_format requires use_ust"
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
    a = a_ref
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED, batch_dims=batch_b)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED, batch_dims=batch_c)
    c_backup = copy_array(c)

    try:
        if use_ust:
            a = ust.Tensor.from_package(a_ref)
            b = ust.Tensor.from_package(b)
            c = ust.Tensor.from_package(c)
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    if named_format is not None:
        a = a.convert(tensor_format=named_format)

    prologs = {}
    if prolog_a is not None:
        prologs["a"] = prolog_a
    if prolog_b is not None:
        prologs["b"] = prolog_b
    if prolog_c is not None:
        prologs["c"] = prolog_c
    prologs = prologs or None

    try:
        with Matmul(a, b, c=c, options=options, alpha=alpha, beta=beta, qualifiers=qualifiers) as mm:
            _matmul_plan(mm, prologs=prologs, epilog=epilog, semiring=semiring)
            result = mm.execute()
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")

    if use_ust:
        a = a_ref if named_format else a.to_package()
        b = b.to_package()
        result = result.to_package()

    check_meta_data(
        a, b, result, framework, operand_placement, None, dtype, (*batch_a, m, n), (*batch_b, n, k), (*batch_c, m, k)
    )
    return result, a, b, c_backup


def example_prolog(x):
    return 2 * x + 3


def example_epilog(x):
    return x**2


@lru_cache
def get_example_prolog(dtype, operand, cc=None):
    compile_dtype = _TEST_TO_COMPILE_DTYPE[dtype]
    return compile_matmul_prolog(example_prolog, operand_label=operand, dtype=compile_dtype, compute_capability=cc)


@lru_cache
def get_example_epilog(dtype, cc=None):
    compile_dtype = _TEST_TO_COMPILE_DTYPE[dtype]
    return compile_matmul_epilog(example_epilog, dtype=compile_dtype, compute_capability=cc)


def example_prolog1(x):
    return -32 * x + 11


def example_epilog1(x):
    return x / 8


@lru_cache
def get_example_prolog1(dtype, operand, cc=None):
    compile_dtype = _TEST_TO_COMPILE_DTYPE[dtype]
    return compile_matmul_prolog(example_prolog1, operand_label=operand, dtype=compile_dtype, compute_capability=cc)


@lru_cache
def get_example_epilog1(dtype, cc=None):
    compile_dtype = _TEST_TO_COMPILE_DTYPE[dtype]
    return compile_matmul_epilog(example_epilog1, dtype=compile_dtype, compute_capability=cc)


@lru_cache
def get_tropical_semiring(dtype, cc=None):
    assert numba is not None and nc is not None
    from numba.cuda.np.numpy_support import farray

    compile_dtype = _TEST_TO_COMPILE_DTYPE[dtype]
    numba_dtype = _TEST_TO_NUMBA_DTYPE[dtype]

    def mul(a, b):
        return a + b

    def add_tropical(a, b):
        return max(a, b)

    def add_atomic_tropical(a, b):
        a_array = farray(a, (1,), dtype=numba_dtype)
        return nc.atomic.max(a_array, 0, b)

    return {
        "mul": nvmath.sparse.compile_matmul_mul(mul, dtype=compile_dtype, compute_capability=cc),
        "add": nvmath.sparse.compile_matmul_add(add_tropical, dtype=compile_dtype, compute_capability=cc),
        "atomic_add": nvmath.sparse.compile_matmul_atomic_add(add_atomic_tropical, dtype=compile_dtype, compute_capability=cc),
    }, {
        "mul": mul,
        "add": add_tropical,
    }


@lru_cache
def get_arctic_semiring(dtype, cc=None):
    assert numba is not None and nc is not None
    from numba.cuda.np.numpy_support import farray

    compile_dtype = _TEST_TO_COMPILE_DTYPE[dtype]
    numba_dtype = _TEST_TO_NUMBA_DTYPE[dtype]

    def mul(a, b):
        return a + b

    def add_arctic(a, b):
        return min(a, b)

    def add_atomic_arctic(a, b):
        a_array = farray(a, (1,), dtype=numba_dtype)
        return nc.atomic.min(a_array, 0, b)

    return {
        "mul": nvmath.sparse.compile_matmul_mul(mul, dtype=compile_dtype, compute_capability=cc),
        "add": nvmath.sparse.compile_matmul_add(add_arctic, dtype=compile_dtype, compute_capability=cc),
        "atomic_add": nvmath.sparse.compile_matmul_atomic_add(add_atomic_arctic, dtype=compile_dtype, compute_capability=cc),
    }, {
        "mul": mul,
        "add": add_arctic,
    }


def get_reference_result_prologs_epilog(
    a,
    b,
    c,
    dtype,
    use_prolog_a,
    use_prolog_b,
    use_prolog_c,
    use_epilog,
    prolog_a=example_prolog,
    prolog_b=example_prolog,
    prolog_c=example_prolog,
    epilog=example_epilog,
    qualifiers=None,
    device_id=None,
):
    a = transform_sparse_array(a, prolog_a) if use_prolog_a else a
    b = prolog_b(b) if use_prolog_b else b
    c = prolog_c(c) if use_prolog_c else c

    reference = calculate_reference(a, b, c, dtype, qualifiers=qualifiers, device_id=device_id)
    return epilog(reference) if use_epilog else reference


def get_reference_result_semiring(
    a,
    b_np,
    c_np,
    use_prolog_a,
    use_prolog_b,
    use_prolog_c,
    use_epilog,
    semiring,
):
    a = transform_sparse_array(a, example_prolog) if use_prolog_a else a
    a_val = to_dense_numpy(a)
    b_val = np.asarray(example_prolog(b_np) if use_prolog_b else b_np)
    c_val = np.asarray(example_prolog(c_np) if use_prolog_c else c_np)

    add_op = semiring["add"]
    mul_op = semiring["mul"]

    assert a_val.shape[-1] == b_val.shape[-2]
    M, K = a_val.shape[-2], a_val.shape[-1]
    N = b_val.shape[-1]

    a_batch_shape = a_val.shape[:-2]
    b_batch_shape = b_val.shape[:-2]
    c_batch_shape = c_val.shape[:-2]

    batch_shape = ()
    if len(a_batch_shape) > 0:
        batch_shape = tuple(a_batch_shape)
    if len(b_batch_shape) > 0:
        b_bs = tuple(b_batch_shape)
        if len(batch_shape) == 0:
            batch_shape = b_bs
        elif b_bs != batch_shape:
            raise ValueError(f"Batch dimensions of A {batch_shape} and B {b_batch_shape} must match.")

    if (len(c_batch_shape) > 0 or len(batch_shape) > 0) and (tuple(c_batch_shape) != batch_shape):
        raise ValueError(f"Batch dimension of C {tuple(c_batch_shape)} must match other operands {batch_shape}.")

    target_batch = batch_shape if len(batch_shape) > 0 else ()
    a_bc = np.broadcast_to(a_val, target_batch + (M, K))
    b_bc = np.broadcast_to(b_val, target_batch + (K, N))
    c_bc = np.broadcast_to(c_val, target_batch + (M, N))

    # TODO(jlisowski): vectorize this loop somehow
    result = np.array(c_bc, copy=True)
    for batch_idx in np.ndindex(batch_shape):
        for row in range(M):
            for col in range(N):
                for offset in range(K):
                    a_elt = a_bc[batch_idx + (row, offset)]
                    b_elt = b_bc[batch_idx + (offset, col)]
                    if a_elt == 0 or b_elt == 0:
                        continue
                    result[batch_idx + (row, col)] = add_op(
                        result[batch_idx + (row, col)],
                        mul_op(a_elt, b_elt),
                    )

    return example_epilog(result) if use_epilog else result


# ==========================
# Positive tests
# ==========================


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "use_ust",
        "enforce_codegen",
        "prolog_a",
        "prolog_b",
        "prolog_c",
        "epilog",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            dtype,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("prolog_a", prolog_a),
            Param("prolog_b", prolog_b),
            Param("prolog_c", prolog_c),
            Param("epilog", epilog),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for dtype in supported_callback_dtypes
        if dtype in framework2dtype[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for prolog_a in [False, True]
        for prolog_b in [False, True]
        for prolog_c in [False, True]
        for epilog in [False, True]
        if prolog_a or prolog_b or epilog
    ],
    ids=idfn,
)
def test_prologs_epilogs_dtypes(
    framework,
    operand_placement,
    sparse_array_type,
    dtype,
    use_ust,
    enforce_codegen,
    prolog_a,
    prolog_b,
    prolog_c,
    epilog,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    prolog_a = prolog_a.value
    prolog_b = prolog_b.value
    prolog_c = prolog_c.value
    epilog = epilog.value

    result, a, b, c = run_matmul(
        (32, 32, 32),
        framework=framework,
        operand_placement=operand_placement,
        sparse_array_type=sparse_array_type,
        dtype=dtype,
        use_ust=use_ust,
        prolog_a=get_example_prolog(dtype, "a") if prolog_a else None,
        prolog_b=get_example_prolog(dtype, "b") if prolog_b else None,
        prolog_c=get_example_prolog(dtype, "c") if prolog_c else None,
        epilog=get_example_epilog(dtype) if epilog else None,
        options=MatmulOptions(codegen=enforce_codegen),
    )

    reference = get_reference_result_prologs_epilog(a, b, c, dtype, prolog_a, prolog_b, prolog_c, epilog)
    compare_results(result, reference, dtype)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    (
        "named_format",
        "dtype",
        "enforce_codegen",
        "prolog_c",
        "epilog",
    ),
    [
        (
            named_format,
            dtype,
            Param("enforce_codegen", enforce_codegen),
            Param("prolog_c", prolog_c),
            Param("epilog", epilog),
        )
        for named_format in (ust.NamedFormats.DIAI, ust.NamedFormats.DIAJ)
        for dtype in supported_callback_dtypes
        if dtype in framework2dtype[Framework.torch]
        for enforce_codegen in [False, True]
        for prolog_c in [False, True]
        for epilog in [False, True]
        if prolog_c or epilog
    ],
    ids=idfn,
)
def test_dia_prolog_c_epilog(named_format, dtype, enforce_codegen, prolog_c, epilog):
    enforce_codegen = enforce_codegen.value
    use_prolog_c = prolog_c.value
    use_epilog = epilog.value

    m = n = 32
    framework = Framework.torch
    operand_placement = OperandPlacement.device
    sparse_array_type = SparseArrayType.CSR

    a_ref = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b_vec = create_random_dense_matrix(framework, operand_placement, n, 1, dtype, seed=RNG_SEED).squeeze(-1).contiguous()
    c_vec = create_random_dense_matrix(framework, operand_placement, m, 1, dtype, seed=RNG_SEED).squeeze(-1).contiguous()
    c_backup = copy_array(c_vec)

    a = ust.Tensor.from_package(a_ref)
    b = ust.Tensor.from_package(b_vec)
    c = ust.Tensor.from_package(c_vec)
    assert b.tensor_format.name == "DenseVector"
    assert c.tensor_format.name == "DenseVector"
    a = a.convert(tensor_format=named_format)

    prolog_c_cb = get_example_prolog(dtype, "c") if use_prolog_c else None
    epilog_cb = get_example_epilog(dtype) if use_epilog else None
    prologs = {"c": prolog_c_cb} if prolog_c_cb is not None else None
    options = MatmulOptions(codegen=enforce_codegen)

    try:
        with Matmul(a, b, c=c, options=options) as mm:
            _matmul_plan(mm, prologs=prologs, epilog=epilog_cb)
            result_ust = mm.execute()
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")

    result = result_ust.to_package()
    a = a_ref
    b = b.to_package()
    check_meta_data(
        a,
        b,
        result,
        framework,
        operand_placement,
        None,
        dtype,
        (m, n),
        (n,),
        (m,),
    )

    reference = get_reference_result_prologs_epilog(a, b, c_backup, dtype, False, False, use_prolog_c, use_epilog)
    compare_results(result, reference, dtype)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    (
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "prolog_a",
        "prolog_b",
        "prolog_c",
        "epilog",
        "batch_dims",
        "broadcast",
    ),
    [
        (
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("prolog_a", prolog_a),
            Param("prolog_b", prolog_b),
            Param("prolog_c", prolog_c),
            Param("epilog", epilog),
            Param("batch_dims", batch_dims),
            Param("broadcast", broadcast),
        )
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[Framework.torch]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[Framework.torch]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for prolog_a in [False, True]
        for prolog_b in [False, True]
        for prolog_c in [False, True]
        for epilog in [False, True]
        if prolog_a or prolog_b or epilog
        for batch_dims in [(2, 2), ()]
        for broadcast in ["a", "b", ""]
    ],
    ids=idfn,
)
def test_prologs_epilogs_batched(
    operand_placement, sparse_array_type, use_ust, enforce_codegen, prolog_a, prolog_b, prolog_c, epilog, batch_dims, broadcast
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    prolog_a = prolog_a.value
    prolog_b = prolog_b.value
    prolog_c = prolog_c.value
    epilog = epilog.value
    dtype = DType.complex64
    index_dtype = DType.int32
    batch_dims = batch_dims.value

    result, a, b, c = run_matmul(
        (13, 17, 19),
        framework=Framework.torch,
        operand_placement=operand_placement,
        sparse_array_type=sparse_array_type,
        dtype=dtype,
        index_dtype=index_dtype,
        use_ust=use_ust,
        prolog_a=get_example_prolog(dtype, "a") if prolog_a else None,
        prolog_b=get_example_prolog(dtype, "b") if prolog_b else None,
        prolog_c=get_example_prolog(dtype, "c") if prolog_c else None,
        epilog=get_example_epilog(dtype) if epilog else None,
        options=MatmulOptions(codegen=enforce_codegen),
        batch_dims=(
            () if broadcast == "a" else batch_dims,
            () if broadcast == "b" else batch_dims,
            batch_dims,
        ),
    )

    reference = get_reference_result_prologs_epilog(a, b, c, dtype, prolog_a, prolog_b, prolog_c, epilog)
    compare_results(result, reference, dtype)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "prolog_a",
        "prolog_b",
        "prolog_c",
        "epilog",
        "a_qualifier",
        "b_qualifier",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("prolog_a", prolog_a),
            Param("prolog_b", prolog_b),
            Param("prolog_c", prolog_c),
            Param("epilog", epilog),
            a_qualifier,
            b_qualifier,
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for prolog_a in [False, True]
        for prolog_b in [False, True]
        for prolog_c in [False, True]
        for epilog in [False, True]
        if prolog_a or prolog_b or epilog
        for a_qualifier in ["", "trans", "conj_trans"]
        for b_qualifier in ["", "trans", "conj_trans"]
        if a_qualifier or b_qualifier
    ],
    ids=idfn,
)
def test_prologs_epilogs_qualifiers(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    prolog_a,
    prolog_b,
    prolog_c,
    epilog,
    a_qualifier,
    b_qualifier,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    prolog_a = prolog_a.value
    prolog_b = prolog_b.value
    prolog_c = prolog_c.value
    epilog = epilog.value
    dtype = DType.complex64
    index_dtype = DType.int32

    qualifiers = np.zeros((3,), dtype=matmul_matrix_qualifiers_dtype)
    if a_qualifier == "trans":
        qualifiers[0]["is_transpose"] = True
    elif a_qualifier == "conj_trans":
        qualifiers[0]["is_transpose"] = True
        qualifiers[0]["is_conjugate"] = True

    if b_qualifier == "trans":
        qualifiers[1]["is_transpose"] = True
    elif b_qualifier == "conj_trans":
        qualifiers[1]["is_transpose"] = True
        qualifiers[1]["is_conjugate"] = True

    result, a, b, c = run_matmul(
        (32, 32, 32),
        framework=framework,
        operand_placement=operand_placement,
        sparse_array_type=sparse_array_type,
        dtype=dtype,
        index_dtype=index_dtype,
        use_ust=use_ust,
        prolog_a=get_example_prolog(dtype, "a") if prolog_a else None,
        prolog_b=get_example_prolog(dtype, "b") if prolog_b else None,
        prolog_c=get_example_prolog(dtype, "c") if prolog_c else None,
        epilog=get_example_epilog(dtype) if epilog else None,
        options=MatmulOptions(codegen=enforce_codegen),
        qualifiers=qualifiers,
    )

    reference = get_reference_result_prologs_epilog(a, b, c, dtype, prolog_a, prolog_b, prolog_c, epilog, qualifiers=qualifiers)
    compare_results(result, reference, dtype)


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "prolog_a",
        "prolog_b",
        "prolog_c",
        "epilog",
        "replan_prolog_a",
        "replan_prolog_b",
        "replan_prolog_c",
        "replan_epilog",
        "reset_operands_a",
        "reset_operands_b",
        "reset_operands_c",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("prolog_a", prolog_a),
            Param("prolog_b", prolog_b),
            Param("prolog_c", prolog_c),
            Param("epilog", epilog),
            Param("replan_prolog_a", replan_prolog_a),
            Param("replan_prolog_b", replan_prolog_b),
            Param("replan_prolog_c", replan_prolog_c),
            Param("replan_epilog", replan_epilog),
            Param("reset_operands_a", reset_operands_a),
            Param("reset_operands_b", reset_operands_b),
            Param("reset_operands_c", reset_operands_c),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for prolog_a in [False, True]
        for prolog_b in [False, True]
        for prolog_c in [False, True]
        for epilog in [False, True]
        for replan_prolog_a in [False, True]
        for replan_prolog_b in [False, True]
        for replan_prolog_c in [False, True]
        for replan_epilog in [False, True]
        for reset_operands_a in [True]
        for reset_operands_b in [True]
        for reset_operands_c in [True]  # C is overwritten by the result
        if replan_prolog_a or replan_prolog_b or replan_epilog or reset_operands_a or reset_operands_b or reset_operands_c
    ],
    ids=idfn,
)
def test_replan_cbs(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    prolog_a,
    prolog_b,
    prolog_c,
    epilog,
    replan_prolog_a,
    replan_prolog_b,
    replan_prolog_c,
    replan_epilog,
    reset_operands_a,
    reset_operands_b,
    reset_operands_c,
):
    m, n, k = (13, 17, 19)
    density = 0.5
    dtype = DType.complex64
    index_dtype = DType.int32
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    prolog_a = prolog_a.value
    prolog_b = prolog_b.value
    prolog_c = prolog_c.value
    epilog = epilog.value
    replan_prolog_a = replan_prolog_a.value
    replan_prolog_b = replan_prolog_b.value
    replan_prolog_c = replan_prolog_c.value
    replan_epilog = replan_epilog.value
    reset_operands_a = reset_operands_a.value
    reset_operands_b = reset_operands_b.value
    reset_operands_c = reset_operands_c.value

    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        density,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED)
    reference = get_reference_result_prologs_epilog(a, b, c, dtype, prolog_a, prolog_b, prolog_c, epilog)

    a1 = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        density,
        dtype,
        seed=RNG_SEED + 1,
        index_dtype=index_dtype,
    )
    b1 = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED + 1)
    c1 = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED + 1)
    reference1 = get_reference_result_prologs_epilog(
        a1 if reset_operands_a else a,
        b1 if reset_operands_b else b,
        c1,
        dtype,
        prolog_a or replan_prolog_a,
        prolog_b or replan_prolog_b,
        prolog_c or replan_prolog_c,
        epilog or replan_epilog,
        epilog=example_epilog1 if replan_epilog else example_epilog,
        prolog_a=example_prolog1 if replan_prolog_a else example_prolog,
        prolog_b=example_prolog1 if replan_prolog_b else example_prolog,
        prolog_c=example_prolog1 if replan_prolog_c else example_prolog,
    )

    try:
        if use_ust:
            a = ust.Tensor.from_package(a)
            b = ust.Tensor.from_package(b)
            c = ust.Tensor.from_package(c)
            a1 = ust.Tensor.from_package(a1)
            b1 = ust.Tensor.from_package(b1)
            c1 = ust.Tensor.from_package(c1)
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    prologs = {}
    prologs1 = {}

    if prolog_a:
        prologs["a"] = prologs1["a"] = get_example_prolog(dtype, "a")
    if prolog_b:
        prologs["b"] = prologs1["b"] = get_example_prolog(dtype, "b")

    if replan_prolog_a:
        prologs1["a"] = get_example_prolog1(dtype, "a")
    if replan_prolog_b:
        prologs1["b"] = get_example_prolog1(dtype, "b")

    if prolog_c:
        prologs["c"] = prologs1["c"] = get_example_prolog(dtype, "c")
    if replan_prolog_c:
        prologs1["c"] = get_example_prolog1(dtype, "c")

    epilog = epilog1 = get_example_epilog(dtype) if epilog else None

    if replan_epilog:
        epilog1 = get_example_epilog1(dtype)

    prologs = prologs or None
    prologs1 = prologs1 or None

    try:
        with (
            allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR),
            Matmul(a, b, c=c, options=MatmulOptions(codegen=enforce_codegen)) as mm,
        ):
            _matmul_plan(mm, prologs=prologs, epilog=epilog)
            result = mm.execute()

            if use_ust:
                result = result.to_package()
                a = a.to_package()
                b = b.to_package()

            check_meta_data(a, b, result, framework, operand_placement, None, dtype, (m, n), (n, k), (m, k))
            compare_results(result, reference, dtype)

            if reset_operands_a or reset_operands_b or reset_operands_c:
                mm.reset_operands(
                    a=a1 if reset_operands_a else None,
                    b=b1 if reset_operands_b else None,
                    c=c1 if reset_operands_c else None,
                )

            _matmul_plan(mm, prologs=prologs1, epilog=epilog1)
            result1 = mm.execute()

            if use_ust:
                result1 = result1.to_package()
                a1 = a1.to_package()
                b1 = b1.to_package()

            check_meta_data(
                a1 if reset_operands_a else a,
                b1 if reset_operands_b else b,
                result1,
                framework,
                operand_placement,
                None,
                dtype,
                (m, n),
                (n, k),
                (m, k),
            )
            compare_results(result1, reference1, dtype)

    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "prolog_a",
        "prolog_b",
        "prolog_c",
        "epilog",
        "semiring",
        "batch_dims",
        "broadcast",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("prolog_a", prolog_a),
            Param("prolog_b", prolog_b),
            Param("prolog_c", prolog_c),
            Param("epilog", epilog),
            semiring,
            Param("batch_dims", batch_dims),
            Param("broadcast", broadcast),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for prolog_a in [False, True]
        for prolog_b in [False, True]
        for prolog_c in [False, True]
        for epilog in [False, True]
        for semiring in ["tropical", "arctic"]
        for batch_dims in [(2, 2), ()]
        if batch_dims == () or framework == Framework.torch
        for broadcast in ["a", "b", ""]
    ],
    ids=idfn,
)
def test_semiring(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    prolog_a,
    prolog_b,
    prolog_c,
    epilog,
    semiring,
    batch_dims,
    broadcast,
):
    assert semiring in ["tropical", "arctic"]

    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    prolog_a = prolog_a.value
    prolog_b = prolog_b.value
    prolog_c = prolog_c.value
    epilog = epilog.value
    size = (13, 17, 19)
    index_dtype = DType.int32
    batch_dims = batch_dims.value
    broadcast = broadcast.value
    dtype = DType.float32

    semiring_ops = get_tropical_semiring(dtype) if semiring == "tropical" else get_arctic_semiring(dtype)
    result, a, b, c = run_matmul(
        size,
        framework=framework,
        operand_placement=operand_placement,
        sparse_array_type=sparse_array_type,
        dtype=dtype,
        index_dtype=index_dtype,
        use_ust=use_ust,
        prolog_a=get_example_prolog(dtype, "a") if prolog_a else None,
        prolog_b=get_example_prolog(dtype, "b") if prolog_b else None,
        prolog_c=get_example_prolog(dtype, "c") if prolog_c else None,
        epilog=get_example_epilog(dtype) if epilog else None,
        options=MatmulOptions(codegen=enforce_codegen),
        semiring=semiring_ops[0],
        batch_dims=(
            () if broadcast == "a" else batch_dims,
            () if broadcast == "b" else batch_dims,
            batch_dims,
        ),
    )

    reference = get_reference_result_semiring(
        a, to_dense_numpy(b), to_dense_numpy(c), prolog_a, prolog_b, prolog_c, epilog, semiring_ops[1]
    )
    compare_results(result, reference, dtype)
