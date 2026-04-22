# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys

import numpy as np
import pytest

import nvmath.sparse.ust as ust
from nvmath.internal import utils
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.sparse import ExecutionCUDA, Matmul, MatmulOptions, matmul_matrix_qualifiers_dtype

from ....helpers import check_freed_after
from ...utils.common_axes import (
    JIT_AVAILABLE,
    DType,
    Framework,
    OperandPlacement,
    Param,
    SparseArrayType,
    copy_array,
    framework2dtype,
    framework2index_dtype,
    framework2operand_placement,
    framework2sparse_array_type_support,
    sparse_supporting_frameworks,
)
from ...utils.utils import (
    DEVICE_CC,
    allow_cusparse_unsupported,
    assert_snapshot_equal,
    get_custom_stream,
    idfn,
    is_known_linker_error,
    multi_gpu_only,
    transform_sparse_array,
    use_stream_or_dummy_ctx,
    ust_snapshot,
)
from .utils.data_helpers import (
    calculate_reference,
    check_meta_data,
    compare_results,
    create_random_dense_matrix,
    create_random_sparse_matrix,
)
from .utils.support_matrix import (
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

_COMPUTE_CAPABILITIES = [
    70,
    72,
    75,
    80,
    86,
    89,
    90,
    100,
    101,
    103,
    120,
    121,
]


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


def intercept_default_allocations(monkeypatch):
    allocations = {"raw": 0, "cupy": 0, "torch": 0, "cuda": 0}
    from nvmath.memory import _MEMORY_MANAGER

    def get_memalloc_wrapper(manager, alloc_key):
        actual = manager.memalloc_async

        def wrapper(self, *args, **kwargs):
            allocations[alloc_key] += 1
            return actual(self, *args, **kwargs)

        return wrapper

    managers = [
        (_MEMORY_MANAGER["_raw"], "raw"),
        (_MEMORY_MANAGER["cuda"], "cuda"),
    ]

    if Framework.cupy in Framework.enabled():
        managers += [
            (_MEMORY_MANAGER["cupy"], "cupy"),
        ]

    if Framework.torch in Framework.enabled():
        managers += [
            (_MEMORY_MANAGER["torch"], "torch"),
        ]

    for manager, alloc_key in managers:
        monkeypatch.setattr(manager, "memalloc_async", get_memalloc_wrapper(manager, alloc_key))

    return allocations


def framework_to_alloc_key(framework: Framework):
    match framework:
        case Framework.numpy | Framework.scipy:
            return "raw"
        case Framework.cupy | Framework.cupyx:
            return "cupy"
        case Framework.torch:
            return "torch"
        case _:
            raise ValueError(f"Unknown framework: {framework}")


def run_matmul_on_operands(
    a,
    b,
    c,
    reference,
    dtype,
    alpha=1.0,
    beta=1.0,
    qualifiers=None,
    options=None,
    execution=None,
    use_ust=False,
    stream=None,
    cc=None,
    rtol=None,
):
    with Matmul(
        a, b, c=c, options=options, alpha=alpha, beta=beta, qualifiers=qualifiers, stream=stream, execution=execution
    ) as mm:
        _matmul_plan(mm, stream=stream, compute_capability=cc)
        result = mm.execute(stream=stream)
        compare_results(result.to_package(stream=stream) if use_ust else result, reference, dtype, rtol=rtol)


def _check_result(
    a_operand,
    b_operand,
    result,
    ref,
    use_ust,
    framework,
    operand_placement,
    device_id,
    dtype,
    a_shape,
    b_shape,
    c_shape,
    rtol=None,
    stream=None,
):
    """
    Check meta-data and compare values against reference.
    Do it in a helper, to avoid polluting local scope
    of run_matmul.
    """
    if use_ust:
        result = result.to_package(stream=stream)
        a_operand = a_operand.to_package(stream=stream)
        b_operand = b_operand.to_package(stream=stream)

    check_meta_data(
        a_operand,
        b_operand,
        result,
        framework,
        operand_placement,
        device_id,
        dtype,
        a_shape,
        b_shape,
        c_shape,
    )
    compare_results(result, ref, dtype, rtol=rtol)


def run_matmul(
    size,
    qualifiers=None,
    alpha=1.0,
    alpha1=1.0,
    beta=1.0,
    beta1=1.0,
    batch_dims=(),
    cc=None,
    framework=Framework.torch,
    operand_placement=OperandPlacement.device,
    sparse_array_type=SparseArrayType.CSR,
    dtype=DType.float32,
    index_dtype=DType.int32,
    options=None,
    execution=None,
    use_ust=False,
    density=0.5,
    # reset params
    reset_a=False,
    reset_b=False,
    reset_c=True,
    reset_alpha=False,
    reset_beta=False,
    use_unchecked=False,
    release_operands=False,
    # Execution params
    device_id=None,
    stream=None,
    rtol=None,
):
    assert not release_operands or (release_operands and reset_a and reset_b and reset_c)

    # Prepare data
    if isinstance(batch_dims, tuple) and len(batch_dims) == 3 and all(isinstance(b, tuple) for b in batch_dims):
        batch_a, batch_b, batch_c = batch_dims
    else:
        batch_a = batch_b = batch_c = batch_dims
    m, n, k = size

    # Generate random data
    with use_stream_or_dummy_ctx(framework, stream):
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
            batch_dims=batch_a,
            device_id=device_id,
        )
        b = create_random_dense_matrix(
            framework, operand_placement, n, k, dtype, seed=RNG_SEED, batch_dims=batch_b, device_id=device_id
        )
        c = create_random_dense_matrix(
            framework, operand_placement, m, k, dtype, seed=RNG_SEED, batch_dims=batch_c, device_id=device_id
        )
        c_backup = copy_array(c)
        c_backup_1 = copy_array(c)

        # Avoid generating random sparse matrix to not change layout
        a1 = transform_sparse_array(a, lambda x: 3.7 * x + 1.1)
        b1 = create_random_dense_matrix(
            framework, operand_placement, n, k, dtype, seed=RNG_SEED + 1, batch_dims=batch_b, device_id=device_id
        )
        c1 = create_random_dense_matrix(
            framework, operand_placement, m, k, dtype, seed=RNG_SEED + 1, batch_dims=batch_c, device_id=device_id
        )
        c1_backup = copy_array(c1)

        # Compute reference results
        reference = calculate_reference(a, b, c, dtype, alpha=alpha, beta=beta, qualifiers=qualifiers, device_id=device_id)
        reference1 = calculate_reference(
            a1 if reset_a else a,
            b1 if reset_b else b,
            c1 if reset_c else c,
            dtype,
            alpha=(alpha1 if reset_alpha else alpha),
            beta=(beta1 if reset_beta else beta),
            qualifiers=qualifiers,
            device_id=device_id,
        )

    # Convert to UST operands
    try:
        a_operand = ust.Tensor.from_package(a, stream=stream) if use_ust else a
        b_operand = ust.Tensor.from_package(b, stream=stream) if use_ust else b
        c_operand = ust.Tensor.from_package(c, stream=stream) if use_ust else c
        a_snap = ust_snapshot(a_operand) if use_ust else None
        b_snap = ust_snapshot(b_operand) if use_ust else None
        c_snap = ust_snapshot(c_operand) if use_ust else None
        a1_operand = ust.Tensor.from_package(a1, stream=stream) if use_ust else a1
        b1_operand = ust.Tensor.from_package(b1, stream=stream) if use_ust else b1
        c1_operand = ust.Tensor.from_package(c1, stream=stream) if use_ust else c1
        a1_snap = ust_snapshot(a1_operand) if use_ust else None
        b1_snap = ust_snapshot(b1_operand) if use_ust else None
        c1_snap = ust_snapshot(c1_operand) if use_ust else None
        rc_a1_operand = sys.getrefcount(a1_operand)
        rc_b1_operand = sys.getrefcount(b1_operand)
        rc_c1_operand = sys.getrefcount(c1_operand)
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    try:
        with Matmul(
            a_operand,
            b_operand,
            c=c_operand,
            options=options,
            alpha=alpha,
            beta=beta,
            qualifiers=qualifiers,
            stream=stream,
            execution=execution,
        ) as mm:
            # First execution
            _matmul_plan(mm, stream=stream, compute_capability=cc)
            result = mm.execute(stream=stream)

            # Verify first run
            _check_result(
                a_operand,
                b_operand,
                result,
                reference,
                use_ust,
                framework,
                operand_placement,
                device_id,
                dtype,
                (*batch_a, m, n),
                (*batch_b, n, k),
                (*batch_c, m, k),
                rtol=rtol,
                stream=stream,
            )
            del result

            # Release operands
            if release_operands:
                mm.release_operands()

                #  Verify old operands works after release
                run_matmul_on_operands(
                    a_operand,
                    b_operand,
                    ust.Tensor.from_package(c_backup, stream=stream) if use_ust else c_backup,
                    reference,
                    dtype,
                    alpha=alpha,
                    beta=beta,
                    qualifiers=qualifiers,
                    options=options,
                    use_ust=use_ust,
                    stream=stream,
                    execution=execution,
                    rtol=rtol,
                )

            if use_ust:
                assert_snapshot_equal(ust_snapshot(a_operand), a_snap)
                assert_snapshot_equal(ust_snapshot(b_operand), b_snap)
                assert_snapshot_equal(ust_snapshot(c_operand), c_snap)

            if release_operands:
                # note that, here, x_operand is either the same object
                # as x (use_ust=False) or a UST wrapper holding a reference to x.
                # So either way we need to release this extra reference before
                # correctly asserting sole ownership of x.
                a_operand = b_operand = c_operand = None
                with check_freed_after(a, "a should have sole ownership after release"):
                    del a
                with check_freed_after(b, "b should have sole ownership after release"):
                    del b
                with check_freed_after(c, "c should have sole ownership after release"):
                    del c
            # Reset operands
            if reset_a or reset_b or reset_c or reset_alpha or reset_beta:
                if use_unchecked:
                    mm.reset_operands_unchecked(
                        a=a1_operand if reset_a else None,
                        b=b1_operand if reset_b else None,
                        c=c1_operand if reset_c else None,
                        alpha=alpha1 if reset_alpha else None,
                        beta=beta1 if reset_beta else None,
                        stream=stream,
                    )
                else:
                    mm.reset_operands(
                        a=a1_operand if reset_a else None,
                        b=b1_operand if reset_b else None,
                        c=c1_operand if reset_c else None,
                        alpha=alpha1 if reset_alpha else None,
                        beta=beta1 if reset_beta else None,
                        stream=stream,
                    )

                #  Verify old operands works after reset
                if not release_operands:
                    run_matmul_on_operands(
                        a_operand,
                        b_operand,
                        ust.Tensor.from_package(c_backup_1, stream=stream) if use_ust else c_backup_1,
                        reference,
                        dtype,
                        alpha=alpha,
                        beta=beta,
                        qualifiers=qualifiers,
                        options=options,
                        use_ust=use_ust,
                        stream=stream,
                        execution=execution,
                        rtol=rtol,
                    )

            # Check references: note that, here, x_operand is either the same object
            # as x (use_ust=False) or a UST wrapper holding a reference to x.
            # So either way we need to release this extra reference before
            # correctly asserting sole ownership of x.
            if not release_operands and reset_a:
                a_operand = None
                with check_freed_after(a, "a should have sole ownership after reset"):
                    del a

            if not release_operands and reset_b:
                b_operand = None
                with check_freed_after(b, "b should have sole ownership after reset"):
                    del b

            if not release_operands and reset_c:
                c_operand = None
                with check_freed_after(c, "c should have sole ownership after reset"):
                    del c

            # Second execution
            result1 = mm.execute(stream=stream)

            # Verify second run
            _check_result(
                a1_operand if reset_a else a_operand,
                b1_operand if reset_b else b_operand,
                result1,
                reference1,
                use_ust,
                framework,
                operand_placement,
                device_id,
                dtype,
                (*batch_a, m, n),
                (*batch_b, n, k),
                (*batch_c, m, k),
                rtol=rtol,
                stream=stream,
            )
            del result1

            # Release operands
            if release_operands:
                mm.release_operands()

                run_matmul_on_operands(
                    a1_operand if reset_a else a_operand,
                    b1_operand if reset_b else b_operand,
                    ust.Tensor.from_package(c1_backup, stream=stream) if use_ust else c1_backup,
                    reference1,
                    dtype,
                    alpha=alpha1 if reset_alpha else alpha,
                    beta=beta1 if reset_beta else beta,
                    qualifiers=qualifiers,
                    options=options,
                    use_ust=use_ust,
                    stream=stream,
                    execution=execution,
                    rtol=rtol,
                )

                if use_ust:
                    assert_snapshot_equal(ust_snapshot(a1_operand if reset_a else a_operand), a1_snap)
                    assert_snapshot_equal(ust_snapshot(b1_operand if reset_b else b_operand), b1_snap)
                    assert_snapshot_equal(ust_snapshot(c1_operand if reset_c else c_operand), c1_snap)

                # note that, here, x1_operand is either the same object
                # as x1 (use_ust=False) or a UST wrapper holding a reference to x1.
                # So either way we need to release this extra reference before
                # correctly asserting sole ownership of x1.
                a1_operand = b1_operand = c1_operand = None
                with check_freed_after(a1, "a1 should have sole ownership after release"):
                    del a1
                with check_freed_after(b1, "b1 should have sole ownership after release"):
                    del b1
                with check_freed_after(c1, "c1 should have sole ownership after release"):
                    del c1

    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")

    if not release_operands:
        if use_ust:
            assert_snapshot_equal(ust_snapshot(a1_operand), a1_snap)
            assert_snapshot_equal(ust_snapshot(b1_operand), b1_snap)
            assert_snapshot_equal(ust_snapshot(c1_operand), c1_snap)

            # if use_ust, x1_operand and x are different objects
            # let's veryfiy their count, for x we do the check below
            assert sys.getrefcount(a1_operand) == rc_a1_operand, (
                f"Reference count of a1_operand is {sys.getrefcount(a1_operand)}"
            )
            assert sys.getrefcount(b1_operand) == rc_b1_operand, (
                f"Reference count of b1_operand is {sys.getrefcount(b1_operand)}"
            )
            assert sys.getrefcount(c1_operand) == rc_c1_operand, (
                f"Reference count of c1_operand is {sys.getrefcount(c1_operand)}"
            )
        # note that, here, x1_operand is either the same object
        # as x1 (use_ust=False) or a UST wrapper holding a reference to x1.
        # So either way we need to release this extra reference before
        # correctly asserting sole ownership of x1.
        a1_operand = b1_operand = c1_operand = None
        with check_freed_after(a1, "a1 should have sole ownership"):
            del a1
        with check_freed_after(b1, "b1 should have sole ownership"):
            del b1
        with check_freed_after(c1, "c1 should have sole ownership"):
            del c1

        # here, x_operand is either the same object
        # as x (use_ust=False) or a UST wrapper holding a reference to x.
        # So either way we need to release this extra reference before
        # correctly asserting sole ownership of x.
        if not reset_a:
            a_operand = None
            with check_freed_after(a, "a should have sole ownership"):
                del a

        if not reset_b:
            b_operand = None
            with check_freed_after(b, "b should have sole ownership"):
                del b

        if not reset_c:
            c_operand = None
            with check_freed_after(c, "c should have sole ownership"):
                del c

    # Making sure operands ownership is tested in all branches
    assert a_operand is None
    assert b_operand is None
    assert c_operand is None
    assert a1_operand is None
    assert b1_operand is None
    assert c1_operand is None


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
        "use_ust",
        "enforce_codegen",
        "reset_a",
        "reset_b",
        "reset_c",
        "batch_dims",
        "use_unchecked",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            index_dtype,
            dtype,
            Param("size", size),
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("reset_a", reset_a),
            Param("reset_b", reset_b),
            Param("reset_c", reset_c),
            Param("batch_dims", batch_dims),
            Param("use_unchecked", use_unchecked),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [True, False]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for index_dtype in (supported_codegen_index_dtypes if use_ust else supported_index_dtypes)
        if index_dtype in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for size in [(13, 17, 19), (32, 32, 32)]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for reset_a in [False, True]
        for reset_b in [False, True]
        for reset_c in [True]  # C is overwritten by the result
        if reset_a or reset_b or reset_c
        for batch_dims in [(2, 2), ()]
        if batch_dims == () or framework == Framework.torch
        for use_unchecked in [False, True]
    ],
    ids=idfn,
)
def test_reset(
    framework,
    operand_placement,
    sparse_array_type,
    index_dtype,
    dtype,
    size,
    use_ust,
    enforce_codegen,
    reset_a,
    reset_b,
    reset_c,
    batch_dims,
    use_unchecked,
):
    if sparse_array_type == SparseArrayType.COO and framework == Framework.torch and index_dtype != DType.int64:
        pytest.skip("Torch COO currently only supports int64 index dtype")

    size = size.value
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    reset_a = reset_a.value
    reset_b = reset_b.value
    reset_c = reset_c.value
    batch_dims = batch_dims.value
    use_unchecked = use_unchecked.value

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            index_dtype=index_dtype,
            dtype=dtype,
            size=size,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen),
            reset_a=reset_a,
            reset_b=reset_b,
            reset_c=reset_c,
            batch_dims=batch_dims,
            use_unchecked=use_unchecked,
        )


@pytest.mark.parametrize(
    ("framework", "operand_placement", "sparse_array_type", "use_ust", "enforce_codegen", "execute"),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("execute", execute),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for execute in [False, True]
    ],
    ids=idfn,
)
def test_reference_count_context_manager(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    execute,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    execute = execute.value
    m, n, k = (13, 17, 19)
    density = 0.5
    dtype = DType.float32
    index_dtype = DType.int32

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

    try:
        operand_a = ust.Tensor.from_package(a) if use_ust else a
        operand_b = ust.Tensor.from_package(b) if use_ust else b
        operand_c = ust.Tensor.from_package(c) if use_ust else c
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    rc_operand_a = sys.getrefcount(operand_a)
    rc_operand_b = sys.getrefcount(operand_b)
    rc_operand_c = sys.getrefcount(operand_c)

    try:
        with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
            matmul = Matmul(operand_a, operand_b, c=operand_c, options=MatmulOptions(codegen=enforce_codegen))

            with matmul:
                if execute:
                    _matmul_plan(matmul)
                    matmul.execute()
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")

    if use_ust:
        # If we use UST, the locals ``operand_x`` and ``x`` are different objects.
        # We want to make sure we release both.
        assert sys.getrefcount(operand_a) == rc_operand_a, f"Reference count of operand_a is {sys.getrefcount(operand_a)}"
        assert sys.getrefcount(operand_b) == rc_operand_b, f"Reference count of operand_b is {sys.getrefcount(operand_b)}"
        assert sys.getrefcount(operand_c) == rc_operand_c, f"Reference count of operand_c is {sys.getrefcount(operand_c)}"

    # ensure the operands have sole ownership after context manager exit:
    # note that, here, operand_x is either the same object
    # as x (use_ust=False) or a UST wrapper holding a reference to x.
    # So either way we need to release this extra reference before
    # correctly asserting sole ownership of x.
    operand_a = operand_b = operand_c = None

    with check_freed_after(a, "a should have sole ownership after context manager exit"):
        del a
    with check_freed_after(b, "b should have sole ownership after context manager exit"):
        del b
    with check_freed_after(c, "c should have sole ownership after context manager exit"):
        del c


@pytest.mark.parametrize(
    ("framework", "operand_placement", "sparse_array_type", "use_ust", "enforce_codegen", "execute"),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("execute", execute),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [False, True]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for execute in [False, True]
    ],
    ids=idfn,
)
def test_release_operands(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    execute,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    execute = execute.value
    m, n, k = (13, 17, 19)
    density = 0.5
    dtype = DType.float32
    index_dtype = DType.int32

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

    try:
        operand_a = ust.Tensor.from_package(a) if use_ust else a
        operand_b = ust.Tensor.from_package(b) if use_ust else b
        operand_c = ust.Tensor.from_package(c) if use_ust else c
        rc_operand_a = sys.getrefcount(operand_a)
        rc_operand_b = sys.getrefcount(operand_b)
        rc_operand_c = sys.getrefcount(operand_c)

        # Operands should be "read-only", i.e. we don't modify
        # ptrs, allocations, shapes etc of the passed operands.
        # For external packages this basically comes for free,
        # but for nvmath.sparse.ust.Tensor, let's make sure
        # we don't accidentally modify them.
        snap_a = ust_snapshot(operand_a) if use_ust else None
        snap_b = ust_snapshot(operand_b) if use_ust else None
        snap_c = ust_snapshot(operand_c) if use_ust else None
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    matmul = None
    try:
        with (
            allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR),
            Matmul(operand_a, operand_b, c=operand_c, options=MatmulOptions(codegen=enforce_codegen)) as matmul,
        ):
            if execute:
                _matmul_plan(matmul)
                matmul.execute()

            matmul.release_operands()

            if use_ust:
                # If we use UST, the locals ``operand_x`` and ``x`` are different objects.
                # We want to make sure we release both.
                assert sys.getrefcount(operand_a) == rc_operand_a
                assert sys.getrefcount(operand_b) == rc_operand_b
                assert sys.getrefcount(operand_c) == rc_operand_c
                assert_snapshot_equal(ust_snapshot(operand_a), snap_a)
                assert_snapshot_equal(ust_snapshot(operand_b), snap_b)
                assert_snapshot_equal(ust_snapshot(operand_c), snap_c)

            # ensure the operands have sole ownership after context manager exit:
            # note that, here, operand_x is either the same object
            # as x (use_ust=False) or a UST wrapper holding a reference to x.
            # So either way we need to release this extra reference before
            # correctly asserting sole ownership of x.
            operand_a = operand_b = operand_c = None
            with check_freed_after(a, "a should have sole ownership after release_operands"):
                del a
            with check_freed_after(b, "b should have sole ownership after release_operands"):
                del b
            with check_freed_after(c, "c should have sole ownership after release_operands"):
                del c

    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.skipif(not JIT_AVAILABLE, reason="Jitting is required for this test")
@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "reset_a",
        "reset_b",
        "reset_c",
        "use_unchecked",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("reset_a", reset_a),
            Param("reset_b", reset_b),
            Param("reset_c", reset_c),
            Param("use_unchecked", use_unchecked),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [True, False]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for reset_a in [False, True]
        for reset_b in [False, True]
        for reset_c in [True]  # C is overwritten by the result
        if reset_a or reset_b or reset_c
        for use_unchecked in [False, True]
    ],
    ids=idfn,
)
def test_reset_preserves_qualifiers(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    reset_a,
    reset_b,
    reset_c,
    use_unchecked,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    reset_a = reset_a.value
    reset_b = reset_b.value
    reset_c = reset_c.value
    use_unchecked = use_unchecked.value

    qualifiers = np.zeros((3,), dtype=matmul_matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = True
    qualifiers[1]["is_transpose"] = True
    qualifiers[0]["is_conjugate"] = True
    qualifiers[1]["is_conjugate"] = True

    with allow_cusparse_unsupported(enabled=not enforce_codegen):
        run_matmul(
            size=(32, 32, 32),
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            options=MatmulOptions(codegen=enforce_codegen),
            qualifiers=qualifiers,
            dtype=DType.complex64,
            index_dtype=DType.int32,
            use_ust=use_ust,
            use_unchecked=use_unchecked,
            reset_a=reset_a,
            reset_b=reset_b,
            reset_c=reset_c,
        )


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "index_dtype",
        "dtype",
        "size",
        "use_ust",
        "enforce_codegen",
        "use_unchecked",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            index_dtype,
            dtype,
            Param("size", size),
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("use_unchecked", use_unchecked),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [True, False]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for index_dtype in (supported_codegen_index_dtypes if use_ust else supported_index_dtypes)
        if index_dtype in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for size in [(13, 17, 19), (32, 32, 32)]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for use_unchecked in [False, True]
    ],
    ids=idfn,
)
def test_reset_after_release(
    framework,
    operand_placement,
    sparse_array_type,
    index_dtype,
    dtype,
    size,
    use_ust,
    enforce_codegen,
    use_unchecked,
):
    if sparse_array_type == SparseArrayType.COO and framework == Framework.torch and index_dtype != DType.int64:
        pytest.skip("Torch COO currently only supports int64 index dtype")

    size = size.value
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    use_unchecked = use_unchecked.value

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=size,
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            index_dtype=index_dtype,
            dtype=dtype,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen),
            release_operands=True,
            reset_a=True,
            reset_b=True,
            reset_c=True,
            reset_alpha=True,
            reset_beta=True,
            use_unchecked=use_unchecked,
        )


@multi_gpu_only
@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "index_dtype",
        "dtype",
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
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for index_dtype in supported_index_dtypes
        if index_dtype in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for use_ust in [False, True]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
    ],
    ids=idfn,
)
def test_non_default_device(
    framework,
    operand_placement,
    sparse_array_type,
    index_dtype,
    dtype,
    use_ust,
    enforce_codegen,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    device_id = 1

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=(13, 17, 19),
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            index_dtype=index_dtype,
            dtype=dtype,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen),
            execution=ExecutionCUDA(device_id=device_id),
            device_id=device_id if operand_placement == OperandPlacement.device else "cpu",
        )


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "use_unchecked",
        "reset_a",
        "reset_b",
        "reset_c",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("use_unchecked", use_unchecked),
            Param("reset_a", reset_a),
            Param("reset_b", reset_b),
            Param("reset_c", reset_c),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for use_ust in [False, True]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for use_unchecked in [False, True]
        for reset_a in [False, True]
        for reset_b in [False, True]
        for reset_c in [True]
    ],
    ids=idfn,
)
def test_non_default_stream(
    framework, operand_placement, sparse_array_type, use_ust, enforce_codegen, use_unchecked, reset_a, reset_b, reset_c
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    use_unchecked = use_unchecked.value
    reset_a = reset_a.value
    reset_b = reset_b.value
    reset_c = reset_c.value

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=(13, 17, 19),
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen, blocking=True),
            stream=get_custom_stream(framework),
            use_unchecked=use_unchecked,
            reset_a=reset_a,
            reset_b=reset_b,
            reset_c=reset_c,
        )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    (
        "use_ust",
        "enforce_codegen",
        "use_unchecked",
        "reset_alpha",
        "reset_beta",
    ),
    [
        (
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("use_unchecked", use_unchecked),
            Param("reset_alpha", reset_alpha),
            Param("reset_beta", reset_beta),
        )
        for use_ust in [True, False]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for use_unchecked in [False, True]
        for reset_alpha in [True, False]
        for reset_beta in [True, False]
        if reset_alpha or reset_beta
    ],
    ids=idfn,
)
def test_reset_alpha_beta(use_ust, enforce_codegen, use_unchecked, reset_alpha, reset_beta):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    use_unchecked = use_unchecked.value
    reset_alpha = reset_alpha.value
    reset_beta = reset_beta.value
    if enforce_codegen and use_unchecked:
        pytest.skip("Alpha and beta reset is not supported for code generation")

    run_matmul(
        size=(13, 17, 19),
        alpha=0.7,
        alpha1=-3.7,
        beta=3.3,
        beta1=4.4,
        use_ust=use_ust,
        options=MatmulOptions(codegen=enforce_codegen),
        use_unchecked=use_unchecked,
        reset_alpha=reset_alpha,
        reset_beta=reset_beta,
    )


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "use_ust",
        "enforce_codegen",
        "reset_a",
        "reset_b",
        "reset_c",
        "use_unchecked",
        "release_operands",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("reset_a", reset_a),
            Param("reset_b", reset_b),
            Param("reset_c", reset_c),
            Param("use_unchecked", use_unchecked),
            Param("release_operands", release_operands),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [True, False]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for reset_a in [False, True]
        for reset_b in [False, True]
        for reset_c in [True]  # C is overwritten by the result
        if reset_a or reset_b or reset_c
        for use_unchecked in [False, True]
        for release_operands in [False, True]
        if not release_operands or (reset_a and reset_b and reset_c)
    ],
    ids=idfn,
)
def test_reset_release_same_operands(
    framework,
    operand_placement,
    sparse_array_type,
    use_ust,
    enforce_codegen,
    reset_a,
    reset_b,
    reset_c,
    use_unchecked,
    release_operands,
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    reset_a = reset_a.value
    reset_b = reset_b.value
    reset_c = reset_c.value
    use_unchecked = use_unchecked.value
    release_operands = release_operands.value

    m, n, k = (13, 17, 19)
    density = 0.5
    dtype = DType.float32
    index_dtype = DType.int32

    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, m, n, density, dtype, RNG_SEED, index_dtype=index_dtype
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, RNG_SEED)
    c_backup = copy_array(c)

    reference = calculate_reference(a, b, c, dtype)

    try:
        if use_ust:
            a = ust.Tensor.from_package(a)
            b = ust.Tensor.from_package(b)
            c = ust.Tensor.from_package(c)
            c_backup = ust.Tensor.from_package(c_backup)
    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    try:
        with (
            allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR),
            Matmul(a, b, c=c, options=MatmulOptions(codegen=enforce_codegen)) as mm,
        ):
            _matmul_plan(mm)
            result = mm.execute()

            check_meta_data(
                a.to_package() if use_ust else a,
                b.to_package() if use_ust else b,
                result.to_package() if use_ust else result,
                framework,
                operand_placement,
                None,
                dtype,
                (m, n),
                (n, k),
                (m, k),
            )
            compare_results(result.to_package() if use_ust else result, reference, dtype)

            if release_operands:
                mm.release_operands()

            if use_unchecked:
                mm.reset_operands_unchecked(
                    a=a if reset_a else None,
                    b=b if reset_b else None,
                    c=c_backup if reset_c else None,
                )
            else:
                mm.reset_operands(
                    a=a if reset_a else None,
                    b=b if reset_b else None,
                    c=c_backup if reset_c else None,
                )

            result = mm.execute()
            check_meta_data(
                a.to_package() if use_ust else a,
                b.to_package() if use_ust else b,
                result.to_package() if use_ust else result,
                framework,
                operand_placement,
                None,
                dtype,
                (m, n),
                (n, k),
                (m, k),
            )
            compare_results(result.to_package() if use_ust else result, reference, dtype)
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "release_workspace",
        "use_custom_stream",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            Param("release_workspace", release_workspace),
            Param("use_custom_stream", use_custom_stream),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for release_workspace in [False, True]
        for use_custom_stream in [False, True]
    ],
    ids=idfn,
)
def test_execute_release_workspace(
    monkeypatch, framework, operand_placement, sparse_array_type, release_workspace, use_custom_stream
):
    release_workspace = release_workspace.value
    use_custom_stream = use_custom_stream.value

    m, n, k = (13, 17, 19)
    density = 0.5
    dtype = DType.float32
    index_dtype = DType.int32

    stream = get_custom_stream(framework) if use_custom_stream else None

    with use_stream_or_dummy_ctx(framework, stream):
        a = create_random_sparse_matrix(
            framework, operand_placement, sparse_array_type, m, n, density, dtype, RNG_SEED, index_dtype=index_dtype
        )
        b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, RNG_SEED)
        c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, RNG_SEED)
        c_backup = copy_array(c)

        reference = calculate_reference(a, b, c, dtype)

    allocations = intercept_default_allocations(monkeypatch)
    expected_key = framework_to_alloc_key(framework)

    try:
        with (
            allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR),
            Matmul(a, b, c=c, options=MatmulOptions(blocking=True), stream=stream) as mm,
        ):
            _matmul_plan(mm, stream=stream)
            result = mm.execute(stream=stream, release_workspace=release_workspace)

            if mm.workspace_size != 0:
                assert allocations[expected_key] == 1, f"{allocations}, {expected_key}"

            check_meta_data(a, b, result, framework, operand_placement, None, dtype, (m, n), (n, k), (m, k))
            compare_results(result, reference, dtype)

            mm.reset_operands(c=c_backup)

            result = mm.execute(stream=stream, release_workspace=release_workspace)

            if mm.workspace_size != 0:
                assert allocations[expected_key] == (1 + release_workspace), f"{allocations}, {expected_key}"

            check_meta_data(a, b, result, framework, operand_placement, None, dtype, (m, n), (n, k), (m, k))
            compare_results(result, reference, dtype)

    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "index_dtype",
        "dtype",
        "use_ust",
        "enforce_codegen",
        "cc",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            index_dtype,
            dtype,
            Param("use_ust", use_ust),
            Param("enforce_codegen", enforce_codegen),
            Param("cc", cc),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for use_ust in [True, False]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for index_dtype in (supported_codegen_index_dtypes if use_ust else supported_index_dtypes)
        if index_dtype in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for enforce_codegen in [False, True]
        if use_ust or not enforce_codegen
        for cc in _COMPUTE_CAPABILITIES
        if cc <= DEVICE_CC
    ],
    ids=idfn,
)
def test_plan_custom_compute_capability(
    framework, operand_placement, sparse_array_type, index_dtype, dtype, use_ust, enforce_codegen, cc
):
    use_ust = use_ust.value
    enforce_codegen = enforce_codegen.value
    cc = cc.value

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        run_matmul(
            size=(13, 17, 19),
            framework=framework,
            operand_placement=operand_placement,
            sparse_array_type=sparse_array_type,
            use_ust=use_ust,
            options=MatmulOptions(codegen=enforce_codegen),
            cc=cc,
            index_dtype=index_dtype,
            dtype=dtype,
        )


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    ("operand_placement", "options_blocking", "expect_sync_on_execute"),
    [
        (OperandPlacement.host, Param("options_blocking", "auto"), Param("expect_sync_on_execute", True)),
        (OperandPlacement.host, Param("options_blocking", True), Param("expect_sync_on_execute", True)),
        (OperandPlacement.device, Param("options_blocking", "auto"), Param("expect_sync_on_execute", False)),
        (OperandPlacement.device, Param("options_blocking", True), Param("expect_sync_on_execute", True)),
    ],
    ids=idfn,
)
def test_blocking(monkeypatch, operand_placement, options_blocking, expect_sync_on_execute):
    options_blocking = options_blocking.value
    expect_sync_on_execute = expect_sync_on_execute.value

    sync_count = 0
    original_cuda_call_ctx = utils.cuda_call_ctx

    @contextlib.contextmanager
    def counting_cuda_call_ctx(stream_holder, blocking=True, timing=True):
        nonlocal sync_count
        with original_cuda_call_ctx(stream_holder, blocking, timing) as (end, time):
            yield end, time
        if blocking:
            sync_count += 1

    monkeypatch.setattr(utils, "cuda_call_ctx", counting_cuda_call_ctx)

    run_matmul(
        size=(32, 32, 32),
        framework=Framework.torch,
        operand_placement=operand_placement,
        options=MatmulOptions(blocking=options_blocking),
    )

    # Accounts for additional mm reruns done inside of run_matmul
    # (basic run + reset rerun on different matmul object + final rerun)
    expected_syncs = 3 if expect_sync_on_execute else 0
    assert sync_count == expected_syncs, (
        f"operand_placement={operand_placement}, options.blocking={options_blocking}: "
        f"expected {expected_syncs} sync(s) from execute(), got {sync_count}"
    )


# ==========================
# Negative tests
# ==========================


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
def test_reset_operands_all_none():
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

    with pytest.raises(ValueError, match=r"Use release_operands\(\) to release all operands."), Matmul(a, b, c=c) as mm:
        _matmul_plan(mm)
        mm.reset_operands()


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    ("plan_before_release", "execute_after_release", "plan_after_release"),
    [
        (
            Param("plan_before_release", plan_before_release),
            Param("execute_after_release", execute_after_release),
            Param("plan_after_release", plan_after_release),
        )
        for plan_before_release in [False, True]
        for execute_after_release in [False, True]
        for plan_after_release in [False, True]
        if execute_after_release or plan_after_release
    ],
    ids=idfn,
)
def test_fail_after_release_operands(plan_before_release, execute_after_release, plan_after_release):
    plan_before_release = plan_before_release.value
    execute_after_release = execute_after_release.value
    plan_after_release = plan_after_release.value

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

    with pytest.raises(RuntimeError), Matmul(a, b, c=c) as mm:
        if plan_before_release:
            _matmul_plan(mm)

        mm.release_operands()

        if plan_after_release:
            _matmul_plan(mm)

        if execute_after_release:
            mm.execute()


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
def test_reset_invalid_dtype(
    framework,
    operand_placement,
    sparse_array_type,
    base_dtype,
    mismatch_dtype,
    mismatch_operand,
    use_ust,
):
    use_ust = use_ust.value
    if sparse_array_type == SparseArrayType.COO and framework == Framework.torch:
        pytest.skip("Torch COO index dtype handled in test_reset_invalid_index_dtype")

    m, n, k = (32, 32, 32)
    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        base_dtype,
        seed=RNG_SEED,
        index_dtype=DType.int32,
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, base_dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, m, k, base_dtype, seed=RNG_SEED)

    a1 = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        mismatch_dtype,
        seed=RNG_SEED + 1,
        index_dtype=DType.int32,
    )
    b1 = create_random_dense_matrix(framework, operand_placement, n, k, mismatch_dtype, seed=RNG_SEED + 1)
    c1 = create_random_dense_matrix(framework, operand_placement, m, k, mismatch_dtype, seed=RNG_SEED + 1)

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)
        a1 = ust.Tensor.from_package(a1)
        b1 = ust.Tensor.from_package(b1)
        c1 = ust.Tensor.from_package(c1)

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        try:
            with pytest.raises(TypeError, match="doesn't match the original one"), Matmul(a, b, c=c) as mm:
                _matmul_plan(mm)
                mm.reset_operands(
                    a=a1 if mismatch_operand == "a" else None,
                    b=b1 if mismatch_operand == "b" else None,
                    c=c1 if mismatch_operand == "c" else None,
                )
        except NotImplementedError as e:
            pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "base_index_dtype",
        "mismatch_index_dtype",
        "use_ust",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            base_index_dtype,
            mismatch_index_dtype,
            Param("use_ust", use_ust),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for base_index_dtype in supported_index_dtypes
        if base_index_dtype in framework2index_dtype[framework]
        for mismatch_index_dtype in supported_index_dtypes
        if mismatch_index_dtype != base_index_dtype and mismatch_index_dtype in framework2index_dtype[framework]
        for use_ust in [False, True]
    ],
    ids=idfn,
)
def test_reset_invalid_index_dtype(
    framework,
    operand_placement,
    sparse_array_type,
    base_index_dtype,
    mismatch_index_dtype,
    use_ust,
):
    use_ust = use_ust.value
    if sparse_array_type == SparseArrayType.COO and framework == Framework.torch:
        pytest.skip("Torch COO currently only supports int64 index dtype")
    m, n, k = (32, 32, 32)
    dtype = DType.float32

    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=base_index_dtype,
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED)

    a1 = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED + 1,
        index_dtype=mismatch_index_dtype,
    )

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)
        a1 = ust.Tensor.from_package(a1)

    try:
        with pytest.raises(TypeError, match="doesn't match the original one"), Matmul(a, b, c=c) as mm:
            _matmul_plan(mm)
            mm.reset_operands(a=a1)
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for batching")
@pytest.mark.parametrize(
    ("sparse_array_type", "original_batch_dims", "mismatch_batch_dims", "use_ust", "mismatch_operand"),
    [
        (sparse_array_type, original_batch_dims, mismatch_batch_dims, Param("use_ust", use_ust), mismatch_operand)
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[Framework.torch]
        for original_batch_dims in [(2, 2, 2), (2, 2), (2,)]
        for mismatch_batch_dims in [(3, 3), (3,), (3, 3, 3)]
        for use_ust in [False, True]
        for mismatch_operand in ["a", "b", "c"]
    ],
    ids=idfn,
)
def test_reset_invalid_batch_size(sparse_array_type, original_batch_dims, mismatch_batch_dims, use_ust, mismatch_operand):
    use_ust = use_ust.value
    m, n, k = (32, 32, 32)
    framework = Framework.torch
    operand_placement = OperandPlacement.device
    dtype = DType.float32
    index_dtype = DType.int32

    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
        batch_dims=original_batch_dims,
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED, batch_dims=original_batch_dims)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED, batch_dims=original_batch_dims)

    a1 = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED + 1,
        index_dtype=index_dtype,
        batch_dims=mismatch_batch_dims,
    )
    b1 = create_random_dense_matrix(
        framework, operand_placement, n, k, dtype, seed=RNG_SEED + 1, batch_dims=mismatch_batch_dims
    )
    c1 = create_random_dense_matrix(
        framework, operand_placement, m, k, dtype, seed=RNG_SEED + 1, batch_dims=mismatch_batch_dims
    )

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)
        a1 = ust.Tensor.from_package(a1)
        b1 = ust.Tensor.from_package(b1)
        c1 = ust.Tensor.from_package(c1)

    try:
        with pytest.raises(TypeError, match="doesn't match the original one"), Matmul(a, b, c=c) as mm:
            _matmul_plan(mm)
            mm.reset_operands(
                a=a1 if mismatch_operand == "a" else None,
                b=b1 if mismatch_operand == "b" else None,
                c=c1 if mismatch_operand == "c" else None,
            )
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.parametrize(
    (
        "framework",
        "operand_placement",
        "sparse_array_type",
        "shape_mismatch",
        "use_ust",
    ),
    [
        (
            framework,
            operand_placement,
            sparse_array_type,
            shape_mismatch,
            Param("use_ust", use_ust),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[framework]
        for shape_mismatch in ["a_cols", "b_rows", "c_rows", "c_cols"]
        for use_ust in [False, True]
    ],
    ids=idfn,
)
def test_reset_invalid_shape(
    framework,
    operand_placement,
    sparse_array_type,
    shape_mismatch,
    use_ust,
):
    use_ust = use_ust.value
    m, n, k = (13, 17, 19)
    dtype = DType.float32
    index_dtype = DType.int32

    a = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
    )
    b = create_random_dense_matrix(framework, operand_placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(framework, operand_placement, m, k, dtype, seed=RNG_SEED)

    a1 = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        m,
        n + 1,
        0.5,
        dtype,
        seed=RNG_SEED + 1,
        index_dtype=index_dtype,
    )
    b1 = create_random_dense_matrix(framework, operand_placement, n + 1, k, dtype, seed=RNG_SEED + 1)
    c1 = create_random_dense_matrix(
        framework,
        operand_placement,
        m + 1 if shape_mismatch == "c_rows" else m,
        k + 1 if shape_mismatch == "c_cols" else k,
        dtype,
        seed=RNG_SEED + 1,
    )

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)
        a1 = ust.Tensor.from_package(a1)
        b1 = ust.Tensor.from_package(b1)
        c1 = ust.Tensor.from_package(c1)

    try:
        with (
            allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR),
            pytest.raises(TypeError, match="doesn't match the original one"),
            Matmul(a, b, c=c) as mm,
        ):
            _matmul_plan(mm)
            mm.reset_operands(
                a=a1 if shape_mismatch == "a_cols" else None,
                b=b1 if shape_mismatch == "b_rows" else None,
                c=c1 if shape_mismatch == "c_rows" or shape_mismatch == "c_cols" else None,
            )
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize(
    ("sparse_array_type", "basic_placement", "mismatch_operand", "use_ust"),
    [
        (sparse_array_type, basic_placement, mismatch_operand, Param("use_ust", use_ust))
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[Framework.torch]
        for basic_placement in [OperandPlacement.host, OperandPlacement.device]
        for mismatch_operand in ["a", "b", "c"]
        for use_ust in [False, True]
    ],
    ids=idfn,
)
def test_reset_invalid_placement(sparse_array_type, basic_placement, mismatch_operand, use_ust):
    use_ust = use_ust.value
    mismatch_placement = OperandPlacement.device if basic_placement == OperandPlacement.host else OperandPlacement.host

    m, n, k = (32, 32, 32)
    dtype = DType.float32
    index_dtype = DType.int32

    a = create_random_sparse_matrix(
        Framework.torch,
        basic_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
    )
    b = create_random_dense_matrix(Framework.torch, basic_placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(Framework.torch, basic_placement, m, k, dtype, seed=RNG_SEED)

    a1 = create_random_sparse_matrix(
        Framework.torch,
        mismatch_placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
    )
    b1 = create_random_dense_matrix(Framework.torch, mismatch_placement, n, k, dtype, seed=RNG_SEED)
    c1 = create_random_dense_matrix(Framework.torch, mismatch_placement, m, k, dtype, seed=RNG_SEED)

    if use_ust:
        a = ust.Tensor.from_package(a)
        b = ust.Tensor.from_package(b)
        c = ust.Tensor.from_package(c)
        a1 = ust.Tensor.from_package(a1)
        b1 = ust.Tensor.from_package(b1)
        c1 = ust.Tensor.from_package(c1)

    try:
        with pytest.raises(TypeError, match="doesn't match the original one"), Matmul(a, b, c=c) as mm:
            _matmul_plan(mm)
            mm.reset_operands(
                a=a1 if mismatch_operand == "a" else None,
                b=b1 if mismatch_operand == "b" else None,
                c=c1 if mismatch_operand == "c" else None,
            )
    except NotImplementedError as e:
        pytest.skip(f"Unable to perform matmul: {e}")


@pytest.mark.parametrize(
    (
        "placement",
        "original_framework",
        "mismatch_framework",
        "sparse_array_type",
        "mismatch_operand",
        "use_ust",
        "mismatch_ust",
    ),
    [
        (
            placement,
            original_framework,
            mismatch_framework,
            sparse_array_type,
            mismatch_operand,
            Param("use_ust", use_ust),
            Param("mismatch_ust", mismatch_ust),
        )
        for mismatch_ust in [False, True]
        for placement in [OperandPlacement.host, OperandPlacement.device]
        for original_framework in Framework.enabled()
        if original_framework in sparse_supporting_frameworks
        for mismatch_framework in Framework.enabled()
        if mismatch_framework in sparse_supporting_frameworks
        and (mismatch_framework != original_framework or mismatch_ust)
        and placement in framework2operand_placement[original_framework]
        and placement in framework2operand_placement[mismatch_framework]
        for sparse_array_type in supported_formats
        if sparse_array_type in framework2sparse_array_type_support[original_framework]
        and sparse_array_type in framework2sparse_array_type_support[mismatch_framework]
        for mismatch_operand in ["a", "b", "c"]
        for use_ust in [False, True]
    ],
    ids=idfn,
)
def test_reset_invalid_package(
    placement, original_framework, mismatch_framework, sparse_array_type, mismatch_operand, use_ust, mismatch_ust
):
    use_ust = use_ust.value
    mismatch_ust = mismatch_ust.value
    m, n, k = (32, 32, 32)
    dtype = DType.float32
    index_dtype = DType.int32

    a = create_random_sparse_matrix(
        original_framework,
        placement,
        sparse_array_type,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED,
        index_dtype=index_dtype,
    )
    b = create_random_dense_matrix(original_framework, placement, n, k, dtype, seed=RNG_SEED)
    c = create_random_dense_matrix(original_framework, placement, m, k, dtype, seed=RNG_SEED)

    a1 = create_random_sparse_matrix(
        mismatch_framework,
        placement,
        SparseArrayType.CSR,
        m,
        n,
        0.5,
        dtype,
        seed=RNG_SEED + 1,
        index_dtype=index_dtype,
    )
    b1 = create_random_dense_matrix(mismatch_framework, placement, n, k, dtype, seed=RNG_SEED + 1)
    c1 = create_random_dense_matrix(mismatch_framework, placement, m, k, dtype, seed=RNG_SEED + 1)

    try:
        if use_ust:
            a = ust.Tensor.from_package(a)
            b = ust.Tensor.from_package(b)
            c = ust.Tensor.from_package(c)

        if use_ust != mismatch_ust:
            # we need ust either if we want to match
            # inputs that are ust or mismatch inputs
            # that aren't ust
            a1 = ust.Tensor.from_package(a1)
            b1 = ust.Tensor.from_package(b1)
            c1 = ust.Tensor.from_package(c1)

    except (TypeError, NotImplementedError) as e:
        pytest.skip(f"Unable to perform UST conversion: {str(e)}")

    with allow_cusparse_unsupported(enabled=sparse_array_type == SparseArrayType.BSR):
        try:
            with pytest.raises(TypeError, match="doesn't match the original one"), Matmul(a, b, c=c) as mm:
                _matmul_plan(mm)
                mm.reset_operands(
                    a=a1 if mismatch_operand == "a" else None,
                    b=b1 if mismatch_operand == "b" else None,
                    c=c1 if mismatch_operand == "c" else None,
                )
        except NotImplementedError as e:
            pytest.skip(f"Unable to perform matmul: {e}")


def _torch_column_major_last_two_dims(t):
    assert t.ndim >= 2
    return t.transpose(-2, -1).contiguous().transpose(-2, -1)


@pytest.mark.skipif(Framework.torch not in Framework.enabled(), reason="Torch is required for this test")
@pytest.mark.parametrize("operand_placement", framework2operand_placement[Framework.torch])
@pytest.mark.parametrize(
    "mismatch",
    [Param("mismatch", spec) for spec in ("b_row_to_col", "c_row_to_col", "b_col_to_row", "c_col_to_row")],
    ids=idfn,
)
def test_dense_layout_mismatch_after_reset(operand_placement, mismatch):
    mismatch = mismatch.value
    m, n, k = 19, 23, 29
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

    b_repl = create_random_dense_matrix(Framework.torch, operand_placement, n, k, dtype, seed=RNG_SEED + 1)
    c_repl = create_random_dense_matrix(Framework.torch, operand_placement, m, k, dtype, seed=RNG_SEED + 1)

    if mismatch == "b_row_to_col":
        b_repl = _torch_column_major_last_two_dims(b_repl)
    elif mismatch == "c_row_to_col":
        c_repl = _torch_column_major_last_two_dims(c_repl)
    elif mismatch == "b_col_to_row":
        b = _torch_column_major_last_two_dims(b)
    else:
        assert mismatch == "c_col_to_row"
        c = _torch_column_major_last_two_dims(c)

    with Matmul(a, b, c=c) as mm:
        _matmul_plan(mm)

        with pytest.raises(TypeError, match="strides"):
            mm.reset_operands(b=b_repl, c=c_repl)
