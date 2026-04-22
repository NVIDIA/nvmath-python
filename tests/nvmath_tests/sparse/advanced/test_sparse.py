# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import random
import sys
import typing

import pytest

import nvmath
from nvmath.sparse.advanced import DirectSolverMatrixType, DirectSolverMatrixViewType

try:
    import cupy as cp
    import cupyx.scipy.sparse as sp

    HAS_CUPY_SPARSE = True
except ImportError:
    HAS_CUPY_SPARSE = False


from nvmath_tests.helpers import get_framework_device_ctx

from ...helpers import check_freed_after
from ..utils.common_axes import (
    DType,
    ExecutionSpace,
    Framework,
    OperandPlacement,
    Param,
    RHSBatch,
    RHSMatrix,
    RHSVector,
    cp,
    device_id_from_array,
    framework2dtype,
    framework2index_dtype,
    framework2operand_placement,
    framework2tensor_framework,
    framework2value_dtype,
    framework_from_array,
    get_values_dtype_from_array,
    np,
    operand_placement_from_array,
    sparse_supporting_frameworks,
)
from ..utils.utils import get_custom_stream, idfn, multi_gpu_only, to_dense_numpy, use_stream_or_dummy_ctx
from .utils.data_helpers import (
    create_dense_rhs,
    create_random_sparse_alg_matrix,
    create_random_sparse_matrix,
    sparse_matrix_add_const_inplace,
    unsupported_sparse_formats,
)
from .utils.support_matrix import (
    supported_dtypes,
    supported_exec_space_dense_rhs,
    supported_index_dtypes,
    supported_sparse_array_types,
    supported_sparse_type_dtype,
)

rng = random.Random(101)


def check_meta_data(a, b, x, lhs_framework, rhs_framework, operand_placement, device_id, dtype):
    if device_id is None:
        if operand_placement == OperandPlacement.host:
            device_id = "cpu"
        else:
            device_id = 0
    # TODO(ktokarski) check shapes
    a_framework = framework_from_array(a)
    b_framework = framework_from_array(b)
    x_framework = framework_from_array(x)
    assert a_framework == lhs_framework, f"{a_framework} != {lhs_framework}"
    assert b_framework == rhs_framework, f"{b_framework} != {rhs_framework}"
    assert x_framework == rhs_framework, f"{x_framework} != {rhs_framework}"
    assert operand_placement_from_array(a) == operand_placement, f"{operand_placement_from_array(a)} != {operand_placement}"
    assert operand_placement_from_array(b) == operand_placement, f"{operand_placement_from_array(b)} != {operand_placement}"
    assert operand_placement_from_array(x) == operand_placement, f"{operand_placement_from_array(x)} != {operand_placement}"
    assert device_id_from_array(a) == device_id, f"{device_id_from_array(a)} != {device_id}"
    assert device_id_from_array(b) == device_id, f"{device_id_from_array(b)} != {device_id}"
    assert device_id_from_array(x) == device_id, f"{device_id_from_array(x)} != {device_id}"
    assert get_values_dtype_from_array(a) == dtype, f"{get_values_dtype_from_array(a)} != {dtype}"
    assert get_values_dtype_from_array(b) == dtype, f"{get_values_dtype_from_array(b)} != {dtype}"
    assert get_values_dtype_from_array(x) == dtype, f"{get_values_dtype_from_array(x)} != {dtype}"


def _check(a, b, x):
    a = to_dense_numpy(a)
    b = to_dense_numpy(b)
    x = to_dense_numpy(x)
    ref = a @ x
    assert np.allclose(ref, b)


def check(a, b, x, device_id=None):
    if device_id is None or device_id == "cpu":
        _check(a, b, x)
    else:
        assert isinstance(device_id, int)
        with get_framework_device_ctx(device_id, Framework.cupy):
            _check(a, b, x)


def get_exec_cuda_options(
    hybrid_memory_mode,
    hybrid_device_memory_limit,
    register_cuda_memory,
    exec_space_format,
    memory_mode_format,
):
    if memory_mode_format == "dict":
        hybrid_memory_mode_options = {
            "hybrid_memory_mode": hybrid_memory_mode,
            "hybrid_device_memory_limit": hybrid_device_memory_limit,
            "register_cuda_memory": register_cuda_memory,
        }
    else:
        assert memory_mode_format == "object"
        hybrid_memory_mode_options = nvmath.sparse.advanced.HybridMemoryModeOptions(
            hybrid_memory_mode=hybrid_memory_mode,
            hybrid_device_memory_limit=hybrid_device_memory_limit,
            register_cuda_memory=register_cuda_memory,
        )
    if exec_space_format == "dict":
        execution = {
            "name": "cuda",
            "hybrid_memory_mode_options": hybrid_memory_mode_options,
        }
    else:
        assert exec_space_format == "object"
        execution = nvmath.sparse.advanced.ExecutionCUDA(
            hybrid_memory_mode_options=hybrid_memory_mode_options,
        )
    return execution


def get_alg_matrix_type_and_view(sparse_type, sparse_view):
    match sparse_type:
        case DirectSolverMatrixType.GENERAL:
            alg_matrix_type = None
        case DirectSolverMatrixType.SYMMETRIC | DirectSolverMatrixType.HERMITIAN:
            alg_matrix_type = "symmetric"
        case DirectSolverMatrixType.SPD | DirectSolverMatrixType.HPD:
            alg_matrix_type = "positive"
        case _:
            raise ValueError(f"Unsupported sparse type: {sparse_type}")
    if alg_matrix_type is None:
        alg_matrix_view = None
    else:
        match sparse_view:
            case DirectSolverMatrixViewType.FULL:
                alg_matrix_view = None
            case DirectSolverMatrixViewType.LOWER:
                alg_matrix_view = "lower"
            case DirectSolverMatrixViewType.UPPER:
                alg_matrix_view = "upper"
    return alg_matrix_type, alg_matrix_view


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
        "density",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k, Param("density", density))
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_type in supported_index_dtypes
        if index_type in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for n in [1, 10]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 1), RHSMatrix(n, 5)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
        for density in [1, 0.3]
        if int(n * n * density) > n  # we keep the diagonal
    ],
    ids=idfn,
)
def test_matrix_solve(framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k, density):
    density = density.value
    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, density, dtype, seed=42, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    x = nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)
    check_meta_data(a, b, x, framework, tensor_framework, operand_placement, None, dtype)
    check(a, b, x)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
        "density_0",
        "density_1",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            index_type,
            dtype,
            n,
            rhs_k,
            Param("density_0", density_0),
            Param("density_1", density_1),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in [rng.choice(framework2operand_placement[framework])]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [
            rng.choice([index_type for index_type in supported_index_dtypes if index_type in framework2index_dtype[framework]])
        ]
        for dtype in [rng.choice([dtype for dtype in supported_dtypes if dtype in framework2dtype[framework]])]
        for n in [16]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 5)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
        for density_0, density_1 in [(0.25, 0.5), (0.6, 0.3)]
    ],
    ids=idfn,
)
def test_matrix_unsupported_reset_density_change(
    framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k, density_0, density_1
):
    density_0 = density_0.value
    density_1 = density_1.value
    a_0 = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, density_0, dtype, seed=42, index_dtype=index_type
    )
    a_1 = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, density_1, dtype, seed=44, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    with nvmath.sparse.advanced.DirectSolver(a_0, b, execution=exec_space.nvname) as solver:
        solver.plan()
        solver.factorize()
        x = solver.solve()
        check_meta_data(a_0, b, x, framework, tensor_framework, operand_placement, None, dtype)
        check(a_0, b, x)
        del a_0
        with pytest.raises(TypeError, match="The number of non-zeros"):
            solver.reset_operands(a=a_1)
            solver.plan()
            solver.factorize()


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "device_id",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            Param("device_id", 1),
            sparse_array_type,
            index_type,
            dtype,
            n,
            rhs_k,
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_type in supported_index_dtypes
        if index_type in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for n in [3, 12]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 4)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
@multi_gpu_only
def test_matrix_solve_non_default_device_id(
    framework, exec_space, operand_placement, device_id, sparse_array_type, index_type, dtype, n, rhs_k
):
    density = 0.5
    device_id = device_id.value
    options = {"name": exec_space.nvname}
    if operand_placement == OperandPlacement.host:
        options["device_id"] = device_id
        device_id = "cpu"
    tensor_framework = framework2tensor_framework[framework]
    a_0 = create_random_sparse_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        n,
        n,
        density,
        dtype,
        seed=42,
        index_dtype=index_type,
        device_id=device_id,
    )
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype, device_id=device_id, start=1)
    with nvmath.sparse.advanced.DirectSolver(a_0, b, execution=options) as solver:
        solver.plan()
        solver.factorize()
        x = solver.solve()
        check_meta_data(a_0, b, x, framework, tensor_framework, operand_placement, device_id, dtype)
        check(a_0, b, x, device_id=device_id)
        del a_0
        a_1 = create_random_sparse_matrix(
            framework,
            operand_placement,
            sparse_array_type,
            n,
            n,
            density,
            dtype,
            seed=44,
            index_dtype=index_type,
            device_id=device_id,
        )
        solver.reset_operands(a=a_1)
        solver.plan()
        solver.factorize()
        x = solver.solve()
        check_meta_data(a_1, b, x, framework, tensor_framework, operand_placement, device_id, dtype)
        check(a_1, b, x, device_id=device_id)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "device_id_0",
        "device_id_1",
        "sparse_array_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            Param("device_id_0", 0),
            Param("device_id_1", 1),
            sparse_array_type,
            dtype,
            n,
            rhs_k,
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for n in [13, 27]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 1), RHSMatrix(n, 11)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
@multi_gpu_only
def test_matrix_solve_device_id(
    framework, exec_space, operand_placement, device_id_0, device_id_1, sparse_array_type, dtype, n, rhs_k
):
    density = 0.5
    tensor_framework = framework2tensor_framework[framework]
    device_id_0 = device_id_0.value
    device_id_1 = device_id_1.value
    options_0 = {"name": exec_space.nvname}
    options_1 = {"name": exec_space.nvname}
    if operand_placement == OperandPlacement.host:
        options_0["device_id"] = device_id_0
        options_1["device_id"] = device_id_1
        device_id_0 = "cpu"
        device_id_1 = "cpu"
    a_1 = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, density, dtype, seed=42, device_id=device_id_1
    )
    b_01 = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype, device_id=device_id_1, start=1)
    b_11 = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype, device_id=device_id_1, start=2 * n)
    a_0 = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, density, dtype, seed=43, device_id=device_id_0
    )
    b_0 = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype, device_id=device_id_0, start=n)
    solver_1, solver_0 = None, None
    try:
        solver_1 = nvmath.sparse.advanced.DirectSolver(a_1, b_01, execution=exec_space.nvname)
        solver_0 = nvmath.sparse.advanced.DirectSolver(a_0, b_0, execution=exec_space.nvname)
        solver_1.plan()
        solver_0.plan()
        solver_1.factorize()
        solver_0.factorize()
        x_01 = solver_1.solve()
        x_0 = solver_0.solve()
        check_meta_data(a_1, b_01, x_01, framework, tensor_framework, operand_placement, device_id_1, dtype)
        check(a_1, b_01, x_01, device_id=device_id_1)
        del b_01, x_01
        solver_1.reset_operands(b=b_11)
        x_11 = solver_1.solve()
    finally:
        if solver_0 is not None:
            solver_0.free()
        if solver_1 is not None:
            solver_1.free()
    check_meta_data(a_1, b_11, x_11, framework, tensor_framework, operand_placement, device_id_1, dtype)
    check_meta_data(a_0, b_0, x_0, framework, tensor_framework, operand_placement, device_id_0, dtype)
    check(a_1, b_11, x_11, device_id=device_id_1)
    check(a_0, b_0, x_0, device_id=device_id_0)


@pytest.mark.parametrize(
    (
        "hybrid_device_memory_limit",
        "register_cuda_memory",
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
        "exec_space_format",
        "memory_mode_format",
    ),
    [
        (
            Param("hybrid_device_memory_limit", hybrid_device_memory_limit),
            Param("register_cuda_memory", register_cuda_memory),
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            index_type,
            dtype,
            n,
            rhs_k,
            Param("exec_space_format", exec_space_format),
            Param("memory_mode_format", memory_mode_format),
        )
        for hybrid_device_memory_limit in [None, 3 * 2**20, "1.7MB", "10%"]
        for register_cuda_memory in [False, True]
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in [ExecutionSpace.cudss_cuda]
        for operand_placement in [rng.choice(framework2operand_placement[framework])]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [
            rng.choice([index_type for index_type in supported_index_dtypes if index_type in framework2index_dtype[framework]])
        ]
        for dtype in [rng.choice([dtype for dtype in supported_dtypes if dtype in framework2dtype[framework]])]
        for n in [15]
        for rhs_k in [rng.choice([RHSVector(n), RHSMatrix(n, 3)])]
        for exec_space_format in [rng.choice(["dict", "object"])]
        for memory_mode_format in [rng.choice(["dict", "object"])]
    ],
    ids=idfn,
)
def test_matrix_solve_cuda_options(
    hybrid_device_memory_limit,
    register_cuda_memory,
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    index_type,
    dtype,
    n,
    rhs_k,
    exec_space_format,
    memory_mode_format,
):
    assert exec_space == ExecutionSpace.cudss_cuda
    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, 0.5, dtype, seed=42, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    host_memory_estimates = []
    for hybrid_memory_mode in [False, True]:
        execution = get_exec_cuda_options(
            hybrid_memory_mode,
            hybrid_device_memory_limit.value,
            register_cuda_memory.value,
            exec_space_format.value,
            memory_mode_format.value,
        )
        with nvmath.sparse.advanced.DirectSolver(
            a,
            b,
            execution=execution,
        ) as solver:
            solver.plan()
            solver.factorize()
            x = solver.solve()
            host_memory_estimates.append(solver.plan_info.memory_estimates.peak_host_memory)
        check_meta_data(a, b, x, framework, tensor_framework, operand_placement, None, dtype)
        check(a, b, x)
    # TODO(ktokarski) Can we check if other options are in use?
    without_host_memory, with_host_memory = host_memory_estimates
    assert with_host_memory > without_host_memory, (
        f"Without host memory: {without_host_memory}, with host memory: {with_host_memory}"
    )


@pytest.mark.parametrize(
    (
        "sparse_type",
        "sparse_view",
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
        "options_format",
    ),
    [
        (
            Param("sparse_type", sparse_type),
            Param("sparse_view", sparse_view),
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            index_type,
            dtype,
            n,
            rhs_k,
            Param("options_format", options_format),
        )
        for sparse_type in DirectSolverMatrixType
        for sparse_view in DirectSolverMatrixViewType
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in [rng.choice(framework2operand_placement[framework])]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [
            rng.choice([index_type for index_type in supported_index_dtypes if index_type in framework2index_dtype[framework]])
        ]
        for dtype in [
            rng.choice([dtype for dtype in supported_sparse_type_dtype[sparse_type] if dtype in framework2dtype[framework]])
        ]
        for n in [1, 13]
        for rhs_k in [rng.choice([RHSVector(n), RHSMatrix(n, 3)])]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
        for options_format in [rng.choice(["dict", "object"])]
    ],
    ids=idfn,
)
def test_solver_matrix_type_options(
    sparse_type,
    sparse_view,
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    index_type,
    dtype,
    n,
    rhs_k,
    options_format,
):
    sparse_type = sparse_type.value
    sparse_view = sparse_view.value
    options_format = options_format.value
    alg_matrix_type, alg_matrix_view = get_alg_matrix_type_and_view(sparse_type, sparse_view)
    a_full, a_view = create_random_sparse_alg_matrix(
        framework,
        operand_placement,
        sparse_array_type,
        n,
        n,
        None,
        dtype,
        seed=42,
        index_dtype=index_type,
        alg_matrix_type=alg_matrix_type,
        alg_matrix_view=alg_matrix_view,
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    if options_format == "dict":
        options = {
            "sparse_system_type": sparse_type,
            "sparse_system_view": sparse_view,
        }
    else:
        assert options_format == "object"
        options = nvmath.sparse.advanced.DirectSolverOptions(
            sparse_system_type=sparse_type,
            sparse_system_view=sparse_view,
        )
    x = nvmath.sparse.advanced.direct_solver(a_view, b, execution=exec_space.nvname, options=options)
    check_meta_data(a_view, b, x, framework, tensor_framework, operand_placement, None, dtype)
    check(a_full, b, x)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "options_format",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            Param("options_format", options_format),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in [rng.choice(framework2operand_placement[framework])]
        for sparse_array_type in supported_sparse_array_types
        for options_format in [rng.choice(["dict", "object"])]
    ],
    ids=idfn,
)
def test_solver_matrix_options_external_handle(
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    options_format,
):
    handle = None
    try:
        handle = nvmath.bindings.cudss.create()
        tensor_framework = framework2tensor_framework[framework]
        options_format = options_format.value
        for n in [1, 7]:
            for rhs_k in [RHSVector(n), RHSMatrix(n, 5)]:
                if rhs_k.type not in supported_exec_space_dense_rhs[exec_space]:
                    continue
                sparse_type = rng.choice(list(DirectSolverMatrixType))
                sparse_view = rng.choice(list(DirectSolverMatrixViewType))
                index_types = [
                    index_type for index_type in supported_index_dtypes if index_type in framework2index_dtype[framework]
                ]
                index_type = rng.choice(index_types)
                dtypes = [dtype for dtype in supported_sparse_type_dtype[sparse_type] if dtype in framework2dtype[framework]]
                dtype = rng.choice(dtypes)
                alg_matrix_type, alg_matrix_view = get_alg_matrix_type_and_view(sparse_type, sparse_view)
                a_full, a_view = create_random_sparse_alg_matrix(
                    framework,
                    operand_placement,
                    sparse_array_type,
                    n,
                    n,
                    None,
                    dtype,
                    seed=42,
                    index_dtype=index_type,
                    alg_matrix_type=alg_matrix_type,
                    alg_matrix_view=alg_matrix_view,
                )
                b = create_dense_rhs(framework2tensor_framework[framework], operand_placement, rhs_k, dtype)
                if options_format == "dict":
                    options = {
                        "sparse_system_type": sparse_type,
                        "sparse_system_view": sparse_view,
                        "handle": handle,
                    }
                else:
                    assert options_format == "object"
                    options = nvmath.sparse.advanced.DirectSolverOptions(
                        sparse_system_type=sparse_type,
                        sparse_system_view=sparse_view,
                        handle=handle,
                    )
                with nvmath.sparse.advanced.DirectSolver(
                    a_view,
                    b,
                    execution=exec_space.nvname,
                    options=options,
                ) as solver:
                    assert solver.handle == handle, f"{solver.handle} != {handle}"
                    solver.plan()
                    solver.factorize()
                    x = solver.solve()
                check_meta_data(a_view, b, x, framework, tensor_framework, operand_placement, None, dtype)
                check(a_full, b, x)
    finally:
        if handle is not None:
            nvmath.bindings.cudss.destroy(handle)


@pytest.mark.parametrize(
    (
        "hybrid_memory_mode",
        "register_cuda_memory",
        "hybrid_device_memory_limit",
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
        "exec_space_format",
        "memory_mode_format",
    ),
    [
        (
            Param("hybrid_memory_mode", hybrid_memory_mode),
            Param("register_cuda_memory", register_cuda_memory),
            Param("hybrid_device_memory_limit", hybrid_device_memory_limit),
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            index_type,
            dtype,
            n,
            rhs_k,
            Param("exec_space_format", exec_space_format),
            Param("memory_mode_format", memory_mode_format),
        )
        for hybrid_memory_mode in [True]
        for register_cuda_memory in [False, True]
        for hybrid_device_memory_limit in [1, "0.3KB"]
        for framework in [
            rng.choice([framework for framework in Framework.enabled() if framework in sparse_supporting_frameworks])
        ]
        for exec_space in [ExecutionSpace.cudss_cuda]
        for operand_placement in [rng.choice(framework2operand_placement[framework])]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [
            rng.choice([index_type for index_type in supported_index_dtypes if index_type in framework2index_dtype[framework]])
        ]
        for dtype in [rng.choice([dtype for dtype in supported_dtypes if dtype in framework2dtype[framework]])]
        for n in [15]
        for rhs_k in [rng.choice([RHSVector(n), RHSMatrix(n, 3)])]
        for exec_space_format in [rng.choice(["dict", "object"])]
        for memory_mode_format in [rng.choice(["dict", "object"])]
    ],
    ids=idfn,
)
def test_matrix_solve_cuda_options_too_tight_limit(
    hybrid_memory_mode,
    register_cuda_memory,
    hybrid_device_memory_limit,
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    index_type,
    dtype,
    n,
    rhs_k,
    exec_space_format,
    memory_mode_format,
):
    assert exec_space == ExecutionSpace.cudss_cuda
    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    execution = get_exec_cuda_options(
        hybrid_memory_mode.value,
        hybrid_device_memory_limit.value,
        register_cuda_memory.value,
        exec_space_format.value,
        memory_mode_format.value,
    )
    with pytest.raises(nvmath.bindings.cudss.cuDSSError):
        nvmath.sparse.advanced.direct_solver(a, b, execution=execution)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in [ExecutionSpace.cudss_hybrid]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [
            rng.choice([index_type for index_type in supported_index_dtypes if index_type in framework2index_dtype[framework]])
        ]
        for dtype in [rng.choice([dtype for dtype in supported_dtypes if dtype in framework2dtype[framework]])]
        for n in [11]
        for rhs_k in [RHSMatrix(n, 1), RHSMatrix(n, 11)]
    ],
    ids=idfn,
)
def test_matrix_solve_hybrid_multiple_rhs_unsupported(
    framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k
):
    assert rhs_k.type not in supported_exec_space_dense_rhs[exec_space]
    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    with pytest.raises(TypeError, match="multiple RHS"):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in [DType.complex32, DType.bfloat16, DType.uint8, DType.int32, DType.float16]
        if dtype in framework2value_dtype[framework]
        for n in [7]
        for rhs_k in [RHSVector(n)]
    ],
    ids=idfn,
)
def test_matrix_solve_unsupported_dtype(framework, exec_space, operand_placement, sparse_array_type, dtype, n, rhs_k):
    assert dtype not in supported_dtypes
    a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42)
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    with pytest.raises(TypeError, match=f"The dtype \\(value type\\) {dtype.name} is not supported"):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_types in [[dtype for dtype in framework2index_dtype[framework] if dtype not in supported_index_dtypes]]
        if index_types
        for index_type in [rng.choice(index_types)]
        for dtype in [rng.choice([dtype for dtype in supported_dtypes if dtype in framework2dtype[framework]])]
        for n in [13]
        for rhs_k in [RHSVector(n)]
    ],
    ids=idfn,
)
def test_matrix_solve_unsupported_index_dtype(
    framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k
):
    assert index_type not in supported_index_dtypes
    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    with pytest.raises(TypeError, match=f"The index type {index_type.name} is not supported"):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


def _random_sizes(batch_size, max_value):
    sizes = [np.random.randint(1, max_value + 1) for _ in range(batch_size)]
    return sizes


def _generate_n(batch_size, lhs_batching_mode, rhs_batching_mode, max_value):
    if lhs_batching_mode == "sequence" and rhs_batching_mode == "sequence":
        return _random_sizes(batch_size, max_value)
    else:
        return max_value


def _generate_rhs_k(batch_size, lhs_batching_mode, rhs_batching_mode, max_value):
    if rhs_batching_mode == "sequence":
        return _random_sizes(batch_size, max_value)
    else:
        return max_value


def _generate_lhs_batch(
    lhs_batching_mode,
    batch_size,
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    dtype,
    ns,
    seed=42,
    index_dtype=DType.int32,
):
    assert isinstance(ns, int) or len(ns) == batch_size

    if lhs_batching_mode == "sequence":
        if isinstance(ns, int):
            ns = [ns] * batch_size
        a = [
            create_random_sparse_matrix(
                framework, operand_placement, sparse_array_type, n, n, 0.5, dtype, seed=seed + i, index_dtype=index_dtype
            )
            for i, n in enumerate(ns)
        ]
    else:
        if isinstance(ns, int):
            n = ns
        else:
            n = ns[0]
            assert all(m == n for m in ns)
        assert framework == Framework.torch
        a = create_random_sparse_matrix(
            framework,
            operand_placement,
            sparse_array_type,
            n,
            n,
            None,
            dtype,
            seed=42,
            index_dtype=index_dtype,
            batch_dims=batch_size,
        )
    return a


def _generate_rhs_batch(rhs_batching_mode, batch_size, framework, exec_space, operand_placement, dtype, ns, rhs_ks, start=1):
    assert isinstance(rhs_ks, int) or len(rhs_ks) == batch_size

    if rhs_batching_mode in ("sequence", "sequence_of_vectors"):
        if isinstance(rhs_ks, int):
            rhs_ks = [rhs_ks] * batch_size
        if isinstance(ns, int):
            ns = [ns] * batch_size
        b = [
            create_dense_rhs(
                framework2tensor_framework[framework],
                operand_placement,
                RHSMatrix(n, rhs_k) if rhs_batching_mode == "sequence" else RHSVector(n),
                dtype,
                start=start + i,
            )
            for i, (n, rhs_k) in enumerate(zip(ns, rhs_ks, strict=False))
        ]
    else:
        if isinstance(ns, int):
            n = ns
        else:
            n = ns[0]
            assert all(m == n for m in ns)
        if isinstance(rhs_ks, int):
            rhs_k = rhs_ks
        else:
            assert all(m == rhs_k for m in rhs_ks)
            rhs_k = rhs_ks[0]

        b = create_dense_rhs(framework2tensor_framework[framework], operand_placement, RHSBatch(n, rhs_k, batch_size), dtype)

    return b


def _check_batched_result(
    a,
    b,
    x,
    batch_size: int | tuple[int, int],
    expected_x_batching_mode: typing.Literal["sequence", "sequence_of_vectors", "tensor"],
):
    match expected_x_batching_mode:
        case "sequence" | "sequence_of_vectors":
            assert isinstance(batch_size, int)
            assert len(x) == batch_size
            assert isinstance(x, tuple)
            for a_item, b_item, x_item in zip(a, b, x, strict=True):
                check(a_item, b_item, x_item)
        case "tensor":
            check(a, b, x)  # Broadcasting
        case _:
            raise ValueError(f"Unknown expected_x_batching_mode: {expected_x_batching_mode}")


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "batch_size",
        "lhs_batching_mode",
        "rhs_batching_mode",
        "ns",
        "rhs_ks",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            index_type,
            dtype,
            batch_size,
            lhs_batching_mode,
            rhs_batching_mode,
            _generate_n(batch_size, lhs_batching_mode, rhs_batching_mode, max_n),
            _generate_rhs_k(batch_size, lhs_batching_mode, rhs_batching_mode, max_rhs_k),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [DType.int32]  # int64 is not supported with batching.
        if index_type in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for lhs_batching_mode in (["sequence", "tensor"] if framework == Framework.torch else ["sequence"])
        for rhs_batching_mode in ["sequence", "tensor", "sequence_of_vectors"]
        for batch_size in [1, 4] + ([(2, 3), (1, 1)] if lhs_batching_mode == rhs_batching_mode == "tensor" else [])
        for max_n in [5]
        for max_rhs_k in [5]
        if exec_space != ExecutionSpace.cudss_hybrid  # batching not supported for hybrid
    ],
    ids=idfn,
)
def test_batching(
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    index_type,
    dtype,
    batch_size: int | tuple[int, int],
    lhs_batching_mode: typing.Literal["sequence", "tensor"],
    rhs_batching_mode: typing.Literal["sequence", "tensor", "sequence_of_vectors"],
    ns,
    rhs_ks,
):
    a = _generate_lhs_batch(
        lhs_batching_mode,
        batch_size,
        framework,
        exec_space,
        operand_placement,
        sparse_array_type,
        dtype,
        ns,
        index_dtype=index_type,
    )
    b = _generate_rhs_batch(rhs_batching_mode, batch_size, framework, exec_space, operand_placement, dtype, ns, rhs_ks)

    if rhs_batching_mode == "tensor" and framework == Framework.scipy:
        with pytest.raises(TypeError, match="Implicit RHS batching for NumPy.*not supported"):
            nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)
        return

    x = nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)
    expected_x_batching_mode = "tensor" if rhs_batching_mode == "tensor" else "sequence"
    _check_batched_result(a, b, x, batch_size, expected_x_batching_mode)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_type in supported_index_dtypes
        if index_type in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for n in [1, 10]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 1), RHSMatrix(n, 5)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
@pytest.mark.parametrize(
    "reset_lhs,reset_rhs,free",
    [
        (lhs, rhs, free)
        for lhs in (True, False)
        for rhs in (True, False)
        for free in (True, False)
        if not free or (lhs and rhs)  # both operands need to be set after freeing
    ],
)
def test_reset(
    framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k, reset_lhs, reset_rhs, free
):
    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, index_dtype=index_type
    )
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)

    a2 = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42 + 1, index_dtype=index_type
    )
    b2 = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype, start=-100)

    with nvmath.sparse.advanced.DirectSolver(a, b, execution=exec_space.nvname) as solver:
        solver.plan()
        solver.factorize()
        x = solver.solve()
        check_meta_data(a, b, x, framework, tensor_framework, operand_placement, None, dtype)
        check(a, b, x)
        if free:
            solver.release_operands()
        if (reset_lhs, reset_rhs) == (True, False):
            solver.reset_operands(a=a2)
        elif (reset_lhs, reset_rhs) == (False, True):
            solver.reset_operands(b=b2)
        elif (reset_lhs, reset_rhs) == (True, True):
            solver.reset_operands(a=a2, b=b2)
        if not reset_lhs:
            a2 = a
        if not reset_rhs:
            b2 = b
        if reset_lhs:
            # Re-plan since the LHS has changed.
            solver.plan()
            solver.factorize()
        x2 = solver.solve()
        check_meta_data(a2, b2, x2, framework, tensor_framework, operand_placement, None, dtype)
        check(a2, b2, x2)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "batch_size",
        "lhs_batching_mode",
        "rhs_batching_mode",
        "ns",
        "rhs_ks",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            index_type,
            dtype,
            batch_size,
            lhs_batching_mode,
            rhs_batching_mode,
            _generate_n(batch_size, lhs_batching_mode, rhs_batching_mode, max_n),
            _generate_rhs_k(batch_size, lhs_batching_mode, rhs_batching_mode, max_rhs_k),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for index_type in [DType.int32]  # int64 is not supported with batching.
        if index_type in framework2index_dtype[framework]
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for lhs_batching_mode in (["sequence", "tensor"] if framework == Framework.torch else ["sequence"])
        for rhs_batching_mode in ["sequence", "tensor", "sequence_of_vectors"]
        for batch_size in [1, 4] + ([(2, 3), (1, 1)] if lhs_batching_mode == rhs_batching_mode == "tensor" else [])
        for max_n in [5]
        for max_rhs_k in [5]
        if exec_space != ExecutionSpace.cudss_hybrid  # batching not supported for hybrid
        if not (framework == Framework.scipy and rhs_batching_mode == "tensor")  # implicit batching for numpy
    ],
    ids=idfn,
)
@pytest.mark.parametrize("reset_lhs", [True, False])
@pytest.mark.parametrize("reset_rhs", [True, False])
def test_reset_batched(
    framework: Framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    index_type,
    dtype,
    batch_size: int | tuple[int, int],
    lhs_batching_mode: typing.Literal["sequence", "tensor"],
    rhs_batching_mode: typing.Literal["sequence", "tensor", "sequence_of_vectors"],
    ns,
    rhs_ks,
    reset_lhs,
    reset_rhs,
):
    a = _generate_lhs_batch(
        lhs_batching_mode,
        batch_size,
        framework,
        exec_space,
        operand_placement,
        sparse_array_type,
        dtype,
        ns,
        seed=42,
        index_dtype=index_type,
    )
    b = _generate_rhs_batch(rhs_batching_mode, batch_size, framework, exec_space, operand_placement, dtype, ns, rhs_ks)

    a2 = _generate_lhs_batch(
        lhs_batching_mode,
        batch_size,
        framework,
        exec_space,
        operand_placement,
        sparse_array_type,
        dtype,
        ns,
        seed=42 + 1,
        index_dtype=index_type,
    )
    b2 = _generate_rhs_batch(
        rhs_batching_mode, batch_size, framework, exec_space, operand_placement, dtype, ns, rhs_ks, start=-100
    )

    with nvmath.sparse.advanced.DirectSolver(a, b, execution=exec_space.nvname) as solver:
        solver.plan()
        solver.factorize()
        x = solver.solve()
        _check_batched_result(a, b, x, batch_size, rhs_batching_mode)
        if (reset_lhs, reset_rhs) == (True, False):
            solver.reset_operands(a=a2)
        elif (reset_lhs, reset_rhs) == (False, True):
            solver.reset_operands(b=b2)
        elif (reset_lhs, reset_rhs) == (True, True):
            solver.reset_operands(a=a2, b=b2)
        if not reset_lhs:
            a2 = a
        if not reset_rhs:
            b2 = b
        if reset_lhs:
            # Re-plan since the LHS has changed.
            solver.plan()
            solver.factorize()
        x2 = solver.solve()
        _check_batched_result(a2, b2, x2, batch_size, rhs_batching_mode)


@pytest.mark.parametrize("reset_target", ["a", "b"])
def test_reset_implicit_batch_device_pointers(reset_target):
    """
    Verify that reset_operands() with implicit-batch on device actually updates
    the cuDSS device pointer arrays.  Both a1/b1 are kept alive so CUDA cannot
    reuse their addresses for a2/b2.  Fails if the copy_() calls in
    update_cudss_*_implicit_batch_ptr are removed.
    """
    torch = pytest.importorskip("torch")

    n, batch_size = 8, 2

    def make_batch_csr(seed):
        torch.manual_seed(seed)
        dense = torch.rand(batch_size, n, n, dtype=torch.float64, device="cuda")
        dense += 3.0 * torch.eye(n, dtype=torch.float64, device="cuda")
        csr = dense.to_sparse_csr()
        return torch.sparse_csr_tensor(
            csr.crow_indices().to(torch.int32),
            csr.col_indices().to(torch.int32),
            csr.values(),
            size=csr.size(),
            device="cuda",
        )

    a1 = make_batch_csr(seed=1)
    a2 = make_batch_csr(seed=2)
    b1 = torch.ones(batch_size, 1, n, dtype=torch.float64, device="cuda").swapaxes(-2, -1)
    b2 = torch.full((batch_size, 1, n), 2.0, dtype=torch.float64, device="cuda").swapaxes(-2, -1)

    solver = nvmath.sparse.advanced.DirectSolver(a1, b1)
    solver.plan()
    solver.factorize()
    solver.solve()

    if reset_target == "a":
        solver.reset_operands(a=a2)
        solver.plan()
        solver.factorize()
        a_check, b_check = a2, b1
    else:
        solver.reset_operands(b=b2)
        a_check, b_check = a1, b2

    x = solver.solve()
    torch.cuda.current_stream().synchronize()

    for i in range(batch_size):
        residual = torch.sparse.mm(a_check[i], x[i]) - b_check[i]
        assert residual.norm().item() < 1e-10, (
            f"batch {i}: residual {residual.norm().item():.2e} — "
            f"cuDSS likely used stale pointers after resetting {reset_target}"
        )


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "problem",
        "format",
        "when",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, dtype, problem, format, when)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in [ExecutionSpace.cudss_cuda]
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in [DType.float32]
        if dtype in framework2dtype[framework]
        for problem in ["single", "one_in_sequence", "all_in_sequence"]
        + (["implicit_batch"] if framework == Framework.torch else [])
        for when in ["init", "reset_operands"]
        for format in unsupported_sparse_formats[framework]
    ],
    ids=idfn,
)
@pytest.mark.parametrize("n,rhs_k", [(n, rhs_k) for n in [10] for rhs_k in [RHSVector(n), RHSMatrix(n, 5)]])
def test_invalid_sparse_format(
    framework, exec_space, operand_placement, sparse_array_type, dtype, problem, format, when, n, rhs_k
):
    constructor = unsupported_sparse_formats[framework][format]
    match problem:
        case "single":
            invalid_a = constructor(shape=(n, n), dtype=dtype, placement=operand_placement)
            valid_a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42)
            batch_size = None
        case "one_in_sequence":
            bad_a_part = constructor(shape=(n, n), dtype=dtype, placement=operand_placement)
            good_a_part = create_random_sparse_matrix(
                framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42
            )
            invalid_a = [good_a_part, bad_a_part, good_a_part]
            valid_a = [good_a_part] * 3
            batch_size = 3
        case "all_in_sequence":
            batch_size = 3
            invalid_a = [constructor(shape=(n, n), dtype=dtype, placement=operand_placement) for _ in range(batch_size)]
            valid_a = [
                create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42)
                for _ in range(batch_size)
            ]
        case "implicit_batch":
            batch_size = 3
            invalid_a = constructor(shape=(batch_size, n, n), dtype=dtype, placement=operand_placement)
            valid_a = create_random_sparse_matrix(
                framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, batch_dims=(batch_size,)
            )

    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    if batch_size is not None:
        b = [b] * batch_size

    if when == "init":
        with pytest.raises(TypeError, match="The LHS must be.*CSR"):
            nvmath.sparse.advanced.direct_solver(invalid_a, b, execution=exec_space.nvname)
    elif when == "reset_operands":
        with nvmath.sparse.advanced.DirectSolver(valid_a, b, execution=exec_space.nvname) as solver:
            solver.plan()
            solver.factorize()
            solver.solve()
            with pytest.raises(TypeError, match="The LHS.*must.*be.*CSR"):
                solver.reset_operands(a=invalid_a)
    else:
        raise ValueError(f"Bad `when`: {when}")


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        # The operands are copied under the hood, inplace modification
        # won't be visible to the solver.
        if exec_space != ExecutionSpace.cudss_cuda or operand_placement != OperandPlacement.host
        for sparse_array_type in supported_sparse_array_types
        for index_type in supported_index_dtypes
        if index_type in framework2index_dtype[framework]
        for dtype in [DType.float64, DType.complex128]
        if dtype in framework2dtype[framework]
        for n in [105, 333]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 1), RHSMatrix(n, 5)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
def test_matrix_solve_inplace_reset_blocking_auto(
    framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k
):
    stream = get_custom_stream(framework) if operand_placement == OperandPlacement.device else None

    with use_stream_or_dummy_ctx(framework, stream):
        a = create_random_sparse_matrix(
            framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, index_dtype=index_type
        )
        a_orig = a.copy() if framework != Framework.torch else a.clone()
        a_modifed = a.copy() if framework != Framework.torch else a.clone()
        b = create_dense_rhs(framework2tensor_framework[framework], operand_placement, rhs_k, dtype)

    with nvmath.sparse.advanced.DirectSolver(
        a, b, options={"blocking": "auto"}, execution=exec_space.nvname, stream=stream
    ) as solver:
        solver.plan(stream=stream)
        solver.factorize(stream=stream)
        x0 = solver.solve(stream=stream)
        with use_stream_or_dummy_ctx(framework, stream):
            # do some kind-of heavy operations on the operand in non-default stream order
            for _ in range(100):
                sparse_matrix_add_const_inplace(a_modifed, 1)
                sparse_matrix_add_const_inplace(a, 1)
                sparse_matrix_add_const_inplace(a_modifed, -1)
                sparse_matrix_add_const_inplace(a, -1)
            sparse_matrix_add_const_inplace(a_modifed, 3)
            sparse_matrix_add_const_inplace(a, 3)
        solver.factorize(stream=stream)
        x1 = solver.solve(stream=stream)
        # modify the operand in non-default stream order after solver call
        with use_stream_or_dummy_ctx(framework, stream):
            sparse_matrix_add_const_inplace(a, 10**6)
        if stream is not None:
            stream.synchronize()
        check_meta_data(a_orig, b, x0, framework, framework2tensor_framework[framework], operand_placement, None, dtype)
        check(a_orig, b, x0)
        check_meta_data(a, b, x1, framework, framework2tensor_framework[framework], operand_placement, None, dtype)
        check(a_modifed, b, x1)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "index_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        # The operands are copied under the hood, inplace modification
        # won't be visible to the solver.
        if exec_space != ExecutionSpace.cudss_cuda or operand_placement != OperandPlacement.host
        for sparse_array_type in supported_sparse_array_types
        for index_type in supported_index_dtypes
        if index_type in framework2index_dtype[framework]
        for dtype in [DType.float64, DType.complex128]
        if dtype in framework2dtype[framework]
        for n in [105, 333]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 1), RHSMatrix(n, 5)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
def test_matrix_solve_always_blocking(framework, exec_space, operand_placement, sparse_array_type, index_type, dtype, n, rhs_k):
    stream = get_custom_stream(framework) if operand_placement == OperandPlacement.device else None
    other_stream = get_custom_stream(framework) if operand_placement == OperandPlacement.device else None

    a = create_random_sparse_matrix(
        framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42, index_dtype=index_type
    )
    ref_a = a.copy() if framework != Framework.torch else a.clone()
    b = create_dense_rhs(framework2tensor_framework[framework], operand_placement, rhs_k, dtype)
    b_ref = b.copy() if framework != Framework.torch else b.clone()

    with nvmath.sparse.advanced.DirectSolver(
        a, b, options={"blocking": True}, execution=exec_space.nvname, stream=stream
    ) as solver:
        solver.plan(stream=stream)
        solver.factorize(stream=stream)
        for _i in range(32):
            x = solver.solve(stream=stream)
        # modify the operands in place in a different stream,
        # relying on the fact that with blocking True the x should already
        # be computed
        with use_stream_or_dummy_ctx(framework, other_stream):
            b -= 10**6
            sparse_matrix_add_const_inplace(a, 10**6)
        check_meta_data(a, b, x, framework, framework2tensor_framework[framework], operand_placement, None, dtype)
        if stream is not None:
            stream.synchronize()
        check(ref_a, b_ref, x)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "n",
        "m",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, dtype, n, m, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in [DType.float32]
        for n, m, rhs_k in ((1, 2, RHSVector(1)), (2, 1, RHSVector(2)), (2, 3, RHSVector(2)))
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
def test_non_square(framework, exec_space, operand_placement, sparse_array_type, dtype, n, m, rhs_k):
    a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, m, None, dtype, seed=42, ones=True)
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    with pytest.raises(TypeError, match="(?s)The LHS.*must.*be.*(N, N)"):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "lhs_n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, dtype, lhs_n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in [DType.float32]
        for lhs_n, rhs_k in ((2, RHSVector(3)), (2, RHSMatrix(3, 3)))
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
def test_n_mismatch(framework, exec_space, operand_placement, sparse_array_type, dtype, lhs_n, rhs_k):
    a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, lhs_n, lhs_n, None, dtype, seed=42)
    tensor_framework = framework2tensor_framework[framework]
    b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
    with pytest.raises(TypeError, match="The extent N.*is not consistent between the LHS and RHS."):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "n",
        "m",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, dtype, n, m, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in [DType.float32]
        for n, m, rhs_k in ((1, 2, RHSVector(1)), (2, 1, RHSVector(2)), (2, 3, RHSVector(2)))
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
def test_non_square_sequence(framework, exec_space, operand_placement, sparse_array_type, dtype, n, m, rhs_k):
    rect_a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, m, None, dtype, seed=42, ones=True)
    square_a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42)
    a = [square_a, rect_a, square_a]
    tensor_framework = framework2tensor_framework[framework]
    b = [create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)] * 3
    with pytest.raises(TypeError, match="Each object in an explicitly-batched LHS.*must be.*(N, N)"):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "n",
        "rhs_k",
    ),
    [
        (framework, exec_space, operand_placement, sparse_array_type, dtype, n, rhs_k)
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for n in [1, 10]
        for rhs_k in [RHSVector(n), RHSMatrix(n, 1), RHSMatrix(n, 5)]
        if rhs_k.type in supported_exec_space_dense_rhs[exec_space]
    ],
    ids=idfn,
)
def test_logging(framework, exec_space, operand_placement, sparse_array_type, dtype, n, rhs_k):
    """
    Test if enabling logging doesn't trigger any errors.
    """
    import logging

    original_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        a = create_random_sparse_matrix(framework, operand_placement, sparse_array_type, n, n, None, dtype, seed=42)
        tensor_framework = framework2tensor_framework[framework]
        b = create_dense_rhs(tensor_framework, operand_placement, rhs_k, dtype)
        x = nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)
        check_meta_data(a, b, x, framework, tensor_framework, operand_placement, None, dtype)
        check(a, b, x)
    finally:
        logging.getLogger().setLevel(original_level)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_space",
        "operand_placement",
        "sparse_array_type",
        "dtype",
        "batch_size",
        "lhs_batching_mode",
        "rhs_batching_mode",
        "ns",
        "rhs_ks",
    ),
    [
        (
            framework,
            exec_space,
            operand_placement,
            sparse_array_type,
            dtype,
            batch_size,
            lhs_batching_mode,
            rhs_batching_mode,
            _generate_n(batch_size, lhs_batching_mode, rhs_batching_mode, max_n),
            _generate_rhs_k(batch_size, lhs_batching_mode, rhs_batching_mode, max_rhs_k),
        )
        for framework in Framework.enabled()
        if framework in sparse_supporting_frameworks
        for exec_space in ExecutionSpace
        for operand_placement in framework2operand_placement[framework]
        for sparse_array_type in supported_sparse_array_types
        for dtype in supported_dtypes
        if dtype in framework2dtype[framework]
        for lhs_batching_mode in (["sequence", "tensor"] if framework == Framework.torch else ["sequence"])
        for rhs_batching_mode in ["sequence", "tensor", "sequence_of_vectors"]
        for batch_size in [1, 4] + ([(2, 3), (1, 1)] if lhs_batching_mode == rhs_batching_mode == "tensor" else [])
        for max_n in [5]
        for max_rhs_k in [5]
        if exec_space != ExecutionSpace.cudss_hybrid  # batching not supported for hybrid
    ],
    ids=idfn,
)
def test_batch_size_mismatch(
    framework,
    exec_space,
    operand_placement,
    sparse_array_type,
    dtype,
    batch_size: int | tuple[int, int],
    lhs_batching_mode: typing.Literal["sequence", "tensor"],
    rhs_batching_mode: typing.Literal["sequence", "tensor", "sequence_of_vectors"],
    ns,
    rhs_ks,
):
    if rhs_batching_mode == "tensor" and framework == Framework.scipy:
        pytest.skip("Implicit RHS batching for NumPy is not supported")

    a_batch_size = batch_size
    b_batch_size = batch_size - 1 if isinstance(batch_size, int) else (7, 7)
    b_ns = ns if isinstance(ns, int) else ns[:-1]
    b_rhs_ks = rhs_ks if isinstance(rhs_ks, int) else rhs_ks[:-1]
    a = _generate_lhs_batch(
        lhs_batching_mode, a_batch_size, framework, exec_space, operand_placement, sparse_array_type, dtype, ns
    )
    b = _generate_rhs_batch(rhs_batching_mode, b_batch_size, framework, exec_space, operand_placement, dtype, b_ns, b_rhs_ks)

    with pytest.raises(TypeError, match="The batch (count|shapes).*must match"):
        nvmath.sparse.advanced.direct_solver(a, b, execution=exec_space.nvname)


@pytest.mark.skipif(HAS_CUPY_SPARSE is False, reason="cupy.sparse is not available")
@pytest.mark.parametrize("use_plan_solve", [False, True], ids=["no_plan", "with_plan"])
def test_reference_count(use_plan_solve):
    """
    Test reference counts consistency before/after context manager.
    Only need a single scenario with CuPy to test scenario when tensors reside on GPU.
    """

    # The number of equations.
    n = 8
    # Prepare sample input data.
    # Create a diagonally-dominant random CSR matrix.
    a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
    a += sp.diags([2.0] * n, format="csr", dtype="float64")
    # Create the RHS, which can be a matrix or vector in column-major layout.
    b = cp.ones((n,))

    initial_refcount_a = sys.getrefcount(a)
    initial_refcount_b = sys.getrefcount(b)

    # Create and optionally solve
    solver = nvmath.sparse.advanced.DirectSolver(a, b)
    with solver:
        if use_plan_solve:
            solver.plan()
            solver.factorize()
            x = solver.solve()
            cp.cuda.get_current_stream().synchronize()
        else:
            pass

    # After exiting context manager, reference counts should return to initial values
    assert sys.getrefcount(a) == initial_refcount_a, "post op: a refcount changed"
    assert sys.getrefcount(b) == initial_refcount_b, "post op: b refcount changed"
    if use_plan_solve:
        with check_freed_after(x, "post op: x should have sole ownership"):
            del x


@pytest.mark.parametrize("batch", [False, 4])
@pytest.mark.parametrize("unity_strides", [True, False])
def test_unity_strides(batch, unity_strides):
    torch = pytest.importorskip("torch")
    n = 3
    a = torch.rand(n, n, device="cuda") + torch.diag(torch.tensor([10] * n, device="cuda"))

    if batch:
        a = torch.stack([a] * batch, dim=0)
        if unity_strides:
            b = torch.ones(batch, n, 1, device="cuda")
            b = b.mT.contiguous().mT
        else:
            b = torch.ones(batch, 1, n, device="cuda").mT
    else:
        if unity_strides:
            b = torch.ones(n, 1, device="cuda")
            b = b.mT.contiguous().mT
        else:
            b = torch.ones(1, n, device="cuda").mT

    if unity_strides:
        assert tuple(b.stride()[-2:]) == (1, 1)
    else:
        assert tuple(b.stride()[-2:]) == (1, n)

    a = a.to_sparse_csr()
    # Note that torch uses int64 for index buffers, whereas cuDSS currently requires int32.
    a = torch.sparse_csr_tensor(
        a.crow_indices().to(dtype=torch.int32), a.col_indices().to(dtype=torch.int32), a.values(), size=a.size(), device="cuda"
    )

    x = nvmath.sparse.advanced.direct_solver(a, b)
    if batch:
        for i in range(batch):
            assert torch.allclose(a[i] @ x[i], b[i])
    else:
        assert torch.allclose(a @ x, b)


@pytest.mark.parametrize("use_plan_solve", [False, True], ids=["no_plan", "with_plan"])
@pytest.mark.parametrize("batching_mode", ["no_batch", "explicit_batch", "implicit_batch"])
def test_release_operands(use_plan_solve, batching_mode):
    """
    Test that after release_operands(), the refcounts of user-provided
    operands return to their initial values.
    """
    torch = pytest.importorskip("torch")

    def _make_sparse_csr(n, device="cuda"):
        dense = torch.rand(n, n, dtype=torch.float64, device=device)
        dense += 2.0 * torch.eye(n, dtype=torch.float64, device=device)
        csr = dense.to_sparse_csr()
        # cuDSS requires int32 index buffers.
        return torch.sparse_csr_tensor(
            csr.crow_indices().to(dtype=torch.int32),
            csr.col_indices().to(dtype=torch.int32),
            csr.values(),
            size=csr.size(),
            device=device,
        )

    n = 8
    # Starting from Python 3.14, sys.getrefcount() behavior changes due to
    # LOAD_FAST_BORROW, a bytecode optimization that skips the refcount
    # increment when the compiler's lifetime analysis can prove the local
    # variable outlives the stack reference.  Whether the optimization
    # applies can vary between load sites depending on code structure.
    # Pre-assigning a and b here (rather than only inside the if/elif
    # branches) ensures consistent behavior across all sys.getrefcount()
    # calls.
    # See https://github.com/python/cpython/issues/130704 for example.
    a = None
    b = None

    if batching_mode == "no_batch":
        a = _make_sparse_csr(n)
        b = torch.ones(n, dtype=torch.float64, device="cuda")

    elif batching_mode == "explicit_batch":
        batch_size = 3
        a = [_make_sparse_csr(n) for _ in range(batch_size)]
        b = [torch.ones(n, dtype=torch.float64, device="cuda") for _ in range(batch_size)]

    elif batching_mode == "implicit_batch":
        batch_size = 3
        dense_batch = torch.rand(batch_size, n, n, dtype=torch.float64, device="cuda")
        for i in range(batch_size):
            dense_batch[i] += 2.0 * torch.eye(n, dtype=torch.float64, device="cuda")
        csr_batch = dense_batch.to_sparse_csr()
        # cuDSS requires int32 index buffers for batched operations.
        a = torch.sparse_csr_tensor(
            csr_batch.crow_indices().to(dtype=torch.int32),
            csr_batch.col_indices().to(dtype=torch.int32),
            csr_batch.values(),
            size=csr_batch.size(),
            device="cuda",
        )
        # RHS must be col-major: create as (batch, k, n) then swap to (batch, n, k).
        b = torch.ones(batch_size, 1, n, dtype=torch.float64, device="cuda").swapaxes(-2, -1)

    if isinstance(a, list):
        initial_refcounts_a = [sys.getrefcount(ai) for ai in a]
        initial_refcounts_b = [sys.getrefcount(bi) for bi in b]
    else:
        initial_refcounts_a = sys.getrefcount(a)
        initial_refcounts_b = sys.getrefcount(b)

    solver = nvmath.sparse.advanced.DirectSolver(a, b)
    if use_plan_solve:
        solver.plan()
        solver.factorize()
        _ = solver.solve()
        torch.cuda.current_stream().synchronize()

    solver.release_operands()

    # Verify refcounts
    if isinstance(a, list):
        final_refcounts_a = [sys.getrefcount(ai) for ai in a]
        final_refcounts_b = [sys.getrefcount(bi) for bi in b]
        for i in range(len(a)):
            assert final_refcounts_a[i] == initial_refcounts_a[i]
        for i in range(len(b)):
            assert final_refcounts_b[i] == initial_refcounts_b[i]
    else:
        assert sys.getrefcount(a) == initial_refcounts_a
        assert sys.getrefcount(b) == initial_refcounts_b

    solver.free()


@pytest.mark.skipif(HAS_CUPY_SPARSE is False, reason="cupy.sparse is not available")
def test_execute_after_release_operands_raises():
    """
    Test that calling plan/factorize/solve after release_operands() raises a RuntimeError.
    """
    n = 8
    a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
    a += sp.diags([2.0] * n, format="csr", dtype="float64")
    b = cp.ones((n,), dtype="float64")

    solver = nvmath.sparse.advanced.DirectSolver(a, b)
    solver.plan()
    solver.factorize()
    solver.release_operands()

    with pytest.raises(RuntimeError, match="cannot be performed after the operands have been released"):
        solver.plan()

    solver.free()


@pytest.mark.skipif(HAS_CUPY_SPARSE is False, reason="cupy.sparse is not available")
def test_reset_operands_after_release_works():
    """
    Test that calling reset_operands() after release_operands() works correctly.
    """
    n = 8
    a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
    a += sp.diags([2.0] * n, format="csr", dtype="float64")
    b = cp.ones((n,), dtype="float64")

    solver = nvmath.sparse.advanced.DirectSolver(a, b)
    solver.plan()
    solver.factorize()
    _ = solver.solve()

    solver.release_operands()

    # Create new operands with SAME sparsity structure (same nnz)
    # by modifying the values of the original matrix
    a_new = a.copy()
    a_new.data[:] = cp.random.rand(a_new.data.size)  # Change values only
    b_new = cp.random.rand(n).astype("float64")

    # Reset with new operands (must provide both after release)
    solver.reset_operands(a=a_new, b=b_new)
    # Need to re-plan because reset_operands invalidated the plan
    solver.plan()
    # Then refactorize because values changed
    solver.factorize()
    x2 = solver.solve()

    # Verify the result is correct for the new operands
    a_new_dense = cp.asnumpy(a_new.toarray())  # Convert to CPU
    b_new_cpu = cp.asnumpy(b_new)
    x2_cpu = cp.asnumpy(x2)
    expected = np.linalg.solve(a_new_dense, b_new_cpu)
    np.testing.assert_allclose(x2_cpu, expected, rtol=1e-5)

    solver.free()
