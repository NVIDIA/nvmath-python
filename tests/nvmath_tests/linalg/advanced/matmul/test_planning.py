# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests checks basic properties of separated planning.
"""

from nvmath.bindings import cublasLt as cublaslt
from nvmath.linalg.advanced import Matmul, MatmulPlanPreferences
import numpy as np
import pytest

from .utils import sample_matrix, allow_cublas_unsupported, assert_tensors_equal

try:
    import cupy
except ModuleNotFoundError:
    pytest.skip("cupy required for matmul tests", allow_module_level=True)


@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize("dtype", ("float32", "complex64", "float64", "complex128"))
@pytest.mark.parametrize(
    "n,m,k",
    (
        (2, 3, 4),
        (50, 51, 52),
        (64, 32, 32),
        (200, 100, 50),
    ),
)
@pytest.mark.parametrize("max_waves_count", (0.99, 1.0))
@pytest.mark.parametrize("iterations", (1, 5))
@pytest.mark.parametrize("prune", (1, 5, 9))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_autotuning(
    framework,
    dtype,
    n,
    m,
    k,
    max_waves_count,
    iterations,
    prune,
    use_cuda,
):
    a = sample_matrix(framework, dtype, (n, k), use_cuda)
    b = sample_matrix(framework, dtype, (k, m), use_cuda)
    c = sample_matrix(framework, dtype, (n, m), use_cuda)
    mm = Matmul(a, b, beta=0.7, c=c)
    with allow_cublas_unsupported(
        allow_invalid_value=False,
        message=(
            f"Unsupported configuration: {framework}-{dtype}-{n}-{m}-{k}-{max_waves_count}-{iterations}-{prune}-{use_cuda}."
        ),
    ):
        mm.plan(preferences=MatmulPlanPreferences(limit=9, max_waves_count=max_waves_count))
    num_algorithms = len(mm.algorithms)
    mm.autotune(iterations=iterations, prune=prune)
    assert len(mm.algorithms) == min(prune, num_algorithms)
    assert_tensors_equal(mm.execute(), a @ b + c * 0.7)


@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
@pytest.mark.parametrize(
    "n,m,k",
    (
        (1, 1, 1),
        (64, 32, 96),
    ),
)
@pytest.mark.parametrize("max_waves_count", (0.0, 1.0, 2.0))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_plan(framework, dtype, n, m, k, max_waves_count, use_cuda):
    a = sample_matrix(framework, dtype, (n, k), use_cuda)
    b = sample_matrix(framework, dtype, (k, m), use_cuda)
    c = sample_matrix(framework, dtype, (n, m), use_cuda)
    mm = Matmul(a, b, beta=0.7, c=c)
    mm.plan(preferences=MatmulPlanPreferences(limit=6, max_waves_count=max_waves_count))
    assert_tensors_equal(mm.execute(), a @ b + c * 0.7)


def test_multiple_executions():
    """
    Tests if single Matmul object can be reused.
    """
    a = cupy.zeros((10, 10))
    b = cupy.zeros((10, 10))
    mm = Matmul(a, b)
    mm.plan()
    for _ in range(5):
        cupy.copyto(a, cupy.random.rand(*a.shape))
        cupy.copyto(b, cupy.random.rand(*b.shape))
        result = mm.execute()
        assert_tensors_equal(result, a @ b)


def test_limit():
    """
    Tests if limiting the number of algorithms works as expected
    """
    a = cupy.zeros((10, 10))
    b = cupy.zeros((10, 10))
    mm = Matmul(a, b)
    mm.plan(preferences=MatmulPlanPreferences(limit=3))
    assert len(mm.algorithms) <= 3


def test_reduction_scheme():
    """
    Tests if one can specify reduction scheme
    """
    a = cupy.zeros((1000, 1000))
    b = cupy.zeros((1000, 1000))
    mm = Matmul(a, b)
    algos = mm.plan(preferences=MatmulPlanPreferences(reduction_scheme_mask=cublaslt.ReductionScheme.NONE, limit=64))
    assert not any(a.reduction_scheme for a in algos)


def test_capabilities():
    """
    Tests if one can modify algorithm capabilities
    """
    a = cupy.random.rand(1000, 1000, dtype=np.float32)
    b = cupy.random.rand(1000, 1000, dtype=np.float32)
    mm = Matmul(a, b)
    mm.plan()
    best = mm.algorithms[0]
    best.tile = best.capabilities.tile_ids[-1]
    with allow_cublas_unsupported(message=f"Unsupported tile: {best.tile}"):
        # The chosen tile size might not be supported on some platforms
        result = mm.execute()
        assert_tensors_equal(result, a @ b)


@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize("serialize", (True, False))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_algorithms(framework, serialize, use_cuda):
    a = b = sample_matrix(framework, "float32", (20, 20), use_cuda)
    mm = Matmul(a, b)
    algos = mm.plan(preferences=MatmulPlanPreferences(limit=10))
    if serialize:
        import pickle

        algos = pickle.loads(pickle.dumps(algos))
    c = d = sample_matrix(framework, "float32", (20, 20), use_cuda)

    # Test providing multiple algorithms
    mm2 = Matmul(c, d)
    mm2.plan(algorithms=algos)
    assert_tensors_equal(mm2.execute(), c @ d)

    # Test executing a specified algorithm
    mm3 = Matmul(c, d)
    mm3.plan(algorithms=algos)
    assert_tensors_equal(mm3.execute(algorithm=algos[0]), c @ d)


@pytest.mark.parametrize("value", (None, 0, "algo"))
def test_algorithms_invalid(value):
    a = b = sample_matrix("torch", "float32", (20, 20), True)
    mm = Matmul(a, b)
    with pytest.raises(AssertionError):
        mm.plan(algorithms=[value])


@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_algorithm_not_planned(framework, use_cuda):
    a = b = sample_matrix(framework, "float32", (20, 20), use_cuda)
    mm = Matmul(a, b)
    algos = mm.plan(preferences=MatmulPlanPreferences(limit=10))

    mm2 = Matmul(a, b)
    mm2.plan(algorithms=algos[1:])
    with pytest.raises(
        ValueError,
        match=r"Algorithm passed to execute\(\) has to be included in the plan\(\) algorithms",
    ):
        mm2.execute(algorithm=algos[0])

def test_algorithm_ids():
    a = cupy.zeros((10, 10))
    b = cupy.zeros((10, 10))
    with Matmul(a, b) as mm:
        assert len(mm.applicable_algorithm_ids(limit=4)) <= 4

def test_algo_attributes():
    '''
    Test Algorithm class setter/property
    '''
    m, n, k = 24, 24, 24
    a = cupy.random.rand(m, k)
    b = cupy.random.rand(k, n)

    with Matmul(a, b) as mm:
        algos = mm.plan()
        best = algos[0]

        # An attribute may not be supported in all cuBLASLt versions (INVALID_VALUE).

        message = "The attribute '{attr}' is not supported in this version."
        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='stages')
        ):
            if best.capabilities.stages_ids:
                best.stages = best.capabilities.stages_ids[-1]
                assert best.stages == best.capabilities.stages_ids[-1]

        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='split_k')
        ):
            best.split_k = 4
            assert best.split_k == 4

        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='reduction_scheme')
        ):
            best.reduction_scheme = best.capabilities.reduction_scheme_mask
            assert best.reduction_scheme == best.capabilities.reduction_scheme_mask

        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='cta_swizzling')
        ):
            best.cta_swizzling = True
            assert best.cta_swizzling == True

        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='custom_option')
        ):
            best.custom_option = 1
            assert best.custom_option == 1

        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='inner_shape')
        ):
            best.inner_shape = cublaslt.MatmulInnerShape.MMA884
            assert best.inner_shape == cublaslt.MatmulInnerShape.MMA884

        with allow_cublas_unsupported(
            allow_invalid_value=True,
            message=message.format(attr='cluster_shape')
        ):
            best.cluster_shape = (1,1,1)
            assert best.cluster_shape == (1,1,1)
