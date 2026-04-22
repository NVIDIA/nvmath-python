# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Get/set tests for cuDSS config properties.

Each test sets a config property via the setter, then reads it back via the getter
and asserts the value matches. This catches bugs where the getter returns a value
from the wrong internal buffer.

Every test also asserts that the chosen test value differs from the default, so
a broken getter that always returns the default will be caught.
"""

import pytest

from nvmath.bindings import cudss
from nvmath.sparse.advanced import DirectSolver
from nvmath.sparse.advanced.direct_solver import get_threading_lib

cp = pytest.importorskip("cupy")
sp = pytest.importorskip("cupyx.scipy.sparse")


PLAN_CONFIG_CASES = {
    "reordering_algorithm": (cudss.AlgType.ALG_2,),
    "matching_algorithm": (cudss.AlgType.ALG_2,),
    "pivot_type": (cudss.PivotType.PIVOT_ROW,),
    "pivot_threshold": (0.5,),
    "max_nnz": (13,),
    "use_matching": (1, 0),
    "nd_min_levels": (3,),
    "use_superpanels": (0, 1),
}

FACTORIZATION_CONFIG_CASES = {
    "factorization_algorithm": (cudss.AlgType.ALG_2,),
    "pivot_eps_algorithm": (cudss.AlgType.ALG_1,),
    "pivot_eps": (0.25,),
}

SOLUTION_CONFIG_CASES = {
    "solution_algorithm": (cudss.AlgType.ALG_1,),
    "ir_num_steps": (10,),
}

INTERNAL_CONFIG_CASES = {
    "hybrid_mode": (1,),
    "hybrid_device_memory_limit": (3 * 2**20,),
    "use_cuda_register_memory": (0,),
    "hybrid_execute_mode": (1,),
}

# host_nthreads requires a multithreading library (CUDSS_THREADING_LIB).
# Test it only when available; otherwise exclude from the completeness check.
_SKIP_PROPS: set[str] = set()
if get_threading_lib() is not None:
    PLAN_CONFIG_CASES["host_nthreads"] = (2,)
    INTERNAL_CONFIG_CASES["host_nthreads"] = (2,)
else:
    _SKIP_PROPS.add("host_nthreads")


def _test_impl(cfg, cases):
    # Discover all public properties on the config class so the test fails
    # immediately if a new property is added without a corresponding test entry.
    props = {name for name in dir(cfg) if not name.startswith("_")} - _SKIP_PROPS
    assert props == set(cases.keys()), (
        f"Property mismatch on {type(cfg).__name__}: untested={props - set(cases.keys())}, stale={set(cases.keys()) - props}"
    )
    for name, values in cases.items():
        for v in values:
            assert getattr(cfg, name) != v, f"{name}: test value {v!r} unexpectedly matches default"
            setattr(cfg, name, v)
            assert getattr(cfg, name) == v, f"{name}: roundtrip failed for {v!r}"


@pytest.fixture(scope="module")
def sample_problem():
    """Create a small diagonally-dominant CSR system suitable for all config tests."""
    n = 16
    a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
    a += sp.diags([2.0] * n, format="csr", dtype="float64")
    b = cp.ones((n, 1), order="F")
    return a, b


def test_plan_config_roundtrip(sample_problem):
    a, b = sample_problem
    with DirectSolver(a, b) as solver:
        _test_impl(solver.plan_config, PLAN_CONFIG_CASES)


def test_factorization_config_roundtrip(sample_problem):
    a, b = sample_problem
    with DirectSolver(a, b) as solver:
        _test_impl(solver.factorization_config, FACTORIZATION_CONFIG_CASES)


def test_solution_config_roundtrip(sample_problem):
    a, b = sample_problem
    with DirectSolver(a, b) as solver:
        _test_impl(solver.solution_config, SOLUTION_CONFIG_CASES)


def test_internal_config_roundtrip(sample_problem):
    """InternalConfig is not public API but its getters had bugs (bug 5942281)."""
    a, b = sample_problem
    with DirectSolver(a, b) as solver:
        _test_impl(solver._internal_config, INTERNAL_CONFIG_CASES)
