# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from ast import literal_eval

import numpy as np
import pytest

import nvmath
from nvmath.bindings import cufft as cufft_bindings

from .utils.common_axes import (
    DType,
    ExecBackend,
    Framework,
)
from .utils.input_fixtures import (
    get_random_input_data,
    init_assert_exec_backend_specified,
)
from .utils.support_matrix import supported_backends

# DO NOT REMOVE, this call creates a fixture that enforces
# specifying execution option to the FFT calls in tests
# defined in this file
assert_exec_backend_specified = init_assert_exec_backend_specified()


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "technique"),
    [
        (framework, exec_backend, mem_backend, technique)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.fftw]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for technique in ["default", "refined"]
    ],
)
def test_estimate_workspace_size_cpu(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    technique,
):
    """For CPU keys, workspace should always be 0."""
    shape = (128,)
    dtype = DType.complex64
    a = get_random_input_data(framework, shape, dtype, mem_backend)
    key = nvmath.fft.FFT.create_key(a, axes=(0,), execution=exec_backend.nvname)
    estimated = nvmath.fft.estimate_workspace_size(key, technique=technique)
    assert estimated == 0, f"Expected 0 for CPU execution, got {estimated}"


@pytest.mark.parametrize(
    ("framework", "exec_backend", "mem_backend", "shape", "axes", "technique"),
    [
        (framework, exec_backend, mem_backend, shape, repr(axes), technique)
        for framework in Framework.enabled()
        for exec_backend in [ExecBackend.cufft]
        if exec_backend in supported_backends.exec
        for mem_backend in supported_backends.framework_mem[framework]
        for shape, axes in [
            ((4099,), (0,)),
            ((7919,), (0,)),
            ((10007,), (0,)),
            ((509, 521), (0,)),
            ((509, 521), (1,)),
            ((509, 521), (0, 1)),
            ((1013, 1019), (0, 1)),
            ((8, 2039, 1021), (1, 2)),
            ((4, 4099, 509), (1, 2)),
        ]
        for technique in ["default", "refined"]
    ],
)
def test_estimate_workspace_size_cuda(
    seeder,
    framework,
    exec_backend,
    mem_backend,
    shape,
    axes,
    technique,
):
    axes = literal_eval(axes)
    dtype = DType.complex64

    a = get_random_input_data(framework, shape, dtype, mem_backend)
    key = nvmath.fft.FFT.create_key(a, axes=axes, execution=exec_backend.nvname)

    if technique == "refined":
        handle = cufft_bindings.create()
        try:
            estimated = nvmath.fft.estimate_workspace_size(
                key,
                technique=technique,
                handle=handle,
            )
        finally:
            cufft_bindings.destroy(handle)
    else:
        estimated = nvmath.fft.estimate_workspace_size(
            key,
            technique=technique,
        )

    assert isinstance(estimated, int), f"Expected int, got {type(estimated)}"
    assert estimated >= 0, f"Expected non-negative, got {estimated}"

    with nvmath.fft.FFT(a, axes=axes, execution=exec_backend.nvname) as f:
        f.plan()
        actual = f.workspace_size
    assert estimated >= actual, (
        f"Estimated workspace ({estimated}) is smaller than actual "
        f"({actual}) for shape={shape}, axes={axes}, technique={technique}"
    )


@pytest.mark.parametrize(
    ("exec_backend", "technique"),
    [(exec_backend, technique) for exec_backend in supported_backends.exec for technique in ["default", "refined"]],
)
def test_estimate_workspace_size_consistent_across_calls(exec_backend, technique):
    """Calling the function twice with the same key and technique returns the same value."""
    shape = (128, 64)
    a = np.random.randn(*shape).astype(np.complex64)
    key = nvmath.fft.FFT.create_key(a, axes=(0, 1), execution=exec_backend.nvname)

    if technique == "refined" and exec_backend == ExecBackend.cufft:
        handle = cufft_bindings.create()
        try:
            r1 = nvmath.fft.estimate_workspace_size(key, technique=technique, handle=handle)
            r2 = nvmath.fft.estimate_workspace_size(key, technique=technique, handle=handle)
        finally:
            cufft_bindings.destroy(handle)
    else:
        r1 = nvmath.fft.estimate_workspace_size(key, technique=technique)
        r2 = nvmath.fft.estimate_workspace_size(key, technique=technique)

    assert r1 == r2, f"Non-deterministic: first={r1}, second={r2}"


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("not a tuple", id="not_a_tuple"),
        pytest.param((1, 2), id="2_tuple"),
        pytest.param((1, 2, 3, 4), id="4_tuple"),
        pytest.param(("not a tuple", None, ("cuda",)), id="plan_args_not_tuple"),
        pytest.param(((1,) * 11, None, ("cuda",)), id="plan_args_len_11"),
        pytest.param(((1,) * 12, None, ()), id="execution_tuple_empty"),
        pytest.param(((1,) * 12, None, "cuda"), id="execution_tuple_not_tuple"),
    ],
)
def test_estimate_workspace_size_invalid_key(key):
    with pytest.raises(ValueError):
        nvmath.fft.estimate_workspace_size(key)


@pytest.mark.parametrize("exec_backend", [eb for eb in [ExecBackend.cufft] if eb in supported_backends.exec])
def test_estimate_workspace_size_refined_requires_handle(exec_backend):
    """technique='refined' with a CUDA key but no handle should raise ValueError."""
    shape = (64,)
    a = np.random.randn(*shape).astype(np.complex64)
    key = nvmath.fft.FFT.create_key(a, axes=(0,), execution=exec_backend.nvname)
    with pytest.raises(ValueError, match="handle is required"):
        nvmath.fft.estimate_workspace_size(key, technique="refined")
