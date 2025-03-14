# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import pytest

import nvmath

from .utils.common_axes import Framework, ExecBackend
from .utils.axes_utils import is_complex, is_half, get_fft_dtype
from .utils.support_matrix import framework_exec_type_support, supported_backends
from .utils.input_fixtures import get_random_input_data
from .utils.check_helpers import (
    get_fft_ref,
    get_scaled,
    assert_norm_close,
    assert_array_type,
)


@pytest.mark.parametrize(
    (
        "framework",
        "exec_backend",
        "mem_backend",
        "dtype",
    ),
    [
        (
            framework,
            exec_backend,
            exec_backend.mem,
            dtype,
        )
        for framework in Framework.enabled()
        for exec_backend in supported_backends.exec
        if exec_backend.mem in supported_backends.framework_mem[framework]
        for dtype in framework_exec_type_support[framework][exec_backend]
        if not is_half(dtype)
    ],
)
def test_default_backend(monkeypatch, framework, exec_backend, mem_backend, dtype):
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("The fft bindings are not build for windows")

    def fail_fn(fn):
        def wrapper(*args, **kwargs):
            pytest.fail(f"The FFT should not call {fn}")

        return wrapper

    forbidden_module = nvmath.bindings.cufft if exec_backend == ExecBackend.fftw else nvmath.bindings.nvpl.fft

    for el_name in dir(forbidden_module):
        el = getattr(forbidden_module, el_name)
        if inspect.isroutine(el):
            monkeypatch.setattr(forbidden_module, el_name, fail_fn(el))

    shape = (45, 13)
    axes = (0,)
    signal = get_random_input_data(framework, shape, dtype, mem_backend, seed=55)
    fft_fn = nvmath.fft.fft if is_complex(dtype) else nvmath.fft.rfft
    out = fft_fn(signal, axes=axes)
    assert_array_type(out, framework, mem_backend, get_fft_dtype(dtype))
    assert_norm_close(out, get_fft_ref(signal, axes))
    ifft_fn = nvmath.fft.ifft if is_complex(dtype) else nvmath.fft.irfft
    iout = ifft_fn(out, axes=axes, options={"last_axis_parity": "odd"})
    assert_array_type(iout, framework, mem_backend, dtype)
    assert_norm_close(iout, get_scaled(signal, shape[0]))
