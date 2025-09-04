# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import nvmath
from .common_axes import (
    Framework,
    ExecBackend,
    MemBackend,
    DType,
    ShapeKind,
    Direction,
    OptFftType,
)
import cuda.core.experimental as ccx


framework_backend_support = {
    Framework.cupy: [MemBackend.cuda],
    Framework.numpy: [MemBackend.cpu],
    Framework.torch: [MemBackend.cpu, MemBackend.cuda],
}

exec_backend_type_support = {
    ExecBackend.fftw: [
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    ExecBackend.cufft: [
        DType.float16,
        DType.float32,
        DType.float64,
        DType.complex32,
        DType.complex64,
        DType.complex128,
    ],
}

framework_type_support = {
    Framework.cupy: [
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    Framework.numpy: [
        DType.float32,
        DType.float64,
        DType.complex64,
        DType.complex128,
    ],
    Framework.torch: [
        DType.float16,
        DType.float32,
        DType.float64,
        DType.complex32,
        DType.complex64,
        DType.complex128,
    ],
}

framework_exec_type_support = {
    framework: {
        exec_backend: [dtype for dtype in framework_type_support[framework] if dtype in exec_backend_type_support[exec_backend]]
        for exec_backend in ExecBackend
    }
    for framework in Framework
}


cufft_type_shape_support = {
    # Note, there is an undocumented partial support for prime shapes above 128
    DType.float16: [ShapeKind.pow2],
    DType.float32: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
    DType.float64: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
    # Note, there is an undocumented partial support for prime shapes above 128
    DType.complex32: [ShapeKind.pow2],
    DType.complex64: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
    DType.complex128: [
        ShapeKind.pow2,
        ShapeKind.pow2357,
        ShapeKind.prime,
        ShapeKind.random,
    ],
}

fftw_type_shape_support = {dtype: list(ShapeKind) for dtype in exec_backend_type_support[ExecBackend.fftw]}


type_shape_support = {
    ExecBackend.fftw: fftw_type_shape_support,
    ExecBackend.cufft: cufft_type_shape_support,
}


opt_fft_type_direction_support = {
    OptFftType.c2c: [Direction.forward, Direction.inverse],
    OptFftType.r2c: [Direction.forward],
    OptFftType.c2r: [Direction.inverse],
}

opt_fft_type_input_type_support = {
    OptFftType.c2c: [DType.complex32, DType.complex64, DType.complex128],
    OptFftType.r2c: [DType.float16, DType.float32, DType.float64],
    OptFftType.c2r: [DType.complex32, DType.complex64, DType.complex128],
}

inplace_opt_ftt_type_support = {True: [OptFftType.c2c], False: list(OptFftType)}

lto_callback_supperted_types = [DType.float32, DType.float64, DType.complex64, DType.complex128]


class _BackendSupport:
    @functools.cached_property
    def exec(self) -> list[ExecBackend]:
        return self.backends[0]

    @functools.cached_property
    def mem(self) -> list[MemBackend]:
        return self.backends[1]

    @functools.cached_property
    def framework_mem(self) -> dict[Framework, list[MemBackend]]:
        return {framework: [b for b in self.mem if b in framework_backend_support[framework]] for framework in Framework}

    def __call__(self):
        return self.backends

    @functools.cached_property
    def backends(self) -> tuple[list[ExecBackend], list[MemBackend]]:
        import platform
        import sys
        from nvmath.fft._exec_utils import _check_init_fftw

        machine = platform.machine()
        x86 = "x86_64"
        aarch = "aarch64"
        exec_backends, memory_backends = [], [MemBackend.cpu]

        if machine == aarch and sys.platform.startswith("linux"):
            exec_backends.append(ExecBackend.fftw)
        else:
            assert not sys.platform.startswith("linux") or machine == x86
            try:
                _check_init_fftw()
            except RuntimeError as e:
                if "The FFT CPU execution is not available" not in str(e):
                    raise
            else:
                exec_backends.append(ExecBackend.fftw)

        try:
            nvmath.bindings.cufft.get_version()
            exec_backends.append(ExecBackend.cufft)
            memory_backends.append(MemBackend.cuda)
        except nvmath.bindings._internal.utils.NotSupportedError as e:
            if "CUDA driver is not found" not in str(e):
                raise

        return exec_backends, memory_backends


supported_backends = _BackendSupport()


def multi_gpu_only(fn):
    import pytest

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        dev_count = ccx.system.num_devices
        if dev_count < 2:
            pytest.skip(f"Test requires at least two gpus, got {dev_count}")
        else:
            return fn(*args, **kwargs)

    return inner
