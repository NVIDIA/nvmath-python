# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os

from nvmath.bindings import cufft  # type: ignore
from nvmath.bindings._internal import utils as _bindings_utils  # type: ignore

from nvmath.fft import _configuration


IS_EXEC_GPU_AVAILABLE = False
IS_EXEC_CPU_AVAILABLE = False


def _check_init_cufft():
    global IS_EXEC_GPU_AVAILABLE
    if not IS_EXEC_GPU_AVAILABLE:
        try:
            cufft.get_version()
        except (
            _bindings_utils.FunctionNotFoundError,
            _bindings_utils.NotSupportedError,
            RuntimeError,
        ) as e:
            raise RuntimeError(
                "The FFT CUDA execution is not available. "
                "Please check if CUDA toolkit and cuFFT are installed and visible to nvmath."
            ) from e

        try:
            import cupy  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Currently, the FFT CUDA execution requires cupy. Please make sure cupy is installed.") from e

        IS_EXEC_GPU_AVAILABLE = True


def _check_init_fftw():
    global IS_EXEC_CPU_AVAILABLE
    if not IS_EXEC_CPU_AVAILABLE:
        try:
            from nvmath.bindings.nvpl import fft as _fftw
            from nvmath.bindings.nvpl._internal import fft as _internal_fftw
        except ImportError as e:
            import sys

            if sys.platform.startswith("linux"):
                raise
            else:
                raise RuntimeError(
                    "The FFT CPU execution is not available. Currently, it is supported only "
                    "with Linux. To process the CPU samples on GPU, please specify `execution='cuda'` "
                    "when calling fft."
                ) from e

        lib_nvpl = "libnvpl_fftw.so"
        env_lib_name = os.environ.get("NVMATH_FFT_CPU_LIBRARY")

        if env_lib_name:
            _internal_fftw._set_lib_so_names((env_lib_name,))
        try:
            _internal_fftw._inspect_function_pointers()
            loaded_lib_name = _internal_fftw._get_current_lib_so_name()
        except (
            _bindings_utils.FunctionNotFoundError,
            _bindings_utils.NotSupportedError,
            RuntimeError,
        ) as e:
            raise RuntimeError(
                "The FFT CPU execution is not available, because no FFTW-compatible "
                "library was found. Please make sure to install optional dependency required "
                "for FFT execution (NVPL on aarch64 platforms or MKL on x86_64 platforms). "
                "Users can use other FFTW3-compatible libraries. In order to do that, please "
                "specify the library name as "
                "`NVMATH_FFT_CPU_LIBRARY=some_fftw3_compatible_lib.so` and make sure "
                "the library can be found within current `LD_LIBRARY_PATH`"
            ) from e

        if env_lib_name:
            assert env_lib_name == loaded_lib_name

        if not loaded_lib_name.startswith(lib_nvpl):
            _fftw.init_threads_float()
            _fftw.init_threads_double()

        IS_EXEC_CPU_AVAILABLE = True


def _cross_setup_execution_and_options(
    options: _configuration.FFTOptions,
    execution: _configuration.ExecutionCUDA | _configuration.ExecutionCPU,
):
    if execution.name == "cpu":
        _check_init_fftw()
        if options.device_id is not None:
            raise ValueError("The 'device_id' is not a valid option when 'execution' is specified to be 'cpu'.")
        if execution.num_threads is None:
            # `sched_getaffinity` is not supported on Windows, it must be adjusted
            # once the support for Windows is enabled
            execution.num_threads = len(os.sched_getaffinity(0))
        if not isinstance(execution.num_threads, int) or execution.num_threads <= 0:
            raise ValueError("The 'num_threads' must be a positive integer")
    else:
        assert execution.name == "cuda"
        _check_init_cufft()
        if execution.device_id is None:
            execution.device_id = options.device_id or 0
        elif options.device_id is not None and execution.device_id != options.device_id:
            raise ValueError(
                f"Got conflicting 'device_id' passed in 'execution' ({execution.device_id}) "
                f"and 'options' ({options.device_id}). It should be passed as 'execution' "
                f"configuration only."
            )
    return options, execution
