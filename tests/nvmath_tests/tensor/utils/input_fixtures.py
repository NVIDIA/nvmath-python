# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath_tests.helpers import get_custom_stream as _get_custom_stream
from nvmath_tests.helpers import get_random_input_data as _get_random_input_data

from .axes_utils import get_framework_dtype
from .common_axes import DType, Framework, MemBackend


def get_random_input_data(
    framework: Framework,
    shape: int | tuple[int],
    dtype: DType,
    mem_backend: MemBackend,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id=None,
):
    """Generate random input data for tensor tests.

    This is a wrapper around the common get_random_input_data function
    that provides the module-specific utility functions.
    """
    return _get_random_input_data(
        framework,
        shape,
        dtype,
        mem_backend,
        get_framework_dtype,
        lo=lo,
        hi=hi,
        device_id=device_id,
    )


def get_custom_stream(framework: Framework, device_id=None, is_numpy_stream_oriented=False):
    """Get a custom stream for the specified framework.

    This is a wrapper around the common get_custom_stream function.
    """
    return _get_custom_stream(framework, device_id, is_numpy_stream_oriented)
