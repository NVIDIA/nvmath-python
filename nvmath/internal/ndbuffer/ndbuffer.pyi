# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import _cython_3_1_2
from _typeshed import Incomplete
from typing import Any

CPU_DEVICE_ID: int
__pyx_capi__: dict
__test__: dict
copy_into: _cython_3_1_2.cython_function_or_method
empty: _cython_3_1_2.cython_function_or_method
empty_like: _cython_3_1_2.cython_function_or_method
reshaped_view: _cython_3_1_2.cython_function_or_method
wrap_external: _cython_3_1_2.cython_function_or_method

class NDBuffer:
    data: Incomplete
    data_ptr: Incomplete
    device: Incomplete
    device_id: Incomplete
    dtype_name: Incomplete
    itemsize: Incomplete
    ndim: Incomplete
    shape: Incomplete
    size: Incomplete
    size_in_bytes: Incomplete
    strides: Incomplete
    strides_in_bytes: Incomplete
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def __reduce__(self): ...
    def __reduce_cython__(self) -> Any: ...
    def __setstate_cython__(self, __pyx_state) -> Any: ...
