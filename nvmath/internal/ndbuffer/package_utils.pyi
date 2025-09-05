# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import _cython_3_1_2
from typing import Any, ClassVar

__pyx_capi__: dict
__reduce_cython__: _cython_3_1_2.cython_function_or_method
__setstate_cython__: _cython_3_1_2.cython_function_or_method
__test__: dict
empty_numpy_like: _cython_3_1_2.cython_function_or_method
is_c_or_f: _cython_3_1_2.cython_function_or_method
wrap_cupy_array: _cython_3_1_2.cython_function_or_method
wrap_numpy_array: _cython_3_1_2.cython_function_or_method

class _DType2NameCache:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def get(self, dtype) -> str: ...
    def __reduce__(self): ...

class _Name2DTypeCache:
    __pyx_vtable__: ClassVar[PyCapsule] = ...
    @classmethod
    def __init__(cls, *args, **kwargs) -> None: ...
    def get(self, name) -> Any: ...
    def __reduce__(self): ...
