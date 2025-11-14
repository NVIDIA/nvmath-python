# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "Complex",
    "Vector",
    "complex32",
    "complex64",
    "complex128",
    "half2",
    "half4",
    "np_float16x2",
    "np_float16x4",
    "REAL_NP_TYPES",
    "INT_NP_TYPES",
]

import numpy as np
import warnings

from ._deprecated import deprecated

np_float16x2 = np.dtype([("x", np.float16), ("y", np.float16)], align=True)
np_float16x4 = np.dtype([("x", np.float16), ("y", np.float16), ("z", np.float16), ("w", np.float16)], align=True)

REAL_NP_TYPES: list = [np.float16, np.float32, np.float64]
INT_NP_TYPES: list = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]

_alignment_warning_msg = (
    "You are using the host counterpart of a dtype that is under-aligned "
    "compared to what the device function expects. In most cases this will "
    "work, since device memory is typically allocated with at least 256-byte "
    "alignment. For details, please review the alignment guidelines at "
    "https://nvidia.com/docs/nvmath/alignment"
)


class Complex:
    """
    Complex type that can be used to represent complex numbers both
    on host and device side. Numpy does not provide a built-in complex type with
    16-bit real and imaginary parts, so we define our own dtype for that case.
    For 32-bit and 64-bit complex numbers, we can use the built-in numpy dtypes.
    However on device side we expect those types to be aligned to the full size
    of the complex type, so the array defined on host and device side will have
    different type and alignment. :py:const:`np_float16x2`,
    :py:const:`numpy.dtype(numpy.complex64)` and
    :py:const:`numpy.dtype(numpy.complex128)` are the host side dtypes and
    :py:const:`float16x2_type`, :py:const:`float32x2_type` and
    :py:const:`float64x2_type` are the device side types.
    """

    def __init__(self, real_dtype):
        self._real_dtype = real_dtype

    @property
    def real_dtype(self):
        return self._real_dtype

    @property
    def dtype(self):
        warnings.warn(_alignment_warning_msg, UserWarning, stacklevel=2)
        if self._real_dtype == np.float16:
            return np_float16x2
        elif self._real_dtype == np.float32:
            return np.dtype(np.complex64)
        assert self._real_dtype == np.float64
        return np.dtype(np.complex128)

    @property
    def _numba_type(self):
        from .vector_types_numba import float16x2_type, float32x2_type, float64x2_type

        if self.real_dtype == np.float16:
            return float16x2_type
        if self.real_dtype == np.float32:
            return float32x2_type
        assert self.real_dtype == np.float64
        return float64x2_type

    @property
    @deprecated("This is a numba fallback behavior and will be removed in future releases, please use numba types directly")
    def make(self):
        return self._numba_type.make


complex32 = Complex(np.float16)
complex64 = Complex(np.float32)
complex128 = Complex(np.float64)


class Vector:
    """
    Vector type that can be used to represent vector numbers both
    on host and device side. Host side representation uses numpy structured
    dtypes to represent the vector components, while device side representation
    uses custom numba types. This difference is necessary because device
    functions expect alignment of the vector types to be the same as the size of
    the vector, which is not the case for numpy structured dtypes.
    :py:const:`np_float16x2` and :py:const:`np_float16x4` are the host side
    dtypes and :py:const:`float16x2_type` and :py:const:`float16x4_type` are the
    device side types.
    """

    def __init__(self, real_dtype, size):
        if size not in (2, 4):
            raise ValueError(f"Unsupported vector size {size}, only 2 and 4 are supported")
        if real_dtype != np.float16:
            raise ValueError(f"Unsupported vector real dtype {real_dtype}, only float16 is supported")
        self._real_dtype = real_dtype
        self._size = size

    @property
    def real_dtype(self):
        return self._real_dtype

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        warnings.warn(_alignment_warning_msg, UserWarning, stacklevel=2)
        if self._size == 2:
            return np_float16x2
        assert self._size == 4
        return np_float16x4

    @property
    def _numba_type(self):
        from .vector_types_numba import float16x2_type, float16x4_type

        if self._size == 2:
            return float16x2_type
        assert self._size == 4
        return float16x4_type

    @property
    @deprecated("This is a numba fallback behavior and will be removed in future releases, please use numba types directly")
    def make(self):
        return self._numba_type.make


half2 = Vector(np.float16, 2)
half4 = Vector(np.float16, 4)
