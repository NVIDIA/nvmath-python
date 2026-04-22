# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface class to encapsulate low-level calls to get algorithm capabilities information.
"""

__all__ = ["AlgoCapInterface"]

import numpy as np

from nvmath._internal.attribute_ifc_factory import make_cublas_attribute_interface
from nvmath.bindings import cublasLt as cublaslt

from .enum_to_tuples import ENUM_TO_MATMUL_STAGE, ENUM_TO_MATMUL_TILE

CapEnum = cublaslt.MatmulAlgoCapAttribute

# Create a base class, _AlgoCapBase, such that each enum member in
# MatmulAlgoCapAttribute (except TILE_IDS and STAGES_IDS) is exposed
# as a read-only lowercase property. For example, if the class instance
# is stored as ``cap_ifc``, then the enum member ``ATOMIC_SYNC``
# becomes the property ``cap_ifc.atomic_sync``.
#
# The C binding expects a raw pointer to the algo sub-buffer, not the
# handle dict. _get_attribute bridges that gap by extracting
# `handle["algo"].ctypes.data` before forwarding the call.
_raw_get = cublaslt.matmul_algo_cap_get_attribute


def _get_attribute(handle, enum_value, buf_addr, buf_size, size_written_addr):
    _raw_get(handle["algo"].ctypes.data, enum_value, buf_addr, buf_size, size_written_addr)


# TILE_IDS and STAGES_IDS are excluded because they are variable-length
# arrays that require a two-phase size-then-read query. They are implemented
# as extra properties on the AlgoCapInterface subclass below.
_AlgoCapBase = make_cublas_attribute_interface(
    class_module=__name__,
    class_name="_AlgoCapBase",
    attribute_enum=CapEnum,
    get_attribute_dtype_fn=cublaslt.get_matmul_algo_cap_attribute_dtype,
    get_attribute_fn=_get_attribute,
    exclude={"TILE_IDS", "STAGES_IDS"},
)


class AlgoCapInterface(_AlgoCapBase):  # type: ignore[valid-type, misc]
    """The factory base class is extended with two extra properties:
    ``tile_ids`` and ``stages_ids`` are added here because they return
    variable-length arrays via a two-phase size-then-read query."""

    def _get_array_attribute(self, name, enum_to_value_map):
        dtype = cublaslt.get_matmul_algo_cap_attribute_dtype(CapEnum[name])
        size_written = np.zeros((1,), dtype=np.uint64)

        _raw_get(self._handle["algo"].ctypes.data, CapEnum[name].value, 0, 0, size_written.ctypes.data)

        size = size_written[0]
        if size == 0:
            return ()

        num = int(size // dtype().itemsize)
        attribute_buffer = np.zeros((num,), dtype=dtype)
        _raw_get(
            self._handle["algo"].ctypes.data,
            CapEnum[name].value,
            attribute_buffer.ctypes.data,
            size,
            size_written.ctypes.data,
        )

        return tuple(enum_to_value_map[e] for e in attribute_buffer)

    @property
    def tile_ids(self):
        return self._get_array_attribute("TILE_IDS", ENUM_TO_MATMUL_TILE)

    @property
    def stages_ids(self):
        return self._get_array_attribute("STAGES_IDS", ENUM_TO_MATMUL_STAGE)
