# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
import numpy as np

from nvmath.device.common_cuda import ISAVersion
from .types import np_float16x2, np_float16x4

from nvmath.bindings import mathdx

MATHDX_TYPES_TO_NP = {
    mathdx.CommondxValueType.R_16F: np.float16,
    mathdx.CommondxValueType.R_16F2: np_float16x2,
    mathdx.CommondxValueType.R_32F: np.float32,
    mathdx.CommondxValueType.R_64F: np.float64,
    mathdx.CommondxValueType.C_16F: np_float16x2,
    mathdx.CommondxValueType.C_16F2: np_float16x4,
    mathdx.CommondxValueType.C_32F: np.complex64,
    mathdx.CommondxValueType.C_64F: np.complex128,
    mathdx.CommondxValueType.R_8I: np.int8,
    mathdx.CommondxValueType.R_16I: np.int16,
    mathdx.CommondxValueType.R_32I: np.int32,
    mathdx.CommondxValueType.R_64I: np.int64,
    mathdx.CommondxValueType.R_8UI: np.uint8,
    mathdx.CommondxValueType.R_16UI: np.uint16,
    mathdx.CommondxValueType.R_32UI: np.uint32,
    mathdx.CommondxValueType.R_64UI: np.uint64,
}


NP_TYPES_TO_MATHDX_TYPES = {np_type: mathdx_type for mathdx_type, np_type in MATHDX_TYPES_TO_NP.items()}

NP_TYPES_TO_MATHDX_PRECISION = {
    np.float16: mathdx.CommondxPrecision.F16,
    np.float32: mathdx.CommondxPrecision.F32,
    np.float64: mathdx.CommondxPrecision.F64,
    np.int8: mathdx.CommondxPrecision.I8,
    np.int16: mathdx.CommondxPrecision.I16,
    np.int32: mathdx.CommondxPrecision.I32,
    np.int64: mathdx.CommondxPrecision.I64,
    np.uint8: mathdx.CommondxPrecision.UI8,
    np.uint16: mathdx.CommondxPrecision.UI16,
    np.uint32: mathdx.CommondxPrecision.UI32,
    np.uint64: mathdx.CommondxPrecision.UI64,
}

EXECUTION_STR_TO_MATHDX = {
    "Block": mathdx.CommondxExecution.BLOCK,
    "Thread": mathdx.CommondxExecution.THREAD,
}

NVARG_GEN_OPT_LTO = "-gen-opt-lto"


class DescriptorWrapper:
    """
    A smart pointer to a descriptor.

    Lives as long as we need the descriptor (due to lru cache usage). Once the
    descriptor is no longer needed/cached, it will trigger proper resource
    deallocation.
    """

    def __init__(self, descriptor, destructor):
        self.descriptor = descriptor
        self._destructor = destructor

    def __del__(self):
        if not self.descriptor:
            return

        self._destructor(self.descriptor)

        # Safety clean up
        self.descriptor = None
        self._destructor = None


def get_lto(code_descriptor: int) -> bytes:
    """Get lto binary from the mathdx common code descriptor."""
    lto_size = mathdx.commondx_get_code_ltoir_size(code_descriptor)

    lto_fn = bytearray(lto_size)
    mathdx.commondx_get_code_ltoir(code_descriptor, lto_size, lto_fn)
    return bytes(lto_fn)


def get_isa_version(code_descriptor: int) -> ISAVersion:
    """Parse isa version from the mathdx common code descriptor."""
    isa = mathdx.commondx_get_code_option_int64(code_descriptor, mathdx.CommondxOption.CODE_ISA)
    return ISAVersion.from_integer(isa)


def build_get_int_traits(get_trait_int64s: Callable[[int, int, int, int], int]):
    """Generate function that returns mathdx tuple of int traits."""

    def get_int_traits(handle: int, trait_type: int, size: int) -> tuple:
        int_buffer = np.zeros(size, np.int64)
        get_trait_int64s(handle, trait_type, size, int_buffer.ctypes.data)
        return tuple(map(int, int_buffer))

    return get_int_traits


def build_get_str_trait(
    get_trait_str_size: Callable[[int, int], int],
    get_trait_str: Callable[[int, int, int, bytearray], bytearray],
):
    """Generate function that returns mathdx string trait."""

    def get_str_trait(handle: int, trait_type: int) -> str:
        symbol_size = get_trait_str_size(handle, trait_type)

        symbol = bytearray(symbol_size)

        get_trait_str(handle, trait_type, symbol_size, symbol)

        # terminate trailing 0 (indicator of c-string end)
        symbol_str = symbol[:-1].decode()

        return symbol_str

    return get_str_trait
