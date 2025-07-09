# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy

from nvmath.device.common_backend import DescriptorWrapper
from nvmath.device.common_cuda import ComputeCapability
from .common import check_contains, check_in, check_not_in, check_code_type
from .common_backend import (
    NP_TYPES_TO_MATHDX_PRECISION,
    EXECUTION_STR_TO_MATHDX,
    NVARG_GEN_OPT_LTO,
    build_get_int_traits,
    build_get_str_trait,
)
from .types import REAL_NP_TYPES

from nvmath.bindings import mathdx


_FFT_TYPE_TO_MATHDX = {t.name.lower(): t for t in mathdx.CufftdxType}

_FFT_DIRECTION_TO_MATHDX = {d.name.lower(): d for d in mathdx.CufftdxDirection}

_FFT_COMPLEX_LAYOUT_TO_MATHDX = {cl.name.lower(): cl for cl in mathdx.CufftdxComplexLayout}

_FFT_REAL_MODE_TO_MATHDX = {m.name.lower(): m for m in mathdx.CufftdxRealMode}

_FFT_API_STR_TO_MATHDX = {
    "register_memory": mathdx.CufftdxApi.LMEM,
    "shared_memory": mathdx.CufftdxApi.SMEM,
}

_FFT_KNOB_TYPE_TO_MATHDX = {t.name.lower(): t for t in mathdx.CufftdxKnobType}


class CallableGetIntTraits(Protocol):
    def __call__(self, handle: int, trait_type: mathdx.CufftdxTraitType, size: int) -> tuple: ...


class CallableGetStrTrait(Protocol):
    def __call__(self, handle: int, trait_type: mathdx.CufftdxTraitType) -> str: ...


get_int_traits: CallableGetIntTraits = build_get_int_traits(mathdx.cufftdx_get_trait_int64s)

get_str_trait: CallableGetStrTrait = build_get_str_trait(mathdx.cufftdx_get_trait_str_size, mathdx.cufftdx_get_trait_str)


def get_int_trait(handle: int, trait_type: mathdx.CufftdxTraitType) -> int:
    return int(mathdx.cufftdx_get_trait_int64(handle, trait_type))


def get_data_type_trait(handle: int, trait_type: mathdx.CufftdxTraitType) -> mathdx.CommondxValueType:
    return mathdx.CommondxValueType(mathdx.cufftdx_get_trait_commondx_data_type(handle, trait_type))


def validate(
    size,
    precision,
    fft_type,
    execution,
    direction,
    ffts_per_block,
    elements_per_thread,
    real_fft_options,
    code_type,
):
    if size <= 0:
        raise ValueError(f"size must be > 0. Got {size}")
    check_in("precision", precision, REAL_NP_TYPES)
    check_in("fft_type", fft_type, ["c2c", "c2r", "r2c"])
    check_in("execution", execution, ["Block", "Thread"])
    if direction is not None:
        check_in("direction", direction, ["forward", "inverse"])
    if ffts_per_block in (None, "suggested"):
        pass
    else:
        if ffts_per_block <= 0:
            raise ValueError(
                f"ffts_per_block must be None, 'suggested' or a positive integer ; got ffts_per_block = {ffts_per_block}"
            )
    if elements_per_thread in (None, "suggested"):
        pass
    else:
        if elements_per_thread <= 0:
            raise ValueError(
                f"elements_per_thread must be None, 'suggested' or a positive integer ; "
                f"got elements_per_thread = {elements_per_thread}"
            )
    if real_fft_options is None:
        pass
    else:
        check_contains(real_fft_options, "complex_layout")
        check_contains(real_fft_options, "real_mode")
        check_in("real_fft_options['complex_layout']", real_fft_options["complex_layout"], ["natural", "packed", "full"])
        check_in("real_fft_options['real_mode']", real_fft_options["real_mode"], ["normal", "folded"])
    check_code_type(code_type)


def validate_execute_api(execution: str, execute_api: str | None):
    """
    Validate the execute_api argument.
    """
    if execution == "Block":
        check_in("execute_api", execute_api, list(_FFT_API_STR_TO_MATHDX.keys()) + [None])
    else:
        if execute_api is not None:
            raise ValueError(f"api may be set only for block execution ; got execution = {execution}")


@lru_cache
def generate_FFT(
    size,
    precision,
    fft_type,
    direction,
    code_type,
    execution,
    ffts_per_block,
    elements_per_thread,
    real_fft_options,
    execute_api=None,
):
    check_not_in("ffts_per_block", ffts_per_block, ["suggested"])
    check_not_in("elements_per_thread", elements_per_thread, ["suggested"])

    h = mathdx.cufftdx_create_descriptor()

    if execute_api is not None:
        mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.API, _FFT_API_STR_TO_MATHDX[execute_api])

    mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.SIZE, size)
    mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.PRECISION, NP_TYPES_TO_MATHDX_PRECISION[precision])
    mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.TYPE, _FFT_TYPE_TO_MATHDX[fft_type])
    mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.DIRECTION, _FFT_DIRECTION_TO_MATHDX[direction])

    if real_fft_options:
        real_fft_options = dict(real_fft_options)
        ll = real_fft_options["complex_layout"]
        mm = real_fft_options["real_mode"]
        mathdx.cufftdx_set_operator_int64s(
            h, mathdx.CufftdxOperatorType.REAL_FFT_OPTIONS, 2, [_FFT_COMPLEX_LAYOUT_TO_MATHDX[ll], _FFT_REAL_MODE_TO_MATHDX[mm]]
        )

    mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.EXECUTION, EXECUTION_STR_TO_MATHDX[execution])

    if code_type:
        mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.SM, code_type.cc.major * 100 + code_type.cc.minor * 10)

    if ffts_per_block:
        mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.FFTS_PER_BLOCK, ffts_per_block)

    if elements_per_thread:
        mathdx.cufftdx_set_operator_int64(h, mathdx.CufftdxOperatorType.ELEMENTS_PER_THREAD, elements_per_thread)

    return DescriptorWrapper(h, mathdx.cufftdx_destroy_descriptor)


@lru_cache
def generate_code(handle, version: ComputeCapability):
    code = mathdx.commondx_create_code()

    mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, version.integer)
    mathdx.commondx_set_code_option_str(code, mathdx.CommondxOption.EXTRA_NVTRC_ARGS, NVARG_GEN_OPT_LTO)
    mathdx.cufftdx_finalize_code(code, handle)

    return DescriptorWrapper(code, mathdx.commondx_destroy_code)


@lru_cache
def get_knobs(handle, knobs: Sequence[str]):
    knobs_dx: list[mathdx.CufftdxKnobType] = [_FFT_KNOB_TYPE_TO_MATHDX[k] for k in knobs]
    knobs_size = mathdx.cufftdx_get_knob_int64size(handle, len(knobs), knobs_dx)

    knobs_result_dx = numpy.empty(knobs_size, dtype=numpy.int64)
    mathdx.cufftdx_get_knob_int64s(
        handle,
        len(knobs),
        knobs_dx,
        knobs_size,
        knobs_result_dx.ctypes.data,
    )

    knobs_result = [
        tuple(int(knobs_result_dx[j]) for j in range(i, i + len(knobs))) for i in range(0, len(knobs_result_dx), len(knobs))
    ]

    return knobs_result
