# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Sequence
from functools import lru_cache, partial

import numpy as np
from numba import cuda

from nvmath.bindings import mathdx
from nvmath.device.common_cuda import ComputeCapability

from .common import check_in, check_not_in, check_positive_integer_sequence, check_sm
from .common_backend import (
    EXECUTION_STR_TO_MATHDX,
    NP_TYPES_TO_MATHDX_PRECISION,
    NVARG_GEN_OPT_LTO,
    DescriptorWrapper,
    build_get_int_traits,
    build_get_str_trait,
)

_ENABLE_CUSOLVERDX_0_3 = mathdx.get_version_ex() >= (0, 3, 2)

# =======================================================
# Python to mathdx translation
# =======================================================

_SOLVER_FUNCTION_TO_MATHDX = {f.name.lower(): f for f in mathdx.CusolverdxFunction}
_SOLVER_SIDE_TO_MATHDX = {s.name.lower(): s for s in mathdx.CusolverdxSide}
_SOLVER_DIAG_TO_MATHDX = {d.name.lower(): d for d in mathdx.CusolverdxDiag}
_SOLVER_FILL_MODE_TO_MATHDX = {f.name.lower(): f for f in mathdx.CusolverdxFillMode}
_SOLVER_DATA_TYPE_TO_MATHDX = {t.name.lower(): t for t in mathdx.CusolverdxType}
_SOLVER_ARRANGEMENT_TO_MATHDX = {a.name.lower(): a for a in mathdx.CusolverdxArrangement}
_SOLVER_TRANSPOSE_MODE_TO_MATHDX = {t.name.lower(): t for t in mathdx.CusolverdxTransposeMode}
_SOLVER_EXECUTION_API_TO_MATHDX = {
    "compiled_leading_dim": mathdx.CusolverdxApi.SMEM,
    "runtime_leading_dim": mathdx.CusolverdxApi.SMEM_DYNAMIC_LD,
}
_SOLVER_JOB_TO_MATHDX = {j.name.lower(): j for j in mathdx.CusolverdxJob}

# =======================================================
# Allowed input arguments
# =======================================================

ALLOWED_CUSOLVERDX_FUNCTIONS = {
    # Cholesky
    "potrf": "Cholesky factorization",
    "potrs": "Linear system solve after Cholesky factorization",
    "posv": "Fused Cholesky factorization with solve",
    # LU without pivoting
    "getrf_no_pivot": "LU factorization without pivoting",
    "getrs_no_pivot": "LU solve without pivoting",
    "gesv_no_pivot": "Fused LU without pivoting factorization with solve",
    # LU with partial pivoting
    "getrf_partial_pivot": "LU factorization with partial pivoting",
    "getrs_partial_pivot": "LU solve with partial pivoting",
    "gesv_partial_pivot": "Fused LU without pivoting factorization with solve",
    # QR
    "geqrf": "QR factorization",
    "unmqr": "Multiplication of Q from QR factorization",
    # LQ
    "gelqf": "LQ factorization",
    "unmlq": "Multiplication of Q from LQ factorization",
    # Triangular solve
    "trsm": "Triangular matrix-matrix solve",
    # Least squares
    "gels": "Overdetermined or underdetermined least square problems",
    # ======= libmathdx 0.3.2 functions =======
    # QR
    "ungqr": "Q matrix generation from geqrf factorization",
    # LQ
    "unglq": "Q matrix generation from gelqf factorization",
    # Eigenvalues
    "htev": "Eigenvalue solver for hermitian tridiagonal matrices",
    "heev": "Eigenvalue solver for hermitian dense matrices",
    # Tridiagonal matrices
    "gtsv_no_pivot": "General tridiagonal matrix solve",
}

CUSOLVERDX_0_3_ALLOWED_FUNCTIONS = ("ungqr", "unglq", "htev", "heev", "gtsv_no_pivot")

ALLOWED_REAL_NP_TYPES = [np.float32, np.float64]
ALLOWED_ARRANGEMENT = [a.name.lower() for a in mathdx.CusolverdxArrangement]
ALLOWED_DATA_TYPE = [d.name.lower() for d in mathdx.CusolverdxType]
ALLOWED_EXECUTION = ["Block"]
ALLOWED_TRANSPOSE_MODE = [t.name.lower() for t in mathdx.CusolverdxTransposeMode]
ALLOWED_SIDE = [s.name.lower() for s in mathdx.CusolverdxSide]
ALLOWED_DIAG = [d.name.lower() for d in mathdx.CusolverdxDiag]
ALLOWED_FILL_MODE = [f.name.lower() for f in mathdx.CusolverdxFillMode]
ALLOWED_EXECUTION_API = ["compiled_leading_dim", "runtime_leading_dim"]
ALLOWED_JOB = [j.name.lower() for j in mathdx.CusolverdxJob]

# =======================================================
# Supported/Unsupported functions for arguments
# =======================================================

TRANSPOSE_SUPPORTED_FUNCTIONS = [
    "getrf_no_pivot",
    "getrs_no_pivot",
    "gesv_no_pivot",
    "getrf_partial_pivot",
    "getrs_partial_pivot",
    "gesv_partial_pivot",
    "geqrf",
    "gelqf",
    "unmqr",
    "unmlq",
    "gels",
    "trsm",
]
SIDE_SUPPORTED_FUNCTIONS = ["unmqr", "unmlq", "trsm"]
FILL_MODE_SUPPORTED_FUNCTIONS = ["potrf", "potrs", "posv", "trsm", "heev"]
DIAG_SUPPORTED_FUNCTIONS = ["trsm"]
JOB_SUPPORTED_FUNCTIONS = ["htev", "heev"]
JOB_SUPPORT_MAP = {
    "htev": ["no_vectors", "all_vectors", "multiply_vectors"],
    "heev": ["no_vectors", "overwrite_vectors"],
}

# =======================================================
# Helper functions
# =======================================================

check_arg_supported = partial(
    check_in,
    format='For function: "{value}", parameter "{name}" is not supported. It is only supported for functions: {coll_str}',
)


def check_required_optional_provided(function, required_for, arg, arg_name):
    if arg is None:
        check_not_in(
            name=arg_name, value=function, coll=required_for, format='Parameter "{name}" is required for function "{value}".'
        )


get_str_trait = build_get_str_trait(mathdx.cusolverdx_get_trait_str_size, mathdx.cusolverdx_get_trait_str)
get_int_traits = build_get_int_traits(mathdx.cusolverdx_get_trait_int64s)

# =======================================================
# validate()
# =======================================================


def _validate_types_and_values(
    function,
    size,
    precision,
    execution,
    sm,
    arrangement,
    transpose_mode,
    side,
    diag,
    fill_mode,
    batches_per_block,
    data_type,
    leading_dimensions,
    block_dim,
    job,
):
    check_in("function", function, ALLOWED_CUSOLVERDX_FUNCTIONS.keys())
    check_positive_integer_sequence(size, "size", 1, 3)
    check_in("precision", precision, ALLOWED_REAL_NP_TYPES)
    check_in("execution", execution, ALLOWED_EXECUTION)
    check_sm(sm, "sm")

    if arrangement is not None:
        if not isinstance(arrangement, Sequence):
            raise ValueError(f'Parameter "arrangement" must be a sequence. Got: {type(arrangement).__name__}.')
        if len(arrangement) != 2 and len(arrangement) != 1:
            raise ValueError(f'Parameter "arrangement" must be a sequence of length 1 or 2. Got length: {len(arrangement)}.')
        check_in("arrangement[0]", arrangement[0], ALLOWED_ARRANGEMENT)

        if len(arrangement) == 2:
            check_in("arrangement[1]", arrangement[1], ALLOWED_ARRANGEMENT)

    if transpose_mode is not None:
        check_in("transpose_mode", transpose_mode, ALLOWED_TRANSPOSE_MODE)

    if side is not None:
        check_in("side", side, ALLOWED_SIDE)

    if diag is not None:
        check_in("diag", diag, ALLOWED_DIAG)

    if fill_mode is not None:
        check_in("fill_mode", fill_mode, ALLOWED_FILL_MODE)

    if data_type is not None:
        check_in("data_type", data_type, ALLOWED_DATA_TYPE)

    if batches_per_block is not None and batches_per_block != "suggested":
        if not isinstance(batches_per_block, int):
            raise ValueError(
                'Parameter "batches_per_block" must be a positive integer or "suggested". '
                + f"Got: {type(batches_per_block).__name__}."
            )
        if batches_per_block < 1:
            raise ValueError(f'Parameter "batches_per_block" must be a positive integer. Got: {batches_per_block}.')

    if leading_dimensions is not None:
        check_positive_integer_sequence(leading_dimensions, "leading_dimensions", 1, 2)

    if block_dim is not None and block_dim != "suggested":
        check_positive_integer_sequence(block_dim, "block_dim", 1, 3)

    if job is not None:
        check_in("job", job, ALLOWED_JOB)


def _validate_logic(
    function,
    transpose_mode,
    side,
    diag,
    fill_mode,
    job,
):
    if not _ENABLE_CUSOLVERDX_0_3 and job is not None:
        raise RuntimeError("job operator requires libmathdx 0.3.2")

    if transpose_mode is not None:
        check_arg_supported("transpose_mode", function, TRANSPOSE_SUPPORTED_FUNCTIONS)
        if function in {"geqrf", "gelqf", "getrf_no_pivot", "getrf_partial_pivot"}:
            check_in(
                name="transpose_mode",
                value=transpose_mode,
                coll={"non_transposed"},
                format=f'For function "{function}", parameter "transpose_mode" only supports: {{coll_str}}. Got: "{{value}}".',
            )

    if side is not None:
        check_arg_supported("side", function, SIDE_SUPPORTED_FUNCTIONS)

    if fill_mode is not None:
        check_arg_supported("fill_mode", function, FILL_MODE_SUPPORTED_FUNCTIONS)

    if diag is not None:
        check_arg_supported("diag", function, DIAG_SUPPORTED_FUNCTIONS)

    if job is not None:
        check_arg_supported("job", function, JOB_SUPPORTED_FUNCTIONS)

        check_in(
            name="job",
            value=job,
            coll=JOB_SUPPORT_MAP[function],
            format=f'For function "{function}", parameter "job" only supports: {{coll_str}}. Got: "{{value}}".',
        )

    check_required_optional_provided(function, SIDE_SUPPORTED_FUNCTIONS, side, "side")
    check_required_optional_provided(function, FILL_MODE_SUPPORTED_FUNCTIONS, fill_mode, "fill_mode")
    check_required_optional_provided(function, DIAG_SUPPORTED_FUNCTIONS, diag, "diag")
    check_required_optional_provided(function, JOB_SUPPORTED_FUNCTIONS, job, "job")


def validate(
    function,
    size,
    precision,
    execution,
    sm,
    arrangement,
    transpose_mode,
    side,
    diag,
    fill_mode,
    batches_per_block,
    data_type,
    leading_dimensions,
    block_dim,
    job,
):
    if function in CUSOLVERDX_0_3_ALLOWED_FUNCTIONS and not _ENABLE_CUSOLVERDX_0_3:
        raise RuntimeError(f'Function "{function}" requires libmathdx 0.3.2 or later.')

    if (
        function == "gesv_no_pivot"
        and sm.major == 12
        and sm.minor == 0
        and (data_type == "real" or data_type is None)
        and "NVMATH_CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT" not in os.environ
    ):
        raise RuntimeError(
            "NVBUG 5288270: CUDA 12.8, 12.9 and 13.0 could miscompile kernels using "
            "'gesv_no_pivot' function with high register pressure when SM is set to 1200 "
            "and Type is set to 'real'. To ignore this assertion and verify correctness "
            "of the results, set the NVMATH_CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT environment variable."
        )

    _validate_types_and_values(
        function=function,
        size=size,
        precision=precision,
        execution=execution,
        sm=sm,
        arrangement=arrangement,
        transpose_mode=transpose_mode,
        side=side,
        diag=diag,
        fill_mode=fill_mode,
        batches_per_block=batches_per_block,
        data_type=data_type,
        leading_dimensions=leading_dimensions,
        block_dim=block_dim,
        job=job,
    )

    _validate_logic(
        function=function,
        transpose_mode=transpose_mode,
        side=side,
        diag=diag,
        fill_mode=fill_mode,
        job=job,
    )


# =======================================================
# generate_SOLVER()
# =======================================================


@lru_cache
def generate_SOLVER(
    function,
    size,
    precision,
    execution,
    sm,
    arrangement,
    transpose_mode,
    side,
    diag,
    fill_mode,
    batches_per_block,
    data_type,
    leading_dimensions,
    block_dim,
    execution_api,
    job,
):
    check_not_in("leading_dimensions", leading_dimensions, ["suggested"])
    check_not_in("batches_per_block", batches_per_block, ["suggested"])
    check_not_in("block_dim", block_dim, ["suggested"])
    check_in("execution_api", execution_api, ALLOWED_EXECUTION_API)

    h = mathdx.cusolverdx_create_descriptor()

    assert _ENABLE_CUSOLVERDX_0_3 or function not in CUSOLVERDX_0_3_ALLOWED_FUNCTIONS
    mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.FUNCTION, _SOLVER_FUNCTION_TO_MATHDX[function])
    mathdx.cusolverdx_set_operator_int64s(h, mathdx.CusolverdxOperatorType.SIZE, 3, list(size))
    mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.PRECISION, NP_TYPES_TO_MATHDX_PRECISION[precision])
    mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.EXECUTION, EXECUTION_STR_TO_MATHDX[execution])
    mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.SM, sm.major * 100 + sm.minor * 10)

    if arrangement is not None:
        converted_arrangement = [_SOLVER_ARRANGEMENT_TO_MATHDX[a] for a in list(arrangement)]
        mathdx.cusolverdx_set_operator_int64s(
            h, mathdx.CusolverdxOperatorType.ARRANGEMENT, len(arrangement), converted_arrangement
        )

    if side is not None:
        mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.SIDE, _SOLVER_SIDE_TO_MATHDX[side])

    if diag is not None:
        mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.DIAG, _SOLVER_DIAG_TO_MATHDX[diag])

    if fill_mode is not None:
        mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.FILL_MODE, _SOLVER_FILL_MODE_TO_MATHDX[fill_mode])

    if data_type is not None:
        mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.TYPE, _SOLVER_DATA_TYPE_TO_MATHDX[data_type])

    if block_dim is not None:
        mathdx.cusolverdx_set_operator_int64s(h, mathdx.CusolverdxOperatorType.BLOCK_DIM, 3, list(block_dim))

    if transpose_mode is not None:
        mathdx.cusolverdx_set_operator_int64(
            h, mathdx.CusolverdxOperatorType.TRANSPOSE_MODE, _SOLVER_TRANSPOSE_MODE_TO_MATHDX[transpose_mode]
        )

    if leading_dimensions is not None:
        mathdx.cusolverdx_set_operator_int64s(
            h, mathdx.CusolverdxOperatorType.LEADING_DIMENSION, len(leading_dimensions), list(leading_dimensions)
        )

    if batches_per_block is not None:
        mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.BATCHES_PER_BLOCK, batches_per_block)

    if job is not None:
        assert _ENABLE_CUSOLVERDX_0_3
        mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.JOB, _SOLVER_JOB_TO_MATHDX[job])

    mathdx.cusolverdx_set_operator_int64(h, mathdx.CusolverdxOperatorType.API, _SOLVER_EXECUTION_API_TO_MATHDX[execution_api])

    return DescriptorWrapper(h, mathdx.cusolverdx_destroy_descriptor)


# =======================================================
# generate_code()
# =======================================================


@lru_cache
def generate_code(handle, version: ComputeCapability):
    code = mathdx.commondx_create_code()

    mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, version.integer)

    mathdx.commondx_set_code_option_str(code, mathdx.CommondxOption.EXTRA_NVRTC_ARGS, NVARG_GEN_OPT_LTO)
    if "NVMATH_CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT" in os.environ:
        # TODO: use commondx_set_code_option_strs: currently limited to 1 argument
        mathdx.commondx_set_code_option_str(
            code, mathdx.CommondxOption.EXTRA_NVRTC_ARGS, "-DCUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT"
        )

    mathdx.cusolverdx_finalize_code(code, handle)

    return DescriptorWrapper(code, mathdx.commondx_destroy_code)


# =======================================================
# get_universal_fatbin()
# =======================================================


@lru_cache
def get_universal_fatbin():
    h = mathdx.cusolverdx_create_descriptor()
    fatbin_size = mathdx.cusolverdx_get_universal_fatbin_size(h)
    fatbin = bytearray(fatbin_size)
    mathdx.cusolverdx_get_universal_fatbin(h, fatbin_size, fatbin)
    mathdx.cusolverdx_destroy_descriptor(h)

    return cuda.Fatbin(bytes(fatbin))
