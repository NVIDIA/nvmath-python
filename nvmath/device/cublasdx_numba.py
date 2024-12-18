# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.extending import intrinsic, overload
from numba.core.typing import signature

from .common import check_in
from .common_numba import NUMBA_FE_TYPES_TO_NUMBA_IR, make_function_call


#
# Lowering down to function call to cuBLASDx
# kind = 'smem_basic' or 'smem_ldabc' - the kind of API being used
# value_type = real or complex numpy type of the A/B/C and alpha/beta inputs.
# symbols = name of the function, as a string
#
def make_codegen(kind, value_type, symbol, a_type, b_type, c_type):
    check_in("kind", kind, ["smem_basic", "smem_ldabc"])

    assert a_type.dtype == value_type
    assert b_type.dtype == value_type
    assert c_type.dtype == value_type
    ld_type = types.uint32
    return_type = types.void

    if kind == "smem_basic":
        # smem_basic APIs take the 3 input arrays and alpha/beta as argument
        # lda, ldb and ldc are based on the underlying Dx type
        # (void) ( (value_type*)alpha, (value_type*)a, (value_type*)b, (value_type*)beta, (value_type*)c )  # noqa: W505

        return signature(return_type, value_type, a_type, b_type, value_type, c_type), make_function_call(symbol)

    elif kind == "smem_ldabc":
        # smem_ldabc APIs take the 3 input arrays, alpha/beta as argument and lda, ldb and
        # ldc
        # (void) ( (value_type*)alpha, (value_type*)a, (unsigned)lda,
        #                              (value_type*)b, (unsigned)ldb,
        #           (value_type*)beta, (value_type*)c, (unsigned)ldc )

        return signature(
            return_type, value_type, a_type, ld_type, b_type, ld_type, value_type, c_type, ld_type
        ), make_function_call(symbol)

    else:
        raise RuntimeError(f"Invalid kind = {kind}, this should not happen")


def codegen(description, func_to_overload):
    value_type = NUMBA_FE_TYPES_TO_NUMBA_IR[description["value_type"]]
    symbols = description["symbols"]

    #
    # smem_basic version
    #

    @intrinsic
    def intrinsic_smem_basic(typingctx, alpha, a, b, beta, c):
        return make_codegen("smem_basic", value_type, symbols["smem_basic"], a, b, c)

    @overload(func_to_overload, target="cuda")
    def gemm_smem_basic(alpha, a, b, beta, c):
        def impl(alpha, a, b, beta, c):
            return intrinsic_smem_basic(alpha, a, b, beta, c)

        return impl

    #
    # smem_ldabc
    #

    @intrinsic
    def intrinsic_smem_ldabc(typingctx, alpha, a, lda, b, ldb, beta, c, ldc):
        return make_codegen("smem_ldabc", value_type, symbols["smem_ldabc"], a, b, c)

    @overload(func_to_overload, target="cuda")
    def gemm_smem_ldabc(alpha, a, lda, b, ldb, beta, c, ldc):
        def impl(alpha, a, lda, b, ldb, beta, c, ldc):
            return intrinsic_smem_ldabc(alpha, a, lda, b, ldb, beta, c, ldc)

        return impl

    return {"smem_basic": gemm_smem_basic, "smem_ldabc": gemm_smem_ldabc}
