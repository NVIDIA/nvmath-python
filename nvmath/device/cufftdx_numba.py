# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.typing import signature
from numba.extending import intrinsic, overload
from .common_numba import NUMBA_FE_TYPES_TO_NUMBA_IR, make_function_call
from .common import check_in


#
# Lowering down to function call to cuFFTDx
# io = 'thread' or 'smem'
# execution = 'Thread' or 'Block'
# value_type = real or complex numpy type of the input/output thread-private and shared
# memory
# symbols = name of the function, as a string
#
def make_codegen(io, execution, value_type, symbol):
    check_in("io", io, ["thread", "smem"])  # input in thread-private vs shared memory
    check_in("execution", execution, ["Thread", "Block"])  # Thread() or Block() APIs

    array_type = types.Array(value_type, 1, "C")
    return_type = types.void

    # Thread() APIs only work on a single thread-private array
    # no shared memory, no workspace
    # (void) ( (value_type*)thread array )

    if execution == "Thread" and io == "thread":
        return signature(return_type, array_type), make_function_call(symbol)

    # Block() APIs have two variants
    # (void) ( (value_type*)thread array,         (value_type*)shared memory array               )  # noqa: W505
    # (void) ( (value_type*)shared memory array                                                  )  # noqa: W505

    elif execution == "Block" and io == "thread":
        codegen = make_function_call(symbol)

        def wrap_codegen(context, builder, sig, args):
            assert len(args) == 2
            assert len(sig.args) == 2
            codegen(context, builder, sig, [args[0], args[1]])

        return signature(return_type, array_type, array_type), wrap_codegen

    elif execution == "Block" and io == "smem":
        codegen = make_function_call(symbol)

        def wrap_codegen(context, builder, sig, args):
            assert len(args) == 1
            assert len(sig.args) == 1
            codegen(context, builder, sig, args)

        return signature(return_type, array_type), wrap_codegen


def codegen(description, func_to_overload):
    execution = description["execution"]
    execute_api = description["execute_api"]

    check_in("execution", execution, ["Block", "Thread"])

    if execution == "Thread":
        codegen_thread(description, func_to_overload)
    else:
        assert execution == "Block"

        if execute_api == "register_memory":
            codegen_block_lmem(description, func_to_overload)
        else:
            assert execute_api == "shared_memory"

            codegen_block_smem(description, func_to_overload)


def codegen_thread(description, func_to_overload):
    value_type = NUMBA_FE_TYPES_TO_NUMBA_IR[description["value_type"]]
    symbol = description["symbol"]

    @intrinsic
    def intrinsic_1(typingctx, thread):
        return make_codegen("thread", "Thread", value_type, symbol)

    @overload(func_to_overload, target="cuda")
    def fft(thread):
        def impl(thread):
            return intrinsic_1(thread)

        return impl


def codegen_block_lmem(description, func_to_overload):
    value_type = NUMBA_FE_TYPES_TO_NUMBA_IR[description["value_type"]]
    symbol = description["symbol"]

    @intrinsic
    def intrinsic_2(typingctx, thread, smem):
        return make_codegen("thread", "Block", value_type, symbol)

    @overload(func_to_overload, target="cuda")
    def fft(thread, smem):
        def impl(thread, smem):
            return intrinsic_2(thread, smem)

        return impl


def codegen_block_smem(description, func_to_overload):
    value_type = NUMBA_FE_TYPES_TO_NUMBA_IR[description["value_type"]]
    symbol = description["symbol"]

    @intrinsic
    def intrinsic_1(typingctx, smem):
        return make_codegen("smem", "Block", value_type, symbol)

    @overload(func_to_overload, target="cuda")
    def fft(smem):
        def impl(smem):
            return intrinsic_1(smem)

        return impl
