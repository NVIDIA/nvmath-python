# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.typing import signature
from numba.extending import typeof_impl, models, register_model, make_attribute_wrapper, intrinsic, overload
from .common_numba import NUMBA_FE_TYPES_TO_NUMBA_IR, make_function_call
from .common import check_in
from .cufftdx_workspace import Workspace

##
##  Numba Workspace type
##


class WorkspaceType(types.Type):
    def __init__(self):
        super().__init__(name="Workspace")


workspace_type = WorkspaceType()

make_attribute_wrapper(WorkspaceType, "workspace", "workspace")


@register_model(WorkspaceType)
class WorkspaceModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("workspace", types.uint64),  # Workspace
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


##
## Register custom Workspace type
##


@typeof_impl.register(Workspace)
def typeof_workspace(val, c):
    return workspace_type


class WorkspaceArgHandler:
    def prepare_args(self, ty, val, **kwargs):
        if isinstance(val, Workspace):
            assert ty == workspace_type
            return types.uint64, val.workspace
        else:
            return ty, val


#
# Lowering down to function call to cuFFTDx
# io = 'thread' or 'smem'
# execution = 'Thread' or 'Block'
# value_type = real or complex numpy type of the input/output thread-private and shared
# memory
# symbols = name of the function, as a string
# requires_workspace whether we require workspace, or not, as a bool.
#
def make_codegen(io, execution, value_type, symbol, requires_workspace):
    check_in("io", io, ["thread", "smem"])  # input in thread-private vs shared memory
    check_in("execution", execution, ["Thread", "Block"])  # Thread() or Block() APIs
    check_in("requires_workspace", requires_workspace, [False])  # Workspace not supported yet

    array_type = types.Array(value_type, 1, "C")
    workspace_ty = types.int64  # Used to pass void*
    return_type = types.void

    # Thread() APIs only work on a single thread-private array
    # no shared memory, no workspace
    # (void) ( (value_type*)thread array )

    if execution == "Thread" and io == "thread":
        return signature(return_type, array_type), make_function_call(symbol)

    # Block() APIs have four variants
    # (void) ( (value_type*)thread array,         (value_type*)shared memory array               )  # noqa: W505
    # (void) ( (value_type*)shared memory array                                                  )  # noqa: W505
    # (void) ( (value_type*)thread array,         (value_type*)shared memory array,    workspace )  # noqa: W505
    # (void) ( (value_type*)shared memory array,                                       workspace )  # noqa: W505
    # In all cases we pass 3 arguments, with appropriate nullptrs or 0's where needed

    elif execution == "Block" and io == "thread":
        codegen = make_function_call(symbol)

        def wrap_codegen(context, builder, sig, args):
            assert len(args) == 2
            assert len(sig.args) == 3
            codegen(context, builder, sig, [args[0], args[1], None])

        return signature(return_type, array_type, array_type, workspace_ty), wrap_codegen

    elif execution == "Block" and io == "smem":
        codegen = make_function_call(symbol)

        def wrap_codegen(context, builder, sig, args):
            assert len(args) == 1
            assert len(sig.args) == 3
            codegen(context, builder, sig, [None, args[0], None])

        return signature(return_type, array_type, array_type, workspace_ty), wrap_codegen


def codegen(description, func_to_overload, Workspace):
    value_type = NUMBA_FE_TYPES_TO_NUMBA_IR[description["value_type"]]
    execution = description["execution"]
    requires_workspace = description["requires_workspace"]
    symbols = description["symbols"]

    check_in("execution", execution, ["Block", "Thread"])
    check_in("requires_workspace", requires_workspace, [True, False])

    extensions = [WorkspaceArgHandler()]

    if execution == "Thread":

        @intrinsic
        def intrinsic_1(typingctx, thread):
            return make_codegen("thread", "Thread", value_type, symbols["thread"], False)

        @overload(func_to_overload, target="cuda")
        def fft_1(thread):
            def impl(thread):
                return intrinsic_1(thread)

            return impl

        return {"fft_thread": fft_1, "fft_smem": None, "extensions": None}

    else:
        if requires_workspace:

            @intrinsic
            def intrinsic_3(typingctx, thread, smem, workspace):
                return make_codegen("thread", "Block", value_type, symbols["thread"], True)

            @overload(func_to_overload, target="cuda")
            def fft_3(thread, smem, workspace):
                def impl(thread, smem, workspace):
                    return intrinsic_3(thread, smem, workspace)

                return impl

            ####

            @intrinsic
            def intrinsic_2(typingctx, smem, workspace):
                return make_codegen("smem", "Block", value_type, symbols["smem"], True)

            @overload(func_to_overload, target="cuda")
            def fft_2(smem, workspace):
                def impl(smem, workspace):
                    return intrinsic_2(smem, workspace)

                return impl

            return {"fft_thread": fft_3, "fft_smem": fft_2, "extensions": extensions}

        else:

            @intrinsic
            def intrinsic_2(typingctx, thread, smem):
                return make_codegen("thread", "Block", value_type, symbols["thread"], False)

            @overload(func_to_overload, target="cuda")
            def fft_2(thread, smem):
                def impl(thread, smem):
                    return intrinsic_2(thread, smem)

                return impl

            ####

            @intrinsic
            def intrinsic_1(typingctx, smem):
                return make_codegen("smem", "Block", value_type, symbols["smem"], False)

            @overload(func_to_overload, target="cuda")
            def fft_1(smem):
                def impl(smem):
                    return intrinsic_1(smem)

                return impl

            return {"fft_thread": fft_2, "fft_smem": fft_1, "extensions": None}
