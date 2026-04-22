# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal, NamedTuple, NoReturn

from numba import cuda
from numba.core import typing
from numba.extending import intrinsic, overload_method, types

from nvmath.device import (
    CholeskySolver,
    LeastSquaresSolver,
    LQFactorize,
    LQMultiply,
    LUPivotSolver,
    LUSolver,
    QRFactorize,
    QRMultiply,
    Solver,
    TriangularSolver,
    compile_solver_execute,
)
from nvmath.device.common_cuda import get_default_code_type
from nvmath.device.cusolverdx_backend import _ENABLE_CUSOLVERDX_0_3, get_universal_fatbin

from .common_numba import (
    NUMBA_FE_TYPES_TO_NUMBA_IR,
    declare_cabi_device,
    get_array_ptr,
    get_uint32_value_ptr,
    register_dummy_numba_type,
)

# ==========================
# Data: Solver Configuration
# ==========================

_SOLVER_DEFINITION_ARGS = [
    "function",
    "size",
    "precision",
    "execution",
    "sm",
    "arrangement",
    "transpose_mode",
    "side",
    "diag",
    "fill_mode",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
    "block_dim",
]

_SOLVER_PROPS = [
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "info_type",
    "ipiv_type",
    "tau_type",
    "block_size",
]

if _ENABLE_CUSOLVERDX_0_3:
    _SOLVER_PROPS.append("workspace_size")
    _SOLVER_DEFINITION_ARGS.append("job")

# ==========================
# Numba type
# ==========================


class SolverNumbaType(types.Type):
    def __init__(self, solver: Solver) -> None:
        self._solver = solver

        attributes = [f"{attr}={getattr(solver, attr)}" for attr in _SOLVER_DEFINITION_ARGS if getattr(solver, attr)]
        attributes.sort()
        attr_str = ", ".join(attributes)

        super().__init__(name=f"Solver({attr_str})")

    @property
    def solver(self) -> Solver:
        return self._solver


register_dummy_numba_type(
    SolverNumbaType,
    Solver,
    "solver",
    _SOLVER_DEFINITION_ARGS + _SOLVER_PROPS,
)

# ==========================
# Dispatch Data Structures
# ==========================


class _Arg(NamedTuple):
    arg_type: Callable[[SolverNumbaType], types.Type]
    name: str


class _Signature(NamedTuple):
    api: Literal["compiled_leading_dim"] | Literal["runtime_leading_dim"]  # libmathdx API variant
    args: list[_Arg]  # List of expected arguments for the function


class _FunctionSignatures(NamedTuple):
    function: str
    signatures: list[_Signature]


def _numba_solver_value_type_ptr(solver_type: SolverNumbaType) -> types.Type:
    return types.CPointer(NUMBA_FE_TYPES_TO_NUMBA_IR[solver_type.solver.value_type])


def _numba_solver_precision_type_ptr(solver_type: SolverNumbaType) -> types.Type:
    return types.CPointer(NUMBA_FE_TYPES_TO_NUMBA_IR[solver_type.solver.precision])


def _numba_solver_integer_value(solver_type: SolverNumbaType) -> types.Type:
    return types.Integer


def _numba_solver_int32_ptr(solver_type: SolverNumbaType) -> types.Type:
    return types.CPointer(types.int32)


def _validate_arg_type(user_type: types.Type, expected_type: types.Type) -> bool:
    if expected_type == types.Integer:
        return isinstance(user_type, types.Integer)

    assert isinstance(expected_type, types.CPointer)
    dtype, _ = expected_type.key

    return isinstance(user_type, types.Array) and user_type.dtype == dtype


def _get_libmathdx_type(expected_type: types.Type) -> types.Type:
    if expected_type == types.Integer:
        return types.CPointer(types.uint32)
    return expected_type


def _get_error_message_type(numba_type: types.Type) -> types.Type | str:
    """
    Converts a type to its user-friendly representation for error messages.
    """
    if numba_type == types.Integer:
        return numba_type.__name__
    return numba_type


# ==========================
# Dispatch Data: Function Signatures Registry
# ==========================

"""
_DISPATCH_DATA contains the dispatch information for different solver functions.
"""
_DISPATCH_DATA = [
    _FunctionSignatures(
        function="potrs",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="getrs_no_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="gels",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="trsm",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="geqrf",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="gelqf",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="unmqr",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="c"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="c"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldc"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="unmlq",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="c"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="c"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldc"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="potrf",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="getrf_no_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="getrf_partial_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="ipiv"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="ipiv"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="getrs_partial_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="ipiv"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="ipiv"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="posv",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="gesv_no_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="gesv_partial_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="ipiv"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="ipiv"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
]

_DISPATCH_DATA_CUSOLVERDX_0_3 = [
    _FunctionSignatures(
        function="gtsv_no_pivot",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="dl"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="d"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="du"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="dl"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="d"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="du"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="b"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldb"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="htev",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="d"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="e"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="d"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="e"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="V"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="d"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="e"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="V"),
                    _Arg(arg_type=_numba_solver_integer_value, name="ldv"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="heev",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_precision_type_ptr, name="lambda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="workspace"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_precision_type_ptr, name="lambda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="workspace"),
                    _Arg(arg_type=_numba_solver_int32_ptr, name="info"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="ungqr",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
        ],
    ),
    _FunctionSignatures(
        function="unglq",
        signatures=[
            _Signature(
                api="compiled_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
            _Signature(
                api="runtime_leading_dim",
                args=[
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="a"),
                    _Arg(arg_type=_numba_solver_integer_value, name="lda"),
                    _Arg(arg_type=_numba_solver_value_type_ptr, name="tau"),
                ],
            ),
        ],
    ),
]

if _ENABLE_CUSOLVERDX_0_3:
    _DISPATCH_DATA += _DISPATCH_DATA_CUSOLVERDX_0_3

# ==========================
# Intrinsics
# ==========================


@intrinsic
def _link_fatbin(typingctx: typing.Context):
    sig = types.void()

    fatbin = get_universal_fatbin()

    def codegen(context, builder, sig, args):
        context.active_code_library.add_linking_file(fatbin)

    return sig, codegen


@intrinsic
def _do_nothing(typingctx: typing.Context, value: types.Type):
    sig = value(value)

    def codegen(context, builder, sig, args):
        return args[0]

    return sig, codegen


# ==========================
# Overloads: Method Dispatch Implementation
# ==========================


def _get_function_signatures(function: str) -> list[_Signature] | None:
    """Looks up and returns the list of available signatures for a given function."""
    for func_signature in _DISPATCH_DATA:
        if func_signature.function == function:
            return func_signature.signatures

    return None


def _check_all_none(args: list[types.Type]) -> bool:
    """Checks if all elements are None"""
    return all(unwanted_arg in {None, types.Omitted(None), types.none} for unwanted_arg in args)


def _validate_arg_types(solver_type: SolverNumbaType, arg_types: list[_Arg], args_fit: list[types.Type]) -> bool:
    """Validates whether provided arguments match the expected types for a signature."""
    return all(
        _validate_arg_type(arg, arg_spec.arg_type(solver_type)) for arg_spec, arg in zip(arg_types, args_fit, strict=True)
    )


def _find_overload(args: list[types.Type], solver_type: SolverNumbaType) -> _Signature | None:
    """
    Iterates through all available signatures for the solver's function and finds
    the first one where:
    1. The number of arguments matches (no extra arguments provided)
    2. All argument types are valid for the signature
    """

    signatures = _get_function_signatures(solver_type.solver.function)
    assert signatures is not None

    for sig in signatures:
        expected_args_num = len(sig.args)

        args_fit = args[:expected_args_num]
        if not _check_all_none(args[expected_args_num:]):
            continue

        if not _validate_arg_types(solver_type, sig.args, args_fit):
            continue

        return sig

    return None


def _compile_device_function(solver_type: SolverNumbaType, signature: _Signature) -> Callable:
    """
    Compile cusolverdx execute device function and wrap it into numba device function.
    """

    # Call libmathdx to jit the function
    code, symbol = compile_solver_execute(
        solver_type.solver,
        code_type=get_default_code_type(),
        execution_api=signature.api,
    )
    lto = cuda.LTOIR(code.data)

    numba_args = [_get_libmathdx_type(arg_spec.arg_type(solver_type)) for arg_spec in signature.args]
    sig = types.void(*numba_args)

    # Convert to cabi from numba abi
    return declare_cabi_device(symbol, sig, link=lto)


def _format_signature(solver_type: SolverNumbaType, sig: _Signature) -> str:
    arg_list = ", ".join(f"{arg_spec.name}: {_get_error_message_type(arg_spec.arg_type(solver_type))}" for arg_spec in sig.args)
    return f"({arg_list}) -> None"


def _raise_overload_resolution_error(solver_type: SolverNumbaType, args: list[types.Type]) -> NoReturn:
    """
    Raises a detailed TypeError when no matching function signature is found.

    Constructs an informative error message showing:
    - The signature that was provided by the user
    - All available signatures for the solver function
    """
    signatures = _get_function_signatures(solver_type.solver.function)
    assert signatures is not None

    available_sigs = ", ".join(_format_signature(solver_type, sig) for sig in signatures)
    provided_args = [
        types.CPointer(arg.dtype) if isinstance(arg, types.Array) else arg
        for arg in args
        if arg not in {None, types.Omitted(None), types.none}
    ]
    provided_sig = types.void(*provided_args)

    raise TypeError(
        'Failed to find matching overload of "SOLVER.execute()" function.'
        f" Provided signature: {provided_sig}, but only these are available: {available_sigs}"
    )


@overload_method(SolverNumbaType, "execute", target="cuda", jit_options={"forceinline": True}, strict=False)
def _overload_execute(solver_type: SolverNumbaType, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None):
    """Numba overload for Solver.execute(), resolves signature and compiles function."""
    assert isinstance(solver_type, SolverNumbaType)

    # Find matching signature for the provided arguments
    args = [arg0, arg1, arg2, arg3, arg4, arg5]
    overload = _find_overload(args, solver_type)

    if overload is None:
        _raise_overload_resolution_error(solver_type, args)

    # Compile the device function for the matched signature
    device_func = _compile_device_function(solver_type, overload)

    # Generate the implementation wrapper
    def prepare_arg_intrinsic(arg):
        if isinstance(arg, types.Integer):
            return get_uint32_value_ptr
        elif isinstance(arg, types.Array):
            return get_array_ptr
        else:
            return _do_nothing

    intrinsic0 = prepare_arg_intrinsic(arg0)
    intrinsic1 = prepare_arg_intrinsic(arg1)
    intrinsic2 = prepare_arg_intrinsic(arg2)
    intrinsic3 = prepare_arg_intrinsic(arg3)
    intrinsic4 = prepare_arg_intrinsic(arg4)
    intrinsic5 = prepare_arg_intrinsic(arg5)

    num_args = len(overload.args)

    def execute_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None):
        _link_fatbin()
        converted_args = (
            intrinsic0(arg0),
            intrinsic1(arg1),
            intrinsic2(arg2),
            intrinsic3(arg3),
            intrinsic4(arg4),
            intrinsic5(arg5),
        )
        device_func(*converted_args[:num_args])

    return execute_impl


# ==========================
# User API dispatch
# =========================


@cuda.jit(device=True, forceinline=True)
def _calculate_strides(shape: tuple[int, int], ld: int, is_col_major: bool) -> tuple[int, int, int]:
    return (ld * shape[1], 1, ld) if is_col_major else (ld * shape[0], ld, 1)


class _UserApiSolverDispatch(NamedTuple):
    method: str  # name of the method to overload
    solver_attr: str  # name of the solver attribute to forward execution to
    factory: Callable  # factory function to create the overload
    expected_lds: list[Literal["lda", "ldb", "ldc"]]  # list of expected leading dimensions


class _UserApiDeviceMethod(NamedTuple):
    method: str  # name of the method to overload
    factory: Callable  # factory function to create the overload


def _validate_ld_type(ld, name):
    if ld not in {None, types.Omitted(None), types.none} and not isinstance(ld, types.Integer):
        raise RuntimeError(f"{name} must be an Integer!")


def _create_solver_dispatch_overload(numba_type, method, solver_attr, factory, expected_lds):
    @overload_method(numba_type, method, target="cuda", jit_options={"forceinline": True}, strict=False)
    def overload_user_api_method(
        user_api, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None
    ):
        assert isinstance(user_api, numba_type)

        def expected_ensure_provided(ld, name, expected):
            if expected == name and ld is None:
                raise RuntimeError(f"{name} must be provided to use runtime leading dimensions")

        _validate_ld_type(lda, "lda")
        _validate_ld_type(ldb, "ldb")
        _validate_ld_type(ldc, "ldc")

        use_lds = (
            lda not in {None, types.Omitted(None), types.none}
            or ldb not in {None, types.Omitted(None), types.none}
            or ldc not in {None, types.Omitted(None), types.none}
        )
        if use_lds:
            for expected_ld in expected_lds:
                expected_ensure_provided(lda, "lda", expected_ld)
                expected_ensure_provided(ldb, "ldb", expected_ld)
                expected_ensure_provided(ldc, "ldc", expected_ld)

        solver = getattr(user_api.solver, solver_attr)
        return factory(solver, use_lds)

    return overload_user_api_method


def _create_device_method_overload(numba_type, method, factory):
    @overload_method(numba_type, method, target="cuda", jit_options={"forceinline": True}, strict=False)
    def overload_user_api_method(user_api, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        assert isinstance(user_api, numba_type)

        _validate_ld_type(lda, "lda")
        _validate_ld_type(ldb, "ldb")
        _validate_ld_type(ldc, "ldc")

        return factory(user_api, arg0, arg1, arg2, lda, ldb, ldc)

    return overload_user_api_method


def _ensure_no_positional_arguments(arg0, function):
    if arg0 not in {None, types.Omitted(None), types.none}:
        raise RuntimeError(f"Function {function} does not accept positional arguments")


def _register_user_api_dispatch(
    user_api_class: type,
    definition_args: list[str],
    helpers_prop: list[str],
    solver_dispath_methods: list[_UserApiSolverDispatch],
    device_methods: list[_UserApiDeviceMethod],
):
    """
    Registers new numba type for a user API class and overloads the device methods.

    Assumes the API object has hidden fields containing Solver objects.
    Uses device_methods_attrs (which maps method names to solver fields)
    to forward execution to the execute() method of the corresponding solver.
    """

    class UserApiNumbaType(types.Type):
        def __init__(self, solver) -> None:
            self._solver = solver

            attributes = [f"{attr}={getattr(solver, attr)}" for attr in definition_args if getattr(solver, attr)]
            attributes.sort()
            attr_str = ", ".join(attributes)

            super().__init__(name=f"{user_api_class.__name__}({attr_str})")

        @property
        def solver(self):
            return self._solver

    register_dummy_numba_type(UserApiNumbaType, user_api_class, "solver", definition_args + helpers_prop)

    for solver_dispath_method in solver_dispath_methods:
        _create_solver_dispatch_overload(
            UserApiNumbaType,
            solver_dispath_method.method,
            solver_dispath_method.solver_attr,
            solver_dispath_method.factory,
            solver_dispath_method.expected_lds,
        )

    for device_method in device_methods:
        _create_device_method_overload(UserApiNumbaType, device_method.method, device_method.factory)


# ==========================
# Stride factories
# ==========================


def _a_strides_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "a_strides")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return _calculate_strides(
            (solver.a_shape[1], solver.a_shape[2]), lda if lda is not None else solver.lda, solver.a_arrangement == "col_major"
        )

    return impl


_A_STRIDES_DEVICE_METHOD = _UserApiDeviceMethod(
    method="a_strides",
    factory=_a_strides_factory,
)


def _b_strides_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "b_strides")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return _calculate_strides(
            (solver.b_shape[1], solver.b_shape[2]), ldb if ldb is not None else solver.ldb, solver.b_arrangement == "col_major"
        )

    return impl


_B_STRIDES_DEVICE_METHOD = _UserApiDeviceMethod(
    method="b_strides",
    factory=_b_strides_factory,
)


def _c_strides_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "c_strides")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return _calculate_strides(
            (solver.c_shape[1], solver.c_shape[2]), ldc if ldc is not None else solver.ldb, solver.b_arrangement == "col_major"
        )

    return impl


_C_STRIDES_DEVICE_METHOD = _UserApiDeviceMethod(
    method="c_strides",
    factory=_c_strides_factory,
)


def _bx_strides_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "bx_strides")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return _calculate_strides(
            (max(solver.m, solver.n), solver.k), ldb if ldb is not None else solver.ldb, solver.b_arrangement == "col_major"
        )

    return impl


_BX_STRIDES_DEVICE_METHOD = _UserApiDeviceMethod(
    method="bx_strides",
    factory=_bx_strides_factory,
)


# ==========================
# Size (element count) factories
# ==========================


def _bx_size_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "bx_size")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return solver.bx_strides(ldb=ldb)[0] * solver.batches_per_block

    return impl


_BX_SIZE_DEVICE_METHOD = _UserApiDeviceMethod(
    method="bx_size",
    factory=_bx_size_factory,
)


def _a_size_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "a_size")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return solver.a_strides(lda=lda)[0] * solver.a_shape[0]

    return impl


_A_SIZE_DEVICE_METHOD = _UserApiDeviceMethod(
    method="a_size",
    factory=_a_size_factory,
)


def _b_size_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "b_size")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return solver.b_strides(ldb=ldb)[0] * solver.b_shape[0]

    return impl


_B_SIZE_DEVICE_METHOD = _UserApiDeviceMethod(
    method="b_size",
    factory=_b_size_factory,
)


def _c_size_factory(user_api, arg0, arg1, arg2, lda, ldb, ldc) -> Callable:
    _ensure_no_positional_arguments(arg0, "c_size")

    def impl(solver, arg0=None, arg1=None, arg2=None, lda=None, ldb=None, ldc=None):
        return solver.c_strides(ldc=ldc)[0] * solver.c_shape[0]

    return impl


_C_SIZE_DEVICE_METHOD = _UserApiDeviceMethod(
    method="c_size",
    factory=_c_size_factory,
)


# ==========================
# Cholesky Solver dispatch
# ==========================


def _factorize_factory(solver: Solver | None, use_lds: bool) -> Callable:
    assert solver is not None

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1)

    return user_api_method_impl


def _solve_factory(solver: Solver | None, use_lds: bool) -> Callable:
    if solver is None:
        raise RuntimeError(
            "Device function: solve is not available with this configuration: Operation is permitted only for square matrices"
        )
    assert solver.m == solver.n

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1, ldb)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1)

    return user_api_method_impl


_CHOLESKY_SOLVER_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "arrangement",
    "fill_mode",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
    "block_dim",
]

_CHOLESKY_SOLVER_PROPS = [
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "block_size",
    "info_type",
    "info_shape",
    "info_strides",
    "a_shape",
    "b_shape",
]

_CHOLESKY_SOLVER_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="factorize",
        solver_attr="_factorize",
        factory=_factorize_factory,
        expected_lds=["lda"],
    ),
    _UserApiSolverDispatch(
        method="solve",
        solver_attr="_solve",
        factory=_solve_factory,
        expected_lds=["lda", "ldb"],
    ),
]

_register_user_api_dispatch(
    CholeskySolver,
    _CHOLESKY_SOLVER_DEFINITION_ARGS,
    _CHOLESKY_SOLVER_PROPS,
    _CHOLESKY_SOLVER_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _B_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _B_SIZE_DEVICE_METHOD,
    ],
)

# ==========================
# LU Solver dispatch
# ==========================

_LU_SOLVER_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "arrangement",
    "transpose_mode",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
    "block_dim",
]

_LU_SOLVER_PROPS = [
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "block_size",
    "info_type",
    "info_shape",
    "info_strides",
    "a_shape",
    "b_shape",
]

_LU_SOLVER_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="factorize",
        solver_attr="_factorize",
        factory=_factorize_factory,
        expected_lds=["lda"],
    ),
    _UserApiSolverDispatch(
        method="solve",
        solver_attr="_solve",
        factory=_solve_factory,
        expected_lds=["lda", "ldb"],
    ),
]

_register_user_api_dispatch(
    LUSolver,
    _LU_SOLVER_DEFINITION_ARGS,
    _LU_SOLVER_PROPS,
    _LU_SOLVER_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _B_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _B_SIZE_DEVICE_METHOD,
    ],
)

# ==========================
# Triangular Solver dispatch
# ==========================


def _triangular_solve_factory(solver: Solver | None, use_lds: bool) -> Callable:
    assert solver is not None

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1, ldb)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1)

    return user_api_method_impl


_TRIANGULAR_SOLVER_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "side",
    "fill_mode",
    "diag",
    "transpose_mode",
    "arrangement",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
]

_TRIANGULAR_SOLVER_PROPS = [
    "block_dim",
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "block_size",
    "a_shape",
    "b_shape",
]

_TRIANGULAR_SOLVER_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="solve",
        solver_attr="_solve",
        factory=_triangular_solve_factory,
        expected_lds=["lda", "ldb"],
    ),
]

_register_user_api_dispatch(
    TriangularSolver,
    _TRIANGULAR_SOLVER_DEFINITION_ARGS,
    _TRIANGULAR_SOLVER_PROPS,
    _TRIANGULAR_SOLVER_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _B_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _B_SIZE_DEVICE_METHOD,
    ],
)

# ==========================
# LU Pivot Solver dispatch
# ==========================


def _factorize_pivot_factory(solver: Solver | None, use_lds: bool) -> Callable:
    assert solver is not None

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1, arg2)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1, arg2)

    return user_api_method_impl


def _solve_pivot_factory(solver: Solver | None, use_lds: bool) -> Callable:
    if solver is None:
        raise RuntimeError(
            "Device function: solve is not available with this configuration: Operation is permitted only for square matrices"
        )
    assert solver.m == solver.n

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1, arg2, ldb)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1, arg2)

    return user_api_method_impl


_LU_PIVOT_SOLVER_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "arrangement",
    "transpose_mode",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
    "block_dim",
]

_LU_PIVOT_SOLVER_PROPS = [
    "block_size",
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "info_type",
    "info_shape",
    "info_strides",
    "ipiv_type",
    "ipiv_size",
    "ipiv_shape",
    "ipiv_strides",
    "a_shape",
    "b_shape",
]

_LU_PIVOT_SOLVER_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="factorize",
        solver_attr="_factorize",
        factory=_factorize_pivot_factory,
        expected_lds=["lda"],
    ),
    _UserApiSolverDispatch(
        method="solve",
        solver_attr="_solve",
        factory=_solve_pivot_factory,
        expected_lds=["lda", "ldb"],
    ),
]

_register_user_api_dispatch(
    LUPivotSolver,
    _LU_PIVOT_SOLVER_DEFINITION_ARGS,
    _LU_PIVOT_SOLVER_PROPS,
    _LU_PIVOT_SOLVER_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _B_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _B_SIZE_DEVICE_METHOD,
    ],
)

# ==========================
# QRFactors dispatch
# ==========================

_QR_FACTORS_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "batches_per_block",
    "data_type",
    "block_dim",
    "a_arrangement",
    "lda",
]

_QR_FACTORS_PROPS = [
    "value_type",
    "m",
    "n",
    "block_size",
    "tau_type",
    "tau_size",
    "tau_shape",
    "tau_strides",
    "a_shape",
]

_QR_FACTORS_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="factorize",
        solver_attr="_factorize",
        factory=_factorize_factory,
        expected_lds=["lda"],
    ),
]

_register_user_api_dispatch(
    QRFactorize,
    _QR_FACTORS_DEFINITION_ARGS,
    _QR_FACTORS_PROPS,
    _QR_FACTORS_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
    ],
)

_register_user_api_dispatch(
    LQFactorize,
    _QR_FACTORS_DEFINITION_ARGS,
    _QR_FACTORS_PROPS,
    _QR_FACTORS_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
    ],
)

# ==========================
# Orthogonal Multiply dispatch
# ==========================


def _orthogonal_multiply_factory(solver: Solver | None, use_lds: bool) -> Callable:
    assert solver is not None

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1, arg2, ldc)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1, arg2)

    return user_api_method_impl


_ORTHOGONAL_MULTIPLY_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "arrangement",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
    "block_dim",
    "transpose_mode",
    "side",
]

_ORTHOGONAL_MULTIPLY_PROPS = [
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "block_size",
    "tau_type",
    "tau_size",
    "tau_shape",
    "tau_strides",
    "a_shape",
    "c_shape",
]

_QR_MULTIPLY_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="multiply",
        solver_attr="_multiply",
        factory=_orthogonal_multiply_factory,
        expected_lds=["lda", "ldc"],
    ),
]

_register_user_api_dispatch(
    QRMultiply,
    _ORTHOGONAL_MULTIPLY_DEFINITION_ARGS,
    _ORTHOGONAL_MULTIPLY_PROPS,
    _QR_MULTIPLY_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _C_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _C_SIZE_DEVICE_METHOD,
    ],
)

_register_user_api_dispatch(
    LQMultiply,
    _ORTHOGONAL_MULTIPLY_DEFINITION_ARGS,
    _ORTHOGONAL_MULTIPLY_PROPS,
    _QR_MULTIPLY_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _C_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _C_SIZE_DEVICE_METHOD,
    ],
)

# ==========================
# Least Squares Solver dispatch
# ==========================


def _least_squares_solve_factory(solver: Solver | None, use_lds: bool) -> Callable:
    assert solver is not None

    if use_lds:

        def user_api_method_lds_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
            solver.execute(arg0, lda, arg1, arg2, ldb)

        return user_api_method_lds_impl

    def user_api_method_impl(_, arg0, arg1, arg2=None, arg3=None, arg4=None, arg5=None, lda=None, ldb=None, ldc=None):
        solver.execute(arg0, arg1, arg2)

    return user_api_method_impl


_LEAST_SQUARES_SOLVER_DEFINITION_ARGS = [
    "size",
    "precision",
    "execution",
    "sm",
    "arrangement",
    "transpose_mode",
    "batches_per_block",
    "data_type",
    "leading_dimensions",
    "block_dim",
]

_LEAST_SQUARES_SOLVER_PROPS = [
    "value_type",
    "m",
    "n",
    "k",
    "a_arrangement",
    "b_arrangement",
    "lda",
    "ldb",
    "block_size",
    "tau_type",
    "tau_size",
    "tau_shape",
    "tau_strides",
    "a_shape",
    "b_shape",
    "x_shape",
]

_LEAST_SQUARES_SOLVER_DISPATCH_METHODS = [
    _UserApiSolverDispatch(
        method="solve",
        solver_attr="_solve",
        factory=_least_squares_solve_factory,
        expected_lds=["lda", "ldb"],
    ),
]

_register_user_api_dispatch(
    LeastSquaresSolver,
    _LEAST_SQUARES_SOLVER_DEFINITION_ARGS,
    _LEAST_SQUARES_SOLVER_PROPS,
    _LEAST_SQUARES_SOLVER_DISPATCH_METHODS,
    [
        _A_STRIDES_DEVICE_METHOD,
        _BX_STRIDES_DEVICE_METHOD,
        _A_SIZE_DEVICE_METHOD,
        _BX_SIZE_DEVICE_METHOD,
    ],
)
