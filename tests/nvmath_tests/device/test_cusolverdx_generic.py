# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest

from nvmath.device import Code, CodeType, ComputeCapability, Solver, compile_solver_execute
from nvmath.device.cusolverdx_backend import (
    _ENABLE_CUSOLVERDX_0_3,
    ALLOWED_CUSOLVERDX_FUNCTIONS,
    CUSOLVERDX_0_3_ALLOWED_FUNCTIONS,
)

from .cusolverdx_common import (
    ADDITIONAL_REQUIRED_KWARGS,
    CHOLESKY_BASIC_EXAMPLE_KWARGS,
    DIAG_SUPPORTED_FUNCS,
    DIAG_SUPPORTED_VALUES,
    DIAG_UNSUPPORTED_FUNCS,
    FILL_MODE_SUPPORTED_FUNCS,
    FILL_MODE_SUPPORTED_VALUES,
    FILL_MODE_UNSUPPORTED_FUNCS,
    MINIMAL_KWARGS,
    SIDE_SUPPORTED_FUNCS,
    SIDE_SUPPORTED_VALUES,
    SIDE_UNSUPPORTED_FUNCS,
    SOLVER_FUNCTION_TO_CLASS,
    TRANSPOSE_MODE_FULL_SUPPORT_FUNCS,
    TRANSPOSE_MODE_FULL_SUPPORT_VALUES,
    TRANSPOSE_MODE_PARTIAL_SUPPORT_FUNCS,
    TRANSPOSE_MODE_PARTIAL_SUPPORT_UNSUPPORTED_VALUES,
    TRANSPOSE_MODE_PARTIAL_SUPPORT_VALUES,
    TRANSPOSE_MODE_UNSUPPORTED_FUNCS,
)
from .helpers import (
    SM70,
    SM72,
    SM75,
    SM80,
    SM86,
    SM89,
    SM90,
    SM100,
    SM101,
    SM103,
    SM120,
    SM121,
    requires_ctk,
    skip_unsupported_sm,
)

pytestmark = requires_ctk((12, 6, 85))  # CTK 12.6 Update 3


def _skip_if_not_supported(function):
    if function in CUSOLVERDX_0_3_ALLOWED_FUNCTIONS and not _ENABLE_CUSOLVERDX_0_3:
        pytest.skip(f"Function {function} is not supported with libmathdx < 0.3.2")


def _calculate_ld(rows, cols, arr):
    return rows if arr == "col_major" else cols


# ======================================
# Basic class construction test
# ======================================


@pytest.fixture(autouse=True)
def skip_if_nvbug_5288270(request):
    import os

    from nvmath.device.common import parse_sm

    func_name = None
    if hasattr(request.node, "callspec"):
        if "function" in request.node.callspec.params:
            func_name = request.node.callspec.params["function"]
        elif "func" in request.node.callspec.params:
            func_name = request.node.callspec.params["func"]

    if func_name == "gesv_no_pivot":
        sm = parse_sm(None)
        if sm.major == 12 and sm.minor == 0 and "NVMATH_CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT" not in os.environ:
            pytest.skip("NVBUG 5288270")


def test_construct_solver():
    solver = Solver(**CHOLESKY_BASIC_EXAMPLE_KWARGS)

    assert solver.function == "potrf"
    assert solver.size == (32, 32, 1)
    assert solver.precision == np.float64
    assert solver.execution == "Block"
    assert solver.fill_mode == "upper"
    assert solver.leading_dimensions == (33, 32)

    with pytest.raises(RuntimeError):
        solver.execute(1, 2, 3)


# ======================================
# Basic libmathdx integration test
# ======================================


def test_construct_solver_access_traits():
    args = CHOLESKY_BASIC_EXAMPLE_KWARGS.copy()
    args.update({"block_dim": "suggested"})
    solver = Solver(**args)

    assert len(solver.block_dim) == 3
    assert len(solver.size) == 3


def test_suggested_traits():
    args = CHOLESKY_BASIC_EXAMPLE_KWARGS.copy()
    args.update(
        {
            "size": (7, 7),
            "block_dim": "suggested",
            "batches_per_block": "suggested",
        }
    )

    solver = Solver(**args)
    assert solver.block_dim[0] > 0
    assert solver.block_dim[1] > 0
    assert solver.block_dim[2] > 0
    assert solver.leading_dimensions[0] >= 7
    assert solver.batches_per_block > 1
    assert len(solver.block_dim) == 3
    assert len(solver.size) == 3
    assert solver.block_size > 0


@pytest.mark.parametrize("size", [(32,), (32, 32), [32]])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("arrangement", [("col_major", "row_major"), ["col_major", "col_major"]])
@pytest.mark.parametrize("leading_dimensions", [(33, 33), [33, 33]])
@pytest.mark.parametrize(
    "block_dim",
    [
        [
            32,
        ],
        (8, 8),
    ],
)
def test_construct_solver_various_inputs(size, precision, arrangement, leading_dimensions, block_dim):
    args = MINIMAL_KWARGS.copy()
    args.update(ADDITIONAL_REQUIRED_KWARGS.get("trsm", {}))
    args.update(
        {
            "function": "trsm",
            "size": size,
            "precision": precision,
            "arrangement": arrangement,
            "leading_dimensions": leading_dimensions,
            "block_dim": block_dim,
        }
    )

    solver = Solver(**args)
    code, _ = compile_solver_execute(solver, code_type=SM90, execution_api="compiled_leading_dim")
    assert len(code.data) > 0

    assert len(solver.block_dim) == 3
    assert len(solver.size) == 3
    assert solver.block_size > 0


@pytest.mark.parametrize("function", CUSOLVERDX_0_3_ALLOWED_FUNCTIONS)
def test_versioning(function):
    import nvmath.device.cusolverdx_backend as backend

    args = MINIMAL_KWARGS.copy()
    args["function"] = function
    args.update(ADDITIONAL_REQUIRED_KWARGS.get(function, {}))

    if backend._ENABLE_CUSOLVERDX_0_3:
        solver = Solver(**args)
        code, _ = compile_solver_execute(solver, code_type=SM90, execution_api="compiled_leading_dim")
        assert len(code.data) > 0
    else:
        with pytest.raises(RuntimeError, match=r"requires libmathdx 0\.3\.2"):
            Solver(**args)


def test_nvbug_5288270_exception(monkeypatch):
    skip_unsupported_sm(SM120)
    monkeypatch.delenv("NVMATH_CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", raising=False)

    with pytest.raises(RuntimeError, match="NVBUG 5288270"):
        Solver(
            function="gesv_no_pivot",
            size=(10, 10),
            precision=np.float32,
            execution="Block",
            sm=ComputeCapability(12, 0),
            data_type="real",
        )


def test_nvbug_5288270_ignore(monkeypatch):
    skip_unsupported_sm(SM120)
    monkeypatch.setenv("NVMATH_CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT", "1")

    solver = Solver(
        function="gesv_no_pivot",
        size=(10, 10),
        precision=np.float32,
        execution="Block",
        sm=ComputeCapability(12, 0),
        data_type="real",
    )
    code, _ = compile_solver_execute(solver, code_type=SM120, execution_api="compiled_leading_dim")
    assert len(code.data) > 0


@pytest.mark.parametrize("function", ALLOWED_CUSOLVERDX_FUNCTIONS)
def test_all_functions_compile(function):
    _skip_if_not_supported(function)

    args = MINIMAL_KWARGS.copy()
    args["function"] = function
    args.update(ADDITIONAL_REQUIRED_KWARGS.get(function, {}))
    solver = Solver(**args)

    code, _ = compile_solver_execute(solver, code_type=SM90, execution_api="compiled_leading_dim")
    assert len(code.data) > 0


# ======================================
# Argument type validation tests
# ======================================


@pytest.mark.parametrize("size", [(32,), (32, 32), (32, 32, 32), [32], [32, 32], [32, 32, 32]])
@pytest.mark.parametrize("precision", [np.float32, np.float64])
@pytest.mark.parametrize("arrangement", [("col_major", "row_major"), ["col_major", "col_major"]])
@pytest.mark.parametrize("fill_mode", ["upper", "lower"])
@pytest.mark.parametrize("batches_per_block", [1, 32])
@pytest.mark.parametrize("data_type", ["real", "complex"])
@pytest.mark.parametrize("leading_dimensions", [(33, 33), [33, 33]])
@pytest.mark.parametrize("block_dim", [[8, 8, 8], (8, 8, 8)])
def test_cholesky_valid_arguments(
    size, precision, arrangement, fill_mode, batches_per_block, data_type, leading_dimensions, block_dim
):
    solver = Solver(
        function="potrf",
        size=size,
        precision=precision,
        execution="Block",
        fill_mode=fill_mode,
        leading_dimensions=leading_dimensions,
        data_type=data_type,
        batches_per_block=batches_per_block,
        arrangement=arrangement,
        block_dim=block_dim,
    )

    assert solver.function == "potrf"
    assert solver.size[0] == 32
    assert solver.size[1] == 32
    assert solver.size[2] == 32 or solver.size[2] == 1
    assert solver.precision == precision
    assert solver.execution == "Block"
    assert solver.fill_mode == fill_mode
    assert solver.leading_dimensions == tuple(leading_dimensions)
    assert solver.data_type == data_type
    assert solver.batches_per_block == batches_per_block
    assert solver.arrangement == tuple(arrangement)
    assert solver.block_dim == tuple(block_dim)
    assert len(solver.block_dim) == 3
    assert len(solver.size) == 3
    assert solver.block_size > 0


INVALID_ARGUMENT_TYPES = {
    "size": [(0,), (32, 0), (0, 32), (32, 32, 0), (0, 32, 32), (32, 0, 32), 123, "123"],
    "arrangement": [("col_major", "col_major", "col_major"), ("C", "C"), "col_major", 321],
    "batches_per_block": [0, "0", "opt"],
    "leading_dimensions": [32, "opt", "32", (32, 32, 32)],
    "block_dim": [32, (32, 32, "z"), (0, 0, 0), "opt"],
}


@pytest.mark.parametrize(
    "arg_name,invalid_value",
    [(name, val) for name, invalids in INVALID_ARGUMENT_TYPES.items() for val in invalids],
)
# Ensure only one invalid argument in time is set
def test_invalid_argument_types(arg_name, invalid_value):
    with pytest.raises(ValueError):
        args = MINIMAL_KWARGS.copy()
        args.update(ADDITIONAL_REQUIRED_KWARGS.get("trsm", {}))
        args.update({"function": "trsm", arg_name: invalid_value})
        Solver(**args)


INVALID_STRING_ARGUMENTS = {
    "precision": [np.float16, np.int32, np.complex64],
    "execution": ["thread", "async", "sync", 1],
    "transpose_mode": ["T", "Trans", "Non", 1],
    "side": ["leftside", ""],
    "diag": ["diagonal", "non", 123],
    "fill_mode": ["top", "bottom", 1],
    "data_type": [np.float32, np.complex128, "comp", 1],
}


@pytest.mark.parametrize(
    "arg_name,invalid_value",
    [(name, val) for name, invalids in INVALID_STRING_ARGUMENTS.items() for val in invalids],
)
# Ensure only one invalid argument in time is set
def test_invalid_arguments_strings(arg_name, invalid_value):
    with pytest.raises(ValueError):
        args = MINIMAL_KWARGS.copy()
        args.update(ADDITIONAL_REQUIRED_KWARGS.get("trsm", {}))
        args.update({"function": "trsm", arg_name: invalid_value})
        Solver(**args)


# ======================================
# Argument support validation tests
# ======================================


@pytest.mark.parametrize(
    "func,arg_name,value",
    (
        list(itertools.product(SIDE_SUPPORTED_FUNCS, ["side"], SIDE_SUPPORTED_VALUES))
        + list(itertools.product(DIAG_SUPPORTED_FUNCS, ["diag"], DIAG_SUPPORTED_VALUES))
        + list(itertools.product(FILL_MODE_SUPPORTED_FUNCS, ["fill_mode"], FILL_MODE_SUPPORTED_VALUES))
        + list(itertools.product(TRANSPOSE_MODE_FULL_SUPPORT_FUNCS, ["transpose_mode"], TRANSPOSE_MODE_FULL_SUPPORT_VALUES))
        + list(
            itertools.product(TRANSPOSE_MODE_PARTIAL_SUPPORT_FUNCS, ["transpose_mode"], TRANSPOSE_MODE_PARTIAL_SUPPORT_VALUES)
        )
    ),
)
def test_argument_is_supported(func, arg_name, value):
    _skip_if_not_supported(func)

    args = MINIMAL_KWARGS.copy()
    args.update(ADDITIONAL_REQUIRED_KWARGS.get(func, {}))
    args.update({"function": func, arg_name: value})
    Solver(**args)

    if func in SOLVER_FUNCTION_TO_CLASS and not (arg_name == "transpose_mode" and func in {"geqrf", "gelqf"}):
        del args["function"]
        SOLVER_FUNCTION_TO_CLASS[func](**args)


@pytest.mark.parametrize(
    "func,arg_name,value",
    (
        list(itertools.product(SIDE_UNSUPPORTED_FUNCS, ["side"], SIDE_SUPPORTED_VALUES))
        + list(itertools.product(DIAG_UNSUPPORTED_FUNCS, ["diag"], DIAG_SUPPORTED_VALUES))
        + list(itertools.product(FILL_MODE_UNSUPPORTED_FUNCS, ["fill_mode"], FILL_MODE_SUPPORTED_VALUES))
        + list(itertools.product(TRANSPOSE_MODE_UNSUPPORTED_FUNCS, ["transpose_mode"], TRANSPOSE_MODE_FULL_SUPPORT_VALUES))
        + list(
            itertools.product(
                TRANSPOSE_MODE_PARTIAL_SUPPORT_FUNCS, ["transpose_mode"], TRANSPOSE_MODE_PARTIAL_SUPPORT_UNSUPPORTED_VALUES
            )
        )
    ),
)
def test_argument_is_not_supported(func, arg_name, value):
    _skip_if_not_supported(func)

    with pytest.raises(ValueError):
        args = MINIMAL_KWARGS.copy()
        args.update(ADDITIONAL_REQUIRED_KWARGS.get(func, {}))
        args.update({"function": func, arg_name: value})
        Solver(**args)

    # LuSolver class enforces non transposed parameter for those functions
    if func in SOLVER_FUNCTION_TO_CLASS and func not in {"getrf_no_pivot", "getrf_partial_pivot"}:
        with pytest.raises(Exception):  # noqa: B017
            del args["function"]
            SOLVER_FUNCTION_TO_CLASS[func](**args)


# ======================================
# Required optional validation tests
# ======================================


@pytest.mark.parametrize(
    "func,arg_name",
    [(func, arg_name) for func in ALLOWED_CUSOLVERDX_FUNCTIONS for arg_name in ADDITIONAL_REQUIRED_KWARGS.get(func, {})],
)
def test_required_optionals_are_required(func, arg_name):
    _skip_if_not_supported(func)

    args = MINIMAL_KWARGS.copy()
    args["function"] = func
    args.update(ADDITIONAL_REQUIRED_KWARGS.get(func, {}))
    del args[arg_name]

    with pytest.raises(ValueError):
        Solver(**args)

    if func in SOLVER_FUNCTION_TO_CLASS:
        with pytest.raises(Exception):  # noqa: B017
            del args["function"]
            SOLVER_FUNCTION_TO_CLASS[func](**args)


# ======================================
# Compilation tests
# ======================================


@pytest.mark.parametrize("execution_api", ["compiled_leading_dim", "runtime_leading_dim"])
def test_basic_compilation(execution_api):
    solver = Solver(**CHOLESKY_BASIC_EXAMPLE_KWARGS)
    code, symbol = compile_solver_execute(solver, code_type=SM90, execution_api=execution_api)

    assert len(symbol) > 0
    assert isinstance(code, Code)
    assert code.code_type.kind == "lto"
    assert code.code_type.cc.major == 9
    assert code.code_type.cc.minor == 0
    assert isinstance(code.data, bytes)
    assert len(code.data) > 0


@pytest.mark.parametrize("execution_api", ["fast", "async", "sync", "smem"])
def test_invalid_arguments_execution_api(execution_api):
    solver = Solver(**CHOLESKY_BASIC_EXAMPLE_KWARGS)
    with pytest.raises(Exception):  # noqa: B017
        compile_solver_execute(solver, code_type=SM90, execution_api=execution_api)


@pytest.mark.parametrize(
    "code_type",
    [
        None,
        CodeType("lto", ComputeCapability(-1, 0)),
        CodeType("lto", ComputeCapability(5, 0)),
        CodeType("sass", ComputeCapability(7, 0)),
        CodeType("ptx", ComputeCapability(7, 0)),
        CodeType("lto", ComputeCapability(1000, 0)),  # invalid cc > supported Max cc
        ("lto", "lto", ComputeCapability(10, 0)),  # len(code_type) != 2
    ],
)
def test_invalid_arguments_code_type(code_type):
    solver = Solver(**CHOLESKY_BASIC_EXAMPLE_KWARGS)
    with pytest.raises(Exception):  # noqa: B017
        compile_solver_execute(solver, code_type=code_type, execution_api="compiled_leading_dim")


@pytest.mark.parametrize("code_type", [SM70, SM72, SM75, SM80, SM86, SM89, SM90, SM100, SM101, SM103, SM120, SM121])
def test_compile_sm(code_type):
    skip_unsupported_sm(code_type)
    solver = Solver(**CHOLESKY_BASIC_EXAMPLE_KWARGS)
    code, symbol = compile_solver_execute(solver, code_type=code_type, execution_api="compiled_leading_dim")

    assert isinstance(code.data, bytes)
    assert len(code.data) > 0
    assert len(symbol) > 0


def test_default_arguments():
    args = MINIMAL_KWARGS.copy()
    args.update(ADDITIONAL_REQUIRED_KWARGS.get("trsm", {}))
    args.update({"function": "trsm"})
    solver = Solver(**args)

    assert solver.arrangement == ("col_major", "col_major")
    assert solver.batches_per_block == 1
    assert solver.data_type == "real"
    assert solver.block_dim is not None
    assert solver.leading_dimensions is not None


# ======================================
# Default leading dimensions tests
# ======================================


@pytest.mark.parametrize("function", ALLOWED_CUSOLVERDX_FUNCTIONS)
@pytest.mark.parametrize("size", [(32, 32, 8), (16, 24, 8), (24, 16, 12)])
@pytest.mark.parametrize(
    "arrangement",
    [("col_major", "col_major"), ("row_major", "row_major"), ("col_major", "row_major"), ("row_major", "col_major")],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_default_leading_dimensions(function, size, arrangement, side):
    _skip_if_not_supported(function)

    args = MINIMAL_KWARGS.copy()
    args.update(ADDITIONAL_REQUIRED_KWARGS.get(function, {}))
    args.update(
        {
            "function": function,
            "size": size,
            "arrangement": arrangement,
        }
    )

    if function in ["trsm", "unmqr", "unmlq"]:
        args["side"] = side
    solver = Solver(**args)

    m, n, k = size
    arr_a, arr_b = arrangement

    if function in ["geqrf", "gelqf", "htev", "unglq", "ungqr", "heev"]:
        lda = _calculate_ld(m, n, arr_a)
        ldb = 0
    elif function == "trsm":
        lda = m if side == "left" else n
        ldb = _calculate_ld(m, n, arr_b)
    elif function == "unmqr":
        lda = _calculate_ld(m if side == "left" else n, k, arr_a)
        ldb = _calculate_ld(m, n, arr_b)
    elif function == "unmlq":
        lda = _calculate_ld(k, m if side == "left" else n, arr_a)
        ldb = _calculate_ld(m, n, arr_b)
    elif function == "gels":  # mirrors cusolverdx behaviour
        lda = _calculate_ld(m, n, arr_a)
        ldb = _calculate_ld(max(m, n), k, arr_b)
    elif function == "gtsv_no_pivot":
        lda = 0
        ldb = _calculate_ld(n, k, arr_b)
    else:
        lda = _calculate_ld(m, n, arr_a)
        ldb = _calculate_ld(n, k, arr_b)

    assert solver.leading_dimensions == (lda, ldb)
