# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple
import numpy as np
import time
from nvmath.device import (
    Dim3,
    CodeType,
    ComputeCapability,
    matmul,
    TransposeMode,
    LeadingDimension,
    BlasOptions,
)
from nvmath.device.common_cuda import MAX_SUPPORTED_CC, get_default_code_type
from nvmath.device.cublasdx import BlasCompiled, BlasOptionsComplete, SharedStorageCalc
import functools
import pytest
import itertools

from nvmath.device.cublasdx_backend import Alignment, MAX_ALIGNMENT
from .helpers import (
    SM100,
    SM101,
    SM103,
    SM120,
    SM121,
    SM70,
    SM72,
    SM75,
    SM80,
    SM86,
    SM89,
    SM90,
    skip_nvbug_5218000,
    AssertFilesClosed,
    skip_unsupported_sm,
)


def test_files_closed():
    with AssertFilesClosed():
        _ = matmul(
            size=(16, 8, 16),
            data_type="real",
            precision=np.float32,
            transpose_mode=TransposeMode("non_transposed", "transposed"),
            code_type=SM75,
            execution="Block",
        )


@pytest.mark.parametrize("execute_api", ["static_leading_dimensions", "dynamic_leading_dimensions"])
def test_third_party_symbol(execute_api):
    MM = matmul(
        size=(24, 8, 48),
        data_type="real",
        precision=np.float64,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM75,
        execution="Block",
        execute_api=execute_api,
    )

    assert len(MM.symbol) > 0


def test_third_party_code():
    MM = matmul(
        size=(16, 8, 16),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM75,
        execution="Block",
    )

    assert len(MM.codes) > 0
    for code in MM.codes:
        print(code.code_type, code.isa_version)
    assert isinstance(MM, BlasCompiled)
    assert MM.size == (16, 8, 16)
    assert all(f.endswith(".ltoir") for f in MM.files)
    assert all(code.isa_version.major >= 12 for code in MM.codes)
    assert all(code.isa_version.minor >= 0 for code in MM.codes)
    assert all(code.code_type.cc.major == 7 for code in MM.codes)
    assert all(code.code_type.cc.minor == 5 for code in MM.codes)
    assert all(code.code_type.kind == "lto" for code in MM.codes)
    assert all(isinstance(code.data, bytes) for code in MM.codes)
    assert all(len(code.data) > 0 for code in MM.codes)
    assert MM.max_threads_per_block <= 1024


@pytest.mark.parametrize("ta, tb", list(itertools.product(["non_transposed", "transposed", "conj_transposed"], repeat=2)))
def test_transpose_mode(ta, tb):
    MM1 = matmul(
        size=(2, 2, 2),
        data_type="complex",
        precision=np.float32,
        transpose_mode=(ta, tb),
        code_type=SM75,
        execution="Block",
    )

    MM2 = matmul(
        size=(2, 2, 2),
        data_type="complex",
        precision=np.float32,
        transpose_mode=TransposeMode(ta, tb),
        code_type=SM75,
        execution="Block",
    )

    assert isinstance(MM1.transpose_mode, TransposeMode)
    assert MM1.transpose_mode == TransposeMode(ta, tb)
    assert MM1.transpose_mode.a == ta
    assert MM1.transpose_mode.b == tb

    assert isinstance(MM2.transpose_mode, TransposeMode)
    assert MM2.transpose_mode == TransposeMode(ta, tb)
    assert MM2.transpose_mode.a == ta
    assert MM2.transpose_mode.b == tb


def test_suggested_block_dim():
    BO = BlasOptions(
        size=(16, 8, 16),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM75,
        execution="Block",
        block_dim="suggested",
    )  # leading_dimension = None implicit

    # block_dim = suggested    --> Dim3
    # leading_dimension = None --> None
    assert isinstance(BO, BlasOptions)
    assert isinstance(BO.block_dim, Dim3)
    assert BO.leading_dimension is None
    assert BO.size == (16, 8, 16)
    assert BO.block_dim[0] * BO.block_dim[1] * BO.block_dim[2] >= 1

    MM = BO.create()

    # block_dim         --> Dim3 (same as above)
    # leading_dimension --> LeadingDimension (takes a default value)
    assert isinstance(MM, BlasCompiled)
    assert isinstance(MM.block_dim, Dim3)
    assert MM.block_dim == BO.block_dim
    assert isinstance(MM.leading_dimension, LeadingDimension)


def test_suggested_leading_dimension():
    BO = BlasOptions(
        size=(16, 8, 16),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM89,
        block_size=64,
        execution="Block",
        leading_dimension="suggested",
    )

    assert isinstance(BO, BlasOptions)
    assert isinstance(BO.leading_dimension, LeadingDimension)
    assert isinstance(BO.block_dim, Dim3)

    assert len(BO.leading_dimension) == 3
    assert BO.leading_dimension.a >= 1
    assert BO.leading_dimension.b >= 1
    assert BO.leading_dimension.c >= 1

    MM = BO.create()
    assert isinstance(MM, BlasCompiled)
    assert isinstance(MM.leading_dimension, LeadingDimension)
    assert isinstance(MM.block_dim, Dim3)

    assert MM.leading_dimension.a == BO.leading_dimension.a
    assert MM.leading_dimension.b == BO.leading_dimension.b
    assert MM.leading_dimension.c == BO.leading_dimension.c


def test_valid_finalize():
    BO = BlasOptions(
        size=(16, 8, 16),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM75,
        execution="Block",
    )

    assert isinstance(BO, BlasOptions)
    valids = BO.valid("block_dim")

    count = 0
    for (block_dim,) in valids:
        count += 1
        MM = BO.create(block_dim=block_dim, code_type=SM80)
        assert isinstance(MM, BlasCompiled)
        assert MM.block_dim == block_dim
        assert MM.size == (16, 8, 16)
        assert MM.code_type == SM80
    assert count > 0


def test_cached():
    make_mm = functools.partial(
        matmul,
        size=(32, 16, 32),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("transposed", "transposed"),
        block_dim=Dim3(2, 4, 8),
        code_type=SM75,
        execution="Block",
        compiler=None,
    )

    t0 = time.time()
    _ = make_mm()
    t1 = time.time()
    t_base = t1 - t0
    print(f"Base, time {t_base} s.")

    for i in range(3):
        t0 = time.time()
        _ = make_mm()
        t1 = time.time()
        t_cached = t1 - t0
        print(f"Step {i}, time {t_cached} s.")
        assert t_cached < 0.5


@pytest.mark.parametrize(
    "opt, value",
    [
        ("size", (2, 3)),
        ("data_type", "reall"),  # codespell:ignore reall
        ("execution", "CGA"),
        ("size", None),
        ("data_type", None),
        ("precision", None),
        ("precision", (np.float32,)),
        ("precision", (np.float32, np.float32)),
        ("code_type", None),
        ("code_type", CodeType("lto", ComputeCapability(-1, 0))),
        ("code_type", CodeType("lto", ComputeCapability(5, 0))),
        ("code_type", CodeType("sass", ComputeCapability(7, 0))),
        ("code_type", CodeType("ptx", ComputeCapability(7, 0))),
        ("block_dim", (1, 2)),
        ("block_dim", (1025, 1, 1)),
        ("transpose_mode", (3, 2)),
        ("function", "SYRK"),
        ("transpose_mode", "non_transposed2"),
        ("transpose_mode", ("non_transposed2",)),
        ("transpose_mode", ("non_transposed2", "non_transposed")),
        ("transpose_mode", ("non_transposed2", "non_transposed", "I_dont_know")),
        ("transpose_mode", TransposeMode("non_transposed2", "non_transposed")),
    ],
)
def test_negative(opt, value):
    opts = {"size": (24, 8, 48), "data_type": "real", "precision": np.float64, "code_type": SM75, "execution": "Block"}
    if value is None:
        del opts[opt]
    else:
        opts[opt] = value
    with pytest.raises(Exception):
        MM = matmul(**opts)  # noqa: F841


@pytest.mark.parametrize("code_type", [SM70, SM72, SM75, SM80, SM86, SM89, SM90, SM100, SM101, SM103, SM120, SM121])
def test_sm(code_type):
    skip_unsupported_sm(code_type)
    MM = matmul(
        size=(24, 8, 48),
        data_type="real",
        arrangement=("col_major", "col_major", "col_major"),
        precision=np.float32,
        code_type=code_type,
        execution="Block",
    )
    assert all(isinstance(code.data, bytes) for code in MM.codes)
    assert all(len(code.data) > 0 for code in MM.codes)
    assert all(code.code_type == code_type for code in MM.codes)


def test_unsupported_sm():
    code_type = CodeType("lto", ComputeCapability(13, 0))
    assert code_type.cc > MAX_SUPPORTED_CC

    with pytest.raises(
        RuntimeError,
        match="The maximum compute capability currently supported by device APIs is 12.1, got 13.0",
    ):
        matmul(
            size=(24, 8, 48),
            data_type="real",
            arrangement=("col_major", "col_major", "col_major"),
            precision=np.float32,
            code_type=code_type,
            execution="Block",
        )


@pytest.mark.parametrize("code_type", [("lto", (7, 5)), ("lto", (8, 0))])
def test_sm_type(code_type):
    MM = matmul(
        size=(24, 8, 48),
        data_type="real",
        arrangement=("col_major", "col_major", "col_major"),
        precision=np.float32,
        code_type=code_type,
        execution="Block",
    )
    assert all(isinstance(code.data, bytes) for code in MM.codes)
    assert all(len(code.data) > 0 for code in MM.codes)
    assert all(code.code_type.kind == code_type[0] for code in MM.codes)
    assert all(code.code_type.cc.major == code_type[1][0] for code in MM.codes)
    assert all(code.code_type.cc.minor == code_type[1][1] for code in MM.codes)


@pytest.mark.parametrize(
    "data_type,precision,value_type",
    [
        ("real", np.float16, np.float16),
        ("real", np.float32, np.float32),
        ("real", np.float64, np.float64),
        ("real", (np.float16, np.float16, np.float16), np.float16),
        ("real", (np.float32, np.float32, np.float32), np.float32),
        ("complex", np.float16, np.dtype([("x", np.float16), ("y", np.float16)], align=True)),
        ("complex", np.float32, np.complex64),
        ("complex", np.float64, np.complex128),
        ("complex", (np.float32, np.float32, np.float32), np.complex64),
        ("complex", (np.float64, np.float64, np.float64), np.complex128),
    ],
)
def test_value_type(data_type, precision, value_type):
    skip_nvbug_5218000(precision, sm=SM90)
    MM = matmul(
        size=(24, 8, 48),
        data_type=data_type,
        precision=precision,
        arrangement=("col_major", "col_major", "col_major"),
        code_type=SM90,
        execution="Block",
    )
    assert MM.a_value_type == value_type
    assert MM.b_value_type == value_type
    assert MM.c_value_type == value_type


@pytest.mark.parametrize(
    "data_type,precision,value_types",
    [
        ("real", (np.float16, np.float16, np.float32), (np.float16, np.float16, np.float32)),
        ("real", (np.float32, np.float32, np.float64), (np.float32, np.float32, np.float64)),
        ("complex", (np.float32, np.float32, np.float64), (np.complex64, np.complex64, np.complex128)),
        (
            "complex",
            (np.float64, np.float64, np.float16),
            (np.complex128, np.complex128, np.dtype([("x", np.float16), ("y", np.float16)], align=True)),
        ),
    ],
)
def test_value_types(data_type, precision, value_types):
    skip_nvbug_5218000(precision, sm=SM90)
    MM = matmul(
        size=(24, 8, 48),
        data_type=data_type,
        precision=precision,
        arrangement=("col_major", "col_major", "col_major"),
        code_type=SM90,
        execution="Block",
    )
    assert MM.a_value_type == value_types[0]
    assert MM.b_value_type == value_types[1]
    assert MM.c_value_type == value_types[2]


@pytest.mark.parametrize(
    "matrixes, expected_size",
    [
        (
            [
                (4, 40),
                (4, 40),
                (4, 40),
            ],
            40 * 3,
        ),
        (
            [
                (4, 4, 10),
                (4, 4, 10),
                (4, 4, 10),
            ],
            40 * 3,
        ),
        (
            [
                (16, 4, 10),
                (16, 4, 10),
                (16, 4, 10),
            ],
            48 * 2 + 40,
        ),
        (
            [
                (4, 4, 10),
                (4, 4, 10),
                (8, 4, 10),
            ],
            40 * 3,
        ),
        (
            [
                (4, 4, 10),
                (4, 4, 11),
                (8, 4, 10),
            ],
            40 + 48 + 40,
        ),
        (
            [
                (2, 2, 1),
                (4, 4, 1),
                (8, 8, 1),
            ],
            16,
        ),
    ],
)
def test_shared_storage_calc(matrixes, expected_size):
    smem = SharedStorageCalc()
    for m in matrixes:
        smem.add(*m)
    assert smem.get() == expected_size


class TestGetSharedStorageSize(NamedTuple):
    args: tuple = ()
    expected_size: int = 0
    expected_error: str | None = None


@pytest.mark.parametrize(
    "MM_kwargs, compiled, t, t_ab",
    [
        (
            {"size": (1, 1, 1), "precision": np.float16},
            False,
            TestGetSharedStorageSize(expected_size=6),
            TestGetSharedStorageSize(expected_size=4),
        ),
        (
            {"size": (1, 1, 1), "precision": np.float16, "alignment": (8, 8, 8)},
            False,
            TestGetSharedStorageSize(expected_size=18),
            TestGetSharedStorageSize(expected_size=10),
        ),
        (
            {
                "size": (1, 1, 1),
                "precision": (np.float16, np.float64, np.float16),
            },
            False,
            TestGetSharedStorageSize(expected_size=18),
            TestGetSharedStorageSize(expected_size=16),
        ),
        (
            {
                "size": (1, 1, 1),
                "precision": (np.float16, np.float64, np.float16),
                "alignment": (8, 8, 8),
            },
            False,
            TestGetSharedStorageSize(expected_size=18),
            TestGetSharedStorageSize(expected_size=16),
        ),
        (
            {"size": (4, 4, 4), "precision": np.float16},
            False,
            TestGetSharedStorageSize(expected_size=96),
            TestGetSharedStorageSize(expected_size=64),
        ),
        (
            {"size": (4, 4, 4), "precision": np.float16, "alignment": (8, 8, 8)},
            False,
            TestGetSharedStorageSize(expected_size=96),
            TestGetSharedStorageSize(expected_size=64),
        ),
        # Test wrong number of arguments
        (
            {"size": (1, 2, 3), "precision": np.float16, "alignment": (2, 4, 8)},
            False,
            TestGetSharedStorageSize(
                args=(1, 2),
                expected_error=r"get_shared_storage_size\(\) takes either 0 or "
                r"3 arguments\. If .*",
            ),
            TestGetSharedStorageSize(
                args=(1, 2),
                expected_error=r"get_shared_storage_size_ab\(\) takes either 0 "
                r"or 2 arguments\. If .*",
            ),
        ),
        (
            {"size": (1, 2, 3), "precision": np.float16, "alignment": (2, 4, 8)},
            False,
            TestGetSharedStorageSize(
                args=(1, 2, 3, 4),
                expected_error=r"get_shared_storage_size\(\) takes either 0 or "
                r"3 arguments\. If .*",
            ),
            TestGetSharedStorageSize(
                args=(1, 2, 3),
                expected_error=r"get_shared_storage_size_ab\(\) takes either 0 "
                r"or 2 arguments\. If .*",
            ),
        ),
        # Test wrong types of arguments
        (
            {"size": (1, 2, 3), "precision": np.float16, "alignment": (2, 4, 8)},
            False,
            TestGetSharedStorageSize(
                args=(1, 2, "3"),
                expected_error=r"get_shared_storage_size\(\) takes either 0 or "
                r"3 arguments\. If .*",
            ),
            TestGetSharedStorageSize(
                args=(1, "2"),
                expected_error=r"get_shared_storage_size_ab\(\) takes either 0 "
                r"or 2 arguments\. If .*",
            ),
        ),
        (
            {
                "size": (1, 2, 3),
                "precision": np.float16,
                "alignment": (2, 4, 8),
                "execute_api": "tensors",
                "tensor_types": ("smem_a", "smem_b", "smem_c"),
            },
            True,
            TestGetSharedStorageSize(
                args=(1, 2, lambda MM: MM.get_layout_smem_c()),  # wrong type
                expected_error=r"get_shared_storage_size\(\) takes either 0 or "
                r"3 arguments\. If .*",
            ),
            TestGetSharedStorageSize(
                args=(1, lambda MM: MM.get_layout_smem_b()),
                expected_error=r"get_shared_storage_size_ab\(\) takes either 0 "
                r"or 2 arguments\. If .*",
            ),
        ),
        (
            {
                "size": (1, 2, 3),
                "precision": np.float16,
                "alignment": (2, 4, 8),
                "execute_api": "tensors",
                "tensor_types": ("smem_a", "smem_b", "smem_c"),
            },
            True,
            TestGetSharedStorageSize(
                args=(
                    lambda MM: MM.get_layout_smem_a(),
                    lambda MM: MM.get_layout_smem_b(),
                    3,
                ),
                expected_error=r"get_shared_storage_size\(\) takes either 0 or "
                r"3 arguments\. If .*",
            ),
            TestGetSharedStorageSize(
                args=(
                    lambda MM: MM.get_layout_smem_a(),
                    2,
                ),
                expected_error=r"get_shared_storage_size_ab\(\) takes either 0 "
                r"or 2 arguments\. If .*",
            ),
        ),
        # Test with arguments
        (
            {
                "size": (1, 2, 3),
                "precision": np.float16,
                "alignment": (2, 4, 8),
            },
            True,
            TestGetSharedStorageSize(
                args=(5, 5, 5),
                expected_size=76,
            ),
            TestGetSharedStorageSize(
                args=(5, 5),
                expected_size=52,
            ),
        ),
        (
            {
                "size": (1, 2, 3),  # matrix sizes 3, 6, 2
                "precision": np.float16,
                "alignment": (2, 4, 8),
                "execute_api": "tensors",
                "tensor_types": ("smem_a", "smem_b", "smem_c"),
            },
            True,
            TestGetSharedStorageSize(
                args=(
                    lambda MM: MM.get_layout_smem_a(),
                    lambda MM: MM.get_layout_smem_b(),
                    lambda MM: MM.get_layout_smem_c(),
                ),
                expected_size=28,
            ),
            TestGetSharedStorageSize(
                args=(
                    lambda MM: MM.get_layout_smem_a(),
                    lambda MM: MM.get_layout_smem_b(),
                ),
                expected_size=20,
            ),
        ),
    ],
)
def test_cublasdx_get_shared_storage_size_args(
    MM_kwargs: dict,
    compiled: bool,
    t: TestGetSharedStorageSize,
    t_ab: TestGetSharedStorageSize,
):
    if compiled:
        skip_nvbug_5218000(MM_kwargs["precision"], size=MM_kwargs["size"])
    MM_kwargs |= {
        "data_type": "real",
        "arrangement": ("col_major", "col_major", "col_major"),
        "block_size": 128,
        "code_type": get_default_code_type(),
    }

    MM = BlasCompiled(**MM_kwargs) if compiled else BlasOptionsComplete(**MM_kwargs)
    # remove callables from args
    args = tuple(a(MM) if callable(a) else a for a in t.args)
    args_ab = tuple(a(MM) if callable(a) else a for a in t_ab.args)

    if t.expected_error is not None:
        assert t.expected_error != ""  # too generic error message
        with pytest.raises(ValueError, match=t.expected_error):
            MM.get_shared_storage_size(*args)
        return

    assert MM.get_shared_storage_size(*args) == t.expected_size

    if t_ab.expected_error is not None:
        assert t_ab.expected_error != ""  # too generic error message
        with pytest.raises(ValueError, match=t_ab.expected_error):
            MM.get_shared_storage_size_ab(*args_ab)
        return

    assert MM.get_shared_storage_size_ab(*args_ab) == t_ab.expected_size


def test_static_block_dim():
    matmul_base = functools.partial(
        matmul,
        size=(64, 64, 64),
        precision=(np.float16, np.float32, np.float64),
        data_type="real",
        arrangement=("col_major", "col_major", "col_major"),
        block_dim=(128, 1, 1),
        compiler="numba",
    )

    MM1 = matmul_base()
    MM2 = matmul_base()
    MM3 = matmul_base(static_block_dim=True)

    # if input is the same handle will be cached and the same
    assert MM1._handle == MM2._handle
    # here since input is different, handle is different
    assert MM1._handle != MM3._handle


@pytest.mark.parametrize(
    "dtype, alignment, expected, complete, expected_error",
    [
        ("real", (8, 8, 8), Alignment(8, 8, 8), False, None),
        ("real", [8, 8, 8], Alignment(8, 8, 8), False, None),
        ("real", Alignment(8, 8, 8), Alignment(8, 8, 8), False, None),
        ("real", (4, 8, 16), Alignment(4, 8, 16), False, None),
        ("real", (4, 8, 16), Alignment(4, 8, 16), True, None),
        ("real", MAX_ALIGNMENT, Alignment(16, 16, 16), False, None),
        ("real", (8, 2, 8), None, False, "alignment.b must be a multiple of input value type 4. Got 2"),
        ("real", (8, 8, 4), None, False, "alignment.c must be a multiple of input value type 8. Got 4"),
        ("real", (32, 8, 8), None, False, "alignment.a must be less than maximum alignment 16. Got 32"),
        ("real", (-1, 8, 8), None, False, "alignment.a must be > 0. Got -1"),
        ("real", (8, 0, 8), None, False, "alignment.b must be > 0. Got 0"),
        ("real", None, None, False, None),
        ("real", None, Alignment(2, 4, 8), True, None),
        ("real", (4, 8, 16), Alignment(4, 8, 16), True, None),
        ("complex", (4, 8, 16), Alignment(4, 8, 16), False, None),
        ("complex", (4, 8, 16), Alignment(4, 8, 16), True, None),
        ("complex", (8, 8, 8), None, False, "alignment.c must be a multiple of input value type 16. Got 8"),
    ],
)
def test_alignment(dtype, alignment, expected, complete, expected_error):
    matmul = functools.partial(
        BlasOptionsComplete if complete else BlasOptions,
        size=(64, 64, 64),
        precision=(np.float16, np.float32, np.float64),
        data_type=dtype,
        code_type=get_default_code_type(),
        arrangement=("col_major", "col_major", "col_major"),
        alignment=alignment,
    )

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            matmul()
        return

    MM = matmul()

    assert MM.alignment == expected


@pytest.mark.parametrize(
    "param_name, param_value, special_case",
    [
        ("arrangement", (1, 2), None),
        ("alignment", (1, 2), None),
        ("leading_dimension", (1, 2), None),
        ("block_size", 128, "block_dim_conflict"),
        ("block_size", "suggested", "block_size_suggested"),
    ],
)
def test_blas_options_parameter_validation(param_name, param_value, special_case):
    """Test BlasOptions parameter validation"""
    base_kwargs = {
        "size": (16, 8, 16),
        "data_type": "real",
        "precision": np.float32,
        "transpose_mode": TransposeMode("non_transposed", "transposed"),
        "code_type": SM75,
        "execution": "Block",
    }

    if special_case == "block_dim_conflict":
        base_kwargs["block_size"] = param_value
        base_kwargs["block_dim"] = (64, 1, 1)
        with pytest.raises(ValueError):
            BlasOptions(**base_kwargs)
    elif special_case == "block_size_suggested":
        base_kwargs["block_size"] = param_value
        BO = BlasOptions(**base_kwargs)
        assert isinstance(BO.block_dim, Dim3)
        assert BO.block_dim[0] * BO.block_dim[1] * BO.block_dim[2] >= 1
    else:
        base_kwargs[param_name] = param_value
        with pytest.raises(ValueError):
            BlasOptions(**base_kwargs)
