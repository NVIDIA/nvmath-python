# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

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
from nvmath.device.cublasdx import BlasCompiled
import functools
import pytest
import itertools
from .helpers import SM70, SM72, SM75, SM80, SM86, SM89, SM90


def test_third_party_symbols():
    MM = matmul(
        size=(24, 8, 48),
        data_type="real",
        precision=np.float64,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM70,
        execution="Block",
    )

    assert len(MM.symbols) == 2
    assert any([symbol.variant == "smem_basic" for symbol in MM.symbols])
    assert any([symbol.variant == "smem_ldabc" for symbol in MM.symbols])
    assert any(["function" in symbol.name for symbol in MM.symbols])
    assert any(["function" in symbol.name for symbol in MM.symbols])


def test_third_party_code():
    MM = matmul(
        size=(16, 8, 16),
        data_type="real",
        precision=np.float32,
        transpose_mode=TransposeMode("non_transposed", "transposed"),
        code_type=SM70,
        execution="Block",
    )

    assert len(MM.codes) > 0
    for code in MM.codes:
        print(code.code_type, code.isa_version)
    assert isinstance(MM, BlasCompiled)
    assert MM.size == (16, 8, 16)
    assert all(f.endswith(".ltoir") for f in MM.files)
    assert all([code.isa_version.major >= 12 for code in MM.codes])
    assert all([code.isa_version.minor >= 0 for code in MM.codes])
    assert all([code.code_type.cc.major == 7 for code in MM.codes])
    assert all([code.code_type.cc.minor == 0 for code in MM.codes])
    assert all([code.code_type.kind == "lto" for code in MM.codes])
    assert all([isinstance(code.data, bytes) for code in MM.codes])
    assert all([len(code.data) > 0 for code in MM.codes])
    assert MM.max_threads_per_block <= 1024


@pytest.mark.parametrize("ta, tb", list(itertools.product(["non_transposed", "transposed", "conj_transposed"], repeat=2)))
def test_transpose_mode(ta, tb):
    MM1 = matmul(
        size=(2, 2, 2),
        data_type="complex",
        precision=np.float32,
        transpose_mode=(ta, tb),
        code_type=SM70,
        execution="Block",
    )

    MM2 = matmul(
        size=(2, 2, 2),
        data_type="complex",
        precision=np.float32,
        transpose_mode=TransposeMode(ta, tb),
        code_type=SM70,
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
        code_type=SM70,
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
        code_type=SM70,
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
        code_type=SM70,
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
    opts = {"size": (24, 8, 48), "data_type": "real", "precision": np.float64, "code_type": SM70, "execution": "Block"}
    if value is None:
        del opts[opt]
    else:
        opts[opt] = value
    with pytest.raises(Exception):
        MM = matmul(**opts)  # noqa: F841


@pytest.mark.parametrize("code_type", [SM70, SM72, SM75, SM80, SM86, SM89, SM90])
def test_sm(code_type):
    MM = matmul(size=(24, 8, 48), data_type="real", precision=np.float32, code_type=code_type, execution="Block")
    assert all([isinstance(code.data, bytes) for code in MM.codes])
    assert all([len(code.data) > 0 for code in MM.codes])
    assert all([code.code_type == code_type for code in MM.codes])


@pytest.mark.parametrize("code_type", [("lto", (7, 0)), ("lto", (8, 0))])
def test_sm_type(code_type):
    MM = matmul(size=(24, 8, 48), data_type="real", precision=np.float32, code_type=code_type, execution="Block")
    assert all([isinstance(code.data, bytes) for code in MM.codes])
    assert all([len(code.data) > 0 for code in MM.codes])
    assert all([code.code_type.kind == code_type[0] for code in MM.codes])
    assert all([code.code_type.cc.major == code_type[1][0] for code in MM.codes])
    assert all([code.code_type.cc.minor == code_type[1][1] for code in MM.codes])


@pytest.mark.parametrize(
    "data_type,precision,value_type",
    [
        ("real", np.float16, np.float16),
        ("real", np.float32, np.float32),
        ("real", np.float64, np.float64),
        ("complex", np.float16, np.dtype([("x", np.float16), ("y", np.float16)], align=True)),
        ("complex", np.float32, np.complex64),
        ("complex", np.float64, np.complex128),
    ],
)
def test_value_type(data_type, precision, value_type):
    MM = matmul(size=(24, 8, 48), data_type=data_type, precision=precision, code_type=SM90, execution="Block")
    assert MM.value_type == value_type
