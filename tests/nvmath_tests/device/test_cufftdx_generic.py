# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import numpy as np
import pytest

from nvmath.device import FFT, Code, CodeType, ComputeCapability, fft
from nvmath.device.cufftdx import compile_fft_execute
from nvmath.device.types import complex64, complex128, half4

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
    skip_unsupported_sm,
)


@pytest.mark.parametrize("execute_api", ["shared_memory", "register_memory"])
def test_third_party_block_symbol(execute_api):
    fft = FFT(
        fft_type="c2c",
        size=256,
        precision=np.float32,
        direction="forward",
        sm=SM90.cc,
        execution="Block",
    )

    _, symbol = compile_fft_execute(fft, code_type=SM90, execute_api=execute_api)

    assert len(symbol) > 0


@pytest.mark.parametrize(
    "execution",
    ["Block", "Thread"],
)
def test_third_party_symbol(execution):
    fft = FFT(fft_type="c2c", size=16, precision=np.float32, direction="forward", sm=SM90.cc, execution=execution)
    _, symbol = compile_fft_execute(fft, code_type=SM90)

    assert len(symbol) > 0


def test_third_party_code():
    fft = FFT(fft_type="c2c", size=32, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block")
    code, _ = compile_fft_execute(fft, code_type=SM80)

    assert isinstance(code, Code)
    assert code.code_type.kind == "lto"
    assert code.isa_version.major >= 12
    assert code.isa_version.minor >= 0
    assert code.code_type.cc.major == 8
    assert code.code_type.cc.minor == 0
    assert isinstance(code.data, bytes)
    assert len(code.data) > 0


#                                            2      | 2, 2^2, ... | 2, 2^2, ... | 2, 2^2, ...  # noqa: W505
@pytest.mark.parametrize("size, mincount", [(2, 1), (16, 4), (128, 4), (2048, 4)])
def test_knobs_c2c_ept_fpb(size, mincount):
    FO = functools.partial(
        FFT, fft_type="c2c", size=size, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block"
    )
    valid = FO().valid("elements_per_thread", "ffts_per_block")
    assert len(list(valid)) >= mincount
    for ept, fpb in valid:
        print("ept, fpb = ", ept, fpb)
        fft = FO(elements_per_thread=ept, ffts_per_block=fpb)
        assert isinstance(fft, FFT)
        assert fft.elements_per_thread == ept
        assert fft.ffts_per_block == fpb


#                                            3, 3^2 | 11      |2, 2^2, 2^3, ...
@pytest.mark.parametrize("size, mincount", [(9, 2), (121, 1), (2048, 4)])
def test_knobs_c2c_ept_only(size, mincount):
    FO = functools.partial(
        FFT, fft_type="c2c", size=size, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block"
    )
    valid = FO().valid("elements_per_thread")
    assert len(list(valid)) >= mincount
    for (ept,) in valid:
        print("ept = ", ept)
        fft = FO(elements_per_thread=ept)
        assert isinstance(fft, FFT)
        assert fft.elements_per_thread == ept


@pytest.mark.parametrize("size, mincount", [(7, 1), (36, 1), (2048, 1)])
def test_knobs_c2c_fpb_only(size, mincount):
    FO = functools.partial(
        FFT, fft_type="c2c", size=size, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block"
    )
    valid = FO().valid("ffts_per_block")
    assert len(list(valid)) >= mincount
    for (fpb,) in valid:
        print("fpb = ", fpb)
        fft = FO(ffts_per_block=fpb)
        assert isinstance(fft, FFT)
        assert fft.ffts_per_block == fpb


@pytest.mark.parametrize(
    "fft_type, complex_layout, real_mode",
    [
        ("r2c", "natural", "folded"),
        ("r2c", "packed", "folded"),
        ("r2c", "full", "folded"),
        ("r2c", "natural", "normal"),
        ("r2c", "packed", "normal"),
        ("r2c", "full", "normal"),
        ("c2r", "natural", "folded"),
        ("c2r", "packed", "folded"),
        ("c2r", "full", "folded"),
        ("c2r", "natural", "normal"),
        ("c2r", "packed", "normal"),
        ("c2r", "full", "normal"),
    ],
)
def test_knobs_r2c_c2r(fft_type, complex_layout, real_mode):
    FO = functools.partial(
        FFT,
        fft_type=fft_type,
        size=512,
        precision=np.float32,
        sm=SM80.cc,
        execution="Block",
        real_fft_options={"complex_layout": complex_layout, "real_mode": real_mode},
    )
    valid = FO().valid("elements_per_thread", "ffts_per_block")
    assert len(valid) > 2
    for ept, fpb in valid:
        print("ept, fpb = ", ept, fpb)
        fft = FO(elements_per_thread=ept, ffts_per_block=fpb)
        assert isinstance(fft, FFT)
        assert fft.elements_per_thread == ept
        assert fft.ffts_per_block == fpb
    # Max EPT is usually 32, but folded allows for EPT=64
    if real_mode == "folded":
        assert 64 in [e for (e, _) in valid]


def test_knobs_0():
    fft = FFT(fft_type="c2c", size=4, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block")
    val = fft.valid("elements_per_thread", "ffts_per_block")
    print(val)


def test_knobs_1():
    fft = FFT(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
        ffts_per_block="suggested",
        elements_per_thread="suggested",
    )
    assert fft.ffts_per_block is not None
    assert fft.elements_per_thread is not None
    assert fft.ffts_per_block > 1
    assert isinstance(fft, FFT)
    ffts_per_block = fft.ffts_per_block

    fft = FFT(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
        ffts_per_block=ffts_per_block,
    )
    assert isinstance(fft, FFT)
    assert fft.ffts_per_block == ffts_per_block


def test_knobs_2():
    FO = FFT(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
        ffts_per_block="suggested",
    )
    assert FO._ffts_per_block is not None
    assert FO._elements_per_thread is None
    assert FO._ffts_per_block > 1


def test_knobs_3():
    FO = FFT(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
        elements_per_thread="suggested",
    )
    assert FO._ffts_per_block is None
    assert FO._elements_per_thread is not None


def test_functools_partial():
    base = functools.partial(FFT, size=32, precision=np.float32, sm=SM80.cc, execution="Block")
    R2C = base(fft_type="r2c")
    C2R = base(fft_type="c2r")
    assert isinstance(R2C, FFT)
    assert isinstance(C2R, FFT)

    assert R2C.fft_type == "r2c"
    assert C2R.fft_type == "c2r"
    assert R2C.size == C2R.size


def test_partial_fft():
    FO = FFT(
        fft_type="c2c",
        size=32,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
        ffts_per_block="suggested",
    )
    suggested_ffts_per_block = FO.ffts_per_block

    fft = FFT(
        fft_type="c2c",
        size=32,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
        ffts_per_block=suggested_ffts_per_block,
    )
    assert isinstance(fft, FFT)
    assert fft.ffts_per_block == suggested_ffts_per_block


def test_valid_knobs_0():
    FO = FFT(fft_type="c2c", size=32, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block")
    valids = FO.valid("elements_per_thread", "ffts_per_block")
    count = 0
    for ept, bpb in valids:
        count += 1
        fft = FFT(
            fft_type="c2c",
            size=32,
            precision=np.float32,
            direction="forward",
            sm=SM80.cc,
            execution="Block",
            elements_per_thread=ept,
            ffts_per_block=bpb,
        )
        assert isinstance(fft, FFT)
        assert fft.elements_per_thread == ept
        assert fft.ffts_per_block == bpb

    assert count > 0


def test_valid_knobs_1():
    FO = functools.partial(
        FFT, fft_type="c2c", size=32, precision=np.float32, direction="forward", sm=SM80.cc, execution="Block"
    )
    valids = FO().valid("elements_per_thread", "ffts_per_block")
    count = 0
    for ept, bpb in valids:
        count += 1

        fft = FO(elements_per_thread=ept, ffts_per_block=bpb)
        assert isinstance(fft, FFT)
        assert fft.elements_per_thread == ept
        assert fft.ffts_per_block == bpb

    assert count > 0


@pytest.mark.parametrize(
    "sm, ept, bpb",
    [
        (SM80.cc, 2, 128),
        (SM86.cc, 2, 64),
        (SM89.cc, 2, 32),
    ],
)
def test_valid_knob_values(sm, ept, bpb):
    fft = FFT(
        fft_type="c2c",
        size=2,
        precision=np.float32,
        direction="forward",
        sm=sm,
        execution="Block",
    )
    valids = fft.valid("elements_per_thread", "ffts_per_block")

    assert len(valids) == 1
    assert valids[0] == (ept, bpb)


@pytest.mark.parametrize(
    "knobs",
    [
        ("ffts_per_block", "invalid_knob"),
        ("elements_per_thread", "invalid_knob"),
        ("elements_per_thread", -1),
        ("ffts_per_block", "invalid_knob", 1000),
    ],
)
def test_invalid_knob_values(knobs):
    fft = FFT(
        fft_type="c2c",
        size=2,
        precision=np.float32,
        direction="forward",
        sm=SM80.cc,
        execution="Block",
    )
    with pytest.raises(ValueError, match="Unsupported knob"):
        fft.valid(*knobs)


@pytest.mark.parametrize(
    "opt, value",
    [
        ("fft_type", None),
        ("fft_type", "r2r"),
        ("fft_type", "C2C"),
        ("size", None),
        ("size", 0),
        ("size", 147852369852),
        ("precision", None),
        ("direction", None),
        ("direction", "both"),
        ("direction", "INVERSE"),
        ("execution", None),
        ("execution", "CGA"),
        ("ffts_per_block", -1),
        ("ffts_per_block", 0),
        ("elements_per_thread", 1),
        ("elements_per_thread", 0),
        ("elements_per_thread", -1),
        ("real_fft_options", {"test": 1, "test2": 2}),
    ],
)
def test_negative(opt, value):
    opts = {
        "fft_type": "c2c",
        "size": 256,
        "precision": np.float32,
        "direction": "forward",
        "code_type": SM90,
        "execution": "Block",
    }
    if value is None:
        del opts[opt]
    else:
        opts[opt] = value
    with pytest.raises(Exception):  # noqa: B017
        FFT = fft(**opts)
        # trigger compilation
        value_type = FFT.value_type  # noqa: F841


@pytest.mark.parametrize(
    "opt, value",
    [
        ("code_type", None),
        ("code_type", CodeType("lto", ComputeCapability(-1, 0))),
        ("code_type", CodeType("lto", ComputeCapability(5, 0))),
        ("code_type", CodeType("sass", ComputeCapability(7, 0))),
        ("code_type", CodeType("ptx", ComputeCapability(7, 0))),
        ("code_type", CodeType("lto", ComputeCapability(1000, 0))),  # invalid cc > supported Max cc
        ("code_type", ("lto", "lto", ComputeCapability(10, 0))),  # len(code_type) != 2
    ],
)
def test_negative_compile(opt, value):
    fft = FFT(fft_type="c2c", size=256, precision=np.float32, direction="forward", execution="Block")

    with pytest.raises(Exception):  # noqa: B017
        compile_fft_execute(fft, code_type=value)


@pytest.mark.parametrize("code_type", [SM70, SM72, SM75, SM80, SM86, SM89, SM90, SM100, SM101, SM103, SM120, SM121])
def test_sm(code_type):
    skip_unsupported_sm(code_type)
    fft = FFT(fft_type="c2c", size=256, precision=np.float32, direction="forward", execution="Block")
    code, symbol = compile_fft_execute(fft, code_type=code_type)

    assert isinstance(code.data, bytes)
    assert len(code.data) > 0
    assert len(symbol) > 0


@pytest.mark.parametrize(
    "precision,value_type",
    [
        (np.float16, half4),  # ~ complex<__half2>
        (np.float32, complex64),  # complex<float>
        (np.float64, complex128),  # complex<double>
    ],
)
def test_value_type(precision, value_type):
    for fft_type in ["c2r", "r2c", "c2c"]:
        fft = FFT(
            fft_type=fft_type,
            size=256,
            precision=precision,
            direction="forward" if fft_type == "c2c" else None,
            sm=SM90.cc,
            execution="Block",
        )
        assert fft.value_type == value_type


@pytest.mark.parametrize("code_type", [("lto", (7, 5)), ("lto", (8, 0))])
def test_sm_tuple(code_type):
    fft = FFT(fft_type="c2c", size=256, precision=np.float32, direction="forward", execution="Block")
    code, symbol = compile_fft_execute(fft, code_type=code_type)
    assert isinstance(code.data, bytes)
    assert len(code.data) > 0
    assert len(symbol) > 0
    assert code.code_type.kind == code_type[0]
    assert code.code_type.cc.major == code_type[1][0]
    assert code.code_type.cc.minor == code_type[1][1]
