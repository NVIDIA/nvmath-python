# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools
from nvmath.device import fft, CodeType, ComputeCapability, FFTOptions
from nvmath.device.cufftdx import FFTCompiled
import pytest
import numpy as np
from .helpers import SM70, SM72, SM75, SM80, SM86, SM89, SM90


def test_third_party_symbols():
    FFT = fft(fft_type="c2c", size=256, precision=np.float32, direction="forward", code_type=SM90, execution="Block")

    assert len(FFT.symbols) == 2
    assert any([symbol.variant == "thread" for symbol in FFT.symbols])
    assert any([symbol.variant == "smem" for symbol in FFT.symbols])
    assert any(["function" in symbol.name for symbol in FFT.symbols])
    assert any(["function" in symbol.name for symbol in FFT.symbols])

    FFT = fft(fft_type="c2c", size=16, precision=np.float32, direction="forward", code_type=SM90, execution="Thread")

    assert len(FFT.symbols) == 1
    assert any([symbol.variant == "thread" for symbol in FFT.symbols])
    assert any(["function" in symbol.name for symbol in FFT.symbols])


def test_third_party_code():
    FFT = fft(fft_type="c2c", size=32, precision=np.float32, direction="forward", code_type=SM80, execution="Block")

    assert isinstance(FFT, FFTCompiled)
    assert all([code.endswith(".ltoir") for code in FFT.files])
    assert len(FFT.codes) > 0
    for code in FFT.codes:
        print(code.code_type, code.isa_version)
    assert all([code.code_type.kind == "lto" for code in FFT.codes])
    assert all([code.isa_version.major >= 12 for code in FFT.codes])
    assert all([code.isa_version.minor >= 0 for code in FFT.codes])
    assert all([code.code_type.cc.major == 8 for code in FFT.codes])
    assert all([code.code_type.cc.minor == 0 for code in FFT.codes])
    assert all([isinstance(code.data, bytes) for code in FFT.codes])
    assert all([len(code.data) > 0 for code in FFT.codes])


#                                            2      | 2, 2^2, ... | 2, 2^2, ... | 2, 2^2, ...  # noqa: W505
@pytest.mark.parametrize("size, mincount", [(2, 1), (16, 4), (128, 4), (2048, 4)])
def test_knobs_c2c_ept_fpb(size, mincount):
    FO = FFTOptions(fft_type="c2c", size=size, precision=np.float32, direction="forward", code_type=SM80, execution="Block")
    valid = FO.valid("elements_per_thread", "ffts_per_block")
    assert len(list(valid)) >= mincount
    for ept, fpb in valid:
        print("ept, fpb = ", ept, fpb)
        FFT = FO.create(elements_per_thread=ept, ffts_per_block=fpb)
        assert isinstance(FFT, FFTCompiled)
        assert FFT.elements_per_thread == ept
        assert FFT.ffts_per_block == fpb
        assert len(FFT.files) > 0


#                                            3, 3^2 | 11      |2, 2^2, 2^3, ...
@pytest.mark.parametrize("size, mincount", [(9, 2), (121, 1), (2048, 4)])
def test_knobs_c2c_ept_only(size, mincount):
    FO = FFTOptions(fft_type="c2c", size=size, precision=np.float32, direction="forward", code_type=SM80, execution="Block")
    valid = FO.valid("elements_per_thread")
    assert len(list(valid)) >= mincount
    for (ept,) in valid:
        print("ept = ", ept)
        FFT = FO.create(elements_per_thread=ept)
        assert isinstance(FFT, FFTCompiled)
        assert FFT.elements_per_thread == ept
        assert len(FFT.files) > 0


@pytest.mark.parametrize("size, mincount", [(7, 1), (36, 1), (2048, 1)])
def test_knobs_c2c_fpb_only(size, mincount):
    FO = FFTOptions(fft_type="c2c", size=size, precision=np.float32, direction="forward", code_type=SM80, execution="Block")
    valid = FO.valid("ffts_per_block")
    assert len(list(valid)) >= mincount
    for (fpb,) in valid:
        print("fpb = ", fpb)
        FFT = FO.create(ffts_per_block=fpb)
        assert isinstance(FFT, FFTCompiled)
        assert FFT.ffts_per_block == fpb
        assert len(FFT.files) > 0


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
    FO = FFTOptions(
        fft_type=fft_type,
        size=512,
        precision=np.float32,
        code_type=SM80,
        execution="Block",
        real_fft_options={"complex_layout": complex_layout, "real_mode": real_mode},
    )
    valid = FO.valid("elements_per_thread", "ffts_per_block")
    assert len(valid) > 2
    for ept, fpb in valid:
        print("ept, fpb = ", ept, fpb)
        FFT = FO.create(elements_per_thread=ept, ffts_per_block=fpb)
        assert isinstance(FFT, FFTCompiled)
        assert FFT.elements_per_thread == ept
        assert FFT.ffts_per_block == fpb
        assert len(FFT.files) > 0
    # Max EPT is usually 32, but folded allows for EPT=64
    if real_mode == "folded":
        assert 64 in [e for (e, _) in valid]


def test_knobs_0():
    FO = FFTOptions(fft_type="c2c", size=4, precision=np.float32, direction="forward", code_type=SM80, execution="Block")
    val = FO.valid("elements_per_thread", "ffts_per_block")
    print(val)


def test_knobs_1():
    FO = FFTOptions(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        code_type=SM80,
        execution="Block",
        ffts_per_block="suggested",
        elements_per_thread="suggested",
    )
    assert FO.ffts_per_block is not None
    assert FO.elements_per_thread is not None
    assert FO.ffts_per_block > 1
    assert isinstance(FO, FFTOptions)
    ffts_per_block = FO.ffts_per_block

    FFT = fft(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        code_type=SM80,
        execution="Block",
        ffts_per_block=ffts_per_block,
    )
    assert isinstance(FFT, FFTCompiled)
    assert FFT.ffts_per_block == ffts_per_block


def test_knobs_2():
    FO = FFTOptions(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        code_type=SM80,
        execution="Block",
        ffts_per_block="suggested",
    )
    assert FO.ffts_per_block is not None
    assert FO.elements_per_thread is None
    assert FO.ffts_per_block > 1


def test_knobs_3():
    FO = FFTOptions(
        fft_type="c2c",
        size=4,
        precision=np.float32,
        direction="forward",
        code_type=SM80,
        execution="Block",
        elements_per_thread="suggested",
    )
    assert FO.ffts_per_block is None
    assert FO.elements_per_thread is not None


def test_functools_partial():
    base = functools.partial(fft, size=32, precision=np.float32, code_type=SM80, execution="Block")
    R2C = base(fft_type="r2c")
    C2R = base(fft_type="c2r")
    assert isinstance(R2C, FFTCompiled)
    assert isinstance(C2R, FFTCompiled)

    assert R2C.fft_type == "r2c"
    assert C2R.fft_type == "c2r"
    assert R2C.size == C2R.size


def test_partial_fft():
    FO = FFTOptions(
        fft_type="c2c",
        size=32,
        precision=np.float32,
        direction="forward",
        code_type=SM80,
        execution="Block",
        ffts_per_block="suggested",
    )
    suggested_ffts_per_block = FO.ffts_per_block

    FFT = fft(
        fft_type="c2c",
        size=32,
        precision=np.float32,
        direction="forward",
        code_type=SM80,
        execution="Block",
        ffts_per_block=suggested_ffts_per_block,
    )
    assert isinstance(FFT, FFTCompiled)
    assert FFT.ffts_per_block == suggested_ffts_per_block


def test_valid_knobs_0():
    FO = FFTOptions(fft_type="c2c", size=32, precision=np.float32, direction="forward", code_type=SM80, execution="Block")
    valids = FO.valid("elements_per_thread", "ffts_per_block")
    count = 0
    for ept, bpb in valids:
        count += 1
        FFT = fft(
            fft_type="c2c",
            size=32,
            precision=np.float32,
            direction="forward",
            code_type=SM80,
            execution="Block",
            elements_per_thread=ept,
            ffts_per_block=bpb,
        )
        assert isinstance(FFT, FFTCompiled)
        assert FFT.elements_per_thread == ept
        assert FFT.ffts_per_block == bpb

    assert count > 0


def test_valid_knobs_1():
    FO = FFTOptions(fft_type="c2c", size=32, precision=np.float32, direction="forward", code_type=SM80, execution="Block")
    valids = FO.valid("elements_per_thread", "ffts_per_block")
    count = 0
    for ept, bpb in valids:
        count += 1

        FFT = FO.create(elements_per_thread=ept, ffts_per_block=bpb)
        assert isinstance(FFT, FFTCompiled)
        assert FFT.elements_per_thread == ept
        assert FFT.ffts_per_block == bpb
        assert len(FFT.files) > 0

    assert count > 0


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
        ("code_type", None),
        ("code_type", CodeType("lto", ComputeCapability(-1, 0))),
        ("code_type", CodeType("lto", ComputeCapability(5, 0))),
        ("code_type", CodeType("sass", ComputeCapability(7, 0))),
        ("code_type", CodeType("ptx", ComputeCapability(7, 0))),
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
    with pytest.raises(Exception):
        FFT = fft(**opts)  # noqa: F841


@pytest.mark.parametrize("code_type", [SM70, SM72, SM75, SM80, SM86, SM89, SM90])
def test_sm(code_type):
    FFT = fft(fft_type="c2c", size=256, precision=np.float32, direction="forward", code_type=code_type, execution="Block")
    assert all([isinstance(code.data, bytes) for code in FFT.codes])
    assert all([len(code.data) > 0 for code in FFT.codes])


@pytest.mark.parametrize(
    "precision,value_type",
    [
        (
            np.float16,
            np.dtype([("x", np.float16), ("y", np.float16), ("z", np.float16), ("w", np.float16)], align=True),
        ),  # ~ complex<__half2>
        (np.float32, np.complex64),  # complex<float>
        (np.float64, np.complex128),  # complex<double>
    ],
)
def test_value_type(precision, value_type):
    for fft_type in ["c2r", "r2c", "c2c"]:
        FFT = fft(
            fft_type=fft_type,
            size=256,
            precision=precision,
            direction="forward" if fft_type == "c2c" else None,
            code_type=SM90,
            execution="Block",
        )
        assert FFT.value_type == value_type


@pytest.mark.parametrize("code_type", [("lto", (7, 0)), ("lto", (8, 0))])
def test_sm_tuple(code_type):
    FFT = fft(fft_type="c2c", size=256, precision=np.float32, direction="forward", code_type=code_type, execution="Block")
    assert all([isinstance(code.data, bytes) for code in FFT.codes])
    assert all([len(code.data) > 0 for code in FFT.codes])
    assert all([code.code_type.kind == code_type[0] for code in FFT.codes])
    assert all([code.code_type.cc.major == code_type[1][0] for code in FFT.codes])
    assert all([code.code_type.cc.minor == code_type[1][1] for code in FFT.codes])
