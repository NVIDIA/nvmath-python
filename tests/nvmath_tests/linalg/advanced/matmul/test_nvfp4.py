# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Non-batched NVFP4 matmul tests.
"""

import numpy as np
import pytest

from .fp4_utils import NVFP4_SKIP_REASON, NVFP4_SUPPORTED

if not NVFP4_SUPPORTED:
    pytest.skip(NVFP4_SKIP_REASON, allow_module_level=True)

import torch

from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.linalg.advanced import matmul

from ...utils import pad_and_slice
from .fp4_utils import (
    ALPHA_VALUES,
    BETA_VALUES,
    NVFP4_C_TYPES_NAMES,
    NVFP4_D_TYPES_NAMES,
    NVFP4_MNK_DIMENSIONS,
    assert_fp4_matmul_result,
    create_fp4_matrix_a_cyclic,
    create_fp4_matrix_b_cyclic,
    create_uniform_fp4_scales,
    nvfp4_matmul_reference_uniform_scale,
    unpack_matmul,
)

A_UNIFORM_SCALE_RANGE = np.random.uniform(0.25, 2.5)
B_UNIFORM_SCALE_RANGE = np.random.uniform(0.2, 3.0)


@pytest.mark.parametrize("m,n,k", NVFP4_MNK_DIMENSIONS)
@pytest.mark.parametrize("dtype", [*NVFP4_D_TYPES_NAMES, None], ids=[*NVFP4_D_TYPES_NAMES, "default"])
@pytest.mark.parametrize("alpha", ALPHA_VALUES)
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize("pad_a,pad_b", [(True, False), (False, True), (True, True), (False, False)])
def test_nvfp4_matmul_ab(m, n, k, dtype, alpha, use_cuda, pad_a, pad_b):
    """Test FP4 matmul: D = alpha * A @ B, with explicit or default D dtype."""
    if not use_cuda and (pad_a or pad_b):
        # skip this case for two reasons:
        # (a) pad_and_slice raises a "copy_kernel" not implemented
        #     for 'Float4_e2m1fn_x2'
        # (b) helps not having an explosive number of tests
        pytest.skip("Padded FP4 skipped on CPU (copy_kernel not implemented)")

    device = "cuda" if use_cuda else "cpu"
    PAD = 16

    a_orig = create_fp4_matrix_a_cyclic(m, k, device=device)
    b_orig = create_fp4_matrix_b_cyclic(k, n, device=device)
    a = pad_and_slice(a_orig) if pad_a else a_orig
    b = pad_and_slice(b_orig) if pad_b else b_orig
    if pad_a:
        assert a.stride() == (k // 2 + 2 * PAD, 1)
    if pad_b:
        assert b.stride() == (1, k // 2 + 2 * PAD)

    a_scales = create_uniform_fp4_scales(a, A_UNIFORM_SCALE_RANGE, device=device)
    b_scales = create_uniform_fp4_scales(b, B_UNIFORM_SCALE_RANGE, device=device)

    options = {"block_scaling": True}
    if dtype is not None:
        options["result_type"] = NAME_TO_DATA_TYPE[dtype]

    raw_result = matmul(
        a,
        b,
        alpha=alpha,
        quantization_scales={"a": a_scales, "b": b_scales},
        options=options,
    )
    result, d_out_scale, _ = unpack_matmul(raw_result)

    a_uniform_scale_float32 = a_scales[0].float().item()
    b_uniform_scale_float32 = b_scales[0].float().item()
    # important: we use the original matrices for the reference computation
    reference = nvfp4_matmul_reference_uniform_scale(
        a_orig, b_orig, a_uniform_scale_float32, b_uniform_scale_float32, alpha=alpha, d_out_scale=d_out_scale
    )
    assert_fp4_matmul_result(result, reference)


@pytest.mark.parametrize("m,n,k", NVFP4_MNK_DIMENSIONS)
@pytest.mark.parametrize("ctype", NVFP4_C_TYPES_NAMES)
@pytest.mark.parametrize("c_layout", ("row-major", "column-major"))
@pytest.mark.parametrize("alpha", ALPHA_VALUES)
@pytest.mark.parametrize("beta", BETA_VALUES)
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize(
    "pad_a,pad_b,pad_c",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True), (True, True, True)],
)
def test_nvfp4_matmul_abc(m, n, k, ctype, c_layout, alpha, beta, use_cuda, pad_a, pad_b, pad_c):
    """
    Test FP4 matmul: D = alpha * A @ B + beta * C
    with C provided in row-major or column-major layout.
    Since C is provided and result_type is not specified,
    output type should default to C's type.
    """
    if not use_cuda and (pad_a or pad_b):
        # skip this case for two reasons:
        # (a) pad_and_slice raises a "copy_kernel" not implemented
        #     for 'Float4_e2m1fn_x2'
        # (b) helps not having an explosive number of tests
        pytest.skip("Padded FP4 skipped on CPU (copy_kernel not implemented)")

    device = "cuda" if use_cuda else "cpu"

    # 1. setup operands
    a_orig = create_fp4_matrix_a_cyclic(m, k, device=device)
    b_orig = create_fp4_matrix_b_cyclic(k, n, device=device)
    a = pad_and_slice(a_orig) if pad_a else a_orig
    b = pad_and_slice(b_orig) if pad_b else b_orig
    dtype = getattr(torch, ctype)
    if c_layout == "row-major":
        c_orig = torch.full((m, n), 1.0, dtype=dtype, device=device)
    else:
        c_orig = torch.full((n, m), 1.0, dtype=dtype, device=device).T
    c = pad_and_slice(c_orig) if pad_c else c_orig

    # 2. create quantization scales
    a_scales = create_uniform_fp4_scales(a, A_UNIFORM_SCALE_RANGE, device=device)
    b_scales = create_uniform_fp4_scales(b, B_UNIFORM_SCALE_RANGE, device=device)

    # 3. run matmul
    result = matmul(
        a, b, c=c, alpha=alpha, beta=beta, quantization_scales={"a": a_scales, "b": b_scales}, options={"block_scaling": True}
    )

    # 4. validate result
    assert result.dtype == getattr(torch, ctype)
    a_uniform_scale_float32 = a_scales[0].float().item()
    b_uniform_scale_float32 = b_scales[0].float().item()
    # IMPORTANT: we use the original matrices for the reference computation
    reference = nvfp4_matmul_reference_uniform_scale(
        a_orig, b_orig, a_uniform_scale_float32, b_uniform_scale_float32, alpha=alpha, c=c_orig, beta=beta
    )
    assert str(result.dtype).split(".")[-1] == ctype
    assert_fp4_matmul_result(result, reference)


@pytest.mark.parametrize("m,n,k", NVFP4_MNK_DIMENSIONS)
@pytest.mark.parametrize("ctype", NVFP4_C_TYPES_NAMES)
@pytest.mark.parametrize("c_layout", ("row-major", "column-major"))
@pytest.mark.parametrize("alpha", ALPHA_VALUES)
@pytest.mark.parametrize("beta", BETA_VALUES)
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize(
    "pad_a,pad_b,pad_c",
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True), (True, True, True)],
)
def test_nvfp4_matmul_inplace(m, n, k, ctype, c_layout, alpha, beta, use_cuda, pad_a, pad_b, pad_c):
    """
    Test FP4 matmul inplace: C = alpha * A @ B + beta * C

    With inplace=True, the result is written directly into C.
    The result_type option is ignored.
    """
    if not use_cuda and (pad_a or pad_b):
        # skip this case for two reasons:
        # (a) pad_and_slice raises a "copy_kernel" not implemented
        #     for 'Float4_e2m1fn_x2'
        # (b) helps not having an explosive number of tests
        pytest.skip("Padded FP4 skipped on CPU (copy_kernel not implemented)")

    device = "cuda" if use_cuda else "cpu"

    # 1. setup operands
    a_orig = create_fp4_matrix_a_cyclic(m, k, device=device)
    b_orig = create_fp4_matrix_b_cyclic(k, n, device=device)
    a = pad_and_slice(a_orig) if pad_a else a_orig
    b = pad_and_slice(b_orig) if pad_b else b_orig
    dtype = getattr(torch, ctype)
    if c_layout == "row-major":
        c_orig = torch.full((m, n), 1.0, dtype=dtype, device=device)
    else:
        c_orig = torch.full((n, m), 1.0, dtype=dtype, device=device).T
    c = pad_and_slice(c_orig) if pad_c else c_orig

    # 2. create quantization scales
    a_scales = create_uniform_fp4_scales(a, A_UNIFORM_SCALE_RANGE, device=device)
    b_scales = create_uniform_fp4_scales(b, B_UNIFORM_SCALE_RANGE, device=device)

    # 3. compute reference before inplace operation overwrites c
    a_uniform_scale_float32 = a_scales[0].float().item()
    b_uniform_scale_float32 = b_scales[0].float().item()
    # IMPORTANT: we use the original matrices for the reference computation
    reference = nvfp4_matmul_reference_uniform_scale(
        a_orig,
        b_orig,
        a_uniform_scale_float32,
        b_uniform_scale_float32,
        alpha=alpha,
        c=c_orig,
        beta=beta,
    )

    # 4. run matmul (here we must use a,b,c)
    result = matmul(
        a,
        b,
        c=c,
        alpha=alpha,
        beta=beta,
        quantization_scales={"a": a_scales, "b": b_scales},
        options={"inplace": True, "block_scaling": True},
    )

    # 5. validate result
    assert result is c
    assert_fp4_matmul_result(result, reference)
