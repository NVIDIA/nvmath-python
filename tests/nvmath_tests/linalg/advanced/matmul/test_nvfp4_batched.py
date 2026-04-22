# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Batched NVFP4 matmul tests.
"""

import numpy as np
import pytest

from .fp4_utils import NVFP4_SKIP_REASON, NVFP4_SUPPORTED

if not NVFP4_SUPPORTED:
    pytest.skip(NVFP4_SKIP_REASON, allow_module_level=True)

import torch

from nvmath.linalg.advanced import matmul

from ...utils import pad_and_slice
from .fp4_utils import (
    ALPHA_VALUES,
    BETA_VALUES,
    NVFP4_C_TYPES_NAMES,
    assert_fp4_matmul_result,
    create_batched_fp4_matrix_a_cyclic,
    create_batched_fp4_matrix_b_cyclic,
    create_uniform_fp4_scales,
    nvfp4_matmul_reference_uniform_scale,
    unpack_matmul,
)


@pytest.mark.parametrize("ctype", NVFP4_C_TYPES_NAMES[:1])  # bfloat16
@pytest.mark.parametrize("batch_shape", ((2,), (2, 3)))
@pytest.mark.parametrize("m,n,k", ((128, 256, 64),))
@pytest.mark.parametrize("alpha", ALPHA_VALUES)
@pytest.mark.parametrize("beta", BETA_VALUES)
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize("c_layout", ("row-major", "column-major"))
@pytest.mark.parametrize(
    "pad_a,pad_b,pad_c",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_nvfp4_matmul_batched(m, n, k, batch_shape, ctype, alpha, beta, use_cuda, c_layout, pad_a, pad_b, pad_c):
    """
    Test FP4 matmul with batching: D = alpha * A @ B + beta * C.

    Exercises combinations of:
    - pad_a/pad_b: FP4 operands embedded in larger allocations (sliced views).
    - pad_c: C is a slice of a wider allocation (c_wide[..., :n]).
    - c_layout: C in row-major or column-major layout.
    """
    if not use_cuda and (pad_a or pad_b):
        pytest.skip("Padded FP4 skipped on CPU (copy_kernel not implemented)")

    device = "cuda" if use_cuda else "cpu"

    # 1. setup operands
    a_orig = create_batched_fp4_matrix_a_cyclic(batch_shape, m, k, device=device)
    b_orig = create_batched_fp4_matrix_b_cyclic(batch_shape, k, n, device=device)
    a = pad_and_slice(a_orig) if pad_a else a_orig
    b = pad_and_slice(b_orig) if pad_b else b_orig
    if c_layout == "row-major":
        c_orig = torch.randn(*batch_shape, m, n, dtype=getattr(torch, ctype), device=device) * 10
    else:
        c_orig = torch.randn(*batch_shape, n, m, dtype=getattr(torch, ctype), device=device).transpose(-2, -1) * 10
    c = pad_and_slice(c_orig) if pad_c else c_orig

    # 2. create quantization scales
    a_scale_range = np.random.uniform(0.25, 2.5)
    b_scale_range = np.random.uniform(0.2, 3.0)
    a_scales = create_uniform_fp4_scales(a, a_scale_range, device=device)
    b_scales = create_uniform_fp4_scales(b, b_scale_range, device=device)

    # 3. run matmul
    raw_result = matmul(
        a, b, c=c, alpha=alpha, beta=beta, quantization_scales={"a": a_scales, "b": b_scales}, options={"block_scaling": True}
    )
    result, _, _ = unpack_matmul(raw_result)

    # 4. validate result
    a_uniform_scale_float32 = a_scales[0].float().item()
    b_uniform_scale_float32 = b_scales[0].float().item()
    expected_shape = (*batch_shape, m, n)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    assert result.dtype == getattr(torch, ctype), f"Expected dtype {ctype}, got {result.dtype}"
    result_float = result.float()
    assert torch.isfinite(result_float).all(), "Result contains non-finite values"
    assert result_float.abs().max() > 0, "Result is all zeros"

    # important: we use the original matrices for the reference computation
    reference = nvfp4_matmul_reference_uniform_scale(
        a_orig, b_orig, a_uniform_scale_float32, b_uniform_scale_float32, alpha=alpha, c=c_orig, beta=beta
    )
    for batch_flat_idx in range(int(np.prod(batch_shape))):
        batch_idx = np.unravel_index(batch_flat_idx, batch_shape)
        result_2d = result[batch_idx]
        reference_2d = reference[batch_idx]
        assert_fp4_matmul_result(result_2d, reference_2d)
