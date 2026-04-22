# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Non-batched NVFP4 matmul with epilogs tests.
"""

from contextlib import nullcontext

import numpy as np
import pytest

from .fp4_utils import NVFP4_SKIP_REASON, NVFP4_SUPPORTED

if not NVFP4_SUPPORTED:
    pytest.skip(NVFP4_SKIP_REASON, allow_module_level=True)

import torch

from nvmath.bindings import cublasLt as cublaslt
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.internal.utils import check_or_create_options
from nvmath.linalg.advanced import MatmulEpilog as Epilog
from nvmath.linalg.advanced import _configuration, matmul
from nvmath.linalg.advanced.helpers.matmul import (
    expand_block_scale,
    unpack_fp4,
)

from ...utils import allow_cublas_unsupported, sample_matrix
from .fp4_utils import (
    NVFP4_C_TYPES_NAMES,
    NVFP4_D_TYPES_NAMES,
    NVFP4_MNK_DIMENSIONS,
    assert_fp4_matmul_result,
    create_fp4_matrix_a_cyclic,
    create_fp4_matrix_b_cyclic,
    create_uniform_fp4_scales,
    expand_nvfp4_scales_for_matmul_input,
    unpack_matmul,
)

_rng = np.random.default_rng(42)
A_UNIFORM_SCALE_RANGE = _rng.uniform(0.25, 2.5)
B_UNIFORM_SCALE_RANGE = _rng.uniform(0.2, 3.0)


def nvfp4_matmul_reference(
    a, b, c=None, *args, d_out_scale=None, quantization_scales=None, epilog_inputs=None, options=None, **kwargs
):
    """
    Computes NVFP4-like matmul with epilog, but in higher precision (float32).

    This is the reference implementation for testing NVFP4 matmul with epilogs.
    It decodes FP4 inputs to float32, applies block scales, runs the matmul
    with epilog in float32, and optionally applies output scaling.

    Args:
        a: FP4 matrix A (float4_e2m1fn_x2), row-major packed
        b: FP4 matrix B (float4_e2m1fn_x2), column-major packed
        c: Optional matrix C for alpha*A@B + beta*C
        d_out_scale: Optional output scale tensor (for FP4/FP8 output)
        quantization_scales: Dict or MatmulQuantizationScales with 'a' and 'b' scales
        epilog_inputs: Dict of epilog inputs (bias, relu_aux, gelu_aux, etc.)
        options: MatmulOptions
        *args, **kwargs: Passed to matmul

    Returns:
        Reference result tensor (or tuple with aux outputs if epilog produces them)
    """
    scales = check_or_create_options(_configuration.MatmulQuantizationScales, quantization_scales, "Matmul scales")

    # Decode FP4 to float32 (decode returns CPU tensors)
    a_decoded_float32 = unpack_fp4(a, axis=-1)
    b_decoded_float32 = unpack_fp4(b, axis=-2)

    # Expand and apply block scales
    a_scale_float32 = expand_nvfp4_scales_for_matmul_input(a, scales.a, is_b_matrix=False)
    b_scale_float32 = expand_nvfp4_scales_for_matmul_input(b, scales.b, is_b_matrix=True)

    ascaled = a_decoded_float32 * a_scale_float32
    bscaled = b_decoded_float32 * b_scale_float32

    # Convert epilog inputs to float32 if present
    if epilog_inputs:
        epilog_inputs = dict(epilog_inputs)  # Copy to avoid mutating
        for key in ("bias", "gelu_aux"):
            if key in epilog_inputs:
                epilog_inputs[key] = epilog_inputs[key].type(torch.float32)

    # Run matmul with epilog in float32
    options = check_or_create_options(_configuration.MatmulOptions, options, "Matmul options")
    # For the reference computation, we use float32 without block scaling
    options.result_type = None
    options.block_scaling = False
    result = matmul(
        ascaled,
        bscaled,
        c.type(torch.float32) if c is not None else None,
        *args,
        quantization_scales=None,  # No block scaling for float32 reference
        epilog_inputs=epilog_inputs,
        options=options,
        **kwargs,
    )
    d, _, aux_dic = unpack_matmul(result)

    # If d_out_scale is provided, apply quantization by dividing
    # so that the result is suitable for FP4 precision comparison
    # as done in assert_fp4_matmul_result.
    if d_out_scale is not None:
        m, n = d.shape[-2:]
        device = "cuda" if d.is_cuda else "cpu"
        d_scale = expand_block_scale(d_out_scale, (m, n), "NVFP4", axis=-1, device=device, output_dtype=torch.float32)
        d /= d_scale

    return d, aux_dic


def bias_type_from_ctype_and_dtype(ctype_name: str, dtype_name: str) -> str | None:
    """Determine the appropriate bias type based on C and D types.
    See https://docs.nvidia.com/cuda/cublas/index.html#id105
    """
    assert ctype_name in NVFP4_C_TYPES_NAMES
    assert dtype_name in NVFP4_D_TYPES_NAMES

    if (ctype_name == "bfloat16" and dtype_name == "bfloat16") or (
        ctype_name == "bfloat16" and dtype_name == "float4_e2m1fn_x2"
    ):
        return "bfloat16"
    elif (ctype_name == "float16" and dtype_name == "float16") or (
        ctype_name == "float16" and dtype_name == "float4_e2m1fn_x2"
    ):
        return "float16"
    elif ctype_name == "float32" and dtype_name == "float32":
        return "bfloat16"
    else:
        return None


# Valid (ctype, dtype) combinations for NVFP4 epilog tests.
# Based on cuBLAS documentation Table 4: "When A, B, C, and D Use Layouts for FP4"
# Rule: When dtype is not FP4, ctype must equal dtype.
NVFP4_EPILOG_CD_TYPE_COMBOS = (
    ("bfloat16", "bfloat16"),
    ("bfloat16", "float4_e2m1fn_x2"),
    ("float16", "float16"),
    ("float16", "float4_e2m1fn_x2"),
    ("float32", "float32"),
)

# =============================================================================
# epilogs with no extra outputs
# =============================================================================
EPILOGS_WITH_NO_EXTRA_OUTPUTS = ("RELU", "GELU", "BIAS", "RELU_BIAS", "GELU_BIAS")


@pytest.mark.parametrize("m,n,k", NVFP4_MNK_DIMENSIONS)
@pytest.mark.parametrize("ctype,dtype", NVFP4_EPILOG_CD_TYPE_COMBOS)
@pytest.mark.parametrize("epilog_name", EPILOGS_WITH_NO_EXTRA_OUTPUTS)
@pytest.mark.parametrize("c_layout", ("row_major", "col_major"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_nvfp4_epilogs_with_no_extra_outputs(m, n, k, ctype, dtype, epilog_name, c_layout, use_cuda):
    """Test epilogs with no extra outputs (RELU, GELU, BIAS, RELU_BIAS, GELU_BIAS).

    RELU, GELU:
      Apply a pointwise activation function to the matmul result.
      No extra inputs required, no extra outputs produced.

    BIAS, RELU_BIAS, GELU_BIAS:
      Add a bias vector to each row of the result, optionally
      followed by an activation function.
      Required input: bias vector of size (m,): m is the matmul result's outer dim.
      No extra outputs produced.

      IMPORTANT cuBLASLt constraint: BIAS epilogs require column-major output.
      Row-major output tests are skipped for these epilogs.
    """
    device = "cuda" if use_cuda else "cpu"

    # cuBLASLt constraint: BIAS epilogs require column-major output
    # Skip row-major tests for BIAS epilogs
    if "BIAS" in epilog_name and c_layout == "row_major":
        pytest.skip(f"cuBLASLt does not support {epilog_name} with row-major output")

    a = create_fp4_matrix_a_cyclic(m, k, device=device)
    b = create_fp4_matrix_b_cyclic(k, n, device=device)
    a_scales = create_uniform_fp4_scales(a, A_UNIFORM_SCALE_RANGE, device=device)
    b_scales = create_uniform_fp4_scales(b, B_UNIFORM_SCALE_RANGE, device=device)
    scales = {"a": a_scales, "b": b_scales}

    # Create C matrix with appropriate layout (row-major or column-major)
    # Note that this will affect the layout of D (output) because the layout of
    # D is decided internally by matmul to inherit its layout from C when C is provided.
    # Also, since the datatype of C can never be FP4, we don't have to worry
    # about the packing direction for C so we can create C with just plain torch.zeros
    # and transpose it if needed.
    if c_layout == "col_major":
        c = torch.zeros((n, m), dtype=getattr(torch, ctype), device=device).T
    else:
        c = torch.zeros((m, n), dtype=getattr(torch, ctype), device=device)

    alpha, beta = 1.0, 0.5
    options = {"result_type": NAME_TO_DATA_TYPE[dtype], "block_scaling": True}
    epilog = getattr(Epilog, epilog_name)
    epilog_inputs = {}
    if "BIAS" in epilog_name:
        bias_type = bias_type_from_ctype_and_dtype(ctype, dtype)
        if bias_type is None:
            pytest.skip(f"Unsupported combination of ctype: {ctype} and dtype: {dtype}")
        bias = sample_matrix("torch", bias_type, (m,), use_cuda=use_cuda, min=0, max=1)
        epilog_inputs["bias"] = bias

    # Handle gaps in cuBLAS support. cuBLAS docs: "limited forward compatibility for narrow
    # precisions (FP4 and FP8)" - https://docs.nvidia.com/cuda/cublas/index.html#forward-compatibility
    # NVFP4 BIAS + float16 + col_major fails with NOT_SUPPORTED
    # on cuBLAS 12.8, works on 13.0+.
    allow_not_supported = False
    allow_not_supported |= (
        epilog_name == "BIAS"
        and ctype == "float16"
        and dtype == "float16"
        and c_layout == "col_major"
        and cublaslt.get_version() < 130000
    )

    with (
        nullcontext()
        if not allow_not_supported
        else allow_cublas_unsupported(
            message=f"NVFP4 epilog not supported by cuBLAS: {epilog_name} for C:{ctype} D:{dtype} layout:{c_layout}",
            allow_invalid_value=True,
        )
    ):
        d_computed, d_out_scale, aux_dic = unpack_matmul(
            matmul(
                a,
                b,
                c,
                alpha=alpha,
                beta=beta,
                epilog=epilog,
                quantization_scales=scales,
                options=options,
                epilog_inputs=epilog_inputs,
            )
        )
    # there are no auxiliary outputs
    assert aux_dic == {}

    # Check that the layout of D (output) is what we expect
    # Use negative indexing to handle both batched and non-batched cases
    # Row-major: last dimension (N/2 for FP4) is contiguous, stride[-1] == 1
    # Column-major: second-to-last dimension (M/2 for FP4) is contiguous, stride[-2] == 1
    if c_layout == "col_major":
        assert d_computed.stride(-2) == 1, f"Expected column-major layout (stride[-2]=1), got strides {d_computed.stride()}"
    else:
        assert d_computed.stride(-1) == 1, f"Expected row-major layout (stride[-1]=1), got strides {d_computed.stride()}"

    d_gold, _ = nvfp4_matmul_reference(
        a,
        b,
        c,
        d_out_scale=d_out_scale,
        alpha=alpha,
        beta=beta,
        epilog=epilog,
        quantization_scales=scales,
        options=options,
        epilog_inputs=epilog_inputs,
    )
    assert_fp4_matmul_result(d_computed, d_gold)


# =============================================================================
# epilogs with auxiliary outputs
# =============================================================================
EPILOGS_WITH_AUX_OUTPUT = ("RELU_AUX", "GELU_AUX", "RELU_AUX_BIAS", "GELU_AUX_BIAS")


@pytest.mark.parametrize("m,n,k", NVFP4_MNK_DIMENSIONS)
@pytest.mark.parametrize("ctype,dtype", NVFP4_EPILOG_CD_TYPE_COMBOS)
@pytest.mark.parametrize("epilog_name", EPILOGS_WITH_AUX_OUTPUT)
@pytest.mark.parametrize("c_layout", ("row_major", "col_major"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_nvfp4_aux_epilogs(m, n, k, ctype, dtype, epilog_name, c_layout, use_cuda):
    """Test epilogs producing auxiliary output (RELU_AUX, GELU_AUX, etc.).

    All AUX epilogs seem to be UNSUPPORTED with FP4 inputs (A, B).

    cuBLASLt Constraint 1 - Layout restriction:
        When using row-major output with AUX epilogs:
        "Unsupported combination: CUBLASLT_ORDER_ROW for output matrix and ReLu mask
         or GELU input (CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER)."

        Conclusion: AUX epilogs require column-major output (ORDER_COL) only.

    cuBLASLt Constraint 2 - Precision restrictions:
        When using FP4 inputs with AUX epilogs, ALL type combinations fail.

        Example errors (tested with column-major output):

        - bfloat16 C/D:
          "Unsupported combination of precisions. Got input matrix A type (R_4F_E2M1),
           A scale mode (VEC16_UE4M3), input matrix B type (R_4F_E2M1), B scale mode
           (VEC16_UE4M3), input matrix C type (R_16BF), C scale mode (SCALAR_32F),
           output matrix D type (R_16BF), D scale mode (SCALAR_32F), D output scale mode
           (SCALAR_32F), epilogue AUX type (R_16BF),
           and epilogue AUX scale mode (SCALAR_32F)"

        - float16 C/D:
          "Unsupported combination of precisions. Got input matrix A type (R_4F_E2M1),
           A scale mode (VEC16_UE4M3), input matrix B type (R_4F_E2M1), B scale mode
           (VEC16_UE4M3), input matrix C type (R_16F), C scale mode (SCALAR_32F),
           output matrix D type (R_16F), D scale mode (SCALAR_32F), D output scale mode
           (SCALAR_32F), epilogue AUX type (R_16F),
           and epilogue AUX scale mode (SCALAR_32F)"

        - float32 C/D:
          "Unsupported combination of precisions. Got input matrix A type (R_4F_E2M1),
           A scale mode (VEC16_UE4M3), input matrix B type (R_4F_E2M1), B scale mode
           (VEC16_UE4M3), input matrix C type (R_32F), C scale mode (SCALAR_32F),
           output matrix D type (R_32F), D scale mode (SCALAR_32F), D output scale mode
           (SCALAR_32F), epilogue AUX type (R_32F),
           and epilogue AUX scale mode (SCALAR_32F)"

        - FP4 output (bfloat16 C):
          "Unsupported combination of precisions. Got input matrix A type (R_4F_E2M1),
           A scale mode (VEC16_UE4M3), input matrix B type (R_4F_E2M1), B scale mode
           (VEC16_UE4M3), input matrix C type (R_16BF), C scale mode (SCALAR_32F),
           output matrix D type (R_4F_E2M1), D scale mode (SCALAR_32F), D output scale
           mode (VEC16_UE4M3), epilogue AUX type (R_4F_E2M1),
           and epilogue AUX scale mode (SCALAR_32F)"

        - FP4 output (float16 C):
          "Unsupported combination of precisions. Got input matrix A type (R_4F_E2M1),
           A scale mode (VEC16_UE4M3), input matrix B type (R_4F_E2M1), B scale mode
           (VEC16_UE4M3), input matrix C type (R_16F), C scale mode (SCALAR_32F),
           output matrix D type (R_4F_E2M1), D scale mode (SCALAR_32F), D output scale
           mode (VEC16_UE4M3), epilogue AUX type (R_4F_E2M1),
           and epilogue AUX scale mode (SCALAR_32F)"
    """
    # Skip all AUX epilog tests with FP4 inputs - not supported by cuBLASLt
    pytest.skip(
        f"cuBLASLt does not support {epilog_name} with FP4 inputs (A, B). "
        f"due to both layout and precision combination constraints. "
        f"See function docstring for detailed error messages."
    )


# =============================================================================
# Category 4: Gradient Epilogs (backward pass, requires aux input)
# =============================================================================
EPILOGS_GRADIENT = ("DRELU", "DGELU", "DRELU_BGRAD", "DGELU_BGRAD", "BGRADA", "BGRADB")


@pytest.mark.parametrize("m,n,k", NVFP4_MNK_DIMENSIONS)
@pytest.mark.parametrize("ctype,dtype", NVFP4_EPILOG_CD_TYPE_COMBOS)
@pytest.mark.parametrize("epilog_name", EPILOGS_GRADIENT)
@pytest.mark.parametrize("c_layout", ("row_major", "col_major"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_nvfp4_gradient_epilogs(m, n, k, ctype, dtype, epilog_name, c_layout, use_cuda):
    """Test gradient epilogs (DRELU, DGELU, DRELU_BGRAD, DGELU_BGRAD, BGRADA, BGRADB)."""
    # similar errors as in test_nvfp4_aux_epilogs above
    pytest.skip(
        f"cuBLASLt does not support {epilog_name} with FP4 inputs (A, B). "
        f"due to both layout and precision combination constraints. "
        f"See function docstring for detailed error messages."
    )
