# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for NVFP4 tests.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
except ImportError:
    torch = None
    TORCH_VERSION = (0, 0)

from nvmath.bindings import cublasLt as cublaslt

# === Skip-logic flags (always safe to import) ===

HAS_TORCH_2_9 = torch is not None and TORCH_VERSION >= (2, 9)


def _check_nvfp4_support():
    if torch is None:
        return False, "Torch is required for NVFP4 tests"
    if TORCH_VERSION < (2, 9):
        return False, "Torch >= 2.9 is required for NVFP4 tests (float4_e2m1fn_x2 support)"
    try:
        if torch.cuda.get_device_properties(0).major < 10:
            return False, "CC>=10.0 is required for NVFP4 tests"
    except Exception as e:
        return False, f"Could not get CUDA device properties: {e}"
    if cublaslt.get_version() < 120800:
        return False, f"cuBLAS 12.8 is required for NVFP4 tests (got {cublaslt.get_version()})"
    return True, "NVFP4 supported"


NVFP4_SUPPORTED, NVFP4_SKIP_REASON = _check_nvfp4_support()


# === Imports that require full NVFP4 support (torch >= 2.9, CC >= 10, cuBLAS >= 12.8) ===
# Utility functions below reference these names but are only called when NVFP4 is supported.

if NVFP4_SUPPORTED:
    from nvmath.linalg.advanced.helpers.matmul import (
        _FP4_DECODE_VALUES,
        expand_block_scale,
        unpack_fp4,
    )

    from ...utils import assert_tensors_equal


# - M (outer dim of A): must be multiple of 128
# - N (outer dim of B): must be multiple of 128
# - K (inner dim): must be multiple of 64
NVFP4_MNK_DIMENSIONS = ((128, 256, 64), (384, 256, 128))

# Alpha and beta values to test
ALPHA_VALUES = (1.0, 2.0)
BETA_VALUES = (0.5, 1.0)

# Supported C and D types for FP4 matmul (https://docs.nvidia.com/cuda/cublas/index.html#id105))
NVFP4_C_TYPES_NAMES = ("bfloat16", "float16", "float32")
NVFP4_D_TYPES_NAMES = ("bfloat16", "float16", "float32", "float4_e2m1fn_x2")


def unpack_matmul(result):
    """
    Unpack the result of `matmul` into D, d_out_scale, and aux outputs.

    When output is FP4/FP8, matmul returns (tensor, aux_dict) where
    aux_dict contains 'd_out_scale' and potentially other epilog outputs.

    Returns:
        tuple: (d, d_out_scale, aux) where aux contains remaining auxiliary outputs
    """
    if isinstance(result, tuple):
        d, aux = result
        d_out_scale = aux.pop("d_out_scale", None)
        return d, d_out_scale, aux
    else:
        return result, None, {}


# =============================================================================
# Helper functions for creating FP4 matrix inputs
# =============================================================================


def _fp4_pack_byte(counter: int) -> int:
    """Pack two FP4 values into a single byte based on counter.

    Uses a deterministic pattern where:
    - Low bits: counter % 5 (values 0-4)
    - High bits: counter % 5 + 1 (values 1-5)
    """
    # inspired by
    # https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/LtNvfp4Matmul/main.cpp

    # FP4 E2M1 encoding: integer value -> packed byte representation
    FP4_ENCODE = {0: 0x0, 1: 0x2, 2: 0x4, 3: 0x5, 4: 0x6, 5: 0x7}

    val1 = counter % 5
    val2 = counter % 5 + 1
    return (FP4_ENCODE[val1] & 0xF) | ((FP4_ENCODE[val2] & 0xF) << 4)


def create_fp4_matrix_a_cyclic(m, k, device="cuda"):
    """
    Create FP4 matrix A as (M, K//2) row-major with cycling values.
    FP4 data is packed: 2 FP4 values per byte.
    Values cycle through a deterministic pattern based on position.
    """
    # inspired by
    # https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/LtNvfp4Matmul/main.cpp

    a_bytes = torch.zeros((m, k // 2), dtype=torch.uint8, device="cpu")
    counter = 0
    for i in range(m):
        for j in range(k // 2):
            a_bytes[i, j] = _fp4_pack_byte(counter)
            counter += 1
    return a_bytes.to(device).view(torch.float4_e2m1fn_x2)


def create_fp4_matrix_b_cyclic(k, n, device="cuda"):
    """
    Create FP4 matrix B as (K//2, N) column-major with cycling values.
    FP4 data is packed: 2 FP4 values per byte.
    Column-major via torch.as_strided.
    Values cycle through a deterministic pattern based on position.
    """
    # this is inspired by
    # https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/LtNvfp4Matmul/main.cpp

    b_packed_size = (k * n) // 2
    b_bytes = torch.zeros(b_packed_size, dtype=torch.uint8, device="cpu")
    for i in range(b_packed_size):
        b_bytes[i] = _fp4_pack_byte(i)
    b_fp4_1d = b_bytes.to(device).view(torch.float4_e2m1fn_x2)
    # must have (1, k // 2) stride
    return torch.as_strided(b_fp4_1d, size=(k // 2, n), stride=(1, k // 2))


# =============================================================================
# Helpers for creating batched FP4 input tensors
# =============================================================================


def create_batched_fp4_matrix_a_cyclic(batch_shape, m, k, device="cuda"):
    """
    Create batched FP4 matrix A as (*batch_shape, M, K//2) row-wise packed
    with cycling values.
    FP4 data is packed row-wise: 2 FP4 values per byte.
    Values cycle through a deterministic pattern based on position.
    """
    # inspired by
    # https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/LtNvfp4Matmul/main.cpp

    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    full_shape = (*batch_shape, m, k // 2) if batch_shape else (m, k // 2)

    a_bytes = torch.zeros(full_shape, dtype=torch.uint8, device="cpu")

    counter = 0
    for b in range(batch_size):
        batch_idx = np.unravel_index(b, batch_shape)
        for i in range(m):
            for j in range(k // 2):
                byte_val = _fp4_pack_byte(counter)
                if batch_shape:
                    a_bytes[batch_idx + (i, j)] = byte_val
                else:
                    a_bytes[i, j] = byte_val
                counter += 1

    return a_bytes.to(device).view(torch.float4_e2m1fn_x2)


def create_batched_fp4_matrix_b_cyclic(batch_shape, k, n, device="cuda"):
    """
    Create batched FP4 matrix B as (*batch_shape, K//2, N) with cycling values.

    Each matrix is column-wise packed. The data is laid out so that for each
    batch element, the B matrix is stored column-major in memory.
    Values cycle through a deterministic pattern based on position.
    """
    # inspired by
    # https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLASLt/LtNvfp4Matmul/main.cpp

    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    b_packed_size = (k * n) // 2  # bytes per batch element

    total_elements = batch_size * b_packed_size
    b_bytes = torch.zeros(total_elements, dtype=torch.uint8, device="cpu")

    idx = 0
    for r in range(batch_size):
        for j in range(n):  # columns
            for i in range(k // 2):  # rows within column
                pos = r * b_packed_size + i + j * (k // 2)
                b_bytes[pos] = _fp4_pack_byte(idx)
                idx += 1

    b_fp4_1d = b_bytes.to(device).view(torch.float4_e2m1fn_x2)

    # Create strided view: (*batch_shape, K//2, N) with column-wise
    # packing layout.
    if batch_shape:
        full_shape = (*batch_shape, k // 2, n)
        # Strides: batch dimensions are contiguous, matrix is column-major
        matrix_stride = k // 2  # stride between columns
        batch_strides = tuple(int(np.prod(batch_shape[i + 1 :])) * b_packed_size for i in range(len(batch_shape)))
        full_strides = (*batch_strides, 1, matrix_stride)
    else:
        full_shape = (k // 2, n)
        full_strides = (1, k // 2)

    return torch.as_strided(b_fp4_1d, size=full_shape, stride=full_strides)


# =============================================================================
# Helpers for creating FP4 scales
# =============================================================================


def _create_nvfp4_uniform_torch_scale_from_dims(
    inner_dim: int, outer_dim: int, scale_value: float, device: str = "cuda"
) -> torch.Tensor:
    """
    Create uniform NVFP4 block scale torch tensor factors.
    NVFP4 uses VEC16_UE4M3 scaling mode with 16-element blocks.
    Scale factors are arranged in a 2D tiled layout with 128x64 tiles.

    Args:
        inner_dim: The inner or contraction dimension of the matrix.
        outer_dim: The outer dimension of the matrix: for A it is M, for B it is N.
        scale_value: Uniform scale value to use for all blocks (float).
        device: The device for the tensor.

    Returns:
        A ``float8_e4m3fn`` tensor with shape that matches the 2D tiled layout
        required by cuBLASLt's VEC16_UE4M3 mode.
        The scale tensor shape is (inner_dim // 16) * outer_dim.
        Each 16-element block gets one scale factor.
    """

    if inner_dim % 64 != 0:
        raise ValueError(f"inner_dim ({inner_dim}) must be divisible by 64 for FP4 block scaling")
    if outer_dim % 128 != 0:
        raise ValueError(f"outer_dim ({outer_dim}) must be divisible by 128 for FP4 block scaling")

    # Calculate the number of scales
    # s_inner: number of scales in inner dimension (1 per 16 elements)
    # s_outer: number of scales in outer dimension (matches outer_dim)
    s_inner = inner_dim // 16
    s_outer = outer_dim
    num_scales = s_inner * s_outer
    scale_tensor = torch.full((num_scales,), scale_value, dtype=torch.float8_e4m3fn, device=device)

    return scale_tensor


def create_uniform_fp4_scales(x, scale_value, device="cuda"):
    """
    Create FP4 block scales for tensor x with uniform scale value.

    Supports both 2D matrices and batched tensors (ndim >= 3).

    When doing block-scaled FP4 matmul, x should be:
    - A: (*batch, M, K//2) row-wise packed -> inner_dim=K, outer_dim=M
    - B: (*batch, K//2, N) column-wise packed -> inner_dim=K, outer_dim=N

    Args:
        x: FP4 tensor (2D matrix or batched with ndim >= 3)
        scale_value: Uniform scale value to use for all blocks
        device: Device for the tensor

    Returns:
        1D tensor of FP8 e4m3fn scale values with proper tiled layout.
        For batched inputs, scales are repeated for each batch element.
    """
    # Determine matrix layout from strides
    if x.stride(-1) == 1 or (x.ndim > 2 and x.stride(-1) < x.stride(-2)):
        # Row-wise packed A: (*batch, M, K//2)
        outer_dim, k_packed = x.shape[-2:]
        inner_dim = k_packed * 2
    elif x.stride(-2) == 1:
        # Column-wise packed B: (*batch, K//2, N)
        k_packed, outer_dim = x.shape[-2:]
        inner_dim = k_packed * 2
    else:
        raise ValueError(f"Unexpected stride pattern: {x.stride()}")

    single_scale_tensor = _create_nvfp4_uniform_torch_scale_from_dims(
        inner_dim=inner_dim, outer_dim=outer_dim, scale_value=scale_value, device=device
    )

    if x.ndim == 2:
        return single_scale_tensor
    else:
        batch_shape = x.shape[:-2]
        batch_size = int(np.prod(batch_shape))
        return single_scale_tensor.repeat(batch_size)


# =============================================================================
# Helpers for expanding FP4 scales to match result shape for dequantization
# =============================================================================


def expand_nvfp4_scales_for_matmul_input(x: torch.Tensor, scales: torch.Tensor, is_b_matrix: bool = False) -> torch.Tensor:
    """
    Expand NVFP4 block scales for matmul input matrices A or B.

    For matmul D = A x B where A is (M, K) and B is (K, N):
        - A matrix: (M, K//2) packed row-major -> scales expand to (M, K)
        - B matrix: (K//2, N) packed column-major -> scales expand to (K, N)

    Scales are organized along the K dimension (the contracted dimension).
    Each scale applies to 16 consecutive elements along K.

    Args:
        x: FP4 input tensor (float4_e2m1fn_x2)
           - For A: shape (M, K//2) row-major packed
           - For B: shape (K//2, N) column-major packed
        scales: uint8 tensor representing float8_e4m3fn scale values
        is_b_matrix: True if x is matrix B (column-major packed), False for A

    Returns:
        Float32 tensor with scale factors expanded:
            - For A: shape (M, K) where K is the logical dimension
            - For B: shape (K, N) where K is the logical dimension

    Example:
        >>> # For A matrix (128, 32) packed -> logical (128, 64)
        >>> a_fp4 = torch.zeros((128, 32), dtype=torch.float4_e2m1fn_x2)
        >>> a_scale = torch.zeros(128 * 64 // 16, dtype=torch.uint8)
        >>> expanded = expand_nvfp4_scales_for_matmul_input(
        a_fp4, a_scale, is_b_matrix=False)
        >>> expanded.shape
        torch.Size([128, 64])

        >>> # For B matrix (32, 256) packed -> logical (64, 256)
        >>> b_fp4 = torch.zeros((32, 256), dtype=torch.float4_e2m1fn_x2)
        >>> b_scale = torch.zeros(256 * 64 // 16, dtype=torch.uint8)
        >>> expanded = expand_nvfp4_scales_for_matmul_input(
        b_fp4, b_scale, is_b_matrix=True)
        >>> expanded.shape
        torch.Size([64, 256])
    """
    # Determine logical dimensions from packed FP4 tensor
    if is_b_matrix:
        # B is column-major: (K//2, N) -> logical (K, N)
        k_packed, n = x.shape[-2:]
        k = k_packed * 2
        operand_shape = (k, n)
        axis = -2  # blocked along K
    else:
        # A is row-major: (M, K//2) -> logical (M, K)
        m, k_packed = x.shape[-2:]
        k = k_packed * 2
        operand_shape = (m, k)
        axis = -1  # blocked along K

    device = "cuda" if x.is_cuda else "cpu"
    return expand_block_scale(scales, operand_shape, "NVFP4", axis=axis, device=device, output_dtype=torch.float32)


# =============================================================================
# Helpers for computing reference FP4 matmul results
# =============================================================================


def _nvfp4_matmul_reference_uniform_scale_2d(a, b, a_scale, b_scale, *, alpha=1.0, c=None, beta=0.0, d_out_scale=None):
    """
    Compute reference FP4 matmul result in high precision (2D only) for uniform scale.

    D = alpha * (A_scaled @ B_scaled) + beta * C
    If d_out_scale is provided: D /= d_scale (simulates quantization)
    """
    # a should have fp4 dtype and should be row-major
    assert a.dtype == torch.float4_e2m1fn_x2, "Expected a to be float4_e2m1fn_x2"
    assert a.stride(-2) > a.stride(-1), "Expected a to be row-major"

    # b should have fp4 dtype and should be column-major
    assert b.dtype == torch.float4_e2m1fn_x2, "Expected b to be float4_e2m1fn_x2"
    assert b.stride(-2) < b.stride(-1), "Expected b to be column-major"

    a_decoded_float32 = unpack_fp4(a, axis=-1)
    b_decoded_float32 = unpack_fp4(b, axis=-2)

    a_scaled_float32 = a_decoded_float32 * a_scale
    b_scaled_float32 = b_decoded_float32 * b_scale

    d_float32 = alpha * torch.matmul(a_scaled_float32, b_scaled_float32)
    if c is not None and beta != 0.0:
        d_float32 = d_float32 + beta * c.float()

    # If d_out_scale is provided, apply quantization by dividing
    # so that the result is suitable for FP4 precision comparison
    # as done in assert_fp4_matmul_result.
    if d_out_scale is not None:
        m, n = d_float32.shape[-2:]
        device = "cuda" if d_float32.is_cuda else "cpu"
        d_scale = expand_block_scale(d_out_scale, (m, n), "NVFP4", axis=-1, device=device, output_dtype=torch.float32)
        d_float32 = d_float32 / d_scale

    return d_float32


def _nvfp4_matmul_reference_uniform_scale_batched(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: float,
    b_scale: float,
    alpha: float = 1.0,
    c: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """
    Compute reference FP4 matmul result for batched tensors.

    Loops over batch dimensions and computes each 2D slice separately

    Args:
        a: Batched FP4 tensor A with shape (..., M, K//2) row-major packed
        b: Batched FP4 tensor B with shape (..., K//2, N) column-major packed
        a_scale: Scalar scale for A (float)
        b_scale: Scalar scale for B (float)
        alpha: Scalar coefficient
        c: Optional batched bias tensor with shape (..., M, N)
        beta: Scalar coefficient

    Returns:
        Batched reference result with shape (..., M, N)
    """
    # Determine batch shape from A
    batch_shape = a.shape[:-2]
    m = a.shape[-2]
    n = b.shape[-1]

    # Create output tensor
    result_shape = (*batch_shape, m, n)
    result = torch.zeros(result_shape, dtype=torch.float32, device=a.device)

    # Loop over all batch indices
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1
    for batch_flat_idx in range(batch_size):
        # Convert flat index to tuple index
        if batch_shape:
            batch_idx = np.unravel_index(batch_flat_idx, batch_shape)
        else:
            batch_idx = ()

        a_2d = a[batch_idx]  # Shape (M, K//2), potentially strided
        b_2d = b[batch_idx]  # Shape (K//2, N), potentially strided
        c_2d = c[batch_idx] if c is not None else None
        ref_2d = _nvfp4_matmul_reference_uniform_scale_2d(a_2d, b_2d, a_scale, b_scale, alpha=alpha, c=c_2d, beta=beta)
        result[batch_idx] = ref_2d

    return result


def nvfp4_matmul_reference_uniform_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: float,
    b_scale: float,
    *,
    alpha: float = 1.0,
    c: torch.Tensor | None = None,
    beta: float = 0.0,
    d_out_scale=None,
):
    """
    Compute reference FP4 matmul result for uniform scale.
    """
    # a and b should have fp4 dtype
    assert a.dtype == torch.float4_e2m1fn_x2, "Expected a to be float4_e2m1fn_x2"
    assert b.dtype == torch.float4_e2m1fn_x2, "Expected b to be float4_e2m1fn_x2"
    # if provided, c should not have fp4 dtype
    if c is not None:
        assert c.dtype != torch.float4_e2m1fn_x2, "Expected c to not be float4_e2m1fn_x2"

    assert a.dim() == b.dim(), "Expected a and b to have the same number of dimensions"
    if c is not None:
        assert c.dim() == a.dim(), "Expected c to have the same number of dimensions as a"

    if a.ndim == 2:
        return _nvfp4_matmul_reference_uniform_scale_2d(
            a, b, a_scale, b_scale, alpha=alpha, c=c, beta=beta, d_out_scale=d_out_scale
        )
    else:
        return _nvfp4_matmul_reference_uniform_scale_batched(a, b, a_scale, b_scale, alpha=alpha, c=c, beta=beta)


# =============================================================================
# Helper class for comparing results related
# =============================================================================


class Fp4Helper:
    """
    A helper class to compare FP4 E2M1 quantized results.
    Similar to Fp8Helper.
    """

    def __init__(self):
        # FP4 E2M1 representable values (excluding -0.0 which equals 0.0)
        values = sorted(set(_FP4_DECODE_VALUES))
        self.values = np.asarray(values)

        # For each value, calculate the range it covers (midpoints between adjacent values)
        middles = (self.values[1:] + self.values[:-1]) / 2
        self.lranges = np.append(np.asarray([-np.inf]), middles)
        self.rranges = np.append(middles, np.asarray([np.inf]))

    def range(self, value):
        """
        Finds a representable value closest to `value` and returns its range.
        """
        i = np.abs(self.values - value).argmin()
        return self.lranges[i], self.rranges[i]

    def absdiff(self, quantized, expected):
        """
        Returns absolute difference between the ranges of quantized numbers
        and the expected values.

        If expected falls within the quantized value's range, returns 0.
        """
        left, right = np.vectorize(self.range)(quantized)
        diff = np.minimum(abs(left - expected), abs(right - expected))
        diff[(left <= expected) & (right >= expected)] = 0.0
        return diff

    def allclose(self, quantized, expected, atol=1e-2, rtol=1e-2, return_info=False):
        """
        Checks if quantized values are close enough to the expected ones.

        Args:
            quantized: The quantized FP4 result (as numpy array or torch tensor)
            expected: The expected reference values (as numpy array or torch tensor)
            atol: Absolute tolerance
            rtol: Relative tolerance
            return_info: If True, returns (ok, info_dict) with error details

        Returns:
            bool or (bool, dict): Whether values are close, optionally with error info
        """
        if torch is not None and hasattr(quantized, "numpy"):
            quantized = quantized.cpu().numpy() if hasattr(quantized, "cpu") else quantized.numpy()
        if torch is not None and hasattr(expected, "numpy"):
            expected = expected.cpu().numpy() if hasattr(expected, "cpu") else expected.numpy()

        quantized = np.asarray(quantized, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)

        aerr = self.absdiff(quantized, expected)
        ok = np.all(aerr <= atol + rtol * np.abs(expected))

        if not return_info:
            return ok
        else:
            rerr = aerr / (np.abs(expected) + 1e-10)
            return ok, {
                "aerr": np.max(aerr),
                "atol": atol,
                "rerr": np.max(rerr),
                "rtol": rtol,
            }


def assert_fp4_matmul_result(result_possibly_quantized: torch.Tensor, reference: torch.Tensor) -> None:
    """
    Assert FP4 matmul result matches reference.

    Args:
        result_possibly_quantized: The matmul result tensor, possibly quantized to FP4
        reference: The expected reference tensor. NOTE: this should be already scaled
                   for precision comparison with FP4.

    Currently only supports 2D tensors.
    """
    # inputs should be both torch tensors
    assert isinstance(result_possibly_quantized, torch.Tensor), "Expected result_possibly_quantized to be a torch tensor"
    assert isinstance(reference, torch.Tensor), "Expected reference to be a torch tensor"

    # check that the inputs are both 2D tensors
    if result_possibly_quantized.ndim != 2 or reference.ndim != 2:
        msg = f"Currently, only 2D tensors are supported. Got {result_possibly_quantized.ndim}D and {reference.ndim}D"
        raise ValueError(msg)

    # reference dtype should not be fp4
    assert reference.dtype != torch.float4_e2m1fn_x2, "Expected reference dtype != float4_e2m1fn_x2"

    # result_possibly_quantized and reference on the same device
    assert result_possibly_quantized.device == reference.device, (
        "Expected result_possibly_quantized and reference to live on the same device"
    )

    # figure out the dtype of the computed result so that we can decide
    # how to do the numerical comparison
    result_type_name = str(result_possibly_quantized.dtype).split(".")[-1]

    reference_float32 = reference.cpu().float()

    if result_type_name == "float4_e2m1fn_x2":
        s = result_possibly_quantized.stride()
        assert s[-1] != s[-2], (
            f"Ambiguous strides {s} for FP4 result with shape {result_possibly_quantized.shape}: "
            f"cannot infer packing axis. Pass axis explicitly."
        )
        result_axis = -2 if s[-2] == 1 else -1
        result_float32 = unpack_fp4(result_possibly_quantized, axis=result_axis)

        # Only after decoding, check that logical shapes match
        # otherwise the comparison will fail.
        assert result_float32.shape == reference.shape, (
            f"Shape mismatch after FP4 decode: result {result_float32.shape} vs reference {reference.shape}"
        )

        # Use fp4_helper like Fp8Helper for FP8.
        # NOTE: the reference should be already scaled for precision comparison with FP4.
        # This means that the reference should have been computed using one of
        # functions this file above like compute_reference_uniform_scale_2d ,
        # etc which automatically scales the reference tensor and returns
        # it in float32 for FP4 precision comparison.
        fp4_helper = Fp4Helper()
        ok, info = fp4_helper.allclose(result_float32, reference_float32, atol=0.5, rtol=1e-1, return_info=True)
        if not ok:
            print("\nFP4 matmul comparison failed:")
            print(f"  Result type: {result_type_name}")
            print(f"  Absolute error: {info['aerr']:.4f} (tolerance {info['atol']})")
            print(f"  Relative error: {info['rerr']:.4f} (tolerance {info['rtol']})")
        assert ok
    else:
        # reference shape should be the same as the result_possibly_quantized shape
        assert reference.shape == result_possibly_quantized.shape, "Expected reference shape == result_possibly_quantized shape"

        result_float32 = result_possibly_quantized.cpu().float()
        # accumulation in float32 keeps output accurate, so we can use smaller tolerances.
        if result_type_name == "bfloat16" or result_type_name == "float16":
            atol, rtol = 0.1, 0.05
        else:  # float32
            atol, rtol = 0.1, 0.05
        assert_tensors_equal(result_float32, reference_float32, atol=atol, rtol=rtol)
