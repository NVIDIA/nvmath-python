# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Non-batched NVFP4 matmul tests with analytical results.
"""

import pytest

from .fp4_utils import NVFP4_SKIP_REASON, NVFP4_SUPPORTED

if not NVFP4_SUPPORTED:
    pytest.skip(NVFP4_SKIP_REASON, allow_module_level=True)

import torch

from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.linalg.advanced import Matmul, matmul
from nvmath.linalg.advanced.helpers.matmul import (
    expand_block_scale,
    get_block_scale_offset,
    unpack_fp4,
)

from ...utils import pad_and_slice
from .fp4_utils import unpack_matmul


def create_nvfp4_zero_scale(outer_dim, inner_dim, device="cuda"):
    """
    Create a zeroed NVFP4 scale tensor with right shape that
    we can use for manual population.
    """

    n_inner_groups = inner_dim // 16
    num_scales = outer_dim * n_inner_groups
    return torch.zeros((num_scales,), dtype=torch.float8_e4m3fn, device=device)


def set_nvfp4_scale_value(scale_tensor, outer_idx, inner_group_idx, scale_value, inner_dim, *, axis=-1):
    """
    Set a single scale value in an NVFP4 scale tensor.

    Args:
        scale_tensor: The 1D scale tensor.
        outer_idx: The outer dimension index (row for A, column for B).
        inner_group_idx: Which 16-element scale group along K.
        scale_value: The scale value to set (will be converted to float8_e4m3fn).
        inner_dim: Total inner dimension (K).
        axis: -1 for A (row-major, K blocked); -2 for B (col-major, K blocked).
    """
    outer_dim = scale_tensor.shape[0] * 16 // inner_dim
    if axis == -1:
        # A: operand shape (M, K), index = (row, any col in block)
        scale_index = get_block_scale_offset((outer_idx, inner_group_idx), (outer_dim, inner_dim), "NVFP4", axis=-1)
    else:
        # B: operand shape (K, N), index = (any row in block, col)
        scale_index = get_block_scale_offset((inner_group_idx, outer_idx), (inner_dim, outer_dim), "NVFP4", axis=-2)
    scale_tensor[scale_index] = scale_value


# FP4 e2m1fn encoding for 1.0
# Bits [S][E1][E0][M] = 0010 -> S=0, E=01, M=0
#
# This is used below to pack two 1.0 values into one byte
# (float4_e2m1fn_x2 format):
# (FP4_ENCODE_ONE & 0xF) | ((FP4_ENCODE_ONE & 0xF) << 4)
# = 0x2 | 0x20 = 0x22
# -> lower 4 bits = 1.0, upper 4 bits = 1.0
FP4_ENCODE_ONE = 0x2


def create_fp4_matrix_a_with_ones(m: int, k: int, blocks: list[tuple[int, int]], device: str = "cuda") -> torch.Tensor:
    """
    Create FP4 matrix A (M x K) with ones in specific scale groups.

    Args:
        m: Number of rows (outer dimension for A)
        k: Number of columns (inner/contraction dimension)
        blocks: List of (outer_idx, inner_group_idx) tuples specifying which
                16-element blocks to fill with ones.
                - outer_idx: Row index (0 to M-1)
                - inner_group_idx: Which 16-element group along K (0 to K//16 - 1)
        device
    """
    assert m % 128 == 0
    assert k % 64 == 0

    n_scale_groups = k // 16  # 16-element scale groups along K

    # Build a set for fast lookup
    block_set = set()
    for outer_idx, inner_group_idx in blocks:
        assert 0 <= outer_idx < m, f"outer_idx {outer_idx} out of range [0, {m})"
        assert 0 <= inner_group_idx < n_scale_groups, f"inner_group_idx {inner_group_idx} out of range [0, {n_scale_groups})"
        block_set.add((outer_idx, inner_group_idx))

    a_bytes = torch.zeros((m, k // 2), dtype=torch.uint8, device="cpu")
    for i in range(m):
        for j in range(k // 2):
            # j // 8 gives the scale group index (since 16 elements = 8 packed bytes)
            group_idx = j // 8
            if (i, group_idx) in block_set:
                a_bytes[i, j] = (FP4_ENCODE_ONE & 0xF) | ((FP4_ENCODE_ONE & 0xF) << 4)
    return a_bytes.to(device).view(torch.float4_e2m1fn_x2)


def create_fp4_matrix_b_with_ones(k: int, n: int, blocks: list[tuple[int, int]], device: str = "cuda") -> torch.Tensor:
    """
    Create FP4 matrix B (K x N) with ones in specific scale groups.

    Args:
        k: Number of rows (inner/contraction dimension)
        n: Number of columns (outer dimension for B)
        blocks: List of (outer_idx, inner_group_idx) tuples specifying which
                16-element blocks to fill with ones.
                - outer_idx: Column index (0 to N-1)
                - inner_group_idx: Which 16-element group along K (0 to K//16 - 1)
        device
    """
    assert k % 64 == 0
    assert n % 128 == 0

    n_scale_groups = k // 16  # 16-element scale groups along K

    # Build a set for fast lookup
    block_set = set()
    for outer_idx, inner_group_idx in blocks:
        assert 0 <= outer_idx < n, f"outer_idx {outer_idx} out of range [0, {n})"
        assert 0 <= inner_group_idx < n_scale_groups, f"inner_group_idx {inner_group_idx} out of range [0, {n_scale_groups})"
        block_set.add((outer_idx, inner_group_idx))

    b_packed_size = (k * n) // 2
    b_bytes = torch.zeros(b_packed_size, dtype=torch.uint8, device="cpu")
    counter = 0
    for j in range(n):
        for i in range(k // 2):
            # i // 8 gives the scale group index (since 16 elements = 8 packed bytes)
            group_idx = i // 8
            if (j, group_idx) in block_set:
                b_bytes[counter] = (FP4_ENCODE_ONE & 0xF) | ((FP4_ENCODE_ONE & 0xF) << 4)
            counter += 1
    b_fp4_1d = b_bytes.to(device).view(torch.float4_e2m1fn_x2)
    return torch.as_strided(b_fp4_1d, size=(k // 2, n), stride=(1, k // 2))


def create_batched_fp4_matrix_a_with_ones(
    batch_size: int, m: int, k: int, batch_blocks: list[tuple[int, int]], device: str = "cuda"
) -> torch.Tensor:
    """
    Create batched FP4 matrix A (batch, M, K//2) with ones
    in specific scale groups per batch.

    Args:
        batch_size: Number of batch slices
        m: Number of rows (outer dimension for A)
        k: Number of columns (inner/contraction dimension)
        batch_blocks: List of length batch_size, where each element is a list of
                      (outer_idx, inner_group_idx) tuples for that batch slice.
        device

    Returns:
        FP4 tensor of shape (batch, M, K//2) row-major.
    """
    assert len(batch_blocks) == batch_size
    assert m % 128 == 0
    assert k % 64 == 0

    n_scale_groups = k // 16

    # Create batched bytes tensor
    a_bytes = torch.zeros((batch_size, m, k // 2), dtype=torch.uint8, device="cpu")

    for batch_idx, blocks in enumerate(batch_blocks):
        block_set = set()
        for outer_idx, inner_group_idx in blocks:
            assert 0 <= outer_idx < m, f"outer_idx {outer_idx} out of range [0, {m})"
            assert 0 <= inner_group_idx < n_scale_groups
            block_set.add((outer_idx, inner_group_idx))

        for i in range(m):
            for j in range(k // 2):
                group_idx = j // 8
                if (i, group_idx) in block_set:
                    a_bytes[batch_idx, i, j] = (FP4_ENCODE_ONE & 0xF) | ((FP4_ENCODE_ONE & 0xF) << 4)

    return a_bytes.to(device).view(torch.float4_e2m1fn_x2)


def create_batched_fp4_matrix_b_with_ones(
    batch_size: int, k: int, n: int, batch_blocks: list[tuple[int, int]], device: str = "cuda"
) -> torch.Tensor:
    """
    Create batched FP4 matrix B (batch, K//2, N) column-major
    with ones in specific scale groups per batch.

    Args:
        batch_size: Number of batch slices
        k: Number of rows (inner/contraction dimension)
        n: Number of columns (outer dimension for B)
        batch_blocks: List of length batch_size, where each element is a list of
                      (outer_idx, inner_group_idx) tuples for that batch slice.
        device

    Returns:
        FP4 tensor of shape (batch, K//2, N) column-major (stride[-2]=1).
    """
    assert len(batch_blocks) == batch_size
    assert k % 64 == 0
    assert n % 128 == 0

    n_scale_groups = k // 16
    k_packed = k // 2

    # For column-major B with shape (batch, K//2, N) and stride (K//2 * N, 1, K//2),
    # we store data in column-major order per batch slice.
    # Total bytes per batch slice = K//2 * N, stored column-by-column.
    b_bytes = torch.zeros((batch_size, k_packed * n), dtype=torch.uint8, device="cpu")

    for batch_idx, blocks in enumerate(batch_blocks):
        block_set = set()
        for outer_idx, inner_group_idx in blocks:
            assert 0 <= outer_idx < n, f"outer_idx {outer_idx} out of range [0, {n})"
            assert 0 <= inner_group_idx < n_scale_groups
            block_set.add((outer_idx, inner_group_idx))

        counter = 0
        for j in range(n):  # columns
            for i in range(k_packed):  # packed rows
                group_idx = i // 8
                if (j, group_idx) in block_set:
                    b_bytes[batch_idx, counter] = (FP4_ENCODE_ONE & 0xF) | ((FP4_ENCODE_ONE & 0xF) << 4)
                counter += 1

    # Convert to FP4 and create strided view for column-major layout
    b_fp4_flat = b_bytes.to(device).view(torch.float4_e2m1fn_x2)
    # Shape: (batch, K//2 * N) -> strided view (batch, K//2, N)
    # with stride (K//2*N, 1, K//2)
    return torch.as_strided(b_fp4_flat, size=(batch_size, k_packed, n), stride=(k_packed * n, 1, k_packed))


@pytest.mark.parametrize("use_cuda", [True, False])
@pytest.mark.parametrize("pad_a,pad_b", [(False, False), (True, False), (False, True), (True, True)])
def test_nvfp4_matmul_sparse_analytic_single_tile(use_cuda, pad_a, pad_b):
    """
    Test NVFP4 matmul with sparse matrices containing ones in a single scale group.

    This test checks that NVFP4 block scaling works correctly by creating
    matrices where we leverage the block scaling and only set specific
    16-element blocks to non-zero values,
    making the expected result easy to compute analytically.

    When pad_a/pad_b=True, the operand is embedded in a wider allocation
    (ld > K//2) to verify that scale addressing is not affected by the padded stride.

    The test makes sense if one looks at the block scaling layout
    as described in the cuBLASLt documentation:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Setup:
        - A: 128x64 matrix with 16 ones in row 5, K-group 2 (cols 32-47)
        - B: 64x128 matrix with 16 ones in col 32, K-group 2 (rows 32-47)
        - A scale = 1.0 for the active block, B scale = 4.0 for the active block

    Matrix A (M=128 x K=64):

              K-group 0    K-group 1    K-group 2    K-group 3
              cols 0-15    cols 16-31   cols 32-47   cols 48-63
            +------------+------------+------------+------------+
    row 0   |     0      |     0      |     0      |     0      |
    row 1   |     0      |     0      |     0      |     0      |
      ...   |     0      |     0      |     0      |     0      |
    row 5   |     0      |     0      |  1 1...1   |     0      | <- 16 ones
      ...   |     0      |     0      |     0      |     0      |    here
    row 127 |     0      |     0      |     0      |     0      |
            +------------+------------+------------+------------+
                                            ^
                            a_scale[5, group 2] = 1.0

    Matrix B (K=64 x N=128):

                        col 0    col 1   ...  col 32  ...  col 127
                      +--------+--------+---+--------+---+--------+
    K-group 0 row 0   |   0    |   0    |   |   0    |   |   0    |
              ...     |   0    |   0    |   |   0    |   |   0    |
              row 15  |   0    |   0    |   |   0    |   |   0    |
                      +--------+--------+---+--------+---+--------+
    K-group 1 row 16  |   0    |   0    |   |   0    |   |   0    |
              ...     |   0    |   0    |   |   0    |   |   0    |
              row 31  |   0    |   0    |   |   0    |   |   0    |
                      +--------+--------+---+--------+---+--------+
    K-group 2 row 32  |   0    |   0    |   |   1    |   |   0    |  <- 16 ones
              ...     |   0    |   0    |   |   1    |   |   0    |     in this
              row 47  |   0    |   0    |   |   1    |   |   0    |     column
                      +--------+--------+---+--------+---+--------+
    K-group 3 row 48  |   0    |   0    |   |   0    |   |   0    |
              ...     |   0    |   0    |   |   0    |   |   0    |
              row 63  |   0    |   0    |   |   0    |   |   0    |
                      +--------+--------+---+--------+---+--------+
                                              ^
                               b_scale[col 32, group 2] = 4.0

    With scales applied, result is:
    C[5, 32] = alpha * 16 * a_scale * b_scale = alpha * 16 * 1.0 * 4.0 = 64.0
    """
    if not use_cuda and (pad_a or pad_b):
        # skip this case for two reasons:
        # (a) pad_and_slice raises a "copy_kernel" not implemented
        #     for 'Float4_e2m1fn_x2'
        # (b) helps not having an explosive number of tests
        pytest.skip("Padded FP4 skipped on CPU (copy_kernel not implemented)")

    device = "cuda" if use_cuda else "cpu"

    m = 128
    k = 64
    n = 128

    # A: row 5, K scale group 2 (elements 32-47 of K)
    a_outer_idx = 5
    a_inner_group_idx = 2
    a_fp4 = create_fp4_matrix_a_with_ones(m, k, blocks=[(a_outer_idx, a_inner_group_idx)], device=device)
    if pad_a:
        a_fp4 = pad_and_slice(a_fp4)

    a_scale_value = 1.0
    a_scale = create_nvfp4_zero_scale(outer_dim=m, inner_dim=k, device=device)
    set_nvfp4_scale_value(a_scale, a_outer_idx, a_inner_group_idx, a_scale_value, inner_dim=k)

    # B: col 32, K scale group 2 (elements 32-47 of K) - same group so they multiply!
    b_outer_idx = 32
    b_inner_group_idx = 2
    b_fp4 = create_fp4_matrix_b_with_ones(k, n, blocks=[(b_outer_idx, b_inner_group_idx)], device=device)
    if pad_b:
        b_fp4 = pad_and_slice(b_fp4)

    b_scale_value = 4.0
    b_scale = create_nvfp4_zero_scale(outer_dim=n, inner_dim=k, device=device)
    set_nvfp4_scale_value(b_scale, b_outer_idx, b_inner_group_idx, b_scale_value, inner_dim=k, axis=-2)

    alpha = 2.0
    raw_result = matmul(
        a_fp4,
        b_fp4,
        alpha=alpha,
        quantization_scales={"a": a_scale, "b": b_scale},
        options={"result_type": NAME_TO_DATA_TYPE["float32"], "block_scaling": True},
    )
    expected_value = 16 * alpha * a_scale_value * b_scale_value
    expected = torch.zeros_like(raw_result)
    expected[5, 32] = expected_value
    assert torch.equal(raw_result, expected), "Expected only position [5, 32] to be non-zero, but found other non-zero values"


@pytest.mark.parametrize("use_cuda", [True, False])
@pytest.mark.parametrize("pad_a,pad_b", [(False, False), (True, False), (False, True), (True, True)])
def test_nvfp4_matmul_sparse_analytic_two_k_tiles(use_cuda, pad_a, pad_b):
    """
    Test NVFP4 matmul with K=128, spanning two K-tiles (each tile covers 64 K elements).

    This test verifies that block scaling works correctly across tile boundaries
    by placing non-zero blocks in both K-tiles and checking that contributions
    from both tiles accumulate correctly.

    Tile structure (128x64 tiles):
        - K-tile 0: K elements 0-63   (K-groups 0-3)
        - K-tile 1: K elements 64-127 (K-groups 4-7)

    Setup:
        - A: 128x128 matrix with ones in row 5 at K-groups 2 and 6
        - B: 128x128 matrix with ones in col 32 at K-groups 2 and 6
        - Different scales for each K-tile to verify correct scale application

    Matrix A (M=128 x K=128):

            |              K-tile 0             |              K-tile 1             |
              K-grp 0  K-grp 1  K-grp 2  K-grp 3  K-grp 4  K-grp 5  K-grp 6  K-grp 7
              0-15     16-31    32-47    48-63    64-79    80-95    96-111   112-127
            +--------+--------+--------+--------+--------+--------+--------+--------+
    row 0   |   0    |   0    |   0    |   0    |   0    |   0    |   0    |   0    |
      ...   |   0    |   0    |   0    |   0    |   0    |   0    |   0    |   0    |
    row 5   |   0    |   0    | 1...1  |   0    |   0    |   0    | 1...1  |   0    |
      ...   |   0    |   0    |   0    |   0    |   0    |   0    |   0    |   0    |
    row 127 |   0    |   0    |   0    |   0    |   0    |   0    |   0    |   0    |
            +--------+--------+--------+--------+--------+--------+--------+--------+
                                  ^                                    ^
                         a_scale[5,2]=1.0                      a_scale[5,6]=2.0

    Matrix B (K=128 x N=128):

                                  col 32
                                    v
                      +---------+-------+---------+
    K-tile 0  grp 0   |    0    |   0   |    0    |
              grp 1   |    0    |   0   |    0    |
              grp 2   |    0    |   1   |    0    |  <- 16 ones, b_scale[32,2]=4.0
              grp 3   |    0    |   0   |    0    |
                      +---------+-------+---------+
    K-tile 1  grp 4   |    0    |   0   |    0    |
              grp 5   |    0    |   0   |    0    |
              grp 6   |    0    |   1   |    0    |  <- 16 ones, b_scale[32,6]=8.0
              grp 7   |    0    |   0   |    0    |
                      +---------+-------+---------+

    Result C[5, 32] = alpha * (16 * 1.0 * 4.0 + 16 * 2.0 * 8.0)
    """
    if not use_cuda and (pad_a or pad_b):
        # skip this case for two reasons:
        # (a) pad_and_slice raises a "copy_kernel" not implemented
        #     for 'Float4_e2m1fn_x2'
        # (b) helps not having an explosive number of tests
        pytest.skip("Padded FP4 skipped on CPU (copy_kernel not implemented)")

    device = "cuda" if use_cuda else "cpu"

    m = 128
    k = 64 * 2  # tiles along K
    n = 128

    # A: row 5, K-groups 2 (tile 0) and 6 (tile 1)
    a_blocks = [
        (5, 2),  # K-tile 0: K elements 32-47
        (5, 6),  # K-tile 1: K elements 96-111
    ]
    a_fp4 = create_fp4_matrix_a_with_ones(m, k, blocks=a_blocks, device=device)
    if pad_a:
        a_fp4 = pad_and_slice(a_fp4)

    a_scale = create_nvfp4_zero_scale(outer_dim=m, inner_dim=k, device=device)
    a_scale_tile0 = 1.0
    a_scale_tile1 = 2.0
    set_nvfp4_scale_value(a_scale, outer_idx=5, inner_group_idx=2, scale_value=a_scale_tile0, inner_dim=k)
    set_nvfp4_scale_value(a_scale, outer_idx=5, inner_group_idx=6, scale_value=a_scale_tile1, inner_dim=k)

    # B: col 32, K-groups 2 (tile 0) and 6 (tile 1)
    b_blocks = [
        (32, 2),  # K-tile 0
        (32, 6),  # K-tile 1
    ]
    b_fp4 = create_fp4_matrix_b_with_ones(k, n, blocks=b_blocks, device=device)
    if pad_b:
        b_fp4 = pad_and_slice(b_fp4)

    b_scale = create_nvfp4_zero_scale(outer_dim=n, inner_dim=k, device=device)
    b_scale_tile0 = 4.0
    b_scale_tile1 = 8.0
    set_nvfp4_scale_value(b_scale, outer_idx=32, inner_group_idx=2, scale_value=b_scale_tile0, inner_dim=k, axis=-2)
    set_nvfp4_scale_value(b_scale, outer_idx=32, inner_group_idx=6, scale_value=b_scale_tile1, inner_dim=k, axis=-2)

    alpha = 1.0
    raw_result = matmul(
        a_fp4,
        b_fp4,
        alpha=alpha,
        quantization_scales={"a": a_scale, "b": b_scale},
        options={"result_type": NAME_TO_DATA_TYPE["float32"], "block_scaling": True},
    )

    # Expected: 16 * a_scale_tile0 * b_scale_tile0 + 16 * a_scale_tile1 * b_scale_tile1
    expected_value = alpha * (16 * a_scale_tile0 * b_scale_tile0 + 16 * a_scale_tile1 * b_scale_tile1)
    expected = torch.zeros_like(raw_result)
    expected[5, 32] = expected_value
    assert torch.equal(raw_result, expected), "Expected only position [5, 32] to be non-zero, but found other non-zero values"


@pytest.mark.parametrize("use_cuda", [True, False])
def test_nvfp4_matmul_sparse_analytic_two_outer_tiles(use_cuda):
    """
    Test NVFP4 matmul with M=256, spanning two outer tiles for A.

    This test verifies that block scaling works correctly when A spans multiple
    outer tiles (each tile covers 128 rows). We place non-zeros in two different
    rows (one per outer tile) to produce two non-zero output elements.

    Dimensions:
        - A: 256x64 (M=256, K=64) -> 2 outer tiles, 1 K-tile
        - B: 64x128 (K=64, N=128) -> 1 K-tile, 1 outer tile
        - C: 256x128

    Setup:
        - A: ones at row 5 (tile 0) and row 133 (tile 1), both at K-group 2
        - B: ones at col 32, K-group 2
        - Different scales for each A row to verify correct scale application

    Matrix A (M=256 x K=64):

            |<--------- K-tile 0 --------->|
              K-grp 0  K-grp 1  K-grp 2  K-grp 3
              0-15     16-31    32-47    48-63
            +--------+--------+--------+--------+
    row 0   |   0    |   0    |   0    |   0    |  -+
      ...   |   0    |   0    |   0    |   0    |   | outer
    row 5   |   0    |   0    | 1...1  |   0    |   | tile 0
      ...   |   0    |   0    |   0    |   0    |   | (rows
    row 127 |   0    |   0    |   0    |   0    |  -+  0-127)
            +--------+--------+--------+--------+
    row 128 |   0    |   0    |   0    |   0    |  -+
      ...   |   0    |   0    |   0    |   0    |   | outer
    row 133 |   0    |   0    | 1...1  |   0    |   | tile 1
      ...   |   0    |   0    |   0    |   0    |   | (rows
    row 255 |   0    |   0    |   0    |   0    |  -+  128-255)
            +--------+--------+--------+--------+
                               ^
                     a_scale[5,2]=1.0, a_scale[133,2]=2.0

    Matrix B (K=64 x N=128):

                                  col 32
                                    v
                      +---------+-------+---------+
    K-tile 0  grp 0   |    0    |   0   |    0    |
              grp 1   |    0    |   0   |    0    |
              grp 2   |    0    |   1   |    0    |  <- 16 ones, b_scale[32,2]=4.0
              grp 3   |    0    |   0   |    0    |
                      +---------+-------+---------+

    Result:
        C[5, 32]   = 16 * a_scale[5,2] * b_scale[32,2]   = 16 * 1.0 * 4.0 = 64.0
        C[133, 32] = 16 * a_scale[133,2] * b_scale[32,2] = 16 * 2.0 * 4.0 = 128.0
    """
    device = "cuda" if use_cuda else "cpu"

    m = 256  # 2 outer tiles
    k = 64  # 1 K-tile
    n = 128  # 1 outer tile

    # A: row 5 (outer tile 0) and row 133 (outer tile 1), both at K-group 2
    a_blocks = [
        (5, 2),  # outer tile 0, K-group 2
        (133, 2),  # outer tile 1, K-group 2
    ]
    a_fp4 = create_fp4_matrix_a_with_ones(m, k, blocks=a_blocks, device=device)

    a_scale = create_nvfp4_zero_scale(outer_dim=m, inner_dim=k, device=device)
    a_scale_row5 = 1.0
    a_scale_row133 = 2.0
    set_nvfp4_scale_value(a_scale, outer_idx=5, inner_group_idx=2, scale_value=a_scale_row5, inner_dim=k)
    set_nvfp4_scale_value(a_scale, outer_idx=133, inner_group_idx=2, scale_value=a_scale_row133, inner_dim=k)

    # B: col 32, K-group 2
    b_blocks = [
        (32, 2),
    ]
    b_fp4 = create_fp4_matrix_b_with_ones(k, n, blocks=b_blocks, device=device)

    b_scale = create_nvfp4_zero_scale(outer_dim=n, inner_dim=k, device=device)
    b_scale_col32 = 4.0
    set_nvfp4_scale_value(b_scale, outer_idx=32, inner_group_idx=2, scale_value=b_scale_col32, inner_dim=k, axis=-2)

    alpha = 1.0
    raw_result = matmul(
        a_fp4,
        b_fp4,
        alpha=alpha,
        quantization_scales={"a": a_scale, "b": b_scale},
        options={"result_type": NAME_TO_DATA_TYPE["float32"], "block_scaling": True},
    )

    # Verify two non-zero outputs
    expected_5_32 = alpha * 16 * a_scale_row5 * b_scale_col32
    expected_133_32 = alpha * 16 * a_scale_row133 * b_scale_col32

    expected = torch.zeros_like(raw_result)
    expected[5, 32] = expected_5_32
    expected[133, 32] = expected_133_32
    assert torch.equal(raw_result, expected), (
        "Expected only positions [5, 32] and [133, 32] to be non-zero, but found other non-zero values"
    )


@pytest.mark.parametrize("use_cuda", [True, False])
def test_nvfp4_matmul_sparse_analytic_fp4_output(use_cuda):
    """
    Test NVFP4 matmul with FP4 output dtype and verify d_out_scale formula.

    This test is similar to test_nvfp4_matmul_sparse_analytic_single_tile but
    uses FP4 output (float4_e2m1fn_x2) and verifies the d_out_scale
    is d_out_scale \approx max_value / 6.0, and the dequantized
    result matches the expected float value

    Setup:
        - A: 128x64 matrix with 16 ones in row 5, K-group 2
        - B: 64x128 matrix with 16 ones in col 32, K-group 2
        - A scale = 2.0, B scale = 3.0, alpha = 1.0

    Expected computation:
        - float_result = 16 * a_scale * b_scale * alpha = 16 * 2.0 * 3.0 * 1.0 = 96.0
        - d_out_scale = float_result / max_fp4_value = 96.0 / 6.0 = 16.0
        - fp4_quantized = 6.0 (max FP4 value, since 96/16 = 6)
        - dequantized = fp4_quantized * d_out_scale = 6.0 * 16.0 = 96.0

    Note: These scale values (2.0, 3.0) are chosen deliberately so that the true
    result 96.0 decomposes exactly into an FP4 value (6.0) times a
    float8_e4m3fn scale (16.0). This avoids quantization error and makes exact
    assertions possible.
    """
    device = "cuda" if use_cuda else "cpu"

    m = 128
    k = 64
    n = 128

    # A: row 5, K scale group 2 (elements 32-47 of K)
    a_outer_idx = 5
    a_inner_group_idx = 2
    a_fp4 = create_fp4_matrix_a_with_ones(m, k, blocks=[(a_outer_idx, a_inner_group_idx)], device=device)

    a_scale_value = 2.0
    a_scale = create_nvfp4_zero_scale(outer_dim=m, inner_dim=k, device=device)
    set_nvfp4_scale_value(a_scale, a_outer_idx, a_inner_group_idx, a_scale_value, inner_dim=k)

    # B: col 32, K scale group 2 (elements 32-47 of K)
    b_outer_idx = 32
    b_inner_group_idx = 2
    b_fp4 = create_fp4_matrix_b_with_ones(k, n, blocks=[(b_outer_idx, b_inner_group_idx)], device=device)

    b_scale_value = 3.0
    b_scale = create_nvfp4_zero_scale(outer_dim=n, inner_dim=k, device=device)
    set_nvfp4_scale_value(b_scale, b_outer_idx, b_inner_group_idx, b_scale_value, inner_dim=k, axis=-2)

    alpha = 1.0
    raw_result = matmul(
        a_fp4,
        b_fp4,
        alpha=alpha,
        quantization_scales={"a": a_scale, "b": b_scale},
        options={"result_type": NAME_TO_DATA_TYPE["float4_e2m1fn_x2"], "block_scaling": True},
    )

    # Unpack FP4 result and d_out_scale
    result_fp4, d_out_scale, _ = unpack_matmul(raw_result)
    assert result_fp4.dtype == torch.float4_e2m1fn_x2, f"Expected FP4 output, got {result_fp4.dtype}"
    assert d_out_scale is not None, "Expected d_out_scale for FP4 output"

    # Expected float result before quantization
    expected_float = 16 * alpha * a_scale_value * b_scale_value

    # Decode FP4 to float, result shape is (M, N/2) packed,
    # decode gives (M, N)
    result_decoded = unpack_fp4(result_fp4, axis=-1)

    # Expand d_out_scale to match result shape
    m, n = result_decoded.shape[-2:]
    device = "cuda" if result_decoded.is_cuda else "cpu"
    scales_expanded = expand_block_scale(d_out_scale, (m, n), "NVFP4", axis=-1, device=device, output_dtype=torch.float16)
    dequantized = result_decoded * scales_expanded
    actual_value = dequantized[5, 32].item()
    assert abs(actual_value - expected_float) < 1e-3, f"Dequantized C[5,32]: Expected {expected_float}, got {actual_value}"

    # Verify the FP4 quantized value is 6.0 (max FP4 value, since 96/16 = 6)
    fp4_value = result_decoded[5, 32].item()
    assert fp4_value == 6.0, f"FP4 C[5,32]: Expected 6.0, got {fp4_value}"

    # Verify d_out_scale for the block containing C[5,32]
    # d_out_scale = expected_float / 6.0 = 96.0 / 6.0 = 16.0
    expected_scale = expected_float / 6.0  # 16.0
    actual_scale = scales_expanded[5, 32].item()
    assert abs(actual_scale - expected_scale) < 1e-3, (
        f"d_out_scale for C[5,32] block: Expected {expected_scale}, got {actual_scale}"
    )

    # Verify all other values are zero
    expected_dequantized = torch.zeros_like(dequantized)
    expected_dequantized[5, 32] = expected_float
    assert torch.equal(dequantized, expected_dequantized), (
        "Expected only position [5, 32] to be non-zero, but found other non-zero values"
    )


@pytest.mark.parametrize("use_cuda", [True, False])
def test_nvfp4_matmul_sparse_analytic_batched(use_cuda):
    """
    Test NVFP4 matmul with batching (3 slices), each slice having a different
    non-zero block configuration with analytically known results.

    This test verifies that batched NVFP4 matmul correctly handles independent
    computations per batch slice, with each slice exercising different blocks
    and scale factors.

    Dimensions:
        - Batch size: 3
        - A: (3, 128, 64) -> 3 slices of 128x64 matrices
        - B: (3, 64, 128) -> 3 slices of 64x128 matrices (column-major packed)
        - C: (3, 128, 128)

    Setup for each batch slice:

    Slice 0: Non-zero at A[row=5, K-group=2], B[col=32, K-group=2]
             Scales: a_scale=1.0, b_scale=2.0
             Result: C[0, 5, 32] = 16 * 1.0 * 2.0 = 32.0

    Slice 1: Non-zero at A[row=10, K-group=1], B[col=64, K-group=1]
             Scales: a_scale=2.0, b_scale=3.0
             Result: C[1, 10, 64] = 16 * 2.0 * 3.0 = 96.0

    Slice 2: Non-zero at A[row=100, K-group=3], B[col=100, K-group=3]
             Scales: a_scale=4.0, b_scale=1.5
             Result: C[2, 100, 100] = 16 * 4.0 * 1.5 = 96.0

    Visual representation of slice 0:

    Matrix A[0] (M=128 x K=64):
              K-group 0    K-group 1    K-group 2    K-group 3
            +------------+------------+------------+------------+
    row 5   |     0      |     0      |  1 1...1   |     0      |
            +------------+------------+------------+------------+
                                            ^
                               a_scale[0, row5, grp2] = 1.0

    Matrix B[0] (K=64 x N=128):
                                  col 32
                      +---------+-------+---------+
    K-group 2         |    0    | 1...1 |    0    |
                      +---------+-------+---------+
                                    ^
                       b_scale[0, col32, grp2] = 2.0

    Each slice exercises a different (row, col, K-group) combination with
    different scale factors, ensuring the batched computation correctly
    separates the per-slice scale lookups and accumulations.
    """
    device = "cuda" if use_cuda else "cpu"

    batch_size = 3
    m = 128
    k = 64
    n = 128

    # Configuration for each batch slice
    slice_configs = [
        (5, 2, 32, 2, 1.0, 2.0),  # Slice 0: row 5, col 32, K-group 2
        (10, 1, 64, 1, 2.0, 3.0),  # Slice 1: row 10, col 64, K-group 1
        (100, 3, 100, 3, 4.0, 1.5),  # Slice 2: row 100, col 100, K-group 3
    ]

    # Build batch_blocks for A and B from slice_configs
    a_batch_blocks = [[(cfg[0], cfg[1])] for cfg in slice_configs]  # (a_row, a_k_group)
    b_batch_blocks = [[(cfg[2], cfg[3])] for cfg in slice_configs]  # (b_col, b_k_group)

    # Create batched A and B matrices using helpers that handle FP4 at bytes level
    a_fp4 = create_batched_fp4_matrix_a_with_ones(batch_size, m, k, a_batch_blocks, device=device)
    b_fp4 = create_batched_fp4_matrix_b_with_ones(batch_size, k, n, b_batch_blocks, device=device)

    # Create batched scale tensors (1D flattened: batch_size * scales_per_slice)
    # For batched NVFP4, scales are stored as contiguous 1D array
    num_a_scales_per_slice = m * (k // 16)
    num_b_scales_per_slice = n * (k // 16)

    a_scale = torch.zeros((batch_size * num_a_scales_per_slice,), dtype=torch.float8_e4m3fn, device=device)
    b_scale = torch.zeros((batch_size * num_b_scales_per_slice,), dtype=torch.float8_e4m3fn, device=device)

    for batch_idx, (a_row, a_k_group, b_col, b_k_group, a_scale_val, b_scale_val) in enumerate(slice_configs):
        # Set A scale for this batch slice (batch_offset + intra-slice index)
        # Helper expects operand indices (blocked dim = element index, not group index)
        a_scale_index = get_block_scale_offset((batch_idx, a_row, a_k_group), (batch_size, m, k), "NVFP4", axis=-1)
        a_scale[a_scale_index] = a_scale_val

        # Set B scale for this batch slice (batch_offset + intra-slice index)
        b_scale_index = get_block_scale_offset((batch_idx, b_k_group, b_col), (batch_size, k, n), "NVFP4", axis=-2)
        b_scale[b_scale_index] = b_scale_val

    alpha = 1.0
    raw_result = matmul(
        a_fp4,
        b_fp4,
        alpha=alpha,
        quantization_scales={"a": a_scale, "b": b_scale},
        options={"result_type": NAME_TO_DATA_TYPE["float32"], "block_scaling": True},
    )

    expected_shape = (batch_size, m, n)
    assert raw_result.shape == expected_shape, f"Expected shape {expected_shape}, got {raw_result.shape}"

    expected = torch.zeros_like(raw_result)
    for batch_idx, (a_row, _, b_col, _, a_scale_val, b_scale_val) in enumerate(slice_configs):
        expected_value = 16 * alpha * a_scale_val * b_scale_val
        expected[batch_idx, a_row, b_col] = expected_value

    # Verify results
    assert torch.equal(raw_result, expected), (
        f"Batched result mismatch.\n"
        f"Expected non-zeros at: {[(i, cfg[0], cfg[2]) for i, cfg in enumerate(slice_configs)]}\n"
        f"Expected values: {[16 * alpha * cfg[4] * cfg[5] for cfg in slice_configs]}\n"
        f"Actual non-zero count: {(raw_result != 0).sum().item()}\n"
        f"Actual non-zero positions: {torch.nonzero(raw_result).tolist()}"
    )


@pytest.mark.parametrize("use_cuda", [True, False])
def test_nvfp4_matmul_reset_operands(use_cuda):
    """
    Test that reset_operands correctly swaps A, B, and their scales for NVFP4.
    This test's goal is to verify that reset_operands works correctly,
    so it does not really matter what data we are using. We can use any data we want.
    To make the test simpler, we use a fixed set of configurations
    that are easy to understand: each configuration places ones in a different
    (row, K-group) for A and (col, K-group) for B with different scale values,
    producing a single non-zero output element at a predictable position.
    """
    device = "cuda" if use_cuda else "cpu"

    m, k, n = 128, 64, 128
    # Each tuple: (a_row, a_k_group, b_col, b_k_group, a_scale_val, b_scale_val)
    configs = [
        (5, 2, 32, 2, 1.0, 4.0),  # C[5, 32]   = 16 * 1.0 * 4.0 = 64.0
        (10, 1, 64, 1, 2.0, 3.0),  # C[10, 64]  = 16 * 2.0 * 3.0 = 96.0
        (100, 3, 100, 3, 0.5, 1.5),  # C[100,100] = 16 * 0.5 * 1.5 = 12.0
        (0, 0, 0, 0, 3.0, 0.5),  # C[0, 0]    = 16 * 3.0 * 0.5 = 24.0
    ]

    options = {"result_type": NAME_TO_DATA_TYPE["float32"], "block_scaling": True}

    # Build the initial operands from the first configuration.
    a_row, a_kg, b_col, b_kg, a_sv, b_sv = configs[0]

    a_fp4 = create_fp4_matrix_a_with_ones(m, k, blocks=[(a_row, a_kg)], device=device)
    b_fp4 = create_fp4_matrix_b_with_ones(k, n, blocks=[(b_col, b_kg)], device=device)
    a_scale = create_nvfp4_zero_scale(outer_dim=m, inner_dim=k, device=device)
    set_nvfp4_scale_value(a_scale, a_row, a_kg, a_sv, inner_dim=k)
    b_scale = create_nvfp4_zero_scale(outer_dim=n, inner_dim=k, device=device)
    set_nvfp4_scale_value(b_scale, b_col, b_kg, b_sv, inner_dim=k, axis=-2)

    with Matmul(
        a_fp4,
        b_fp4,
        quantization_scales={"a": a_scale, "b": b_scale},
        options=options,
    ) as mm:
        mm.plan()

        # initial configuration.
        result = mm.execute()
        expected = torch.zeros_like(result)
        expected[a_row, b_col] = 16 * a_sv * b_sv
        assert torch.equal(result, expected), "Config 0 mismatch"

        # Loop through remaining configurations, resetting operands each time.
        for i, (a_row, a_kg, b_col, b_kg, a_sv, b_sv) in enumerate(configs[1:], start=1):
            new_a = create_fp4_matrix_a_with_ones(m, k, blocks=[(a_row, a_kg)], device=device)
            new_b = create_fp4_matrix_b_with_ones(k, n, blocks=[(b_col, b_kg)], device=device)

            new_a_scale = create_nvfp4_zero_scale(outer_dim=m, inner_dim=k, device=device)
            set_nvfp4_scale_value(new_a_scale, a_row, a_kg, a_sv, inner_dim=k)
            new_b_scale = create_nvfp4_zero_scale(outer_dim=n, inner_dim=k, device=device)
            set_nvfp4_scale_value(new_b_scale, b_col, b_kg, b_sv, inner_dim=k, axis=-2)

            mm.reset_operands(
                a=new_a,
                b=new_b,
                quantization_scales={"a": new_a_scale, "b": new_b_scale},
            )

            result = mm.execute()
            expected = torch.zeros_like(result)
            expected[a_row, b_col] = 16 * a_sv * b_sv
            assert torch.equal(result, expected), (
                f"Config {i} mismatch: expected non-zero at [{a_row}, {b_col}] = {expected[a_row, b_col].item()}, "
                f"got {result[a_row, b_col].item()}"
            )
