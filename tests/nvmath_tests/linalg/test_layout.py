# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
import pprint
import pytest
import time

from nvmath.bindings import cublasLt as cublaslt
from nvmath.linalg._internal.layout import BLASMatrixTraits, InputMMTraits, BLASMMTraitsView
from nvmath.linalg.generic._configuration.layout import (
    is_supported_gemm_layout,
    is_supported_symm_layout_left,
    is_supported_symm_layout_right,
    is_supported_trmm_layout_left,
    is_supported_trmm_layout_right,
)

COL = cublaslt.Order.COL
ROW = cublaslt.Order.ROW


def create_blas_view_transformation_lookup(layout_checker=None, logger=None):
    """Create a lookup table of all unique BLASMMTraitsView transformations.

    This function exhaustively enumerates different input layout combinations
    (COL/ROW orders) and uses BLASMMTraitsView.find_supported_layout to
    determine what transformations are applied for each input configuration.

    For simplicity, we assume all inputs are non-transpose (is_transpose=False) because
    the user should do a no-op transpose if they want to using their tensor library.

    Args:
        layout_checker: A callable that determines if a layout is supported.
                       If None, accepts all COL-order layouts.
        logger: A logger instance. If None, creates a default logger.

    Returns:
        dict: A lookup table mapping input configurations to output transformations.
              Key format: (a_order, b_order, c_order)
              Value format: dict with keys:
                  'output_view': The resulting BLASMMTraitsView
                  'is_swapped': Whether A and B were swapped
                  'transformations': List of transformations applied
                  'input_traits': The input BLASMMTraits
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if layout_checker is None:
        # Default: accept only if all operands are COL-order
        def layout_checker(view):
            return view.a_layout_traits.order == COL and view.b_layout_traits.order == COL and view.c_layout_traits.order == COL

    lookup_table = {}
    search_time = []
    lookup_time = []
    order_options = [COL, ROW]
    conjugate_options = [True, False]

    # The shape is irrelevant, so just use the same shape for all matrices
    shape = (3, 3)

    for a_order, b_order, c_order in itertools.product(order_options, order_options, order_options):
        for a_conjugate, b_conjugate, c_conjugate in itertools.product(conjugate_options, conjugate_options, conjugate_options):
            # Create strides based on order
            a_strides = (1, shape[0]) if a_order == COL else (shape[1], 1)
            b_strides = (1, shape[0]) if b_order == COL else (shape[1], 1)
            c_strides = (1, shape[0]) if c_order == COL else (shape[1], 1)

            try:
                a_traits = BLASMatrixTraits(
                    dtype=0,
                    shape=shape,
                    strides=a_strides,
                    is_conjugate=a_conjugate,
                    is_transpose=False,
                    is_lower=False,
                )
                b_traits = BLASMatrixTraits(
                    dtype=0,
                    shape=shape,
                    strides=b_strides,
                    is_conjugate=b_conjugate,
                    is_transpose=False,
                    is_lower=False,
                )
                c_traits = BLASMatrixTraits(
                    dtype=0,
                    shape=shape,
                    strides=c_strides,
                    is_conjugate=c_conjugate,
                    is_transpose=False,
                    is_lower=False,
                )

                # Create initial view
                initial_view = BLASMMTraitsView(
                    M=3,
                    N=3,
                    K=3,
                    a_layout_traits=a_traits,
                    b_layout_traits=b_traits,
                    c_layout_traits=c_traits,
                    is_swapped_AB=False,
                )

                # Find supported layout
                start = time.perf_counter_ns()
                result_view = initial_view.find_supported_layout(layout_checker, logger)
                stop = time.perf_counter_ns()

                # Determine what transformations were applied
                transformations = []
                if result_view.a_layout_traits.order != a_traits.order:
                    transformations.append("transpose_and_reorder_A")
                if result_view.b_layout_traits.order != b_traits.order:
                    transformations.append("transpose_and_reorder_B")
                if result_view.c_layout_traits.order != c_traits.order:
                    transformations.append("transpose_and_reorder_C")
                if result_view.is_swapped_AB:
                    transformations.append("swap_AB_and_transpose_ABC")

                # Create lookup key (simplified - only orders matter and conjugates matter)
                key = (
                    a_order,
                    b_order,
                    c_order,
                    a_conjugate,
                    b_conjugate,
                    c_conjugate,
                )

                # Store result
                lookup_table[key] = transformations
                search_time.append(stop - start)

                start = time.perf_counter_ns()
                _ = lookup_table[key]
                stop = time.perf_counter_ns()
                lookup_time.append(stop - start)

            except ValueError:
                pass
                # No supported layout found for this combination

    return (
        lookup_table,
        (sum(search_time) / len(search_time)) / 1_000,
        max(search_time) / 1_000,
        (sum(lookup_time) / len(lookup_time)) / 1_000,
        max(lookup_time) / 1_000,
    )


def test_create_blas_view_transformation_lookup_gemm():
    """Test the create_blas_view_transformation_lookup with the gemm layout checker."""
    lookup_table = create_blas_view_transformation_lookup(
        layout_checker=is_supported_gemm_layout,
    )
    print("Lookup table:\n")
    pprint.pprint(lookup_table)


def test_create_blas_view_transformation_lookup_symm_left():
    """Test the create_blas_view_transformation_lookup with the symm layout checker."""
    lookup_table = create_blas_view_transformation_lookup(
        layout_checker=is_supported_symm_layout_left,
    )
    print("Lookup table:\n")
    pprint.pprint(lookup_table)


def test_create_blas_view_transformation_lookup_symm_right():
    """Test the create_blas_view_transformation_lookup with the symm layout checker."""
    lookup_table = create_blas_view_transformation_lookup(
        layout_checker=is_supported_symm_layout_right,
    )
    print("Lookup table:\n")
    pprint.pprint(lookup_table)


def test_create_blas_view_transformation_lookup_trmm_left():
    """Test the create_blas_view_transformation_lookup with the trmm layout checker."""
    lookup_table = create_blas_view_transformation_lookup(
        layout_checker=is_supported_trmm_layout_left,
    )
    print("Lookup table:\n")
    pprint.pprint(lookup_table)


def test_create_blas_view_transformation_lookup_trmm_right():
    """Test the create_blas_view_transformation_lookup with the trmm layout checker."""
    lookup_table = create_blas_view_transformation_lookup(
        layout_checker=is_supported_trmm_layout_right,
    )
    print("Lookup table:\n")
    pprint.pprint(lookup_table)


order_cases = [
    # 0D
    ((), (), COL, None),
    # 1D - overlapping
    ((1,), (0,), COL, None),
    ((4,), (0,), COL, None),
    # 1D - dense
    ((1,), (1,), COL, None),
    ((4,), (1,), COL, None),
    # 1D - strided
    ((1,), (8,), COL, None),
    ((4,), (8,), COL, None),
    #
    # 2D - overlapping, overlapping
    ((1, 1), (0, 0), COL, 1),
    ((1, 4), (0, 0), COL, None),
    ((4, 1), (0, 0), COL, None),
    ((4, 4), (0, 0), COL, None),
    # 2D - overlapping, dense
    ((1, 1), (0, 1), COL, 1),
    ((1, 4), (0, 1), COL, 1),
    ((4, 1), (0, 1), COL, None),
    ((4, 4), (0, 1), COL, None),
    # 2D - overlapping, strided
    ((1, 1), (0, 8), COL, 1),
    ((1, 4), (0, 8), COL, 8),
    ((4, 1), (0, 8), COL, None),
    ((4, 4), (0, 8), COL, None),
    #
    # 2D - dense, overlapping
    ((1, 1), (1, 0), ROW, None),
    ((1, 4), (1, 0), ROW, None),
    ((4, 1), (1, 0), ROW, None),
    ((4, 4), (1, 0), ROW, None),
    ((4, 4), (1, 1), None, None),
    # 2D - dense, dense
    ((1, 1), (1, 1), COL, 1),
    ((1, 4), (1, 1), COL, 1),
    ((4, 1), (1, 4), COL, 4),
    ((4, 4), (1, 4), COL, 4),
    ((1, 4), (4, 1), ROW, None),
    ((4, 1), (1, 1), ROW, None),
    ((4, 4), (4, 1), ROW, None),
    # 2D - dense, strided
    ((1, 1), (1, 8), COL, 1),
    ((1, 4), (1, 8), COL, 8),
    ((4, 1), (1, 8), COL, 8),
    ((4, 4), (1, 8), COL, 8),
    #
    # 2D - strided, overlapping
    ((1, 1), (8, 0), ROW, None),
    ((1, 4), (8, 0), ROW, None),
    ((4, 1), (8, 0), ROW, None),
    ((4, 4), (8, 0), ROW, None),
    # 2D - strided, dense
    ((1, 1), (8, 1), ROW, None),
    ((1, 4), (8, 1), ROW, None),
    ((4, 1), (8, 1), ROW, None),
    ((4, 4), (8, 1), ROW, None),
    # 2D - strided, strided
    ((1, 1), (8, 128), COL, 1),
    ((1, 4), (8, 128), COL, 128),
    ((4, 1), (8, 128), ROW, None),
    ((4, 4), (8, 128), None, None),
    ((1, 1), (128, 8), COL, 1),
    ((1, 4), (128, 8), COL, 8),
    ((4, 1), (128, 8), ROW, None),
    ((4, 4), (128, 8), None, None),
]


@pytest.mark.parametrize("shape, strides, order, ld", order_cases)
def test_matrix_traits_order(shape, strides, order, ld):
    t = BLASMatrixTraits(dtype=0, shape=shape, strides=strides, is_conjugate=False, is_lower=False, is_transpose=False)
    if order is None:
        with pytest.raises(ValueError, match="Unsupported layout"):
            _ = t.order
        return
    assert t.order == order

    if ld is None:
        with pytest.raises((ValueError, AssertionError)):
            _ = t.ld
        return
    assert t.ld == ld


@pytest.mark.parametrize(
    "shape, strides, expected_ld",
    [
        ((1, 4), (1, 4), 4),
        ((1, 10), (1, 20), 20),
        ((2, 5), (1, 8), 8),
        ((4, 3), (1, 4), 4),
        ((4, 1), (1, 4), 4),
        ((1, 10), (1, 1), 1),
    ],
)
def test_blas_matrix_traits_ld(shape, strides, expected_ld):
    t = BLASMatrixTraits(dtype=0, shape=shape, strides=strides, is_conjugate=False, is_lower=False, is_transpose=False)
    # The order property will check if the matrix is COL
    assert t.order == COL
    # Test ld
    assert t.ld == expected_ld


@pytest.mark.parametrize(
    "shape, strides",
    [
        ((10, 1), (1, 1)),
        ((10, 1), (2, 8)),
        ((10, 1), (8, 3)),
        # Square matrix nontrivial stride
        ((3, 3), (2, 5)),
        # Nonsquare, stride 0 in non-singleton dimension
        ((2, 2), (0, 5)),
        ((2, 2), (2, 0)),
        # 2D, scalar extent requires stride 0, but non-singleton
        # bad stride config, expected to fail ld
        ((2, 2), (3, 4)),
        ((2, 2), (4, 3)),
        # Not COL-major by definition (row-major)
        ((2, 2), (5, 1)),
        # 1D shapes
        ((1,), (1,)),
    ],
)
def test_blas_matrix_traits_ld_invalid(shape, strides):
    t = BLASMatrixTraits(dtype=0, shape=shape, strides=strides, is_conjugate=False, is_lower=False, is_transpose=False)
    # The order property may error, else ld must error
    if len(shape) != 2:
        with pytest.raises(AssertionError):
            _ = t.ld
    else:
        # Could also raise ValueError for unsupported layouts
        with pytest.raises((ValueError, AssertionError)):
            _ = t.ld


def make_traits(is_conjugate=False, is_transpose=False):
    # Minimal 2D COL-major shape and strides, other args are dummies for test
    return BLASMatrixTraits(
        dtype=0,
        shape=(2, 2),
        strides=(1, 2),
        is_conjugate=is_conjugate,
        is_transpose=is_transpose,
        is_lower=False,
    )


@pytest.mark.parametrize(
    "is_conjugate, is_transpose, expected_name, expect_error",
    [
        (False, False, "N", False),
        (False, True, "T", False),
        (True, True, "C", False),
        (True, False, None, True),
    ],
)
def test_blas_matrix_traits_operation(is_conjugate, is_transpose, expected_name, expect_error):
    t = make_traits(is_conjugate=is_conjugate, is_transpose=is_transpose)
    if expect_error:
        with pytest.raises(NotImplementedError):
            _ = t.operation
    else:
        assert t.operation.name == expected_name


@pytest.mark.parametrize(
    "shape, strides, expected",
    [
        # Two dimensions - contiguous matrices, should return same strides
        ((2, 2), (1, 2), (1, 2)),
        ((2, 2), (2, 1), (2, 1)),
        ((2, 1), (1, 2), (1, 2)),
        ((1, 2), (2, 1), (2, 1)),
        # Singleton dimension, stride for singleton should be 0
        ((1, 1), (5, 7), (0, 0)),
        ((1, 1), (9, 3), (0, 0)),
        # Non-contiguous matrix, should return None
        ((2, 2), (2, 2), None),
        # 1D contiguous matrix
        ((4,), (1,), (1,)),
        # 1D singleton, stride for singleton should be 0
        ((1,), (7,), (0,)),
        # 0D
        ((), (), ()),
    ],
)
def test_blas_matrix_traits_strides_contiguous_perhaps(shape, strides, expected):
    t = BLASMatrixTraits(
        dtype=0,
        shape=shape,
        strides=strides,
        is_conjugate=False,
        is_transpose=False,
        is_lower=False,
    )
    if expected is None:
        assert t.strides_contiguous_perhaps is None
    else:
        assert t.strides_contiguous_perhaps == expected


# InputMMTraits tests
@pytest.fixture
def logger():
    """Fixture to provide a logger for tests."""
    return logging.getLogger(__name__)


class TestInputMMTraits:
    """Test suite for InputMMTraits class."""

    def test_2d_by_2d_matmul(self, logger):
        """Test standard 2D x 2D matrix multiplication."""
        # A: (4, 3), B: (3, 5) -> C: (4, 5)
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M == 4
        assert traits.N == 5
        assert traits.K == 3
        assert traits.inplace is False
        assert traits.a_layout_traits == a
        assert traits.b_layout_traits == b
        assert traits.d_layout_traits.shape == (4, 5)

    def test_2d_by_2d_matmul_with_transpose(self, logger):
        """Test 2D x 2D matrix multiplication with transpose flag."""
        # A: (3, 4)^T = (4, 3), B: (3, 5) -> C: (4, 5)
        a = BLASMatrixTraits(dtype=0, shape=(3, 4), strides=(1, 3), is_conjugate=False, is_transpose=True, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M == 4
        assert traits.N == 5
        assert traits.K == 3
        assert traits.inplace is False
        assert traits.a_layout_traits == a
        assert traits.b_layout_traits == b
        assert traits.d_layout_traits.shape == (4, 5)

    def test_1d_by_1d_dot_product(self, logger):
        """Test 1D x 1D vector dot product (results in scalar)."""
        # A: (5,), B: (5,) -> C: scalar
        a = BLASMatrixTraits(dtype=0, shape=(5,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(5,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M is None
        assert traits.N is None
        assert traits.K == 5
        assert traits.d_layout_traits.shape == ()

    def test_2d_by_1d_matvec(self, logger):
        """Test 2D x 1D matrix-vector multiplication."""
        # A: (4, 3), B: (3,) -> C: (4,)
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M == 4
        assert traits.N is None
        assert traits.K == 3
        assert traits.d_layout_traits.shape == (4,)

    def test_1d_by_2d_vecmat(self, logger):
        """Test 1D x 2D vector-matrix multiplication."""
        # A: (3,), B: (3, 5) -> C: (5,)
        a = BLASMatrixTraits(dtype=0, shape=(3,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M is None
        assert traits.N == 5
        assert traits.K == 3
        assert traits.d_layout_traits.shape == (5,)

    def test_inplace_operation(self, logger):
        """Test in-place matrix multiplication (C = alpha*A*B + beta*C)."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, c, inplace=True, logger=logger)

        assert traits.inplace is True
        assert traits.d_layout_traits == c
        assert traits.c_layout_traits == c

    def test_out_of_place_with_c(self, logger):
        """Test out-of-place operation with C provided."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, c, inplace=False, logger=logger)

        assert traits.inplace is False
        assert traits.c_layout_traits == c
        assert traits.d_layout_traits.shape == (4, 5)

    def test_mismatched_k_dimension_error(self, logger):
        """Test that mismatched K dimensions raise an error."""
        # A: (4, 3), B: (5, 7) -> K mismatch (3 != 5)
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(5, 7), strides=(1, 5), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="'K' extent must match"):
            InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

    def test_mismatched_m_dimension_error(self, logger):
        """Test that mismatched M dimensions raise an error."""
        # A: (4, 3), B: (3, 5), C: (6, 5) -> M mismatch (4 != 6)
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(6, 5), strides=(1, 6), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="'M' extent must match"):
            InputMMTraits.from_layouts(a, b, c, inplace=False, logger=logger)

    def test_mismatched_n_dimension_error(self, logger):
        """Test that mismatched N dimensions raise an error."""
        # A: (4, 3), B: (3, 5), C: (4, 7) -> N mismatch (5 != 7)
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 7), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="'N' extent must match"):
            InputMMTraits.from_layouts(a, b, c, inplace=False, logger=logger)

    def test_non_contiguous_a_error(self, logger):
        """Test that non-contiguous A raises an error."""
        # A with non-contiguous strides
        a = BLASMatrixTraits(dtype=0, shape=(4, 4), strides=(8, 128), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="Operand A is not contiguous"):
            InputMMTraits.from_layouts(a, b, c, inplace=False, logger=logger)

    def test_non_contiguous_b_error(self, logger):
        """Test that non-contiguous B raises an error."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        # B with non-contiguous strides
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(8, 128), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="Operand B is not contiguous"):
            InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

    def test_non_contiguous_c_error(self, logger):
        """Test that non-contiguous C raises an error."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        # C with non-contiguous strides
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(8, 128), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="Operand C is not contiguous"):
            InputMMTraits.from_layouts(a, b, c, inplace=True, logger=logger)

    def test_row_order_output(self, logger):
        """Test that ROW order inputs produce ROW order output by default."""
        # Both A and B are ROW order
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(3, 1), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.d_layout_traits.order == ROW
        assert traits.d_layout_traits.strides == (5, 1)

    def test_col_order_output(self, logger):
        """Test that mixed order inputs produce COL order output by default."""
        # A is COL, B is ROW
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.d_layout_traits.order == COL
        assert traits.d_layout_traits.strides == (1, 4)

    def test_output_order_matches_c_when_provided(self, logger):
        """Test that output order matches C when C is provided."""
        # A and B are COL, but C is ROW
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, c, inplace=False, logger=logger)

        assert traits.d_layout_traits.order == ROW

    def test_vector_vector_with_wrong_c_shape_error(self, logger):
        """Test that vector-vector with non-scalar C raises error."""
        # A: (5,), B: (5,) -> scalar, but C is (5,)
        a = BLASMatrixTraits(dtype=0, shape=(5,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(5,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(5,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)

        with pytest.raises(ValueError, match="operand C must be scalar-like"):
            InputMMTraits.from_layouts(a, b, c, inplace=True, logger=logger)

    def test_singleton_dimensions(self, logger):
        """Test matrix multiplication with singleton dimensions."""
        # A: (1, 3), B: (3, 1) -> C: (1, 1)
        a = BLASMatrixTraits(dtype=0, shape=(1, 3), strides=(0, 1), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 1), strides=(1, 0), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M == 1
        assert traits.N == 1
        assert traits.K == 3
        assert traits.d_layout_traits.shape == (1, 1)

    def test_scalar_operands(self, logger):
        """Test scalar operands (0D tensors)."""
        # A: scalar, B: scalar -> C: scalar
        a = BLASMatrixTraits(dtype=0, shape=(), strides=(), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(), strides=(), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.M is None
        assert traits.N is None
        assert traits.K is None
        assert traits.d_layout_traits.shape == ()

    def test_conjugate_flag_preserved(self, logger):
        """Test that conjugate flags are preserved in the traits."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=True, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        assert traits.a_layout_traits.is_conjugate is True
        assert traits.b_layout_traits.is_conjugate is False


class TestBLASMMTraitsView:
    """Test suite for BLASMMTraitsView class."""

    def test_from_input_traits_col_col_col(self, logger):
        """Test creating BLAS view from COL-order inputs."""
        # All operands are COL order - should not need transformations
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, c, inplace=True, logger=logger)

        # Simple layout checker that accepts all COL layouts
        def layout_checker(view):
            return view.a_layout_traits.order == COL and view.b_layout_traits.order == COL and view.c_layout_traits.order == COL

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        assert view.M == 4
        assert view.N == 5
        assert view.K == 3
        assert view.a_layout_traits.order == COL
        assert view.a_layout_traits.is_transpose is False
        assert view.b_layout_traits.order == COL
        assert view.b_layout_traits.is_transpose is False
        assert view.c_layout_traits.order == COL
        assert view.c_layout_traits.is_transpose is False
        assert view.is_swapped_AB is False

    def test_from_input_traits_row_row_row(self, logger):
        """Test creating BLAS view from ROW-order inputs."""
        # All operands are ROW order - should need all transformations
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(3, 1), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(5, 1), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, c, inplace=True, logger=logger)

        # Layout checker that accepts all COL layouts
        def layout_checker(view):
            return view.a_layout_traits.order == COL and view.b_layout_traits.order == COL and view.c_layout_traits.order == COL

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        assert view.M == 5
        assert view.N == 4
        assert view.K == 3
        assert view.a_layout_traits.order == COL
        assert view.a_layout_traits.is_transpose is False
        assert view.b_layout_traits.order == COL
        assert view.b_layout_traits.is_transpose is False
        assert view.c_layout_traits.order == COL
        assert view.c_layout_traits.is_transpose is False
        assert view.is_swapped_AB is True

    def test_from_input_traits_promotes_1d_operands(self, logger):
        """Test that 1D operands are promoted to 2D."""
        # A is 2D, B is 1D (vector)
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        def layout_checker(view):
            return True  # Accept any layout

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        # B should be promoted to 2D with right promotion (3,) -> (3, 1)
        assert len(view.a_layout_traits.shape) == 2
        assert len(view.b_layout_traits.shape) == 2
        assert len(view.c_layout_traits.shape) == 2
        assert view.M == 4
        assert view.K == 3
        assert view.N == 1

    def test_from_input_traits_1d_by_2d(self, logger):
        """Test that 1D x 2D (vector x matrix) is handled correctly."""
        # A is 1D (vector), B is 2D
        a = BLASMatrixTraits(dtype=0, shape=(3,), strides=(1,), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        def layout_checker(view):
            return True  # Accept any layout

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        # A should be promoted to 2D with left promotion (3,) -> (1, 3)
        assert len(view.a_layout_traits.shape) == 2
        assert len(view.b_layout_traits.shape) == 2
        assert len(view.c_layout_traits.shape) == 2
        assert view.M == 1
        assert view.N == 5
        assert view.K == 3

    def test_from_input_traits_with_transpose(self, logger):
        """Test BLAS view creation with transposed operands."""
        # A is transposed
        a = BLASMatrixTraits(dtype=0, shape=(3, 4), strides=(1, 3), is_conjugate=False, is_transpose=True, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        def layout_checker(view):
            return view.a_layout_traits.order == COL and view.b_layout_traits.order == COL and view.c_layout_traits.order == COL

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        assert view.M == 4
        assert view.N == 5
        assert view.K == 3

    def test_from_input_traits_mixed_orders(self, logger):
        """Test BLAS view creation with mixed COL/ROW orders."""
        # A is ROW, B is COL, C is COL
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(3, 1), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)
        c = BLASMatrixTraits(dtype=0, shape=(4, 5), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, c, inplace=True, logger=logger)

        def layout_checker(view):
            return view.a_layout_traits.order == COL and view.b_layout_traits.order == COL and view.c_layout_traits.order == COL

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        assert view.M == 4
        assert view.N == 5
        assert view.K == 3
        assert view.a_layout_traits.order == COL
        assert view.b_layout_traits.order == COL
        assert view.c_layout_traits.order == COL

    def test_from_input_traits_no_supported_layout(self, logger):
        """Test that error is raised when no supported layout can be found."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        # Layout checker that rejects everything
        def layout_checker(view):
            return False

        with pytest.raises(ValueError, match="No BLAS compatible view"):
            BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

    def test_from_input_traits_preserves_conjugate(self, logger):
        """Test that conjugate flags are preserved through BLAS view creation."""
        a = BLASMatrixTraits(dtype=0, shape=(4, 3), strides=(1, 4), is_conjugate=True, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 5), strides=(1, 3), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        def layout_checker(view):
            return True

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        assert view.a_layout_traits.is_conjugate is True
        assert view.b_layout_traits.is_conjugate is False

    def test_from_input_traits_singleton_dimensions(self, logger):
        """Test BLAS view creation with singleton dimensions."""
        # A: (1, 3), B: (3, 1)
        a = BLASMatrixTraits(dtype=0, shape=(1, 3), strides=(0, 1), is_conjugate=False, is_transpose=False, is_lower=False)
        b = BLASMatrixTraits(dtype=0, shape=(3, 1), strides=(1, 0), is_conjugate=False, is_transpose=False, is_lower=False)

        input_traits = InputMMTraits.from_layouts(a, b, None, inplace=False, logger=logger)

        def layout_checker(view):
            return True

        view = BLASMMTraitsView.from_input_traits(input_traits, layout_checker, logger)

        assert view.M == 1
        assert view.N == 1
        assert view.K == 3
