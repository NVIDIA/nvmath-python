# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import cublasLt as cublaslt
from nvmath.bindings.cublasLt import Order
from nvmath.linalg._internal.layout import BLASMatrixTraits, BLASMMTraitsView, MMLayoutChecker, MMLayoutCheckerLookupTable

from .qualifiers import MatrixQualifier


def mm_layout_checker_getter(qualifiers: MatrixQualifier) -> MMLayoutChecker:
    """Return a function which checks if a layout is supported for a BLAS matrix
    multiplication."""
    match (qualifiers[0]["abbreviation"], qualifiers[1]["abbreviation"], qualifiers[2]["abbreviation"]):
        case ("ge", "ge", "ge"):
            return is_supported_gemm_layout
        case ("sy" | "he" | "dg", "ge", "ge"):
            return is_supported_symm_layout_left
        case ("ge", "sy" | "he" | "dg", "ge"):
            return is_supported_symm_layout_right
        case ("tr", "ge", "ge"):
            return is_supported_trmm_layout_left
        case ("ge", "tr", "ge"):
            return is_supported_trmm_layout_right
        case _:
            raise NotImplementedError(
                f"Layout supported checker for {qualifiers[0]['abbreviation']}, {qualifiers[1]['abbreviation']}, "
                f"{qualifiers[2]['abbreviation']} is not implemented."
            )


def is_supported_gemm_layout(traits: BLASMMTraitsView) -> bool:
    """Return True if the layout is supported for a BLAS matrix multiplication.

    Rules:
    - All operands must be COL order.
    - C must be non-conjugate non-transpose.
    - The general matrix, B, must not be conjugate non-transpose.
    - The general matrix, A, must not be conjugate non-transpose.
    """
    a_layout = traits.a_layout_traits
    b_layout = traits.b_layout_traits
    c_layout = traits.c_layout_traits
    match (a_layout, b_layout, c_layout):
        case (BLASMatrixTraits(is_conjugate=True, is_transpose=False), _, _):
            # If A is conjugate non-transpose that is unsupported.
            return False
        case (_, BLASMatrixTraits(is_conjugate=True, is_transpose=False), _):
            # If B is conjugate non-transpose that is unsupported.
            return False
        case (
            BLASMatrixTraits(order=cublaslt.Order.COL),
            BLASMatrixTraits(order=cublaslt.Order.COL),
            BLASMatrixTraits(is_conjugate=False, is_transpose=False, order=cublaslt.Order.COL),
        ):
            return True
        case _:
            return False


GEMM_LAYOUT_LOOKUP_TABLE: MMLayoutCheckerLookupTable = {
    (Order.COL, Order.COL, Order.COL, False, False, False): [],
    (Order.COL, Order.COL, Order.ROW, False, False, False): ["transpose_and_reorder_C", "swap_AB_and_transpose_ABC"],
    (Order.COL, Order.COL, Order.ROW, False, True, False): ["transpose_and_reorder_C", "swap_AB_and_transpose_ABC"],
    (Order.COL, Order.COL, Order.ROW, True, False, False): ["transpose_and_reorder_C", "swap_AB_and_transpose_ABC"],
    (Order.COL, Order.COL, Order.ROW, True, True, False): ["transpose_and_reorder_C", "swap_AB_and_transpose_ABC"],
    (Order.COL, Order.ROW, Order.COL, False, False, False): ["transpose_and_reorder_B"],
    (Order.COL, Order.ROW, Order.COL, False, True, False): ["transpose_and_reorder_B"],
    (Order.COL, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.COL, Order.ROW, Order.ROW, True, False, False): [
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.COL, Order.COL, False, False, False): ["transpose_and_reorder_A"],
    (Order.ROW, Order.COL, Order.COL, True, False, False): ["transpose_and_reorder_A"],
    (Order.ROW, Order.COL, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.COL, Order.ROW, False, True, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.ROW, Order.COL, False, False, False): ["transpose_and_reorder_A", "transpose_and_reorder_B"],
    (Order.ROW, Order.ROW, Order.COL, False, True, False): ["transpose_and_reorder_A", "transpose_and_reorder_B"],
    (Order.ROW, Order.ROW, Order.COL, True, False, False): ["transpose_and_reorder_A", "transpose_and_reorder_B"],
    (Order.ROW, Order.ROW, Order.COL, True, True, False): ["transpose_and_reorder_A", "transpose_and_reorder_B"],
    (Order.ROW, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
}


def is_supported_symm_layout(traits: BLASMMTraitsView, left_side: bool) -> bool:
    """Return True if the layout is supported for a BLAS symmetric matrix multiplication.

    Rules:
    - All operands must be COL order.
    - The general matrix, C, must be non-conjugate and non-transpose.
    - The general matrix, B, must be non-conjugate and non-transpose.
    - The symmetric matrix, A, must be non-conjugate; transpose is allowed.
    """
    if left_side:
        t_layout = traits.a_layout_traits
        g_layout = traits.b_layout_traits
    else:
        g_layout = traits.a_layout_traits
        t_layout = traits.b_layout_traits
    c_layout = traits.c_layout_traits
    match (t_layout, g_layout, c_layout):
        case (
            BLASMatrixTraits(is_conjugate=False, order=cublaslt.Order.COL),
            BLASMatrixTraits(is_conjugate=False, is_transpose=False, order=cublaslt.Order.COL),
            BLASMatrixTraits(is_conjugate=False, is_transpose=False, order=cublaslt.Order.COL),
        ):
            return True
        case _:
            return False


def is_supported_symm_layout_left(traits: BLASMMTraitsView) -> bool:
    return is_supported_symm_layout(traits, left_side=True)


SYMM_LAYOUT_LOOKUP_TABLE_LEFT: MMLayoutCheckerLookupTable = {
    (Order.COL, Order.COL, Order.COL, False, False, False): [],
    (Order.ROW, Order.COL, Order.COL, False, False, False): ["transpose_and_reorder_A"],
    (Order.ROW, Order.COL, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
}


def is_supported_symm_layout_right(traits: BLASMMTraitsView) -> bool:
    return is_supported_symm_layout(traits, left_side=False)


SYMM_LAYOUT_LOOKUP_TABLE_RIGHT: MMLayoutCheckerLookupTable = {
    (Order.COL, Order.COL, Order.COL, False, False, False): [],
    (Order.COL, Order.ROW, Order.COL, False, False, False): ["transpose_and_reorder_B"],
    (Order.COL, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
}


def is_supported_trmm_layout(traits: BLASMMTraitsView, left_side: bool) -> bool:
    """Return True if the layout is supported for a BLAS triangular matrix multiplication.

    Rules:
    - All operands must be COL order.
    - The general matrix, C, must be non-conjugate and non-transpose.
    - The general matrix, B, must be non-conjugate and non-transpose.
    - The triangular matrix, A, must not be conjugate non-transpose.
    """
    if left_side:
        t_layout = traits.a_layout_traits
        g_layout = traits.b_layout_traits
    else:
        g_layout = traits.a_layout_traits
        t_layout = traits.b_layout_traits
    c_layout = traits.c_layout_traits
    match (t_layout, g_layout, c_layout):
        case (BLASMatrixTraits(is_conjugate=True, is_transpose=False), _, _):
            # If A is conjugate non-transpose that is unsupported.
            return False
        case (
            BLASMatrixTraits(order=cublaslt.Order.COL),
            BLASMatrixTraits(is_conjugate=False, is_transpose=False, order=cublaslt.Order.COL),
            BLASMatrixTraits(is_conjugate=False, is_transpose=False, order=cublaslt.Order.COL),
        ):
            return True
        case _:
            return False


def is_supported_trmm_layout_left(traits: BLASMMTraitsView) -> bool:
    return is_supported_trmm_layout(traits, left_side=True)


TRMM_LAYOUT_LOOKUP_TABLE_LEFT: MMLayoutCheckerLookupTable = {
    (Order.COL, Order.COL, Order.COL, False, False, False): [],
    (Order.ROW, Order.COL, Order.COL, False, False, False): ["transpose_and_reorder_A"],
    (Order.ROW, Order.COL, Order.COL, True, False, False): ["transpose_and_reorder_A"],
    (Order.ROW, Order.COL, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.COL, Order.ROW, False, True, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
}


def is_supported_trmm_layout_right(traits: BLASMMTraitsView) -> bool:
    return is_supported_trmm_layout(traits, left_side=False)


TRMM_LAYOUT_LOOKUP_TABLE_RIGHT: MMLayoutCheckerLookupTable = {
    (Order.COL, Order.COL, Order.COL, False, False, False): [],
    (Order.COL, Order.ROW, Order.COL, False, False, False): ["transpose_and_reorder_B"],
    (Order.COL, Order.ROW, Order.COL, False, True, False): ["transpose_and_reorder_B"],
    (Order.COL, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.COL, Order.ROW, Order.ROW, True, False, False): [
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
    (Order.ROW, Order.ROW, Order.ROW, False, False, False): [
        "transpose_and_reorder_A",
        "transpose_and_reorder_B",
        "transpose_and_reorder_C",
        "swap_AB_and_transpose_ABC",
    ],
}

CACHED_LAYOUT_CHECKERS: dict[MMLayoutChecker, MMLayoutCheckerLookupTable] = {
    is_supported_gemm_layout: GEMM_LAYOUT_LOOKUP_TABLE,
    is_supported_symm_layout_left: SYMM_LAYOUT_LOOKUP_TABLE_LEFT,
    is_supported_symm_layout_right: SYMM_LAYOUT_LOOKUP_TABLE_RIGHT,
    is_supported_trmm_layout_left: TRMM_LAYOUT_LOOKUP_TABLE_LEFT,
    is_supported_trmm_layout_right: TRMM_LAYOUT_LOOKUP_TABLE_RIGHT,
}
