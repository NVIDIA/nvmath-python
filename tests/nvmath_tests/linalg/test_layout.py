import pytest

from nvmath.bindings import cublasLt as cublaslt
from nvmath.linalg._internal.layout import BLASMatrixTraits

COL = cublaslt.Order.COL
ROW = cublaslt.Order.ROW


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
    ((1, 1), (1, 0), COL, 1),
    ((1, 4), (1, 0), COL, None),
    ((4, 1), (1, 0), ROW, None),
    ((4, 4), (1, 0), ROW, None),
    # 2D - dense, dense
    ((1, 1), (1, 1), COL, 1),
    ((1, 4), (1, 1), COL, 1),
    ((4, 1), (1, 4), ROW, None),
    ((4, 4), (1, 4), COL, 4),
    ((1, 4), (4, 1), COL, 1),
    ((4, 1), (1, 1), ROW, None),
    ((4, 4), (4, 1), ROW, None),
    # 2D - dense, strided
    ((1, 1), (1, 8), COL, 1),
    ((1, 4), (1, 8), COL, 8),
    ((4, 1), (1, 8), ROW, None),
    ((4, 4), (1, 8), COL, 8),
    #
    # 2D - strided, overlapping
    ((1, 1), (8, 0), COL, 1),
    ((1, 4), (8, 0), COL, None),
    ((4, 1), (8, 0), ROW, None),
    ((4, 4), (8, 0), ROW, None),
    # 2D - strided, dense
    ((1, 1), (8, 1), COL, 1),
    ((1, 4), (8, 1), COL, 1),
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
