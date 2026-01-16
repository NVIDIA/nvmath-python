# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
"""Concept exploration for NumPy custom-dtype-backed MatrixQualifiers."""

import numpy as np
import pytest

import nvmath.bindings.cublas as cublas

from nvmath.linalg.generic import (
    DiagonalMatrixQualifier,
    GeneralMatrixQualifier,
    HermitianMatrixQualifier,
    matrix_qualifiers_dtype,
    SymmetricMatrixQualifier,
    TriangularMatrixQualifier,
)


@pytest.mark.parametrize(
    "constructor",
    [
        DiagonalMatrixQualifier,
        GeneralMatrixQualifier,
        HermitianMatrixQualifier,
        SymmetricMatrixQualifier,
        TriangularMatrixQualifier,
    ],
)
def test_matrix_qualifier_constructor(constructor):
    q = constructor.create()
    assert isinstance(q, np.ndarray)
    assert q.size == 1
    assert not q.shape
    assert q.dtype == matrix_qualifiers_dtype
    print(q)


def test_matrix_qualifier_set_ranges():
    q = np.empty(10, dtype=matrix_qualifiers_dtype)
    print(q)
    q[...] = GeneralMatrixQualifier.create(conjugate=True)
    print(q)
    q[6] = HermitianMatrixQualifier.create(conjugate=False)
    print(q)

    print(q["abbreviation"])

    print(q.dtype)


def test_matrix_qualifier_validity():
    g = GeneralMatrixQualifier.create()
    assert GeneralMatrixQualifier.is_valid(g)

    h = HermitianMatrixQualifier.create()
    assert HermitianMatrixQualifier.is_valid(h)

    assert not HermitianMatrixQualifier.is_valid(g)
    assert not GeneralMatrixQualifier.is_valid(h)

    g["abbreviation"] = "xx"
    assert not GeneralMatrixQualifier.is_valid(g)

    h["uplo"] = cublas.FillMode.UPPER
    assert HermitianMatrixQualifier.is_valid(h)
    h["uplo"] = -1
    assert not HermitianMatrixQualifier.is_valid(h)


def test_matrix_qualifier_attributes():
    t = TriangularMatrixQualifier.create(
        conjugate=False,
        uplo=cublas.FillMode.UPPER,
        diag=cublas.DiagType.UNIT,
    )
    assert not t["conjugate"]
    assert t["uplo"] == cublas.FillMode.UPPER
    assert t["diag"] == cublas.DiagType.UNIT
