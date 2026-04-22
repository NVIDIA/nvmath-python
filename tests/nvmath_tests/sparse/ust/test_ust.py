# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import numpy as np
import pytest
import scipy.sparse as sps

from nvmath.sparse.ust import (
    Dimension,
    LevelExpr,
    LevelFormat,
    LevelProperty,
    NamedFormats,
    Tensor,
    TensorFormat,
    is_ordered,
    is_unique,
)
from nvmath.sparse.ust._converters import TensorDecomposer


def _round_trip(a):
    ust = Tensor.from_package(a)
    return ust.to_package()


def _assert_same_numpy(a, b):
    assert a.ctypes.data == b.ctypes.data


def _assert_same_cupy(a, b):
    assert a.data.ptr == b.data.ptr


def _assert_same_torch(a, b):
    assert a.data_ptr() == b.data_ptr()


def _assert_lexsort3(a, b, c, x, y, z):
    return a < x or (a == x and (b < y or (b == y and c <= z)))


def test_format_properties():
    coo = NamedFormats.COO
    assert coo.name == "COO"
    assert coo.num_dimensions == 2
    assert coo.num_levels == 2
    assert coo.is_identity
    assert coo.is_ordered
    assert coo.is_unique  # sic!
    csr = NamedFormats.CSR
    assert csr.name == "CSR"
    assert csr.num_dimensions == 2
    assert csr.num_levels == 2
    assert csr.is_identity
    assert csr.is_ordered
    assert csr.is_unique
    csc = NamedFormats.CSC
    assert csc.name == "CSC"
    assert csc.num_dimensions == 2
    assert csc.num_levels == 2
    assert not csc.is_identity
    assert csc.is_ordered
    assert csc.is_unique
    bsr = NamedFormats.BSRRight((4, 8))
    assert bsr.name == "BSRRight4x8"
    assert bsr.num_dimensions == 2
    assert bsr.num_levels == 4
    assert not bsr.is_identity
    assert bsr.is_ordered
    assert bsr.is_unique


def test_format_string():
    assert (
        str(NamedFormats.COO)
        == "[i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)"
    )
    assert str(NamedFormats.COOd(5)) == (
        "[i, j, k, l, m] -> ("
        "i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), "
        "j: <LevelFormat.SINGLETON>, "
        "k: <LevelFormat.SINGLETON>, "
        "l: <LevelFormat.SINGLETON>, "
        "m: <LevelFormat.SINGLETON>)"
    )


def test_bad_formats():
    i = Dimension(dimension_name="i")
    j = Dimension(dimension_name="j")
    k = Dimension(dimension_name="k")

    with pytest.raises(TypeError) as e:
        TensorFormat([i, i], {i: LevelFormat.DENSE})
    assert str(e.value) == "Dimension specifications [i, i] has repeated identifiers."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i: (LevelFormat.DENSE, (LevelProperty.NONUNIQUE, LevelProperty.UNIQUE))})
    assert str(e.value) == "Invalid level property combination: [<LevelProperty.NONUNIQUE>, <LevelProperty.UNIQUE>]."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {j: LevelFormat.DENSE})
    assert str(e.value) == "Dimension j does not appear in [i]."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i + 1: LevelFormat.DENSE})
    assert str(e.value) == "RHS 1 in (i + 1) does not appear in [i]."

    with pytest.raises(TypeError) as e:
        TensorFormat([i, j], {i // j: LevelFormat.DENSE})
    assert str(e.value) == "RHS j in (i // j) must be an integer."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i // 0: LevelFormat.DENSE})
    assert str(e.value) == "RHS 0 in (i // 0) must be strictly positive integer."

    with pytest.raises(TypeError) as e:
        TensorFormat([i, j, k], {i: LevelFormat.DENSE, j: LevelFormat.COMPRESSED})
    assert str(e.value) == "The following dimensions are not used: {k}."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i % 4: LevelFormat.DENSE})
    assert str(e.value) == "Modulo (i % 4) does not match any prior division."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i // 4: LevelFormat.DENSE, i % 8: LevelFormat.DENSE})
    assert str(e.value) == "Modulo (i % 8) does not match any prior division."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i // 4: LevelFormat.DENSE, i // 8: LevelFormat.DENSE})
    assert str(e.value) == "Division (i // 8) reuses dimension of a prior division."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i // 4: LevelFormat.DENSE})
    assert str(e.value) == "Some division dimensions are not matched by modulo: [i]."

    with pytest.raises(TypeError) as e:
        TensorFormat([i, j], {i + j: LevelFormat.DENSE, j - i: LevelFormat.DENSE})
    assert str(e.value) == "Operation (j - i) reuses dimension of a prior range computation."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i // 4: LevelFormat.RANGE})
    assert str(e.value) == "Range uses compound level expression (i // 4)."

    with pytest.raises(TypeError) as e:
        TensorFormat([i], {i: LevelFormat.RANGE})
    assert str(e.value) == "Range uses dimension i that is not uniquely defined."

    with pytest.raises(TypeError) as e:
        TensorFormat([i, j], {i + j: LevelFormat.DENSE})
    assert str(e.value) == "Some add/sub dimensions are not matched by range: [i, j]."


def test_properties():
    assert is_ordered(None)
    assert is_ordered(LevelProperty.ORDERED)
    assert not is_ordered(LevelProperty.UNORDERED)
    assert is_ordered(LevelProperty.UNIQUE)
    assert is_ordered(LevelProperty.NONUNIQUE)
    assert not is_ordered((LevelProperty.UNIQUE, LevelProperty.UNORDERED))
    assert is_ordered((LevelProperty.UNIQUE, LevelProperty.ORDERED))
    assert is_unique(None)
    assert is_unique(LevelProperty.ORDERED)
    assert is_unique(LevelProperty.UNORDERED)
    assert is_unique(LevelProperty.UNIQUE)
    assert not is_unique(LevelProperty.NONUNIQUE)
    assert not is_unique((LevelProperty.ORDERED, LevelProperty.NONUNIQUE))
    assert is_unique((LevelProperty.ORDERED, LevelProperty.UNIQUE))


def test_dim2lvl_2d():
    assert NamedFormats.COO.dim2lvl([2, 1]) == [2, 1]
    assert NamedFormats.CSR.dim2lvl([2, 1]) == [2, 1]
    assert NamedFormats.CSC.dim2lvl([2, 1]) == [1, 2]
    assert NamedFormats.DCSR.dim2lvl([2, 1]) == [2, 1]
    assert NamedFormats.DCSC.dim2lvl([2, 1]) == [1, 2]
    assert NamedFormats.CROW.dim2lvl([2, 1]) == [2, 1]
    assert NamedFormats.CCOL.dim2lvl([2, 1]) == [1, 2]
    assert NamedFormats.DIAI.dim2lvl([2, 1]) == [-1, 2]
    assert NamedFormats.DIAJ.dim2lvl([2, 1]) == [-1, 1]
    assert NamedFormats.SkewDIAI.dim2lvl([2, 1]) == [3, 2]
    assert NamedFormats.SkewDIAJ.dim2lvl([2, 1]) == [3, 1]
    assert NamedFormats.BSRRight((2, 2)).dim2lvl([2, 1]) == [1, 0, 0, 1]
    assert NamedFormats.BSRLeft((2, 2)).dim2lvl([2, 1]) == [1, 0, 1, 0]
    assert NamedFormats.BSCRight((2, 2)).dim2lvl([2, 1]) == [0, 1, 0, 1]
    assert NamedFormats.BSCLeft((2, 2)).dim2lvl([2, 1]) == [0, 1, 1, 0]
    # As size
    assert NamedFormats.COO.dim2lvl([8, 4], True) == [8, 4]
    assert NamedFormats.CSR.dim2lvl([8, 4], True) == [8, 4]
    assert NamedFormats.CSC.dim2lvl([8, 4], True) == [4, 8]
    assert NamedFormats.DIAI.dim2lvl([8, 4], True) == [11, 8]
    assert NamedFormats.DIAJ.dim2lvl([8, 4], True) == [11, 4]
    assert NamedFormats.BSRRight((2, 2)).dim2lvl([8, 4], True) == [4, 2, 2, 2]
    assert NamedFormats.BSCRight((2, 2)).dim2lvl([8, 4], True) == [2, 4, 2, 2]


def test_lvl2dim_2d():
    assert NamedFormats.COO.lvl2dim([2, 1]) == [2, 1]
    assert NamedFormats.CSR.lvl2dim([2, 1]) == [2, 1]
    assert NamedFormats.CSC.lvl2dim([1, 2]) == [2, 1]
    assert NamedFormats.DCSR.lvl2dim([2, 1]) == [2, 1]
    assert NamedFormats.DCSC.lvl2dim([1, 2]) == [2, 1]
    assert NamedFormats.CROW.lvl2dim([2, 1]) == [2, 1]
    assert NamedFormats.CCOL.lvl2dim([1, 2]) == [2, 1]
    assert NamedFormats.DIAI.lvl2dim([-1, 2]) == [2, 1]
    assert NamedFormats.DIAJ.lvl2dim([-1, 1]) == [2, 1]
    assert NamedFormats.SkewDIAI.lvl2dim([3, 2]) == [2, 1]
    assert NamedFormats.SkewDIAJ.lvl2dim([3, 1]) == [2, 1]
    assert NamedFormats.BSRRight((2, 2)).lvl2dim([1, 0, 0, 1]) == [2, 1]
    assert NamedFormats.BSRLeft((2, 2)).lvl2dim([1, 0, 1, 0]) == [2, 1]
    assert NamedFormats.BSCRight((2, 2)).lvl2dim([0, 1, 0, 1]) == [2, 1]
    assert NamedFormats.BSCLeft((2, 2)).lvl2dim([0, 1, 1, 0]) == [2, 1]


def test_dim2lvl2dim_2d():
    for i in range(17):
        for j in range(111):
            crd = [i, j]
            assert NamedFormats.CSR.lvl2dim(NamedFormats.CSR.dim2lvl(crd)) == crd
            assert NamedFormats.CSC.lvl2dim(NamedFormats.CSC.dim2lvl(crd)) == crd
            assert NamedFormats.DIAI.lvl2dim(NamedFormats.DIAI.dim2lvl(crd)) == crd
            assert NamedFormats.DIAJ.lvl2dim(NamedFormats.DIAJ.dim2lvl(crd)) == crd
            assert NamedFormats.SkewDIAI.lvl2dim(NamedFormats.SkewDIAI.dim2lvl(crd)) == crd
            assert NamedFormats.SkewDIAJ.lvl2dim(NamedFormats.SkewDIAJ.dim2lvl(crd)) == crd
            assert NamedFormats.DELTA(2).lvl2dim(NamedFormats.DELTA(2).dim2lvl(crd)) == crd


def test_dim2lvl2dim_3d():
    for i in range(17):
        for j in range(11):
            for k in range(21):
                crd = [i, j, k]
                assert NamedFormats.COOd(3).lvl2dim(NamedFormats.COOd(3).dim2lvl(crd)) == crd
                assert NamedFormats.CSFd(3).lvl2dim(NamedFormats.CSFd(3).dim2lvl(crd)) == crd


def test_tensor_properties():
    tensor = Tensor((8, 4), tensor_format=NamedFormats.CSR, index_type=np.int32, dtype=np.float32)
    assert tensor.num_dimensions == 2
    assert tensor.extents == [8, 4]
    assert tensor.tensor_format == NamedFormats.CSR
    assert tensor.index_type == np.int32
    assert tensor.dtype == np.float32


def test_bad_tensors():
    with pytest.raises(TypeError) as e:
        Tensor((8, 4, 2))
    assert str(e.value) == "The tensor format must be specified."

    with pytest.raises(TypeError) as e:
        Tensor((8, 4, 2), tensor_format="COO")
    assert str(e.value) == "The tensor format COO must be a TensorFormat object."

    with pytest.raises(TypeError) as e:
        Tensor((8, 4, 2), tensor_format=NamedFormats.COO)
    assert str(e.value) == "The tensor rank 3 must equal the format dimensionality 2."

    with pytest.raises(TypeError) as e:
        Tensor((4, 0), tensor_format=NamedFormats.COO)
    assert str(e.value) == "The tensor extents (4, 0) do not define any elements."


def test_from_dense_1d():
    a = np.array([1, 2, 3, 4], dtype=np.float64)
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=1,LVL=1>\n"
        "format   : [i] -> (i: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [4]\n"
        "lvl      : [4]\n"
        "nse      : 4\n"
        "values   : [1.0, 2.0, 3.0, 4.0] #4\n"
        "data     : 32 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([1]) == 2.0


def test_from_dense_1d_cupy():
    cp = pytest.importorskip("cupy")
    a = cp.array([4, 3, 2, 1], dtype=np.float64)
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=1,LVL=1>\n"
        "format   : [i] -> (i: <LevelFormat.DENSE>)\n"
        "device   : cuda\n"
        "dim      : [4]\n"
        "lvl      : [4]\n"
        "nse      : 4\n"
        "values   : [4.0, 3.0, 2.0, 1.0] #4\n"
        "data     : 32 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([1]) == 3.0


def test_from_dense_1d_torch():
    torch = pytest.importorskip("torch")
    a = torch.tensor([10, 20, 30, 40], dtype=torch.float64)
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=1,LVL=1>\n"
        "format   : [i] -> (i: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [4]\n"
        "lvl      : [4]\n"
        "nse      : 4\n"
        "values   : [10.0, 20.0, 30.0, 40.0] #4\n"
        "data     : 32 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([1]) == 20.0
    a = a.cuda()
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=1,LVL=1>\n"
        "format   : [i] -> (i: <LevelFormat.DENSE>)\n"
        "device   : cuda\n"
        "dim      : [4]\n"
        "lvl      : [4]\n"
        "nse      : 4\n"
        "values   : [10.0, 20.0, 30.0, 40.0] #4\n"
        "data     : 32 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([1]) == 20.0


@pytest.mark.parametrize("package,device", [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cpu")])
def test_from_dense_2d(package, device):
    module = pytest.importorskip(package)
    a = module.arange(6, dtype=module.int32).reshape(2, 3)
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=int32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>)\n"
        f"device   : {device}\n"
        "dim      : [2, 3]\n"
        "lvl      : [2, 3]\n"
        "nse      : 6\n"
        "values   : [0, 1, 2, 3, 4, 5] #6\n"
        "data     : 24 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([1, 2]) == 5
    b = ust.to_package()
    assert (a == b).all(), "Error: the recreated package tensor doesn't match the original."

    a = a.transpose(1, 0)
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=int32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)\n"
        f"device   : {device}\n"
        "dim      : [3, 2]\n"
        "lvl      : [2, 3]\n"
        "nse      : 6\n"
        "values   : [0, 1, 2, 3, 4, 5] #6\n"
        "data     : 24 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([2, 1]) == 5
    b = ust.to_package()
    assert (a == b).all(), "Error: the recreated package tensor doesn't match the original."

    a = module.arange(6.0, dtype=module.float32).reshape(2, 3) + 1j
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=complex64,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>)\n"
        f"device   : {device}\n"
        "dim      : [2, 3]\n"
        "lvl      : [2, 3]\n"
        "nse      : 6\n"
        "values   : [1j, (1+1j), (2+1j), (3+1j), (4+1j), (5+1j)] #6\n"
        "data     : 48 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([1, 2]) == 5 + 1j
    b = ust.to_package()
    assert (a == b).all(), "Error: the recreated package tensor doesn't match the original."

    a = a.transpose(1, 0)
    ust = Tensor.from_package(a)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=complex64,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)\n"
        f"device   : {device}\n"
        "dim      : [3, 2]\n"
        "lvl      : [2, 3]\n"
        "nse      : 6\n"
        "values   : [1j, (1+1j), (2+1j), (3+1j), (4+1j), (5+1j)] #6\n"
        "data     : 48 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert ust.get_value([2, 1]) == 5 + 1j
    b = ust.to_package()
    assert (a == b).all(), "Error: the recreated package tensor doesn't match the original."


@pytest.mark.parametrize("package,device", [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cpu")])
def test_from_dense_3d(package, device):
    module = pytest.importorskip(package)
    a = module.arange(24, dtype=module.int64).reshape(2, 3, 4)

    axes = (0, 1, 2)
    b = np.transpose(a, axes=axes)
    ust = Tensor.from_package(b)
    ref = f"""\
---- Sparse Tensor<VAL=int64,POS=int32,CRD=int32,DIM=3,LVL=3>
format   : [i, j, k] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)
device   : {device}
dim      : [2, 3, 4]
lvl      : [2, 3, 4]
nse      : 24
values   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] #24
data     : 192 bytes
sparsity : 0.00%
----"""
    assert str(ust) == ref, "Error: incorrect UST conversion."
    assert ust.get_value([1, 2, 3]) == 23
    c = ust.to_package()
    assert (c == b).all(), "Error: the recreated package tensor doesn't match the original."

    axes = (0, 2, 1)
    b = np.transpose(a, axes=axes)
    ust = Tensor.from_package(b)
    ref = f"""\
---- Sparse Tensor<VAL=int64,POS=int32,CRD=int32,DIM=3,LVL=3>
format   : [i, j, k] -> (i: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>)
device   : {device}
dim      : [2, 4, 3]
lvl      : [2, 3, 4]
nse      : 24
values   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] #24
data     : 192 bytes
sparsity : 0.00%
----"""
    assert str(ust) == ref, "Error: incorrect UST conversion."
    assert ust.get_value([1, 3, 2]) == 23
    c = ust.to_package()
    assert (c == b).all(), "Error: the recreated package tensor doesn't match the original."

    axes = (1, 0, 2)
    b = np.transpose(a, axes=axes)
    ust = Tensor.from_package(b)
    ref = f"""\
---- Sparse Tensor<VAL=int64,POS=int32,CRD=int32,DIM=3,LVL=3>
format   : [i, j, k] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)
device   : {device}
dim      : [3, 2, 4]
lvl      : [2, 3, 4]
nse      : 24
values   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] #24
data     : 192 bytes
sparsity : 0.00%
----"""
    assert str(ust) == ref, "Error: incorrect UST conversion."
    assert ust.get_value([2, 1, 3]) == 23
    c = ust.to_package()
    assert (c == b).all(), "Error: the recreated package tensor doesn't match the original."

    axes = (1, 2, 0)
    b = np.transpose(a, axes=axes)
    ust = Tensor.from_package(b)
    ref = f"""\
---- Sparse Tensor<VAL=int64,POS=int32,CRD=int32,DIM=3,LVL=3>
format   : [i, j, k] -> (k: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>)
device   : {device}
dim      : [3, 4, 2]
lvl      : [2, 3, 4]
nse      : 24
values   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] #24
data     : 192 bytes
sparsity : 0.00%
----"""
    assert str(ust) == ref, "Error: incorrect UST conversion."
    assert ust.get_value([2, 3, 1]) == 23
    c = ust.to_package()
    assert (c == b).all(), "Error: the recreated package tensor doesn't match the original."

    axes = (2, 0, 1)
    b = np.transpose(a, axes=axes)
    ust = Tensor.from_package(b)
    ref = f"""\
---- Sparse Tensor<VAL=int64,POS=int32,CRD=int32,DIM=3,LVL=3>
format   : [i, j, k] -> (j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)
device   : {device}
dim      : [4, 2, 3]
lvl      : [2, 3, 4]
nse      : 24
values   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] #24
data     : 192 bytes
sparsity : 0.00%
----"""
    assert str(ust) == ref, "Error: incorrect UST conversion."
    assert ust.get_value([3, 1, 2]) == 23
    c = ust.to_package()
    assert (c == b).all(), "Error: the recreated package tensor doesn't match the original."

    axes = (2, 1, 0)
    b = np.transpose(a, axes=axes)
    ust = Tensor.from_package(b)
    ref = f"""\
---- Sparse Tensor<VAL=int64,POS=int32,CRD=int32,DIM=3,LVL=3>
format   : [i, j, k] -> (k: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)
device   : {device}
dim      : [4, 3, 2]
lvl      : [2, 3, 4]
nse      : 24
values   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] #24
data     : 192 bytes
sparsity : 0.00%
----"""
    assert str(ust) == ref, "Error: incorrect UST conversion."
    assert ust.get_value([3, 2, 1]) == 23
    c = ust.to_package()
    assert (c == b).all(), "Error: the recreated package tensor doesn't match the original."


def test_from_scipy_basic_2d():
    row = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
    col = np.array([0, 1, 1, 3, 2, 3], dtype=np.int32)
    val = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    coo = sps.coo_array((val, (row, col)), shape=(4, 8))
    coo.sum_duplicates()
    ust = Tensor.from_package(coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cpu\n"
        "dim      : [4, 8]\n"
        "lvl      : [4, 8]\n"
        "nse      : 6\n"
        "pos[0]   : [0, 6] #2\n"
        "crd[0]   : [0, 0, 1, 1, 2, 3] #6\n"
        "crd[1]   : [0, 1, 1, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 80 bytes\n"
        "sparsity : 81.25%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) == 2.0
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([3, 3]) == 6.0
    csr = sps.csr_array(coo)
    ust = Tensor.from_package(csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 8]\n"
        "lvl      : [4, 8]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 2, 4, 5, 6] #5\n"
        "crd[1]   : [0, 1, 1, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 68 bytes\n"
        "sparsity : 81.25%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) == 2.0
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([3, 3]) == 6.0
    csc = sps.csc_array(coo)
    ust = Tensor.from_package(csc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 8]\n"
        "lvl      : [8, 4]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 1, 3, 4, 6, 6, 6, 6, 6] #9\n"
        "crd[1]   : [0, 0, 1, 2, 1, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 5.0, 4.0, 6.0] #6\n"
        "data     : 84 bytes\n"
        "sparsity : 81.25%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) == 2.0
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([3, 3]) == 6.0
    bsr = sps.bsr_array(coo, blocksize=(2, 2))
    ust = Tensor.from_package(bsr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=4>\n"
        "format   : [i, j] -> ((i // 2): <LevelFormat.DENSE>, (j // 2): <LevelFormat.COMPRESSED>,"
        " (i % 2): <LevelFormat.DENSE>, (j % 2): <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [4, 8]\n"
        "lvl      : [2, 4, 2, 2]\n"
        "nse      : 12\n"
        "pos[1]   : [0, 2, 3] #3\n"
        "crd[1]   : [0, 1, 1] #3\n"
        "values   : [1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0] #12\n"
        "data     : 72 bytes\n"
        "sparsity : 62.50%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) == 2.0
    assert ust.get_value([0, 2]) == 0.0
    assert ust.get_value([3, 3]) == 6.0
    a = np.array([[1, 0, 0, 7], [1, 2, 0, 0], [0, -2, 3, 0], [0, 0, -3, 4]], dtype=np.float32)
    dia = sps.dia_array(a)
    ust = Tensor.from_package(dia)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, j: <LevelFormat.RANGE>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [7, 4]\n"
        "nse      : 12\n"
        "pos[0]   : [0, 3] #2\n"
        "crd[0]   : [-1, 0, 3] #3\n"
        "values   : [1.0, -2.0, -3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 7.0] #12\n"
        "data     : 68 bytes\n"
        "sparsity : 25.00%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) is None
    assert ust.get_value([0, 3]) == 7.0
    assert ust.get_value([3, 2]) == -3.0


def test_from_scipy_rectangular_diaj():
    D = np.array([[1, 0, 0, 0, 7], [0, 2, 0, 0, 0], [5, 0, 3, 0, 0], [0, 6, 0, 4, 0]], dtype=np.float64)
    Adia = sps.dia_matrix(D)
    ust = Tensor.from_package(Adia)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, j: <LevelFormat.RANGE>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [8, 5]\n"
        "nse      : 15\n"
        "pos[0]   : [0, 3] #2\n"
        "crd[0]   : [-2, 0, 4] #3\n"
        "values   : [5.0, 6.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0] #15\n"
        "data     : 140 bytes\n"
        "sparsity : 25.00%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) is None
    assert ust.get_value([0, 4]) == 7.0
    assert ust.get_value([1, 1]) == 2.0
    assert ust.get_value([2, 0]) == 5.0
    assert ust.get_value([2, 1]) is None
    assert ust.get_value([2, 2]) == 3.0
    assert ust.get_value([3, 1]) == 6.0
    assert ust.get_value([3, 3]) == 4.0

    stream = StringIO()

    def visit(idx, val):
        print(idx, val, file=stream)

    TensorDecomposer(ust, visit).run()
    assert stream.getvalue() == ("[2, 0] 5.0\n[3, 1] 6.0\n[0, 0] 1.0\n[1, 1] 2.0\n[2, 2] 3.0\n[3, 3] 4.0\n[0, 4] 7.0\n")


def test_from_scipy_sum_dup_sorted_indices():
    row = np.array([3, 2, 1, 0, 0], dtype=np.int32)
    col = np.array([3, 2, 1, 0, 0], dtype=np.int32)
    val = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    coo = sps.coo_array((val, (row, col)), shape=(4, 4))
    coo.sum_duplicates()
    ust = Tensor.from_package(coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 4\n"
        "pos[0]   : [0, 4] #2\n"
        "crd[0]   : [0, 1, 2, 3] #4\n"
        "crd[1]   : [0, 1, 2, 3] #4\n"
        "values   : [9.0, 3.0, 2.0, 1.0] #4\n"
        "data     : 56 bytes\n"
        "sparsity : 75.00%\n"
        "----"
    )
    csr = sps.csr_array(
        (np.array([1, 2, 3], dtype=np.float32), np.array([1, 0, 0], dtype=np.int32), np.array([0, 3, 3, 3, 3], dtype=np.int32)),
        shape=(4, 4),
    )
    csr.sum_duplicates()
    ust = Tensor.from_package(csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 2\n"
        "pos[1]   : [0, 2, 2, 2, 2] #5\n"
        "crd[1]   : [0, 1] #2\n"
        "values   : [5.0, 1.0] #2\n"
        "data     : 36 bytes\n"
        "sparsity : 87.50%\n"
        "----"
    )


def test_from_cupy_basic_2d():
    cp = pytest.importorskip("cupy")
    cps = pytest.importorskip("cupyx.scipy.sparse")
    row = cp.array([0, 0, 1, 1, 2, 3], dtype=np.int32)
    col = cp.array([0, 1, 1, 3, 2, 3], dtype=np.int32)
    val = cp.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    coo = cps.coo_matrix((val, (row, col)), shape=(4, 8))
    coo.sum_duplicates()
    ust = Tensor.from_package(coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cuda\n"
        "dim      : [4, 8]\n"
        "lvl      : [4, 8]\n"
        "nse      : 6\n"
        "pos[0]   : [0, 6] #2\n"
        "crd[0]   : [0, 0, 1, 1, 2, 3] #6\n"
        "crd[1]   : [0, 1, 1, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 80 bytes\n"
        "sparsity : 81.25%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 1]) == 2.0
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([3, 3]) == 6.0
    csr = cps.csr_matrix(coo)
    ust = Tensor.from_package(csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cuda\n"
        "dim      : [4, 8]\n"
        "lvl      : [4, 8]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 2, 4, 5, 6] #5\n"
        "crd[1]   : [0, 1, 1, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 68 bytes\n"
        "sparsity : 81.25%\n"
        "----"
    )
    csc = cps.csc_matrix(coo)
    ust = Tensor.from_package(csc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.COMPRESSED>)\n"
        "device   : cuda\n"
        "dim      : [4, 8]\n"
        "lvl      : [8, 4]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 1, 3, 4, 6, 6, 6, 6, 6] #9\n"
        "crd[1]   : [0, 0, 1, 2, 1, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 5.0, 4.0, 6.0] #6\n"
        "data     : 84 bytes\n"
        "sparsity : 81.25%\n"
        "----"
    )
    data = cp.array(
        [
            [-1, -2, -3, 0],
            [1, 2, 3, 4],
            [0, 0, 0, 7],
        ],
        dtype=np.float64,
    )
    offsets = cp.array([-1, 0, 3], dtype=np.int32)
    dia = cps.dia_matrix((data, offsets), shape=(4, 4))
    ust = Tensor.from_package(dia)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, j: <LevelFormat.RANGE>)\n"
        "device   : cuda\n"
        "dim      : [4, 4]\n"
        "lvl      : [7, 4]\n"
        "nse      : 12\n"
        "pos[0]   : [0, 3] #2\n"
        "crd[0]   : [-1, 0, 3] #3\n"
        "values   : [-1.0, -2.0, -3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 7.0] #12\n"
        "data     : 116 bytes\n"
        "sparsity : 25.00%\n"
        "----"
    )


def test_from_cupy_sum_dup_sorted_indices():
    cp = pytest.importorskip("cupy")
    cps = pytest.importorskip("cupyx.scipy.sparse")
    row = cp.array([3, 2, 1, 0, 0], dtype=np.int32)
    col = cp.array([3, 2, 1, 0, 0], dtype=np.int32)
    val = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
    coo = cps.coo_matrix((val, (row, col)), shape=(4, 4))
    coo.sum_duplicates()
    ust = Tensor.from_package(coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cuda\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 4\n"
        "pos[0]   : [0, 4] #2\n"
        "crd[0]   : [0, 1, 2, 3] #4\n"
        "crd[1]   : [0, 1, 2, 3] #4\n"
        "values   : [9.0, 3.0, 2.0, 1.0] #4\n"
        "data     : 56 bytes\n"
        "sparsity : 75.00%\n"
        "----"
    )
    csr = cps.csr_matrix(
        (cp.array([1, 2, 3], dtype=np.float32), cp.array([1, 0, 0], dtype=np.int32), cp.array([0, 3, 3, 3, 3], dtype=np.int32)),
        shape=(4, 4),
    )
    csr.sum_duplicates()
    ust = Tensor.from_package(csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cuda\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 2\n"
        "pos[1]   : [0, 2, 2, 2, 2] #5\n"
        "crd[1]   : [0, 1] #2\n"
        "values   : [5.0, 1.0] #2\n"
        "data     : 36 bytes\n"
        "sparsity : 87.50%\n"
        "----"
    )


def test_from_torch_basic_2d():
    torch = pytest.importorskip("torch")
    sparse_coo = torch.sparse_coo_tensor(
        torch.tensor([[0, 1, 2, 2, 3, 3], [0, 1, 2, 3, 2, 3]]), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), size=(4, 4)
    ).coalesce()
    ust = Tensor.from_package(sparse_coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 6\n"
        "pos[0]   : [0, 6] #2\n"
        "crd[0]   : [0, 1, 2, 2, 3, 3] #6\n"
        "crd[1]   : [0, 1, 2, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 136 bytes\n"
        "sparsity : 62.50%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([1, 1]) == 2.0
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([3, 3]) == 6.0
    sparse_csr = sparse_coo.to_sparse_csr()
    ust = Tensor.from_package(sparse_csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 1, 2, 4, 6] #5\n"
        "crd[1]   : [0, 1, 2, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 112 bytes\n"
        "sparsity : 62.50%\n"
        "----"
    )
    sparse_csc = sparse_coo.to_sparse_csc()
    ust = Tensor.from_package(sparse_csc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 1, 2, 4, 6] #5\n"
        "crd[1]   : [0, 1, 2, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 5.0, 4.0, 6.0] #6\n"
        "data     : 112 bytes\n"
        "sparsity : 62.50%\n"
        "----"
    )
    sparse_bsr = sparse_coo.to_sparse_bsr((2, 2))
    ust = Tensor.from_package(sparse_bsr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=2,LVL=4>\n"
        "format   : [i, j] -> ((i // 2): <LevelFormat.DENSE>, (j // 2): <LevelFormat.COMPRESSED>,"
        " (i % 2): <LevelFormat.DENSE>, (j % 2): <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [2, 2, 2, 2]\n"
        "nse      : 8\n"
        "pos[1]   : [0, 1, 2] #3\n"
        "crd[1]   : [0, 1] #2\n"
        "values   : [1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0] #8\n"
        "data     : 72 bytes\n"
        "sparsity : 50.00%\n"
        "----"
    )
    sparse_bsc = sparse_coo.to_sparse_bsc((2, 2))
    ust = Tensor.from_package(sparse_bsc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=2,LVL=4>\n"
        "format   : [i, j] -> ((j // 2): <LevelFormat.DENSE>, (i // 2): <LevelFormat.COMPRESSED>,"
        " (i % 2): <LevelFormat.DENSE>, (j % 2): <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [4, 4]\n"
        "lvl      : [2, 2, 2, 2]\n"
        "nse      : 8\n"
        "pos[1]   : [0, 1, 2] #3\n"
        "crd[1]   : [0, 1] #2\n"
        "values   : [1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0] #8\n"
        "data     : 72 bytes\n"
        "sparsity : 50.00%\n"
        "----"
    )


def test_from_torch_basic_2d_cuda():
    torch = pytest.importorskip("torch")
    sparse_coo = (
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 2, 2, 3, 3], [0, 1, 2, 3, 2, 3]]), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), size=(4, 4)
        )
        .coalesce()
        .cuda()
    )
    ust = Tensor.from_package(sparse_coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cuda\n"
        "dim      : [4, 4]\n"
        "lvl      : [4, 4]\n"
        "nse      : 6\n"
        "pos[0]   : [0, 6] #2\n"
        "crd[0]   : [0, 1, 2, 2, 3, 3] #6\n"
        "crd[1]   : [0, 1, 2, 3, 2, 3] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 136 bytes\n"
        "sparsity : 62.50%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([1, 1]) == 2.0
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([3, 3]) == 6.0

    stream = StringIO()

    def visit(idx, val):
        print(idx, val, file=stream)

    TensorDecomposer(ust, visit).run()
    assert stream.getvalue() == ("[0, 0] 1.0\n[1, 1] 2.0\n[2, 2] 3.0\n[2, 3] 4.0\n[3, 2] 5.0\n[3, 3] 6.0\n")


# Hybrid COO: sparse_dim + dense_dim == ndim, dense dimensions follow sparse dimensions.
def test_from_torch_hybrid_coo():
    torch = pytest.importorskip("torch")
    dense = torch.ones(2, 3, 4)
    # Defaults to a regular COO3 format (with innermost dense 'scalars').
    hybrid_coo = dense.to_sparse(sparse_dim=3)  # 3 sparse dim = COO3
    ust = Tensor.from_package(hybrid_coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>),"
        " j: <LevelFormat.SINGLETON>, k: <LevelFormat.SINGLETON>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [0, 24] #2\n"
        "crd[0]   : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] #24\n"
        "crd[1]   : [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2] #24\n"
        "crd[2]   : [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] #24\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #24\n"
        "data     : 688 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Hybrid COO3 format (with innermost dense 'vectors').
    hybrid_coo = dense.to_sparse(sparse_dim=2)  # 2 sparse dim
    ust = Tensor.from_package(hybrid_coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>),"
        " j: <LevelFormat.SINGLETON>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [0, 6] #2\n"
        "crd[0]   : [0, 0, 0, 1, 1, 1] #6\n"
        "crd[1]   : [0, 1, 2, 0, 1, 2] #6\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #24\n"
        "data     : 208 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Hybrid COO3 format (with innermost dense 'matrices').
    hybrid_coo = dense.to_sparse(sparse_dim=1)  # 1 sparse dim
    ust = Tensor.from_package(hybrid_coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.COMPRESSED>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [0, 2] #2\n"
        "crd[0]   : [0, 1] #2\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #24\n"
        "data     : 128 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    hybrid_coo = dense.to_sparse(dense_dim=2)  # same effect
    ust = Tensor.from_package(hybrid_coo)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.COMPRESSED>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [0, 2] #2\n"
        "crd[0]   : [0, 1] #2\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #24\n"
        "data     : 128 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )


# Hybrid CSR: batch_dim + sparse_dim (=2) + dense_dim == ndim
def test_from_torch_batched_hybrid_compressed():
    torch = pytest.importorskip("torch")
    dense_tensor = torch.ones(2, 2, 3)
    # Make it less regular, but same #nnz per batch.
    # This has interesting results in torch formats.
    dense_tensor[0, 0, 0] = 0
    dense_tensor[1, 1, 2] = 0
    # Batched CSR format (with innermost dense 'scalars').
    batched_csr = dense_tensor.to_sparse_csr(dense_dim=0)
    ust = Tensor.from_package(batched_csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.BATCH>, j: <LevelFormat.DENSE>, k: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 3]\n"
        "lvl      : [2, 2, 3]\n"
        "nse      : 10\n"
        "pos[2]   : [0, 2, 5, 0, 3, 5] #6\n"
        "crd[2]   : [1, 2, 0, 1, 2, 0, 1, 2, 0, 1] #10\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #10\n"
        "data     : 168 bytes\n"
        "sparsity : 16.67%\n"
        "----"
    )
    # Hybrid CSR format (with innermost dense 'vectors').
    hybrid_csr = dense_tensor.to_sparse_csr(dense_dim=1)
    ust = Tensor.from_package(hybrid_csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 3]\n"
        "lvl      : [2, 2, 3]\n"
        "nse      : 12\n"
        "pos[1]   : [0, 2, 4] #3\n"
        "crd[1]   : [0, 1, 0, 1] #4\n"
        "values   : [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0] #12\n"
        "data     : 104 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Batched CSC format (with innermost dense 'scalars').
    batched_csc = dense_tensor.to_sparse_csc(dense_dim=0)
    ust = Tensor.from_package(batched_csc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.BATCH>, k: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 3]\n"
        "lvl      : [2, 3, 2]\n"
        "nse      : 10\n"
        "pos[2]   : [0, 1, 3, 5, 0, 2, 4, 5] #8\n"
        "crd[2]   : [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] #10\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #10\n"
        "data     : 184 bytes\n"
        "sparsity : 16.67%\n"
        "----"
    )
    # Hybrid CSC format (with innermost dense 'vectors').
    hybrid_csc = dense_tensor.to_sparse_csc(dense_dim=1)
    ust = Tensor.from_package(hybrid_csc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.COMPRESSED>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 3]\n"
        "lvl      : [2, 2, 3]\n"
        "nse      : 12\n"
        "pos[1]   : [0, 2, 4] #3\n"
        "crd[1]   : [0, 1, 0, 1] #4\n"
        "values   : [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0] #12\n"
        "data     : 104 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )


def test_from_torch_batched_hybrid_compressed_inspection():
    torch = pytest.importorskip("torch")
    dense = (1 + torch.arange(2 * 3 * 4 * 5)).reshape(2, 3, 4, 5)
    batched_csr = dense.to_sparse_csr(dense_dim=0)
    ust = Tensor.from_package(batched_csr)

    stream = StringIO()

    def visit(idx, val):
        print(idx, val, file=stream)

    TensorDecomposer(ust, visit).run()
    result = stream.getvalue()

    assert "[0, 0, 0, 0] 1" in result
    assert "[0, 2, 2, 4] 55" in result
    assert "[1, 1, 3, 3] 99" in result
    assert "[1, 2, 3, 1] 117" in result
    assert "[1, 2, 3, 4] 120" in result

    assert ust.get_value([0, 0, 0, 0]) == 1
    assert ust.get_value([0, 2, 2, 4]) == 55
    assert ust.get_value([1, 1, 3, 3]) == 99
    assert ust.get_value([1, 2, 3, 1]) == 117
    assert ust.get_value([1, 2, 3, 4]) == 120


def test_from_torch_batched_hybrid_blocked():
    torch = pytest.importorskip("torch")
    dense_tensor = torch.ones(2, 2, 2)
    # Batched BSR format.
    batched_bsr = dense_tensor.to_sparse_bsr((2, 2), dense_dim=0)
    ust = Tensor.from_package(batched_bsr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=5>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.BATCH>, (j // 2): <LevelFormat.DENSE>,"
        " (k // 2): <LevelFormat.COMPRESSED>, (j % 2): <LevelFormat.DENSE>, (k % 2): <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 2]\n"
        "lvl      : [2, 1, 1, 2, 2]\n"
        "nse      : 8\n"
        "pos[2]   : [0, 1, 0, 1] #4\n"
        "crd[2]   : [0, 0] #2\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #8\n"
        "data     : 80 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Hybrid BSR format.
    hybrid_bsr = dense_tensor.to_sparse_bsr((2, 2), dense_dim=1)
    ust = Tensor.from_package(hybrid_bsr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=5>\n"
        "format   : [i, j, k] -> ((i // 2): <LevelFormat.DENSE>, (j // 2): <LevelFormat.COMPRESSED>,"
        " (i % 2): <LevelFormat.DENSE>, (j % 2): <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 2]\n"
        "lvl      : [1, 1, 2, 2, 2]\n"
        "nse      : 8\n"
        "pos[1]   : [0, 1] #2\n"
        "crd[1]   : [0] #1\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #8\n"
        "data     : 56 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Batched BSC format.
    batched_bsc = dense_tensor.to_sparse_bsc((2, 2), dense_dim=0)
    ust = Tensor.from_package(batched_bsc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=5>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.BATCH>, (k // 2): <LevelFormat.DENSE>,"
        " (j // 2): <LevelFormat.COMPRESSED>, (j % 2): <LevelFormat.DENSE>, (k % 2): <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 2]\n"
        "lvl      : [2, 1, 1, 2, 2]\n"
        "nse      : 8\n"
        "pos[2]   : [0, 1, 0, 1] #4\n"
        "crd[2]   : [0, 0] #2\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #8\n"
        "data     : 80 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Hybrid BSC format.
    hybrid_bsc = dense_tensor.to_sparse_bsc((2, 2), dense_dim=1)
    ust = Tensor.from_package(hybrid_bsc)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int64,CRD=int64,DIM=3,LVL=5>\n"
        "format   : [i, j, k] -> ((j // 2): <LevelFormat.DENSE>, (i // 2): <LevelFormat.COMPRESSED>,"
        " (i % 2): <LevelFormat.DENSE>, (j % 2): <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 2, 2]\n"
        "lvl      : [1, 1, 2, 2, 2]\n"
        "nse      : 8\n"
        "pos[1]   : [0, 1] #2\n"
        "crd[1]   : [0] #1\n"
        "values   : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #8\n"
        "data     : 56 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )


def test_zero_copy_roundtrip_scipy():
    Adns = np.ones((2, 3, 4), dtype=np.float64)
    Adia = sps.eye(m=10, n=10, dtype=np.float64)
    Acoo = sps.coo_array(Adia)
    Acoo.sum_duplicates()  # no surprises
    Acsr = sps.csr_array(Adia)
    Acsr.sort_indices()  # no surprises
    Acsc = sps.csc_array(Adia)
    Acsc.sort_indices()  # no surprises
    Absr = sps.bsr_array(Adia, blocksize=(5, 5))

    A = _round_trip(Adns)
    _assert_same_numpy(A, Adns)

    A = _round_trip(Adia)
    _assert_same_numpy(A.offsets, Adia.offsets)
    _assert_same_numpy(A.data, Adia.data)

    A = _round_trip(Acoo)
    _assert_same_numpy(A.row, Acoo.row)
    _assert_same_numpy(A.col, Acoo.col)
    _assert_same_numpy(A.data, Acoo.data)

    A = _round_trip(Acsr)
    _assert_same_numpy(A.indptr, Acsr.indptr)
    _assert_same_numpy(A.indices, Acsr.indices)
    _assert_same_numpy(A.data, Acsr.data)

    A = _round_trip(Acsc)
    _assert_same_numpy(A.indptr, Acsc.indptr)
    _assert_same_numpy(A.indices, Acsc.indices)
    _assert_same_numpy(A.data, Acsc.data)

    A = _round_trip(Absr)
    _assert_same_numpy(A.indptr, Absr.indptr)
    _assert_same_numpy(A.indices, Absr.indices)
    _assert_same_numpy(A.data, Absr.data)


def test_zero_copy_roundtrip_cupy():
    cp = pytest.importorskip("cupy")
    cps = pytest.importorskip("cupyx.scipy.sparse")
    Adns = cp.ones((2, 3, 4), dtype=cp.float64)
    Adia = cps.eye(m=10, n=10, dtype=cp.float64)
    Acoo = cps.coo_matrix(Adia)
    Acoo.sum_duplicates()  # no surprises
    Acsr = cps.csr_matrix(Adia)
    Acsr.sort_indices()  # no surprises
    Acsc = cps.csc_matrix(Adia)
    Acsc.sort_indices()  # no surprises

    A = _round_trip(Adns)
    _assert_same_cupy(A, Adns)

    A = _round_trip(Adia)
    _assert_same_cupy(A.offsets, Adia.offsets)
    _assert_same_cupy(A.data, Adia.data)

    A = _round_trip(Acoo)
    _assert_same_cupy(A.row, Acoo.row)
    _assert_same_cupy(A.col, Acoo.col)
    _assert_same_cupy(A.data, Acoo.data)

    A = _round_trip(Acsr)
    _assert_same_cupy(A.indptr, Acsr.indptr)
    _assert_same_cupy(A.indices, Acsr.indices)
    _assert_same_cupy(A.data, Acsr.data)

    A = _round_trip(Acsc)
    _assert_same_cupy(A.indptr, Acsc.indptr)
    _assert_same_cupy(A.indices, Acsc.indices)
    _assert_same_cupy(A.data, Acsc.data)


def test_zero_copy_roundtrip_torch():
    torch = pytest.importorskip("torch")
    Adns = torch.ones((2, 3, 4), dtype=torch.float64)
    Adia = torch.eye(10, m=10, dtype=torch.float64)
    Acoo = Adia.to_sparse()
    Acoo = Acoo.coalesce()  # no surprises
    Acsr = Adia.to_sparse_csr()
    Acsc = Adia.to_sparse_csc()
    Absr = Adia.to_sparse_bsr((5, 5))
    Absc = Adia.to_sparse_bsc((5, 5))

    A = _round_trip(Adns)
    _assert_same_torch(A, Adns)

    A = _round_trip(Acoo)
    _assert_same_torch(A.indices(), Acoo.indices())
    _assert_same_torch(A.values(), Acoo.values())

    A = _round_trip(Acsr)
    _assert_same_torch(A.crow_indices(), Acsr.crow_indices())
    _assert_same_torch(A.col_indices(), Acsr.col_indices())
    _assert_same_torch(A.values(), Acsr.values())

    A = _round_trip(Acsc)
    _assert_same_torch(A.ccol_indices(), Acsc.ccol_indices())
    _assert_same_torch(A.row_indices(), Acsc.row_indices())
    _assert_same_torch(A.values(), Acsc.values())

    A = _round_trip(Absr)
    _assert_same_torch(A.crow_indices(), Absr.crow_indices())
    _assert_same_torch(A.col_indices(), Absr.col_indices())
    _assert_same_torch(A.values(), Absr.values())

    A = _round_trip(Absc)
    _assert_same_torch(A.ccol_indices(), Absc.ccol_indices())
    _assert_same_torch(A.row_indices(), Absc.row_indices())
    _assert_same_torch(A.values(), Absc.values())


def test_scipy_convert():
    row = np.array([0, 1, 2, 3, 0, 1], dtype=np.int32)
    col = np.array([0, 1, 2, 3, 2, 3], dtype=np.int32)
    val = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    coo = sps.coo_array((val, (row, col)), shape=(4, 5))
    coo.sum_duplicates()
    ust = Tensor.from_package(coo)
    csr = ust.convert(tensor_format=NamedFormats.CSR)
    assert str(csr) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [4, 5]\n"
        "nse      : 6\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [0, 2, 4, 5, 6] #5\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [0, 2, 1, 3, 2, 3] #6\n"
        "values   : [1.0, 5.0, 2.0, 6.0, 3.0, 4.0] #6\n"
        "data     : 68 bytes\n"
        "sparsity : 70.00%\n"
        "----"
    )
    csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert str(csc) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (j: <LevelFormat.DENSE>, i: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [5, 4]\n"
        "nse      : 6\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [0, 1, 2, 4, 6, 6] #6\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [0, 1, 0, 2, 1, 3] #6\n"
        "values   : [1.0, 2.0, 5.0, 3.0, 6.0, 4.0] #6\n"
        "data     : 72 bytes\n"
        "sparsity : 70.00%\n"
        "----"
    )
    diai = ust.convert(tensor_format=NamedFormats.DIAI)
    assert str(diai) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, i: <LevelFormat.RANGE>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [8, 4]\n"
        "nse      : 8\n"
        "pos[0]   : [0, 2] #2\n"
        "pos[1]   : [] #0\n"
        "crd[0]   : [0, 2] #2\n"
        "crd[1]   : [] #0\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0] #8\n"
        "data     : 48 bytes\n"
        "sparsity : 60.00%\n"
        "----"
    )
    diaj = ust.convert(tensor_format=NamedFormats.DIAJ)
    assert str(diaj) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> ((j - i): <LevelFormat.COMPRESSED>, j: <LevelFormat.RANGE>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [8, 5]\n"
        "nse      : 10\n"
        "pos[0]   : [0, 2] #2\n"
        "pos[1]   : [] #0\n"
        "crd[0]   : [0, 2] #2\n"
        "crd[1]   : [] #0\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0] #10\n"
        "data     : 56 bytes\n"
        "sparsity : 50.00%\n"
        "----"
    )
    dense = ust.convert(tensor_format=NamedFormats.DensedRight(2))
    assert str(dense) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [4, 5]\n"
        "nse      : 20\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "values   : [1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 6.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0] #20\n"
        "data     : 80 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    # Dense back to sparse drops the zero padding.
    coo = dense.convert(tensor_format=NamedFormats.COO)
    assert str(coo) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: (<LevelFormat.COMPRESSED>, <LevelProperty.NONUNIQUE>), j: <LevelFormat.SINGLETON>)\n"
        "device   : cpu\n"
        "dim      : [4, 5]\n"
        "lvl      : [4, 5]\n"
        "nse      : 6\n"
        "pos[0]   : [0, 6] #2\n"
        "pos[1]   : [] #0\n"
        "crd[0]   : [0, 0, 1, 1, 2, 3] #6\n"
        "crd[1]   : [0, 2, 1, 3, 2, 3] #6\n"
        "values   : [1.0, 5.0, 2.0, 6.0, 3.0, 4.0] #6\n"
        "data     : 80 bytes\n"
        "sparsity : 70.00%\n"
        "----"
    )


def test_scipy_convert_3d():
    a = (1.0 + np.arange(2 * 3 * 4)).reshape(2, 3, 4)
    ust = Tensor.from_package(a)

    ar = ust.convert(tensor_format=NamedFormats.DensedRight(3))
    assert str(ar) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,"
        " 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0] #24\n"
        "data     : 192 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    al = ust.convert(tensor_format=NamedFormats.DensedLeft(3))
    assert str(al) == (
        "---- Sparse Tensor<VAL=float64,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (k: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [4, 3, 2]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0, 10.0,"
        " 22.0, 3.0, 15.0, 7.0, 19.0, 11.0, 23.0, 4.0, 16.0, 8.0, 20.0, 12.0, 24.0] #24\n"
        "data     : 192 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )


def test_torch_convert_3d():
    torch = pytest.importorskip("torch")
    a = (1.0 + torch.arange(2 * 3 * 4)).reshape(2, 3, 4)
    ust = Tensor.from_package(a)
    ar = ust.convert(tensor_format=NamedFormats.DensedRight(3))
    al = ust.convert(tensor_format=NamedFormats.DensedLeft(3))

    assert str(ar) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,"
        " 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0] #24\n"
        "data     : 96 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert str(al) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (k: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)\n"
        "device   : cpu\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [4, 3, 2]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0, 10.0,"
        " 22.0, 3.0, 15.0, 7.0, 19.0, 11.0, 23.0, 4.0, 16.0, 8.0, 20.0, 12.0, 24.0] #24\n"
        "data     : 96 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )

    a = a.cuda()
    ust = Tensor.from_package(a)
    ar = ust.convert(tensor_format=NamedFormats.DensedRight(3))
    al = ust.convert(tensor_format=NamedFormats.DensedLeft(3))

    assert str(ar) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cuda\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,"
        " 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0] #24\n"
        "data     : 96 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert str(al) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (k: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)\n"
        "device   : cuda\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [4, 3, 2]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0, 10.0,"
        " 22.0, 3.0, 15.0, 7.0, 19.0, 11.0, 23.0, 4.0, 16.0, 8.0, 20.0, 12.0, 24.0] #24\n"
        "data     : 96 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )


def test_cupy_convert_3d():
    cp = pytest.importorskip("cupy")
    a = (1.0 + cp.arange(2 * 3 * 4, dtype=cp.float32)).reshape(2, 3, 4)
    ust = Tensor.from_package(a)
    ar = ust.convert(tensor_format=NamedFormats.DensedRight(3))
    al = ust.convert(tensor_format=NamedFormats.DensedLeft(3))

    assert str(ar) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, k: <LevelFormat.DENSE>)\n"
        "device   : cuda\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [2, 3, 4]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,"
        " 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0] #24\n"
        "data     : 96 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )
    assert str(al) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=3,LVL=3>\n"
        "format   : [i, j, k] -> (k: <LevelFormat.DENSE>, j: <LevelFormat.DENSE>, i: <LevelFormat.DENSE>)\n"
        "device   : cuda\n"
        "dim      : [2, 3, 4]\n"
        "lvl      : [4, 3, 2]\n"
        "nse      : 24\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [] #0\n"
        "pos[2]   : [] #0\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [] #0\n"
        "crd[2]   : [] #0\n"
        "values   : [1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0, 10.0,"
        " 22.0, 3.0, 15.0, 7.0, 19.0, 11.0, 23.0, 4.0, 16.0, 8.0, 20.0, 12.0, 24.0] #24\n"
        "data     : 96 bytes\n"
        "sparsity : 0.00%\n"
        "----"
    )


def test_delta():
    row = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    col = np.array([0, 15, 0, 1, 7, 15], dtype=np.int32)
    val = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    csr = sps.coo_array((val, (row, col)), shape=(3, 16)).tocsr()
    ust = Tensor.from_package(csr)
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: <LevelFormat.COMPRESSED>)\n"
        "device   : cpu\n"
        "dim      : [3, 16]\n"
        "lvl      : [3, 16]\n"
        "nse      : 6\n"
        "pos[1]   : [0, 2, 5, 6] #4\n"
        "crd[1]   : [0, 15, 0, 1, 7, 15] #6\n"
        "values   : [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] #6\n"
        "data     : 64 bytes\n"
        "sparsity : 87.50%\n"
        "----"
    )
    ust = ust.convert(tensor_format=NamedFormats.DELTA(2))
    assert str(ust) == (
        "---- Sparse Tensor<VAL=float32,POS=int32,CRD=int32,DIM=2,LVL=2>\n"
        "format   : [i, j] -> (i: <LevelFormat.DENSE>, j: (<LevelFormat.DELTA>, 2))\n"
        "device   : cpu\n"
        "dim      : [3, 16]\n"
        "lvl      : [3, 16]\n"
        "nse      : 13\n"
        "pos[0]   : [] #0\n"
        "pos[1]   : [0, 5, 9, 13] #4\n"
        "crd[0]   : [] #0\n"
        "crd[1]   : [0, 3, 3, 3, 2, 0, 0, 3, 1, 3, 3, 3, 3] #13\n"
        "values   : [1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0] #13\n"
        "data     : 120 bytes\n"
        "sparsity : 72.92%\n"
        "----"
    )
    assert ust.get_value([0, 0]) == 1.0
    assert ust.get_value([0, 3]) is None
    assert ust.get_value([0, 4]) == 0.0  # padded!
    assert ust.get_value([0, 5]) is None
    assert ust.get_value([0, 15]) == 2.0
    assert ust.get_value([1, 0]) == 3.0
    assert ust.get_value([1, 1]) == 4.0
    assert ust.get_value([1, 7]) == 5.0
    assert ust.get_value([2, 0]) is None
    assert ust.get_value([2, 15]) == 6.0

    stream = StringIO()

    def visit(idx, val):
        print(idx, val, file=stream)

    TensorDecomposer(ust, visit).run()
    assert stream.getvalue() == (
        "[0, 0] 1.0\n"
        "[0, 4] 0.0\n"
        "[0, 8] 0.0\n"
        "[0, 12] 0.0\n"
        "[0, 15] 2.0\n"
        "[1, 0] 3.0\n"
        "[1, 1] 4.0\n"
        "[1, 5] 0.0\n"
        "[1, 7] 5.0\n"
        "[2, 3] 0.0\n"
        "[2, 7] 0.0\n"
        "[2, 11] 0.0\n"
        "[2, 15] 6.0\n"
    )


def test_torch_to():
    torch = pytest.importorskip("torch")
    a = 1.0 + torch.arange(2 * 3).reshape(2, 3)
    ust = Tensor.from_package(a.to_sparse_csr())
    ust_dev = ust.to(0)
    assert ust.device == "cpu"
    assert ust_dev.device == "cuda"
    assert ust.get_value([1, 2]) == 6
    assert ust_dev.get_value([1, 2]) == 6


def test_torch_copy_():
    torch = pytest.importorskip("torch")
    a = 1.0 + torch.arange(2 * 3).reshape(2, 3)
    b = 2.0 + torch.arange(2 * 3).reshape(2, 3)
    ust1 = Tensor.from_package(a.to_sparse_csr())
    ust2 = Tensor.from_package(b.to_sparse_csr())
    assert ust1.device == "cpu"
    assert ust2.device == "cpu"
    assert ust1.get_value([1, 2]) == 6
    assert ust2.get_value([1, 2]) == 7
    ust2.copy_(ust1)
    assert ust1.device == "cpu"
    assert ust2.device == "cpu"
    assert ust1.get_value([1, 2]) == 6
    assert ust2.get_value([1, 2]) == 6


def test_torch_clone():
    torch = pytest.importorskip("torch")
    a = 1.0 + torch.arange(2 * 3).reshape(2, 3)
    ust1 = Tensor.from_package(a.to_sparse_csr())
    ust2 = ust1.clone()
    assert ust1.device == "cpu"
    assert ust2.device == "cpu"
    assert ust1.get_value([1, 2]) == 6
    assert ust2.get_value([1, 2]) == 6


def test_invalid_copy_():
    a = np.ones((10, 10), dtype=np.float64)
    b = np.ones((10, 11), dtype=np.float64)
    ust1 = Tensor.from_package(a)
    ust2 = Tensor.from_package(b)
    with pytest.raises(ValueError) as e:
        ust2.copy_(ust1)
    assert str(e.value) == "The source [10, 10] and target [10, 11] extents are not compatible."


@pytest.mark.parametrize("package,device", [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cpu")])
def test_format_then_type_conversion(package, device):
    module = pytest.importorskip(package)
    a = 1 + module.arange(16, dtype=module.float64).reshape(4, 4)
    ust = Tensor.from_package(a)
    assert ust.dtype == "float64"
    assert ust.index_type == "int32"
    assert ust.get_value([3, 3]) == 16
    assert ust.device == device
    ust = ust.convert(tensor_format=NamedFormats.CSR)
    assert ust.dtype == "float64"
    assert ust.index_type == "int32"
    assert ust.get_value([3, 3]) == 16
    assert ust.device == device
    ust = ust.convert(index_type="int8", dtype="float32")
    assert ust.dtype == "float32"
    assert ust.index_type == "int8"
    assert ust.get_value([3, 3]) == 16
    assert ust.device == device


@pytest.mark.parametrize("package,device", [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cpu")])
def test_conversions(package, device):
    module = pytest.importorskip(package)
    a = module.eye(32, 32)
    ust = Tensor.from_package(a)
    assert ust.nse == 32 * 32  # dense
    ust = ust.convert(tensor_format=NamedFormats.CSR)
    assert ust.nse == 32
    ust = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust.nse == 32
    ust = ust.convert(tensor_format=NamedFormats.DIAI)
    assert ust.nse == 32
    ust = ust.convert(tensor_format=NamedFormats.DIAJ)
    assert ust.nse == 32
    ust = ust.convert(tensor_format=NamedFormats.SkewDIAI)
    assert ust.nse == 1024
    ust = ust.convert(tensor_format=NamedFormats.SkewDIAJ)
    assert ust.nse == 1024
    ust = ust.convert(tensor_format=NamedFormats.DELTA(2))
    assert ust.nse == 144
    ust = ust.convert(tensor_format=NamedFormats.COO)
    assert ust.nse == 32
    ust = ust.convert(tensor_format=NamedFormats.DensedLeft(2))
    assert ust.nse == 32 * 32  # dense
    b = ust.to_package()
    assert (a == b).all(), "Error: the recreated package tensor doesn't match the original."


@pytest.mark.parametrize("package,device", [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cpu")])
def test_more_conversions(package, device):
    module = pytest.importorskip(package)
    a = module.arange(32 * 16, dtype=module.float32).reshape(16, 32)
    ust = Tensor.from_package(a)
    assert ust.nse == 16 * 32
    ust = ust.convert(tensor_format=NamedFormats.COO)
    assert ust.nse == 16 * 32 - 1
    ust = ust.convert(tensor_format=NamedFormats.CSR)
    assert ust.nse == 16 * 32 - 1
    ust = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust.nse == 16 * 32 - 1
    ust = ust.convert(tensor_format=NamedFormats.DensedLeft(2))
    assert ust.nse == 16 * 32
    b = ust.to_package()
    assert (a == b).all(), "Error: the recreated package tensor doesn't match the original."


def test_small_tensor_operations():
    A = sps.csr_matrix(([1.0], ([0], [0])), shape=(1, 1))
    ust = Tensor.from_package(A)
    assert ust.nse == 1
    assert ust.get_value([0, 0]) == 1.0
    ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust_csc.nse == 1
    assert ust.get_value([0, 0]) == 1.0
    ust_coo = ust.convert(tensor_format=NamedFormats.COO)
    assert ust_coo.nse == 1
    assert ust.get_value([0, 0]) == 1.0


def test_single_row_column():
    A = sps.csr_matrix(([1, 2, 3], ([0, 0, 0], [0, 1, 9])), shape=(1, 10))
    ust = Tensor.from_package(A)
    assert ust.nse == 3
    assert ust.get_value([0, 0]) == 1
    assert ust.get_value([0, 1]) == 2
    assert ust.get_value([0, 2]) is None
    assert ust.get_value([0, 9]) == 3
    ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust_csc.nse == 3
    assert ust_csc.get_value([0, 0]) == 1
    assert ust_csc.get_value([0, 1]) == 2
    assert ust_csc.get_value([0, 2]) is None
    assert ust_csc.get_value([0, 9]) == 3

    B = sps.csr_matrix(([1, 2, 3], ([0, 1, 9], [0, 0, 0])), shape=(10, 1))
    ust = Tensor.from_package(B)
    assert ust.nse == 3
    assert ust.get_value([0, 0]) == 1
    assert ust.get_value([1, 0]) == 2
    assert ust.get_value([2, 0]) is None
    assert ust.get_value([9, 0]) == 3
    ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust_csc.nse == 3
    assert ust_csc.get_value([0, 0]) == 1
    assert ust_csc.get_value([1, 0]) == 2
    assert ust_csc.get_value([2, 0]) is None
    assert ust_csc.get_value([9, 0]) == 3


def test_diagonal_matrix():
    A = sps.eye(100, format="csr", dtype=np.float32)
    ust = Tensor.from_package(A)
    assert ust.nse == 100
    assert ust.get_value([0, 1]) is None
    assert ust.get_value([99, 99]) == 1
    ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust_csc.nse == 100
    assert ust_csc.get_value([0, 1]) is None
    assert ust_csc.get_value([99, 99]) == 1


def test_may_drop_zeros():
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]
    data = [1.0, 0.0, 0.0, 2.0]  # two zeros
    A = sps.coo_matrix((data, (rows, cols)), shape=(2, 2))
    A.sum_duplicates()
    A = A.tocsr()
    ust = Tensor.from_package(A)
    assert ust.nse == 4  # uses direct data
    ust_csr = ust.convert(tensor_format=NamedFormats.CSR)
    assert ust_csr.nse == 4  # fast-conversion
    assert ust_csr.get_value([0, 0]) == 1.0
    assert ust_csr.get_value([0, 1]) == 0.0
    assert ust_csr.get_value([1, 0]) == 0.0
    assert ust_csr.get_value([1, 1]) == 2.0
    ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust_csc.nse == 2  # drops zeros
    assert ust_csc.get_value([0, 0]) == 1.0
    assert ust_csc.get_value([0, 1]) is None
    assert ust_csc.get_value([1, 0]) is None
    assert ust_csc.get_value([1, 1]) == 2.0


def test_large_sparse_matrix():
    n = 50000
    nnz = 500000
    rows = np.random.randint(0, n, nnz)
    cols = np.random.randint(0, n, nnz)
    data = np.random.randn(nnz).astype(np.float32)

    A = sps.coo_matrix((data, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A = A.tocsr()

    ust = Tensor.from_package(A)
    assert ust.nse == A.nnz
    assert list(ust.extents) == [n, n]
    ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
    assert ust_csc.nse == A.nnz


def test_high_dimensional_tensor():
    # Create via dense (small enough to fit in memory).
    shape = (5, 10, 7, 6, 4)
    A = np.zeros(shape, dtype=np.float32)
    # Add some random non-zeros.
    for _ in range(100):
        idx = tuple(np.random.randint(0, s) for s in shape)
        A[idx] = np.random.randn()

    ust = Tensor.from_package(A)
    assert ust.num_dimensions == 5
    assert list(ust.extents) == [5, 10, 7, 6, 4]


def test_repeated_conversions():
    A = sps.random(1000, 1000, density=0.01, format="csr", dtype=np.float32)
    ust = Tensor.from_package(A)
    nse = ust.nse

    # Convert back and forth many times.
    for _ in range(10):
        ust_csc = ust.convert(tensor_format=NamedFormats.CSC)
        ust_coo = ust_csc.convert(tensor_format=NamedFormats.COO)
        ust = ust_coo.convert(tensor_format=NamedFormats.CSR)
        assert ust_csc.nse == nse
        assert ust_coo.nse == nse
        assert ust.nse == nse


def test_various_formats():
    A = sps.random(500, 500, density=0.02, format="csr", dtype=np.float32)
    ust_csr = Tensor.from_package(A)

    formats = [
        ("COO", NamedFormats.COO),
        ("CSR", NamedFormats.CSR),
        ("CSC", NamedFormats.CSC),
        ("DCSR", NamedFormats.DCSR),
        ("DCSC", NamedFormats.DCSC),
    ]

    for name, fmt in formats:
        ust_converted = ust_csr.convert(tensor_format=fmt)
        assert ust_converted.nse == ust_csr.nse, f"NNZ mismatch for {name}"
        assert ust_converted.extents == ust_csr.extents, f"Shape mismatch for {name}"


def test_pytorch_conversion():
    torch = pytest.importorskip("torch")

    i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    A = torch.sparse_coo_tensor(i, v, (2, 3))

    ust = Tensor.from_package(A.coalesce())
    assert ust.nse == 3
    assert ust.get_value([0, 0]) is None
    assert ust.get_value([0, 2]) == 3.0
    assert ust.get_value([1, 0]) == 4.0
    assert ust.get_value([1, 2]) == 5.0

    ust_csr = ust.convert(tensor_format=NamedFormats.CSR)
    assert ust_csr.nse == 3
    assert ust_csr.get_value([0, 0]) is None
    assert ust_csr.get_value([0, 2]) == 3.0
    assert ust_csr.get_value([1, 0]) == 4.0
    assert ust_csr.get_value([1, 2]) == 5.0


def test_format_property_repeated_access():
    csr = NamedFormats.CSR
    csc = NamedFormats.CSC
    coo = NamedFormats.COO
    for _ in range(10):
        assert coo.is_identity
        assert coo.is_ordered
        assert coo.is_unique
        assert csr.is_identity
        assert csr.is_ordered
        assert csr.is_unique
        assert not csc.is_identity
        assert csc.is_ordered
        assert csc.is_unique


def test_slots():
    assert hasattr(Dimension, "__slots__")
    assert hasattr(LevelExpr, "__slots__")
    assert hasattr(TensorFormat, "__slots__")

    dim = Dimension(dimension_name="test")
    assert not hasattr(dim, "__dict__")
