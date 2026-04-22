# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from nvmath.sparse.ust.interfaces.torch_interface import TorchUST  # noqa: E402
from nvmath.sparse.ust.interfaces.torch_interface import reformat_model as reformat_model  # noqa: E402


def test_like():
    a = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    a_u = TorchUST.from_torch(a)

    z = torch.zeros_like(a_u)
    o = torch.ones_like(a_u)
    e = torch.empty_like(a_u)

    assert torch.equal(torch.zeros(8, 8).cuda(), z)
    assert torch.equal(torch.ones(8, 8).cuda(), o)
    assert e.shape == (8, 8)


def test_dot():
    x = (1.0 + torch.arange(32)).cuda()
    y = (2.0 + torch.arange(32)).cuda()
    z = torch.dot(x, y)

    xu = TorchUST.from_torch(x)
    z1 = torch.dot(xu, y)
    assert torch.equal(z, z1)
    z2 = torch.dot(xu.t(), y)
    assert torch.equal(z, z2)

    yu = TorchUST.from_torch(y)
    z3 = torch.dot(x, yu)
    assert torch.equal(z, z3)
    z4 = torch.dot(x, yu.t())
    assert torch.equal(z, z4)


def test_mv():
    A = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    x = (2.0 + torch.arange(8)).cuda()
    zd = torch.mv(A, x)
    zt = torch.mv(A.t(), x)
    assert zd.ndim == 1
    assert zt.ndim == 1

    A_U = TorchUST.from_torch(A)
    z1 = torch.mv(A_U, x)
    assert z1.ndim == 1
    assert torch.equal(zd, z1)
    z2 = torch.mv(A_U.t(), x)
    assert z2.ndim == 1
    assert torch.equal(zt, z2)

    xu = TorchUST.from_torch(x)
    z3 = torch.mv(A, xu)
    assert z3.ndim == 1
    assert torch.equal(zd, z3)
    z4 = torch.mv(A, xu.t())
    assert z4.ndim == 1
    assert torch.equal(zd, z4)


def test_addmv():
    A = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    x = (2.0 + torch.arange(8)).cuda()
    y = (3.0 + torch.arange(8)).cuda()
    z = torch.addmv(y, A, x)

    A_U = TorchUST.from_torch(A)
    z1 = torch.addmv(y, A_U, x)
    assert torch.equal(z, z1)


def test_mm():
    A = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    B = (2.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    Z1 = torch.mm(A, B)
    Z2 = torch.mm(A.t(), B)
    Z3 = torch.mm(A, B.t())
    assert Z1.ndim == 2
    assert Z2.ndim == 2
    assert Z3.ndim == 2

    A_U = TorchUST.from_torch(A)
    C1 = torch.mm(A_U, B)
    assert C1.ndim == 2
    assert torch.equal(Z1, C1)
    C2 = torch.mm(A_U.t(), B)
    assert C2.ndim == 2
    assert torch.equal(Z2, C2)
    C3 = torch.mm(A_U.t().T, B)
    assert C3.ndim == 2
    assert torch.equal(Z1, C3)

    B_U = TorchUST.from_torch(B)
    D1 = torch.mm(A, B_U)
    assert D1.ndim == 2
    assert torch.equal(Z1, D1)
    D2 = torch.mm(A, B_U.t())
    assert D2.ndim == 2
    assert torch.equal(Z3, D2)
    D3 = torch.mm(A, B_U.T.T.T)
    assert D3.ndim == 2
    assert torch.equal(Z3, D3)


def test_addmm():
    A = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    B = (2.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    C = (3.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    Z = torch.addmm(C, A, B)

    A_U = TorchUST.from_torch(A)
    E = torch.addmm(C, A_U, B)
    assert torch.equal(E, Z)


def test_matmul():
    A = (1.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    B = (2.0 + torch.arange(8 * 8)).reshape(8, 8).cuda()
    Z = torch.matmul(A, B)

    A_U = TorchUST.from_torch(A)
    E = torch.matmul(A_U, B)
    assert torch.equal(E, Z)


def test_linear():
    torch.manual_seed(0)
    linear = torch.nn.Linear(20, 30).cuda()
    A1 = (1.0 + torch.arange(20)).cuda()
    A2 = (2.0 + torch.arange(128 * 20)).reshape(128, 20).cuda()
    A3 = (3.0 + torch.arange(4 * 128 * 20)).reshape(4, 128, 20).cuda()

    with torch.no_grad():
        O1 = linear(A1)
        O2 = linear(A2)
        O3 = linear(A3)

    assert O1.ndim == 1
    assert O2.ndim == 2
    assert O3.ndim == 3

    def func1(weight):
        return TorchUST.from_torch(weight)

    def func2(weight):
        return TorchUST.from_torch(weight.to_sparse())

    def func3(weight):
        return TorchUST.from_torch(weight.to_sparse_csr())

    def func4(weight):
        return TorchUST.from_torch(weight.to_sparse_csc())

    def func5(weight):
        return TorchUST.from_torch(weight.to_sparse_bsr(blocksize=(2, 2)))

    def func6(weight):
        return TorchUST.from_torch(weight.to_sparse_bsc(blocksize=(2, 2)))

    save_weight = linear.weight
    for f in [func1, func2, func3, func4, func5, func6]:
        linear.weight = save_weight

        reformat_model(linear, func=f)

        with torch.no_grad():
            X1 = linear(A1)
            X2 = linear(A2)
            X3 = linear(A3)

        assert X1.ndim == 1
        assert X2.ndim == 2
        assert X3.ndim == 3

        assert torch.allclose(O1, X1, atol=1e-04)
        assert torch.allclose(O2, X2, atol=1e-03)
        assert torch.allclose(O3, X3, atol=1e-03)


def test_bmm():
    A = (1.0 + torch.arange(4 * 3 * 7)).reshape(4, 3, 7).cuda()
    B = (2.0 + torch.arange(4 * 7 * 5)).reshape(4, 7, 5).cuda()
    C = torch.bmm(A, B)

    A_U = TorchUST.from_torch(A)
    D = torch.bmm(A_U, B)
    assert torch.equal(C, D)


def test_addbmm():
    A = (1.0 + torch.arange(4 * 3 * 7)).reshape(4, 3, 7).cuda()
    B = (2.0 + torch.arange(4 * 7 * 5)).reshape(4, 7, 5).cuda()
    X = (3.0 + torch.arange(3 * 5)).reshape(3, 5).cuda()
    C = torch.addbmm(X, A, B)

    # NOTE: addbmm yields 2-D output (sums the batches);
    #       there is currently no TorchUST support for that.
    assert C.ndim == 2


def test_baddbmm():
    A = (1.0 + torch.arange(4 * 3 * 7)).reshape(4, 3, 7).cuda()
    B = (2.0 + torch.arange(4 * 7 * 5)).reshape(4, 7, 5).cuda()
    X = (3.0 + torch.arange(4 * 3 * 5)).reshape(4, 3, 5).cuda()
    C = torch.baddbmm(X, A, B)

    # NOTE: baddbmm yields 3-D output (independent batches).
    assert C.ndim == 3
    A_U = TorchUST.from_torch(A)
    D = torch.baddbmm(X, A_U, B)
    assert torch.equal(C, D)


def test_mul_inplace():
    A1 = (1.0 + torch.arange(4 * 3 * 7)).reshape(4, 3, 7).cuda()
    A1.mul_(2)
    A2 = (1.0 + torch.arange(4 * 3 * 7)).reshape(4, 3, 7).cuda()
    A_U = TorchUST.from_torch(A2)
    A_U.mul_(2)
    assert A1.ndim == 3
    assert A2.ndim == 3
    assert torch.equal(A1, A2)
