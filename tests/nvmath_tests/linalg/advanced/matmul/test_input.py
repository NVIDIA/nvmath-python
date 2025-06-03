# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests checks matmul's behavior for different kinds of inputs.
"""

from nvmath.linalg.advanced import matmul, matrix_qualifiers_dtype
import numpy as np
import pytest
from nvmath.bindings import cublasLt as cublaslt

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

from .utils import compare_tensors, random_torch_complex, sample_matrix, assert_tensors_equal, to_numpy, get_framework


@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16", "float32", "float64", "complex64", "complex128"))
@pytest.mark.parametrize("with_c", (True, False))
@pytest.mark.parametrize(
    "n,m,k",
    (
        (1, 1, 1),
        (2, 3, 4),
        (3, 2, 1),
        (4, 3, 2),
    ),
)
@pytest.mark.parametrize("use_cuda", (True, False))
def test_types(framework, dtype, with_c, n, m, k, use_cuda):
    """
    Tests support for different input data types and frameworks.
    """
    try:
        a = sample_matrix(framework, dtype, (n, k), use_cuda)
        b = sample_matrix(framework, dtype, (k, m), use_cuda)
        c = sample_matrix(framework, dtype, (n, m), use_cuda)
    except NotImplementedError as e:
        pytest.skip(f"Unable to generate sample matrix: {str(e)}")

    if with_c:
        result = matmul(a, b, c, alpha=0.5, beta=0.3)
        reference = 0.5 * to_numpy(a) @ to_numpy(b) + 0.3 * to_numpy(c)
    else:
        result = matmul(a, b, alpha=0.6)
        reference = 0.6 * to_numpy(a) @ to_numpy(b)

    assert_tensors_equal(result, reference)


@pytest.mark.parametrize("with_c", (True, False))
@pytest.mark.parametrize("n", (1, 8, 64, 100, 200, 300))
@pytest.mark.parametrize("m", (1, 8, 64, 100, 200, 300))
@pytest.mark.parametrize("k", (1, 8, 64, 100, 200, 300))
def test_shapes(with_c, n, m, k):
    """
    Tests support for different input data shapes
    """
    try:
        a = sample_matrix("numpy/cupy", "float64", (n, k), True)
        b = sample_matrix("numpy/cupy", "float64", (k, m), True)
        c = sample_matrix("numpy/cupy", "float64", (n, m), True)
    except NotImplementedError as e:
        pytest.skip(f"Unable to generate sample matrix: {str(e)}")

    if with_c:
        result = matmul(a, b, c, alpha=0.5, beta=0.3)
        reference = 0.5 * to_numpy(a) @ to_numpy(b) + 0.3 * to_numpy(c)
    else:
        result = matmul(a, b, alpha=0.6)
        reference = 0.6 * to_numpy(a) @ to_numpy(b)

    assert_tensors_equal(result, reference)


def test_framework_mixing():
    """
    Tests error on inputs from different frameworks.
    """
    a = sample_matrix("torch", "float32", (7, 7), True)
    b = sample_matrix("cupy", "float32", (7, 7), True)
    with pytest.raises(TypeError, match="All tensors in the network must be from the same library"):
        matmul(a, b)


def test_default_alpha():
    """
    Tests if the value of alpha is correct.
    """
    a = sample_matrix("numpy", "float32", (3, 3), False)
    b = sample_matrix("numpy", "float32", (3, 3), False)
    assert compare_tensors(matmul(a, b, alpha=1.0), matmul(a, b))


@pytest.mark.parametrize("a_layout", ("F", "C"))
@pytest.mark.parametrize("b_layout", ("F", "C"))
@pytest.mark.parametrize("c_layout", ("F", "C"))
def test_layouts(a_layout, b_layout, c_layout):
    """
    Tests if matmul works with different layouts.
    """
    a = sample_matrix("numpy", "float32", (3, 4), False)
    b = sample_matrix("numpy", "float32", (4, 5), False)
    c = sample_matrix("numpy", "float32", (3, 5), False)
    if a_layout == "F":
        a = np.asfortranarray(a)
    if b_layout == "F":
        b = np.asfortranarray(b)
    if c_layout == "F":
        c = np.asfortranarray(c)

    assert compare_tensors(matmul(a, b, c, beta=0.2), np.matmul(a, b) + c * 0.2)


@pytest.mark.parametrize(
    "a_batch,b_batch,c_batch,out_batch",
    (
        ((), (), (), ()),
        ((8,), (8,), (8,), (8,)),
        ((3,), (), (), (3,)),
        ((), (4,), (), (4,)),
        # ((), (), (5,), (5,)), # TODO: Should we support batched C?
        ((6,), (), (6,), (6,)),
        ((), (7,), (7,), (7,)),
        ((10, 20), (10, 20), (10, 20), (10, 20)),
    ),
)
def test_batching(a_batch, b_batch, c_batch, out_batch):
    """
    Tests if matmul works with different batch sizes.
    """
    matrix_shape = (7, 7)

    def sample_batch(batch_shape):
        return sample_matrix("numpy/cupy", "float32", (*batch_shape, *matrix_shape), True)

    a = sample_batch(a_batch)
    b = sample_batch(b_batch)
    c = sample_batch(c_batch)

    result = matmul(a, b, c, beta=1)
    assert result.shape == (*out_batch, *matrix_shape)
    assert_tensors_equal(result, a @ b + c)


@pytest.mark.parametrize("c_desc", (None, "M1", "MN"))
@pytest.mark.parametrize("b_desc", ("K", "KN"))
@pytest.mark.parametrize("a_desc", ("K", "MK"))
@pytest.mark.parametrize("a_t", (True, False))
@pytest.mark.parametrize("b_t", (True, False))
@pytest.mark.parametrize("c_t", (True, False))
@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize("M,N,K", ((2, 3, 5),))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_shape_promotion(a_desc, b_desc, c_desc, a_t, b_t, c_t, M, N, K, framework, use_cuda):
    """
    Test shape promotion rules for 1D inputs
    """

    if "M" not in a_desc:
        M = 1
    if "N" not in b_desc:
        N = 1

    def unpack_shape(shape_desc):
        if shape_desc is None:
            return None
        shape_map = {
            "N": N,
            "M": M,
            "K": K,
            "1": 1,
        }
        return tuple(shape_map[c] for c in shape_desc)

    a_shape, b_shape, c_shape = unpack_shape(a_desc), unpack_shape(b_desc), unpack_shape(c_desc)

    def make_matrix(shape, transposed):
        if transposed:
            return sample_matrix(framework, "float32", tuple(reversed(shape)), use_cuda=use_cuda).T
        else:
            return sample_matrix(framework, "float32", shape, use_cuda=use_cuda)

    a = make_matrix(a_shape, a_t)
    b = make_matrix(b_shape, b_t)
    if c_desc:
        c = make_matrix(c_shape, c_t)
        with_c = True
    else:
        c = None
        with_c = False

    a_promoted, b_promoted, c_promoted = a, b, c

    if len(a_shape) == 1:
        # If argument a is 1-D, it is promoted to a matrix by prefixing 1 to its dimensions.
        a_promoted = a_promoted.reshape(1, a_shape[0])

    if len(b_shape) == 1:
        # If argument b is 1-D, it is promoted to a matrix by appending 1 to its dimensions.
        b_promoted = b_promoted.reshape(b_shape[0], 1)

    if with_c and len(c_shape) == 1:
        c_promoted = c_promoted.reshape(c_shape[0], 1)

    if with_c and c_promoted.shape[-1] == 1:
        # If a vector is provided or N = 1, the columns of c are broadcast for the addition.
        c_promoted = get_framework(c_promoted).stack([c_promoted[..., 0]] * N, -1)

    alpha = 0.12
    beta = 0.34 if with_c else None
    result = matmul(a, b, c=c, alpha=alpha, beta=beta)
    reference = matmul(a_promoted, b_promoted, c=c_promoted, alpha=alpha, beta=beta)

    if len(a_shape) == 1:
        assert reference.shape[-2] == 1
        reference = reference.reshape((*reference.shape[:-2], reference.shape[-1]))

    if len(b_shape) == 1:
        assert reference.shape[-1] == 1
        reference = reference.reshape(reference.shape[:-1])

    assert_tensors_equal(result, reference)


@pytest.mark.parametrize(
    "slices",
    (
        ((1, 2), (1, 1), (1, 1)),
        ((1, 1), (1, 3), (1, 1)),
        ((1, 1), (1, 1), (1, 4)),
        ((1, 2), (1, 2), (1, 2)),
        ((2, 3), (4, 5), (6, 7)),
    ),
)
def test_sliced_unsupported(slices):
    """
    Tests if unsupported strided matrices are rejected with appropriate error message.
    (Unsupported strides are the ones with no stride equal to 1)
    """
    (a_step_x, a_step_y), (b_step_x, b_step_y), (c_step_x, c_step_y) = slices

    a = sample_matrix("numpy/cupy", "float32", (a_step_x * 3, a_step_y * 4), True)[::a_step_x, ::a_step_y]
    b = sample_matrix("numpy/cupy", "float32", (b_step_x * 4, b_step_y * 5), True)[::b_step_x, ::b_step_y]
    c = sample_matrix("numpy/cupy", "float32", (c_step_x * 3, c_step_y * 5), True)[::c_step_x, ::c_step_y]

    with pytest.raises(ValueError, match="Unsupported layout."):
        matmul(a, b, c, beta=0.2)


@pytest.mark.parametrize(
    "slices",
    (
        ((2, 1), (1, 1), (1, 1)),
        ((1, 1), (3, 1), (1, 1)),
        ((1, 1), (1, 1), (4, 1)),
        ((2, 1), (2, 1), (2, 1)),
    ),
)
@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_sliced(slices, framework, use_cuda):
    """
    Tests if strided tensors work correctly
    """
    (a_step_x, a_step_y), (b_step_x, b_step_y), (c_step_x, c_step_y) = slices

    a = sample_matrix(framework, "float32", (a_step_x * 3, a_step_y * 4), use_cuda)[::a_step_x, ::a_step_y]
    b = sample_matrix(framework, "float32", (b_step_x * 4, b_step_y * 5), use_cuda)[::b_step_x, ::b_step_y]
    c = sample_matrix(framework, "float32", (c_step_x * 3, c_step_y * 5), use_cuda)[::c_step_x, ::c_step_y]

    assert_tensors_equal(matmul(a, b, c, beta=0.2), a @ b + 0.2 * c)


@pytest.mark.parametrize(
    "slices",
    (
        ((2, 1), (1, 1), (1, 1)),
        ((1, 1), (3, 1), (1, 1)),
        ((1, 1), (1, 1), (4, 1)),
        ((2, 1), (2, 1), (2, 1)),
    ),
)
@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_sliced_batched(slices, framework, use_cuda):
    """
    Tests if strided tensors work correctly
    """
    (a_step_x, a_step_y), (b_step_x, b_step_y), (c_step_x, c_step_y) = slices
    batch = 8

    a = sample_matrix(framework, "float32", (batch, a_step_x * 3, a_step_y * 4), use_cuda)[1::2, ::a_step_x, ::a_step_y]
    b = sample_matrix(framework, "float32", (batch, b_step_x * 4, b_step_y * 5), use_cuda)[1::2, ::b_step_x, ::b_step_y]
    c = sample_matrix(framework, "float32", (batch, c_step_x * 3, c_step_y * 5), use_cuda)[1::2, ::c_step_x, ::c_step_y]

    assert_tensors_equal(matmul(a, b, c, beta=0.2), a @ b + 0.2 * c)


def test_sliced_m1_n1():
    """
    Tests M=1 and N=1, strides are not 1
    """
    a_m1 = sample_matrix("cupy", "float32", (10, 20), True)[2:3, ::2]  # A is [1, 10] with strides [20, 2].
    b_n1 = sample_matrix("cupy", "float32", (15, 20), True).T[::2, 2:3]  # B is [10, 1] with strides [2, 20]

    result = matmul(a_m1, b_n1)
    assert_tensors_equal(result, a_m1 @ b_n1)


@pytest.mark.parametrize("a_conj", (True, False))
@pytest.mark.parametrize("b_conj", (True, False))
@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_conjugate_qualifiers(a_conj, b_conj, framework, use_cuda):
    """
    Test if is_conjugate qualifiers work correctly
    """
    a = random_torch_complex((8, 7), use_cuda, a_conj)
    b = random_torch_complex((7, 11), use_cuda, b_conj)
    c = random_torch_complex((8, 11), use_cuda)

    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_conjugate"] = a_conj
    qualifiers[1]["is_conjugate"] = b_conj

    r = matmul(a, b, c=c, beta=1.0, qualifiers=qualifiers)

    if a_conj:
        a = a.conj()
    if b_conj:
        b = b.conj()
    assert_tensors_equal(r, a @ b + c)


@pytest.mark.parametrize("a_conj", (True, False))
@pytest.mark.parametrize("b_conj", (True, False))
@pytest.mark.parametrize("use_cuda", (True, False))
def test_conjugate_torch_auto(a_conj, b_conj, use_cuda):
    """
    Test if conjugate flag of torch tensors is interpreted correctly
    """

    a = random_torch_complex((8, 7), use_cuda, a_conj)
    b = random_torch_complex((7, 11), use_cuda, b_conj)
    c = random_torch_complex((8, 11), use_cuda)

    if a_conj:
        a = a.conj()
    if b_conj:
        b = b.conj()

    r = matmul(a, b, c=c, beta=1.0)
    assert_tensors_equal(r, a @ b + c)


@pytest.mark.parametrize(
    "a_cuda,b_cuda,c_cuda",
    (
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, False, True),
        (False, True, False),
        (True, False, False),
    ),
)
def test_device_mismatch(a_cuda, b_cuda, c_cuda):
    """
    Tests if a proper error is reported when the devices differ.
    """
    assert not (a_cuda == b_cuda == c_cuda)
    a = sample_matrix("torch", "float32", (2, 2), a_cuda)
    b = sample_matrix("torch", "float32", (2, 2), b_cuda)
    c = sample_matrix("torch", "float32", (2, 2), c_cuda)
    with pytest.raises(ValueError, match=r"not on the same device"):
        matmul(a, b, c, beta=0.42)


@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize(
    "a_dtype,b_dtype,c_dtype",
    (
        ("float16", "bfloat16", "bfloat16"),
        ("float64", "float32", "float64"),
        ("complex64", "complex128", "complex128"),
        ("float16", "float32", "float64"),
    ),
)
def test_dtype_mismatch(framework, a_dtype, b_dtype, c_dtype):
    """
    Tests if a proper error is reported when the data types differ.
    """
    assert not (a_dtype == b_dtype == c_dtype)
    try:
        a = sample_matrix(framework, a_dtype, (2, 2), True)
        b = sample_matrix(framework, b_dtype, (2, 2), True)
        c = sample_matrix(framework, c_dtype, (2, 2), True)
    except NotImplementedError:
        pytest.skip("Unable to generate matrix of this dtype")
    with pytest.raises(ValueError, match=r"Unsupported combination of dtypes"):
        matmul(a, b, c, beta=1)


def test_shape_mismatch_ab():
    """
    Tests if a proper error is reported when the shapes of A and B do not match
    """
    a = np.zeros((3, 2))
    b = c = np.ones((2, 2))
    with pytest.raises(ValueError, match=r"dimension"):
        matmul(a, b, c, beta=1)


def test_shape_mismatch_abc():
    """
    Tests if a proper error is reported when the shapes of AB and C do not match
    """
    a = b = np.zeros((3, 3))
    c = np.zeros((4, 4))
    with pytest.raises(ValueError, match=r"dimension"):
        matmul(a, b, c, beta=1)


def test_missing_beta():
    """
    Tests if a proper error is reported C is provided, but beta is not.
    """
    a = b = c = np.ones((3, 3))
    with pytest.raises(ValueError, match=r"A value for beta must be provided if operand C is provided"):
        matmul(a, b, c)


def test_unsupported_type():
    """
    Tests if a proper error is reported for an unsupported data type.
    """
    a = b = c = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match=r"^The dtype of operand.*not supported"):
        matmul(a, b, c, beta=1)


def test_unsupported_float8():
    """
    Tests if proper error is reported when FP8 is not supported.
    """
    try:
        import torch
    except:
        pytest.skip("Torch is required for FP8 support test.")

    if not hasattr(torch, "float8_e4m3fn"):
        # Old torch versions don't support float8_e4m3fn at all.
        pytest.skip("torch.float8_e4m3fn is required for FP8 support test.")

    a = torch.zeros((16, 16)).type(torch.float8_e4m3fn).cuda()
    b = torch.zeros((16, 16)).type(torch.float8_e4m3fn).cuda()

    if cublaslt.get_version() < 120800:
        with pytest.raises(ValueError, match=r"FP8 is not supported.*cuBLASLt version 12\.8 or higher is required"):
            matmul(a, b, quantization_scales={"a": 1, "b": 1, "d": 1})
    elif (torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor) < (8, 9):
        with pytest.raises(cublaslt.cuBLASLtError):
            matmul(a, b, quantization_scales={"a": 1, "b": 1, "d": 1})


@pytest.mark.parametrize(
    "test_case,expected_error",
    [
        ("not_tileable_a", "batch layout for A .* is not tileable"),
        ("not_tileable_b", "batch layout for B .* is not tileable"),
        ("batch_shape_mismatch", "batch dimensions of operands A .* and B .* must match"),
        ("batch_order_mismatch", "batch order of operands A .* and B .* must match"),
        ("c_m_dimension_mismatch", "The M dimension of the C matrix .* must match the M dimension of A"),
        ("c_n_dimension_mismatch", "The N dimension of the C matrix .* must match the N dimension of B"),
        ("c_batch_shape_mismatch", "The batch dimension of operand C .* must match with that of the other operands"),
        ("c_batch_order_mismatch", "The batch axis order of operand C .* must match with that of the other"),
        (
            "c_not_tileable",
            "The batch layout for C corresponding to shape .* is currently not supported because it is not tileable",
        ),
    ],
)
def test_batch_matrix_negative(test_case, expected_error):
    if cp is None:
        pytest.skip("Cupy is required for this test.")
    M, K, N = 3, 4, 5

    matrices = {
        "not_tileable_a": (
            sample_matrix("cupy", "float32", (2, 3, M, K), True)[:, :2, :, :],
            sample_matrix("cupy", "float32", (2, 2, K, N), True),
            None,
        ),
        "not_tileable_b": (
            sample_matrix("cupy", "float32", (2, 2, M, K), True),
            sample_matrix("cupy", "float32", (2, 3, K, N), True)[:, :2, :, :],
            None,
        ),
        "batch_shape_mismatch": (
            sample_matrix("cupy", "float32", (2, M, K), True),
            sample_matrix("cupy", "float32", (3, K, N), True),
            None,
        ),
        "batch_order_mismatch": (
            sample_matrix("cupy", "float32", (2, 3, M, K), True),
            cp.transpose(sample_matrix("cupy", "float32", (3, 2, K, N), True), (1, 0, 2, 3)),
            None,
        ),
        "c_m_dimension_mismatch": (
            sample_matrix("cupy", "float32", (M, K), True),
            sample_matrix("cupy", "float32", (K, N), True),
            sample_matrix("cupy", "float32", (M + 1, N), True),
        ),
        "c_n_dimension_mismatch": (
            sample_matrix("cupy", "float32", (M, K), True),
            sample_matrix("cupy", "float32", (K, N), True),
            sample_matrix("cupy", "float32", (M, N + 1), True),
        ),
        "c_batch_shape_mismatch": (
            sample_matrix("cupy", "float32", (2, 2, M, K), True),
            sample_matrix("cupy", "float32", (2, 2, K, N), True),
            sample_matrix("cupy", "float32", (3, 2, M, N), True),
        ),
        "c_batch_order_mismatch": (
            sample_matrix("cupy", "float32", (2, 3, M, K), True),
            sample_matrix("cupy", "float32", (2, 3, K, N), True),
            cp.transpose(sample_matrix("cupy", "float32", (3, 2, M, N), True), (1, 0, 2, 3)),
        ),
        "c_not_tileable": (
            sample_matrix("cupy", "float32", (2, 2, M, K), True),
            sample_matrix("cupy", "float32", (2, 2, K, N), True),
            sample_matrix("cupy", "float32", (2, 3, M, N), True)[:, :2, :, :],
        ),
    }

    a, b, c = matrices[test_case]

    with pytest.raises(ValueError, match=expected_error):
        if c is None:
            matmul(a, b)
        else:
            matmul(a, b, c, beta=1.0)
