# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests checks matmul's behavior for different kinds of inputs.
"""

from nvmath.linalg.advanced import matmul, matrix_qualifiers_dtype
import pytest
from .utils import *


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
    except NotImplementedError as e:
        pytest.skip("Unable to generate matrix of this dtype")
    with pytest.raises(ValueError, match=r"The dtype of operands .* must be the same"):
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
    a = b = c = np.asarray(["hello"])
    with pytest.raises(ValueError, match=r"Unsupported dtype."):
        matmul(a, b, c, beta=1)
