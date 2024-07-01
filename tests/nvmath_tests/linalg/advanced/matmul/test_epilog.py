# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests verifies the correctness of the epilog handling.
"""

from nvmath.linalg.advanced import matmul, MatmulEpilog as Epilog
from nvmath.bindings import cublasLt as cublaslt
import pytest
from .utils import *
from cupy import tanh, sqrt, pi, cosh


def relu(x):
    y = x.copy()
    y[y < 0] = 0
    return y


def gelu(x):
    return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.0447115 * x**3)))


simple_epilogs = (
    Epilog.RELU,
    Epilog.GELU,
    Epilog.RELU_AUX,
    Epilog.GELU_AUX,
    Epilog.BGRADA,
    Epilog.BGRADB,
)
epilogs_with_bias = (
    Epilog.RELU_BIAS,
    Epilog.GELU_BIAS,
    Epilog.BIAS,
    Epilog.RELU_AUX_BIAS,
    Epilog.GELU_AUX_BIAS,
)


def verify_relu_bitmask(x, bitmask):
    """
    Verifies if the ReLU bitmask for x is correct.
    """
    n, m = x.shape
    print(n, m)
    for i in range(n):
        for j in range(m):
            if abs(x[i][j]) <= get_tolerance(x):
                # This value is dangerously close to 0 and the bitmask might
                # be incorrect due to precision issues.
                continue
            expected = (x[i][j] >= 0).item()
            actual = bool(bitmask[i // 8][j] & (1 << i % 8))
            if expected != actual:
                return False
    return True


def unpack_bitmask(bitmask, shape):
    result = cupy.zeros(shape)
    n, m = shape
    for i in range(n):
        for j in range(m):
            result[i][j] = bool(bitmask[i // 8][j] & (1 << i % 8))
    return result


def round_up(x, align):
    if x % align != 0:
        return x + align - x % align
    else:
        return x


def simulate_epilog(a, b, epilog, epilog_inputs):
    """
    Simulates behaviour of the provided epilog.

    Returns a tuple: the result, and a dict of functions to verify the correctness of
    corresponding aux output
    """

    a = cupy.asarray(a)
    b = cupy.asarray(b)
    epilog_inputs = {k: cupy.asarray(v) for k, v in epilog_inputs.items()}

    def drelu(mask):
        """
        Derivative of relu from the mask returned by RELU_AUX
        """
        return unpack_bitmask(mask, (a.shape[0], b.shape[1]))

    def dgelu(x):
        """
        Derivative of (tanh-approximated) gelu from the values returned by GELU_AUX
        """
        sech = lambda x: 1 / cosh(x)
        return (
            0.5 * tanh(0.0356774 * x**3 + 0.797885 * x)
            + (0.0535161 * x**3 + 0.398942 * x)
            * (sech(0.0356774 * x**3 + 0.797885 * x)) ** 2
            + 0.5
        )

    def simulate_drelu(a, b, mask):
        return (a @ b) * drelu(mask)

    def simulate_dgelu(a, b, x):
        return (a @ b) * dgelu(x[: a.shape[0], : b.shape[1]])

    x = matmul(a, b)
    if epilog == Epilog.RELU:
        return relu(x), None
    elif epilog == Epilog.GELU:
        return gelu(x), None
    elif epilog == Epilog.RELU_AUX:
        return relu(x), {
            "relu_aux": lambda y: verify_relu_bitmask(to_numpy(x), y)
            and y.shape == (round_up(np.ceil(x.shape[0] / 8), 16), x.shape[1])
        }
    elif epilog == Epilog.GELU_AUX:
        return gelu(x), {
            "gelu_aux": lambda y: compare_tensors(y[: x.shape[0]], x)
            and y.shape[0] == round_up(x.shape[0], 8)
        }
    elif epilog == Epilog.BGRADA:
        return x, {"bgrada": lambda y: compare_tensors(y, a.sum(axis=1))}
    elif epilog == Epilog.BGRADB:
        return x, {"bgradb": lambda y: compare_tensors(y, b.sum(axis=0))}
    elif epilog == Epilog.DRELU:
        return simulate_drelu(a, b, epilog_inputs["relu_aux"]), None
    elif epilog == Epilog.DRELU_BGRAD:
        result = simulate_drelu(a, b, epilog_inputs["relu_aux"])
        return result, {"drelu_bgrad": lambda y: compare_tensors(y, result.sum(axis=1))}
    elif epilog == Epilog.DGELU:
        return simulate_dgelu(a, b, epilog_inputs["gelu_aux"]), None
    elif epilog == Epilog.DGELU_BGRAD:
        result = simulate_dgelu(a, b, epilog_inputs["gelu_aux"])
        return result, {"dgelu_bgrad": lambda y: compare_tensors(y, result.sum(axis=1))}

    b = epilog_inputs["bias"]
    if b.shape[0] == x.shape[0]:
        b = b.reshape((x.shape[0], 1))
    if epilog == Epilog.RELU_BIAS:
        return relu(x + b), None
    elif epilog == Epilog.GELU_BIAS:
        return gelu(x + b), None
    elif epilog == Epilog.BIAS:
        return x + b, None
    elif epilog == Epilog.RELU_AUX_BIAS:
        return relu(x + b), {
            "relu_aux": lambda y: verify_relu_bitmask(to_numpy(x + b), y)
            and y.shape == (round_up(np.ceil(x.shape[0] / 8), 16), x.shape[1])
        }
    elif epilog == Epilog.GELU_AUX_BIAS:
        return gelu(x + b), {
            "gelu_aux": lambda y: compare_tensors(y[: x.shape[0]], x + b)
            and y.shape[0] == round_up(x.shape[0], 8)
        }
    else:
        assert False


@pytest.mark.parametrize("epilog", (*simple_epilogs, *epilogs_with_bias))
@pytest.mark.parametrize("bias_shape", (lambda m: (m,), lambda m: (m, 1)))
@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize("n,m,k", ((40, 50, 60), (1, 1, 1), (8, 16, 32), (65, 43, 21)))
def test_epilogs(epilog, bias_shape, framework, n, m, k, use_cuda):
    if epilog == Epilog.BGRADB and m == 1:
        pytest.skip("BGRADB doesn't support m=1")

    if epilog in (Epilog.GELU, Epilog.BIAS, Epilog.RELU_BIAS, Epilog.GELU_BIAS, Epilog.RELU_AUX, Epilog.GELU_AUX, Epilog.RELU_AUX_BIAS, Epilog.GELU_AUX_BIAS):
        skip_if_cublas_before(11501)

    if epilog in (Epilog.BGRADA, Epilog.BGRADB):
        skip_if_cublas_before(111103)

    def make_matrix(shape, transposed):
        if transposed:
            return sample_matrix(
                framework, "float32", tuple(reversed(shape)), use_cuda=use_cuda
            ).T
        else:
            return sample_matrix(framework, "float32", shape, use_cuda=use_cuda)

    a, b = (
        make_matrix((n, k), transposed=(epilog == Epilog.BGRADA)),
        make_matrix((k, m), transposed=(epilog == Epilog.BGRADA)),
    )
    bias_value = sample_matrix(framework, "float32", bias_shape(n), use_cuda=use_cuda)
    inputs = (
        {
            "bias": bias_value,
        }
        if epilog in epilogs_with_bias
        else {}
    )
    reference, aux_checkers = simulate_epilog(a, b, epilog, inputs)
    if aux_checkers:
        if k == 1 and epilog in [Epilog.BGRADA, Epilog.BGRADB] and cublaslt.get_version() < 120304:
            with pytest.raises(ValueError, match="not supported"):
                matmul(a, b, epilog=epilog, epilog_inputs=inputs)
            return
        result, aux = matmul(a, b, epilog=epilog, epilog_inputs=inputs)
        assert_tensors_equal(result, result)
        assert aux.keys() == aux_checkers.keys()
        for k in aux.keys():
            res, checker = aux[k], aux_checkers[k]
            assert checker(to_numpy(res))
    else:
        result = matmul(a, b, epilog=epilog, epilog_inputs=inputs)
        assert_tensors_equal(result, reference)


@pytest.mark.parametrize(
    "d_epilog,epilog",
    (
        (Epilog.DRELU, Epilog.RELU_AUX_BIAS),
        (Epilog.DRELU_BGRAD, Epilog.RELU_AUX_BIAS),
        (Epilog.DGELU, Epilog.GELU_AUX_BIAS),
        (Epilog.DGELU_BGRAD, Epilog.GELU_AUX_BIAS),
    ),
)
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize("n,m,k", ((41, 33, 29), (2, 2, 2), (64, 32, 16), (65, 43, 21)))
def test_d_epilogs(d_epilog, epilog, n, m, k, use_cuda):
    skip_if_cublas_before(111103, message="DRELU/DGELU not supported")

    a = sample_matrix("torch", "float32", (k, n), use_cuda=use_cuda).T
    b = sample_matrix("torch", "float32", (m, k), use_cuda=use_cuda).T
    bias_value = torch.rand((a.shape[0], 1)) - 0.5
    bias_value = bias_value.cuda() if use_cuda else bias_value
    ab, aux = matmul(a, b, epilog=epilog, epilog_inputs={"bias": bias_value})
    reference, aux_checkers = simulate_epilog(a, b, epilog=d_epilog, epilog_inputs=aux)

    if not aux_checkers:
        result = matmul(a, b, epilog=d_epilog, epilog_inputs=aux)
    else:
        result, aux = matmul(a, b, epilog=d_epilog, epilog_inputs=aux)
        assert aux.keys() == aux_checkers.keys()
        for k in aux.keys():
            assert aux_checkers[k](to_numpy(aux[k]))
    assert_tensors_equal(result, reference)


@pytest.mark.parametrize("epilog", epilogs_with_bias)
@pytest.mark.parametrize(
    "bias",
    (
        lambda m, n: cupy.full((m, n), 0.8),
        lambda m, n: cupy.full((n, m), 0.8),
        lambda m, n: cupy.full((1, 1), 0.8),
        lambda m, n: cupy.full((1,), 0.8),
    ),
)
def test_invalid_bias_shapes(epilog, bias):
    with pytest.raises(ValueError):
        a, b = sample_float_tensor((4, 5)), sample_float_tensor((5, 8))
        matmul(
            a,
            b,
            epilog=epilog,
            epilog_inputs={
                "bias": bias(a.shape[0], b.shape[1]),
            },
        ),


@pytest.mark.parametrize("epilog", epilogs_with_bias)
def test_bias_package_mismatch(epilog):
    with pytest.raises(TypeError):
        a = sample_matrix("torch", "float32", (4, 5), use_cuda=True)
        b = sample_matrix("torch", "float32", (5, 8), use_cuda=True)
        matmul(
            a,
            b,
            epilog=epilog,
            epilog_inputs={
                "bias": np.full((4, 1), 0.7),
            },
        ),


@pytest.mark.parametrize(
    "epilog",
    (
        *epilogs_with_bias,
        Epilog.DRELU,
        Epilog.DGELU,
        Epilog.DRELU_BGRAD,
        Epilog.DGELU_BGRAD,
    ),
)
def test_missing_epilog_inputs(epilog):
    """
    Tests if lack of required epilog inputs is handled correctly
    """
    skip_if_cublas_before(111103, message="DRELU/DGELU not supported")

    with pytest.raises(ValueError):
        a, b = sample_float_tensor((4, 5)), sample_float_tensor((5, 8))
        matmul(a, b, epilog=epilog, epilog_inputs={}),


@pytest.mark.parametrize(
    "epilog",
    (
        *epilogs_with_bias,
        Epilog.DRELU,
        Epilog.DGELU,
        Epilog.DRELU_BGRAD,
        Epilog.DGELU_BGRAD,
    ),
)
def test_extra_epilog_inputs(epilog):
    """
    Tests if extra inputs are handled correctly
    """
    skip_if_cublas_before(111103, message="DRELU/DGELU not supported")

    with pytest.raises(ValueError):
        a, b = sample_float_tensor((4, 5)), sample_float_tensor((5, 8))
        matmul(
            a,
            b,
            epilog=epilog,
            epilog_inputs={
                "bias": cupy.full((4, 1), 0.8),
                "drelu_aux": cupy.zeros((12, 34)),
                "dgelu_aux": cupy.zeros((12, 34)),
                "extra": cupy.zeros((12, 34)),
            },
        ),


def test_renamed_epilog_inputs():
    """
    Tests if input with invalid name is rejected
    """
    with pytest.raises(ValueError):
        a, b = sample_float_tensor((4, 5)), sample_float_tensor((5, 8))
        matmul(
            a,
            b,
            epilog=Epilog.BIAS,
            epilog_inputs={"not_a_bias": cupy.full((4, 1), 0.8)},
        ),
