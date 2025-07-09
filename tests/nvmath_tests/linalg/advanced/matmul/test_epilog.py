# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This set of tests verifies the correctness of the epilog handling.
"""

import numpy as np
import pytest

from nvmath.linalg.advanced import matmul, Matmul, MatmulEpilog as Epilog
from nvmath.bindings import cublasLt as cublaslt

from .utils import (
    compare_tensors,
    get_absolute_tolerance,
    get_framework,
    sample_float_tensor,
    sample_matrix,
    assert_tensors_equal,
    skip_if_cublas_before,
    to_numpy,
)

try:
    import cupy
except ModuleNotFoundError:
    pytest.skip("cupy required for matmul tests", allow_module_level=True)

rng = np.random.default_rng(12345)


def relu(x):
    y = x.copy()
    y[y < 0] = 0
    return y


def gelu(x):
    return 0.5 * x * (1 + cupy.tanh(cupy.sqrt(2 / cupy.pi) * (x + 0.0447115 * x**3)))


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
    for i in range(n):
        for j in range(m):
            if abs(x[i][j]) <= get_absolute_tolerance(x):
                # This value is dangerously close to 0 and the bitmask might
                # be incorrect due to precision issues.
                continue
            expected = (x[i][j] >= 0).item()
            actual = bool(bitmask[i // 8][j].astype(int) & (1 << i % 8))
            if expected != actual:
                return False
    return True


def unpack_bitmask(bitmask, shape):
    result = cupy.zeros(shape)
    n, m = shape
    for i in range(n):
        for j in range(m):
            result[i][j] = bool(bitmask[i // 8][j].astype(int) & (1 << i % 8))
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

    a_batched, b_batched = len(a.shape) > 2, len(b.shape) > 2
    if a_batched or b_batched:
        # For batched input, simulate on each batch element separately and then combine
        results = []
        aux_checkers = {}
        batch_size = a.shape[0] if a_batched else b.shape[0]
        for i in range(batch_size):
            a_slice = a[i] if a_batched else a
            b_slice = b[i] if b_batched else b
            epilog_inputs_slice = {k: v[i] if len(v.shape) > 2 else v for k, v in epilog_inputs.items()}
            r, ac = simulate_epilog(a_slice, b_slice, epilog, epilog_inputs_slice)
            results.append(r)
            if ac:
                for k, v in ac.items():
                    aux_checkers[k] = aux_checkers.get(k, [])
                    aux_checkers[k].append(v)

        def check_all_aux(key):
            checkers = aux_checkers[key]

            def check(aux):
                if epilog in (Epilog.BGRADA, Epilog.BGRADB, Epilog.DRELU_BGRAD, Epilog.DGELU_BGRAD):
                    # Aux outputs of BGRAD were promoted to 3-D
                    assert len(aux.shape) == 3 and aux.shape[-1] == 1
                    aux = aux[:, :, 0]  # Remove the extra dimension
                return all(checker(aux[i]) for i, checker in enumerate(checkers))

            return check

        return cupy.stack(results), {k: check_all_aux(k) for k in aux_checkers}

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
        sech = lambda x: 1 / cupy.cosh(x)
        return (
            0.5 * cupy.tanh(0.0356774 * x**3 + 0.797885 * x)
            + (0.0535161 * x**3 + 0.398942 * x) * (sech(0.0356774 * x**3 + 0.797885 * x)) ** 2
            + 0.5
        )

    def simulate_drelu(a, b, mask):
        return (a @ b) * drelu(mask)

    def simulate_dgelu(a, b, x):
        return (a @ b) * dgelu(x[: a.shape[0], : b.shape[1]])

    x = cupy.matmul(a, b)
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
        return gelu(x), {"gelu_aux": lambda y: compare_tensors(y[: x.shape[0]], x) and y.shape[0] == round_up(x.shape[0], 8)}
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
            "gelu_aux": lambda y: compare_tensors(y[: x.shape[0]], x + b) and y.shape[0] == round_up(x.shape[0], 8)
        }
    else:
        raise AssertionError()


def execute_matmul(a, b, *, epilog=None, epilog_inputs=None, stateful=True, autotune=False):
    if not stateful:
        assert not autotune, "autotune=True requires stateful=True"
        return matmul(a, b, epilog=epilog, epilog_inputs=epilog_inputs)
    with Matmul(a, b) as mm:
        mm.plan(epilog=epilog, epilog_inputs=epilog_inputs)
        if autotune:
            mm.autotune(iterations=3)
        return mm.execute()


def make_matrix(shape, framework, use_cuda, transposed=False):
    if transposed:
        m = sample_matrix(framework, "float32", (*shape[:-2], shape[-1], shape[-2]), use_cuda=use_cuda)
        m = get_framework(m).swapaxes(m, -1, -2)
        return m
    else:
        return sample_matrix(framework, "float32", shape, use_cuda=use_cuda)


@pytest.mark.parametrize("epilog", (*simple_epilogs, *epilogs_with_bias))
@pytest.mark.parametrize("bias_extra_dim", (False, True))
@pytest.mark.parametrize("framework", ("torch", "numpy/cupy"))
@pytest.mark.parametrize("use_cuda", (True, False))
@pytest.mark.parametrize("n,m,k", ((40, 50, 60), (1, 1, 1), (8, 16, 32), (65, 43, 21), (1, 2, 3), (3, 2, 1), (2, 1, 3)))
@pytest.mark.parametrize(
    "a_batch,b_batch",
    (
        (None, 3),
        (5, None),
        (4, 4),
        (None, None),
        (1, 1),
    ),
)
def test_epilogs(epilog, bias_extra_dim, framework, n, m, k, use_cuda, a_batch, b_batch):
    autotune = rng.choice((True, False))
    if epilog == Epilog.BGRADB and m == 1:
        pytest.skip("BGRADB doesn't support m=1")
    if epilog == Epilog.BGRADA and n == 1:
        # TODO: This is a temporary fix. If A has a singleton dimension we change it to COL
        # order (see get_matrix_layout_traits), and COL order is not supported by BGRADA.
        pytest.skip("BGRADA doesn't support n=1")
    if epilog == Epilog.BGRADA and framework == "numpy/cupy" and not use_cuda and (a_batch is not None or b_batch is not None):
        # BGRADA requires COL layout of each matrix in the batch.
        # Also, one of the matrix dimensions needs to have stride one.
        # Transfer to the GPU (which uses cupy.asarray under the hood)
        # won't preserve such layout.
        pytest.skip("It's not possible to create batched COL layout with numpy")
    if bias_extra_dim and epilog not in epilogs_with_bias:
        pytest.skip("bias_extra_dim=False is irrelevant for epilog without bias")

    bias_shape = (lambda m: (m, 1)) if bias_extra_dim else (lambda m: (m,))

    if epilog in (
        Epilog.GELU,
        Epilog.BIAS,
        Epilog.RELU_BIAS,
        Epilog.GELU_BIAS,
        Epilog.RELU_AUX,
        Epilog.GELU_AUX,
        Epilog.RELU_AUX_BIAS,
        Epilog.GELU_AUX_BIAS,
    ):
        skip_if_cublas_before(11501)

    if epilog in (Epilog.BGRADA, Epilog.BGRADB):
        skip_if_cublas_before(111103)

    a, b = (
        make_matrix(
            (n, k) if a_batch is None else (a_batch, n, k),
            transposed=(epilog == Epilog.BGRADA),
            framework=framework,
            use_cuda=use_cuda,
        ),
        make_matrix(
            (k, m) if b_batch is None else (b_batch, k, m),
            transposed=(epilog == Epilog.BGRADA),
            framework=framework,
            use_cuda=use_cuda,
        ),
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

    if (
        cublaslt.get_version() < 11703
        and epilog in (Epilog.RELU_AUX, Epilog.GELU_AUX, Epilog.RELU_AUX_BIAS, Epilog.GELU_AUX_BIAS)
        and (a_batch is not None or b_batch is not None)
    ):
        with pytest.raises(ValueError, match="supports batching in cublaslt >= 11703"):
            execute_matmul(a, b, epilog=epilog, epilog_inputs=inputs, stateful=autotune, autotune=autotune)
        return

    if cublaslt.get_version() < 110902 and epilog in epilogs_with_bias and (a_batch is not None or b_batch is not None):
        with pytest.raises(ValueError, match="Bias broadcasting is not supported"):
            execute_matmul(a, b, epilog=epilog, epilog_inputs=inputs, stateful=autotune, autotune=autotune)
        return

    if aux_checkers:
        if k == 1 and epilog in [Epilog.BGRADA, Epilog.BGRADB] and cublaslt.get_version() < 120304:
            with pytest.raises(ValueError, match="not supported"):
                execute_matmul(a, b, epilog=epilog, epilog_inputs=inputs, stateful=autotune, autotune=autotune)
            return
        result, aux = execute_matmul(a, b, epilog=epilog, epilog_inputs=inputs, stateful=autotune, autotune=autotune)
        assert_tensors_equal(result, reference)
        assert aux.keys() == aux_checkers.keys()
        for k in aux:
            res, checker = aux[k], aux_checkers[k]
            assert checker(to_numpy(res))
    else:
        result = execute_matmul(a, b, epilog=epilog, epilog_inputs=inputs, stateful=autotune, autotune=autotune)
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
@pytest.mark.parametrize(
    "n,m,k", ((41, 33, 29), (2, 2, 2), (64, 32, 16), (65, 43, 21), (4, 1, 2), (1, 1, 1), (9, 2, 1), (1, 2, 3))
)
@pytest.mark.parametrize("framework", ("numpy/cupy", "torch"))
@pytest.mark.parametrize(
    "a_batch,b_batch",
    (
        (None, 2),
        (2, None),
        (5, 5),
        (None, None),
        (1, 1),
    ),
)
def test_d_epilogs(d_epilog, epilog, n, m, k, framework, use_cuda, a_batch, b_batch):
    autotune = rng.choice((True, False))
    skip_if_cublas_before(111103, message="DRELU/DGELU not supported")

    a_shape = (a_batch, n, k) if a_batch is not None else (n, k)
    b_shape = (b_batch, k, m) if b_batch is not None else (k, m)

    if "numpy" in framework and not use_cuda:
        # Transfer to the GPU (which uses cupy.asarray under the hood)
        # won't preserve such layout.
        pytest.skip("It's not possible to create COL-order matrix with numpy")

    a = make_matrix(a_shape, use_cuda=use_cuda, framework=framework, transposed=True)
    b = make_matrix(b_shape, use_cuda=use_cuda, framework=framework, transposed=True)
    bias_value = make_matrix((a.shape[-2], 1), use_cuda=use_cuda, framework=framework)
    ab, aux = execute_matmul(a, b, epilog=epilog, epilog_inputs={"bias": bias_value}, stateful=autotune, autotune=autotune)
    reference, aux_checkers = simulate_epilog(a, b, epilog=d_epilog, epilog_inputs=aux)

    if k == 1:
        with pytest.raises(ValueError, match="not supported"):
            result, aux = execute_matmul(a, b, epilog=d_epilog, epilog_inputs=aux, stateful=autotune, autotune=autotune)
        return

    if not aux_checkers:
        result = execute_matmul(a, b, epilog=d_epilog, epilog_inputs=aux, stateful=autotune, autotune=autotune)
    else:
        result, aux = execute_matmul(a, b, epilog=d_epilog, epilog_inputs=aux, stateful=autotune, autotune=autotune)
        assert aux.keys() == aux_checkers.keys()
        for k in aux:
            assert aux_checkers[k](to_numpy(aux[k]))
    assert_tensors_equal(result, reference)


@pytest.mark.parametrize("epilog", epilogs_with_bias)
@pytest.mark.parametrize(
    "bias",
    (
        lambda m, n: cupy.full((m, n), np.float32(0.8)),
        lambda m, n: cupy.full((n, m), np.float32(0.8)),
        lambda m, n: cupy.full((1, 1), np.float32(0.8)),
        lambda m, n: cupy.full((1,), np.float32(0.8)),
    ),
)
def test_invalid_bias_shapes(epilog, bias):
    with pytest.raises(ValueError):
        a, b = sample_float_tensor((4, 5)), sample_float_tensor((5, 8))
        (
            matmul(
                a,
                b,
                epilog=epilog,
                epilog_inputs={
                    "bias": bias(a.shape[0], b.shape[1]),
                },
            ),
        )


@pytest.mark.parametrize("epilog", epilogs_with_bias)
def test_bias_package_mismatch(epilog):
    with pytest.raises(TypeError):
        a = sample_matrix("torch", "float32", (4, 5), use_cuda=True)
        b = sample_matrix("torch", "float32", (5, 8), use_cuda=True)
        (
            matmul(
                a,
                b,
                epilog=epilog,
                epilog_inputs={
                    "bias": np.full((4, 1), np.float32(0.7)),
                },
            ),
        )


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
        (matmul(a, b, epilog=epilog, epilog_inputs={}),)


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
        (
            matmul(
                a,
                b,
                epilog=epilog,
                epilog_inputs={
                    "bias": cupy.full((4, 1), np.float32(0.8)),
                    "drelu_aux": cupy.zeros((12, 34)),
                    "dgelu_aux": cupy.zeros((12, 34)),
                    "extra": cupy.zeros((12, 34)),
                },
            ),
        )


def test_renamed_epilog_inputs():
    """
    Tests if input with invalid name is rejected
    """
    with pytest.raises(ValueError):
        a, b = sample_float_tensor((4, 5)), sample_float_tensor((5, 8))
        (
            matmul(
                a,
                b,
                epilog=Epilog.BIAS,
                epilog_inputs={"not_a_bias": cupy.full((4, 1), np.float32(0.8))},
            ),
        )


@pytest.mark.parametrize("epilog", epilogs_with_bias)
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "mismatch_batch_size",
            "a_shape": (2, 4, 5),
            "b_shape": (2, 5, 8),
            "bias_shape": (3, 4, 1),
            "error_pattern": "batch dimensions of the bias.*must match",
            "make_bias": lambda shape: cupy.full(shape, np.float32(0.8)),
            "min_cublas_version": 11703,
        },
        {
            "name": "mismatched_batch_axis_order",
            "a_shape": (2, 3, 4, 5),
            "b_shape": (2, 3, 5, 8),
            "bias_shape": (2, 3, 4, 1),
            "error_pattern": "batch axis order of the bias.*must match",
            "make_bias": lambda shape: cupy.lib.stride_tricks.as_strided(
                cupy.full((2, 3, 4, 1), np.float32(0.8)), shape=shape, strides=(4, 12, 1, 4)
            ),
            "min_cublas_version": 11703,
        },
        {
            "name": "non_tileable_batch",
            "a_shape": (2, 3, 4, 5),
            "b_shape": (2, 3, 5, 8),
            "bias_shape": (2, 3, 4, 1),
            "error_pattern": "not supported because it is not tileable",
            "make_bias": lambda shape: cupy.lib.stride_tricks.as_strided(
                cupy.full(shape, np.float32(0.8)), shape=shape, strides=(16, 4, 1, 4)
            ),
            "min_cublas_version": 11703,
        },
        {
            "name": "invalid_stride",
            "a_shape": (4, 5),
            "b_shape": (5, 8),
            "bias_shape": (4, 2),
            "error_pattern": "stride of the bias.*must be 1",
            "make_bias": lambda shape: cupy.lib.stride_tricks.as_strided(
                cupy.full(shape, np.float32(0.8)), shape=(4, 1), strides=(2, 1)
            ),
            "min_cublas_version": 11501,
        },
    ],
)
def test_invalid_bias(epilog, test_case):
    skip_if_cublas_before(test_case["min_cublas_version"])

    a = sample_float_tensor(test_case["a_shape"])
    b = sample_float_tensor(test_case["b_shape"])
    bias = test_case["make_bias"](test_case["bias_shape"])
    with pytest.raises(ValueError, match=test_case["error_pattern"]):
        matmul(a, b, epilog=epilog, epilog_inputs={"bias": bias})


@pytest.mark.parametrize("epilog", [Epilog.DRELU, Epilog.DGELU])
@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "wrong_dtype",
            "a_shape": (16, 32),
            "b_shape": (32, 16),
            "error_pattern": r"The dtype of the .* auxiliary input .* must be",
            "make_aux_input": lambda aux_shape, expected_dtype: cupy.zeros(
                aux_shape, dtype="float16" if expected_dtype == "uint8" else "uint8"
            ),
            "min_cublas_version": 111103,
        },
        {
            "name": "wrong_mm_shape",
            "a_shape": (16, 32),
            "b_shape": (32, 16),
            "error_pattern": r"The auxiliary epilog input .* must have the MM shape",
            "make_aux_input": lambda aux_shape, expected_dtype: cupy.zeros(
                (aux_shape[0] + 8, aux_shape[1] + 8), dtype=expected_dtype
            ),
            "min_cublas_version": 111103,
        },
        {
            "name": "wrong_stride",
            "a_shape": (16, 32),
            "b_shape": (32, 16),
            "error_pattern": (
                r"The stride of the .* auxiliary input .* must be 1 along the dimension "
                r".* which corresponds to the M dimension"
            ),
            "make_aux_input": lambda aux_shape, expected_dtype: cupy.lib.stride_tricks.as_strided(
                cupy.zeros(aux_shape[0] * aux_shape[1] * 2, dtype=expected_dtype),
                shape=aux_shape,
                strides=(2, aux_shape[0]),  # Wrong M stride (should be 1)
            ),
            "min_cublas_version": 111103,
        },
        {
            "name": "batch_shape_mismatch",
            "a_shape": (2, 3, 16, 32),
            "b_shape": (2, 3, 32, 16),
            "error_pattern": (
                r"The batch dimensions of the .* auxiliary input .* must match with that "
                r"of the matrix multiplication definition"
            ),
            "make_aux_input": lambda aux_shape, expected_dtype: cupy.zeros(
                (2, 4, aux_shape[-2], aux_shape[-1]), dtype=expected_dtype
            ),
            "min_cublas_version": 111103,
        },
        {
            "name": "batch_axis_order_mismatch",
            "a_shape": (2, 3, 16, 32),
            "b_shape": (2, 3, 32, 16),
            "error_pattern": (
                r"The batch axis order of the .* auxiliary input .* must match with that "
                r"of the other operands"
            ),
            "make_aux_input": lambda aux_shape, expected_dtype: cupy.lib.stride_tricks.as_strided(
                cupy.zeros(aux_shape[0] * aux_shape[1] * aux_shape[2] * aux_shape[3] * 2, dtype=expected_dtype),
                shape=aux_shape,
                # Swapped batch strides
                strides=(aux_shape[2] * aux_shape[3], aux_shape[0] * aux_shape[2] * aux_shape[3], 1, aux_shape[2]),
            ),
            "min_cublas_version": 111103,
        },
        {
            "name": "non_tileable_batch",
            "a_shape": (2, 3, 16, 32),
            "b_shape": (2, 3, 32, 16),
            "error_pattern": (
                r"The batch layout for .* auxiliary input .* is currently not supported "
                r"because it is not tileable"
            ),
            "make_aux_input": lambda aux_shape, expected_dtype: cupy.lib.stride_tricks.as_strided(
                cupy.zeros(aux_shape[0] * aux_shape[1] * aux_shape[2] * aux_shape[3] * 10, dtype=expected_dtype),
                shape=aux_shape,
                # Irregular strides
                strides=(aux_shape[1] * aux_shape[2] * aux_shape[3] * 7, aux_shape[2] * aux_shape[3] * 5, 1, aux_shape[2]),
            ),
            "min_cublas_version": 111103,
        },
    ],
)
def test_aux_handler_validation(epilog, test_case):
    """Test validation branches in DReluAuxHandler and DGeluAuxHandler validate methods."""
    skip_if_cublas_before(test_case["min_cublas_version"], message=f"{epilog.name} not supported")

    a = sample_matrix("numpy/cupy", "float32", test_case["a_shape"], use_cuda=True)
    b = sample_matrix("numpy/cupy", "float32", test_case["b_shape"], use_cuda=True)

    # Get expected aux shape and dtype from forward pass
    if epilog == Epilog.DRELU:
        aux_key = "relu_aux"
        expected_dtype = "uint8"
        _, aux_output = matmul(a, b, epilog=Epilog.RELU_AUX)
    else:  # DGELU
        aux_key = "gelu_aux"
        expected_dtype = "float32"
        _, aux_output = matmul(a, b, epilog=Epilog.GELU_AUX)

    aux_shape = aux_output[aux_key].shape
    aux_input = test_case["make_aux_input"](aux_shape, expected_dtype)

    with pytest.raises(ValueError, match=test_case["error_pattern"]):
        matmul(a, b, epilog=epilog, epilog_inputs={aux_key: aux_input})
