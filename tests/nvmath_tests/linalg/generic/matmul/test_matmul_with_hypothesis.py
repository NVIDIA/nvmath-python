# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import collections
import logging
import typing

from hypothesis import given, assume, reproduce_failure  # noqa: F401
from hypothesis.extra.numpy import arrays, from_dtype
from hypothesis.strategies import (
    booleans,
    composite,
    integers,
    none,
    one_of,
    sampled_from,
    tuples,
    lists,
)
import numpy as np

from nvmath._internal.templates import ExecutionCPU, ExecutionCUDA
from nvmath.internal.tensor_wrapper import maybe_register_package
from nvmath.memory import _MEMORY_MANAGER
from nvmath.bindings import cublas
from nvmath.linalg._internal.typemaps import (
    NAMES_TO_DEFAULT_COMPUTE_TYPE,
    CUBLAS_COMPUTE_TYPE_TO_NAME,
)
from nvmath.linalg.generic import (
    DiagonalMatrixQualifier,
    GeneralMatrixQualifier,
    HermitianMatrixQualifier,
    matmul,
    MatmulOptions,
    MatrixQualifier,
    matrix_qualifiers_dtype,
    SymmetricMatrixQualifier,
    TriangularMatrixQualifier,
)

from nvmath_tests.helpers import nvmath_seed
from ...utils import get_absolute_tolerance
from . import CUBLAS_AVAILABLE, NVPL_AVAILABLE


AVAILABLE_TENSOR_LIBRARIES: list[str] = ["numpy"]


try:
    import cupy as cp

    maybe_register_package("cupy")

    AVAILABLE_TENSOR_LIBRARIES.append("cupy")
except ModuleNotFoundError:
    pass

try:
    import torch

    maybe_register_package("torch")

    AVAILABLE_TENSOR_LIBRARIES.append("torch-cpu")
    AVAILABLE_TENSOR_LIBRARIES.append("torch-gpu")
except ImportError:
    pass


def compare_result(*, res, ref):
    np.testing.assert_allclose(
        actual=res,
        desired=ref,
        equal_nan=True,
        rtol=(1e-02 if res.dtype == np.float16 else 2e-05),
        atol=2 * get_absolute_tolerance(ref),
    )  # type: ignore


def verify_result(a, b, c, result_c, alpha, beta, qualifiers: typing.Sequence[MatrixQualifier]):
    if (len(qualifiers) > 0 and TriangularMatrixQualifier.is_valid(qualifiers[0])) or (
        len(qualifiers) > 1 and TriangularMatrixQualifier.is_valid(qualifiers[1])
    ):
        # C is not added to A@B for trmm, so we just set beta to 0.0
        beta = 0.0

    if (len(qualifiers) > 0 and DiagonalMatrixQualifier.is_valid(qualifiers[0])) or (
        len(qualifiers) > 1 and DiagonalMatrixQualifier.is_valid(qualifiers[1])
    ):
        # alpha/beta are not used by dgmm, so we just set alpha to None and beta to 0.0
        alpha = None
        beta = 0.0

    if len(qualifiers) > 0 and DiagonalMatrixQualifier.is_valid(qualifiers[0]):
        a = np.diagflat(a[:: qualifiers[0]["incx"]])

    if len(qualifiers) > 1 and DiagonalMatrixQualifier.is_valid(qualifiers[1]):
        b = np.diagflat(b[:: qualifiers[1]["incx"]])

    if len(qualifiers) > 2 and DiagonalMatrixQualifier.is_valid(qualifiers[2]):
        c = np.diagflat(c[:: qualifiers[2]["incx"]])

    if len(qualifiers) > 0 and qualifiers[0]["conjugate"]:
        a = np.conjugate(a)

    if len(qualifiers) > 1 and qualifiers[1]["conjugate"]:
        b = np.conjugate(b)

    if len(qualifiers) > 2 and qualifiers[2]["conjugate"]:
        c = np.conjugate(c)

    logging.debug("Reference matrix A is \n%s", a)
    logging.debug("Reference matrix B is \n%s", b)
    logging.debug("Reference matrix C is \n%s", c)

    possible_dtype = CUBLAS_COMPUTE_TYPE_TO_NAME[NAMES_TO_DEFAULT_COMPUTE_TYPE[(str(a.dtype), str(b.dtype))]]
    compute_dtype = possible_dtype[1] if np.iscomplexobj(a) else possible_dtype[0]
    ab = (
        np.matmul(a, b, dtype=compute_dtype)
        if alpha is None
        else np.matmul(np.multiply(alpha, a, dtype=compute_dtype), b, dtype=compute_dtype)
    )
    ref_c = ab if c is None else np.add(ab, np.multiply(c, beta, dtype=compute_dtype), dtype=compute_dtype)

    result_c_ = result_c[0] if isinstance(result_c, tuple) else result_c
    logging.debug("Reference result is \n%s", ref_c)
    logging.debug("Actual    result is \n%s", result_c_)
    compare_result(res=result_c_, ref=ref_c.astype(a.dtype))


problem_size_mnk = integers(min_value=1, max_value=256)

options_blocking_values = [True, "auto"]
options_allocator_values = [
    None,
    _MEMORY_MANAGER["_raw"](0, logging.getLogger()),
    _MEMORY_MANAGER["cupy"](0, logging.getLogger()),
    _MEMORY_MANAGER["torch"](0, logging.getLogger()) if "torch" in _MEMORY_MANAGER else None,
]

# FIXME: Add integer types to tests
ab_type_values = [
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]

MatmulInputs = collections.namedtuple(
    "MatmulInputs",
    [
        "a",
        "b",
        "c",
        "m",
        "n",
        "k",
        "ab_type",
        "beta",
        "alpha",
        "qualifiers",
        "batches",
    ],
)


def notNone(x):
    return x is not None


@composite
def matrix_qualifiers(draw):
    qual = draw(
        sampled_from(
            [
                GeneralMatrixQualifier,
                HermitianMatrixQualifier,
                SymmetricMatrixQualifier,
                TriangularMatrixQualifier,
                DiagonalMatrixQualifier,
            ]
        )
    )
    kwargs = {"conjugate": draw(booleans())}
    if qual not in (GeneralMatrixQualifier, DiagonalMatrixQualifier):
        kwargs["uplo"] = draw(sampled_from([cublas.FillMode.LOWER, cublas.FillMode.UPPER]))
    if qual is TriangularMatrixQualifier:
        kwargs["diag"] = draw(sampled_from(cublas.DiagType))
    if qual is DiagonalMatrixQualifier:
        kwargs["incx"] = draw(sampled_from([+1, -1]))
    return qual.create(**kwargs)


def enforce_matrix_qualifiers(A: np.ndarray, qualifier: MatrixQualifier | None) -> np.ndarray:
    """For random valued A, force the values of A to be an example of the qualifier."""
    if GeneralMatrixQualifier.is_valid(qualifier):
        pass
    elif HermitianMatrixQualifier.is_valid(qualifier):
        assert A.shape[-2] == A.shape[-1], "Hermitian matrices are square."
        A = 0.5 * (A + np.conj(np.swapaxes(A, -2, -1)))
        assert np.array_equal(A, np.conj(np.swapaxes(A, -2, -1))), "A is not Hermitian."
    elif SymmetricMatrixQualifier.is_valid(qualifier):
        assert A.shape[-2] == A.shape[-1], "Symmetric matrices are square."
        A = np.tril(A) + np.triu(np.swapaxes(A, -2, -1), 1)
        assert np.array_equal(A, np.swapaxes(A, -2, -1)), "A is not symmetric."
    elif TriangularMatrixQualifier.is_valid(qualifier):
        assert A.shape[-2] == A.shape[-1], "Triangular matrices are square."
        if qualifier["diag"] == cublas.DiagType.UNIT:
            # A = A - A * np.identity(A.shape[-1]) + np.identity(A.shape[-1])
            A[..., np.identity(A.shape[-1], dtype=bool)] = 1.0
            assert np.all(np.diagonal(A, offset=0, axis1=-2, axis2=-1) == 1.0), np.diagonal(A, offset=0, axis1=-2, axis2=-1)
        match qualifier["uplo"]:
            case cublas.FillMode.UPPER:
                A = np.triu(A)
                assert np.all(A[np.tril(np.ones_like(A, dtype=np.bool_), k=-1)] == 0.0)
            case cublas.FillMode.LOWER:
                A = np.tril(A)
                assert np.all(A[np.triu(np.ones_like(A, dtype=np.bool_), k=+1)] == 0.0)
            case _:
                raise ValueError(f"{qualifier['uplo']} is not UPPER or LOWER.")
    elif DiagonalMatrixQualifier.is_valid(qualifier):
        assert len(A.shape) == 1, "Diagonal matrix should be vector."
    else:
        raise ValueError(f"{qualifier} describes an unknown matrix type.")
    return A


def destroy_unreferenced_matrix(A: np.ndarray, qualifier: MatrixQualifier | None) -> np.ndarray:
    """Destroy information in the unreferenced portion of the matrix."""
    nan_array = np.empty((1,), A.dtype)
    nan_array[0] = (np.nan + 1j * np.nan) if np.iscomplexobj(A) else np.nan
    nan_value = nan_array[0]
    if GeneralMatrixQualifier.is_valid(qualifier) or DiagonalMatrixQualifier.is_valid(qualifier) or A.size <= 1:
        return A
    if (
        HermitianMatrixQualifier.is_valid(qualifier)
        or TriangularMatrixQualifier.is_valid(qualifier)
        or SymmetricMatrixQualifier.is_valid(qualifier)
    ):
        match qualifier["uplo"]:
            case cublas.FillMode.LOWER:
                A = np.where(np.tril(np.ones_like(A, dtype=np.bool_)), A, nan_value)
            case cublas.FillMode.UPPER:
                A = np.where(np.triu(np.ones_like(A, dtype=np.bool_)), A, nan_value)
            case _:
                raise ValueError(f"{qualifier['uplo']} is not UPPER or LOWER.")
    if TriangularMatrixQualifier.is_valid(qualifier):  # noqa: SIM102
        if qualifier["diag"] == cublas.DiagType.UNIT:
            A = np.where(np.identity(A.shape[-1], dtype=np.bool_), nan_value, A)
            np.testing.assert_equal(actual=np.diagonal(A, offset=0, axis1=-2, axis2=-1), desired=nan_value)
    return A


@composite
def batch_strategy(draw):
    """Generate three tuples of ints which represent valid batching dimensions for a,b,c."""
    batch_shape: tuple[int] = tuple(draw(lists(integers(min_value=1, max_value=4), min_size=1, max_size=4)))
    # () is first because booleans() shrinks to True
    a_batch = () if draw(booleans()) else batch_shape
    b_batch = () if draw(booleans()) else batch_shape
    # Use truthy check here. c_batch must be the larger of a,b batch
    c_batch = batch_shape if a_batch or b_batch else ()
    return a_batch, b_batch, c_batch


@composite
def matrix_multiply_arrays(draw):
    k = draw(problem_size_mnk)
    # Let k be random and then let m,n depend on whether A,B are square matrices
    qualifiers = np.empty(3, dtype=matrix_qualifiers_dtype)
    qualifiers[0] = draw(matrix_qualifiers())
    qualifiers[1] = draw(matrix_qualifiers())
    if GeneralMatrixQualifier.is_valid(qualifiers[0]):
        m = draw(one_of(none(), problem_size_mnk))
    else:
        m = k
    if GeneralMatrixQualifier.is_valid(qualifiers[1]):
        n = draw(one_of(none(), problem_size_mnk))
    else:
        n = k
    ab_type = draw(sampled_from(ab_type_values))
    if HermitianMatrixQualifier.is_valid(qualifiers[0]) or HermitianMatrixQualifier.is_valid(qualifiers[1]):
        assume(np.iscomplexobj(ab_type()))

    a_shape = (k,) if (m is None or DiagonalMatrixQualifier.is_valid(qualifiers[0])) else (m, k)
    b_shape = (k,) if (n is None or DiagonalMatrixQualifier.is_valid(qualifiers[1])) else (k, n)
    c_shape = (m, n)
    if (m is None and not DiagonalMatrixQualifier.is_valid(qualifiers[0])) and (
        n is None and not DiagonalMatrixQualifier.is_valid(qualifiers[1])
    ):
        c_shape = ()
    elif m is None and not DiagonalMatrixQualifier.is_valid(qualifiers[0]):
        c_shape = (n,)
    elif n is None and not DiagonalMatrixQualifier.is_valid(qualifiers[1]):
        c_shape = (m,)

    # TODO: Uncomment when batched inputs are supported
    # if len(a_shape) == 2 and len(b_shape) == 2 and len(c_shape) == 2:
    #     a_batch, b_batch, c_batch = draw(batch_strategy())
    #     a_shape = a_batch + a_shape
    #     b_shape = b_batch + b_shape
    #     c_shape = c_batch + c_shape
    # else:
    a_batch, b_batch, c_batch = (), (), ()

    # Generate data in range [0, 5] to match sample_matrix() from utils
    # Only non-negative reals to avoid catastrophic cancellation
    element_properties: dict[str, typing.Any] = {
        "allow_infinity": False,
        "allow_nan": False,
        "allow_subnormal": False,
        "max_magnitude": np.sqrt(50),
        "min_magnitude": 0,
        "max_value": 5,
        "min_value": 0,
    }
    # NOTE: It is unfeasible for hypothesis to explore a parameter space where
    # all elements of the input arrays are unique, so most of the time, arrays
    # contain just a few unique values
    a = draw(
        arrays(
            dtype=ab_type,
            shape=a_shape,
            elements=element_properties,
        )
    )
    b = draw(
        arrays(
            dtype=ab_type,
            shape=b_shape,
            elements=element_properties,
        )
    )

    # Type promotion can happen unintentionally when enforcing matrix structure.
    a = enforce_matrix_qualifiers(a, qualifier=qualifiers[0]).astype(ab_type)
    b = enforce_matrix_qualifiers(b, qualifier=qualifiers[1]).astype(ab_type)

    # The generic API does not support broadcasting of C, so the shape of must match the
    # output of the matmul exactly.
    c = draw(
        one_of(
            none(),
            arrays(dtype=ab_type, shape=c_shape, elements=element_properties),
        )
    )
    if c is None:
        qualifiers = qualifiers[:2]
    else:
        qualifiers[2] = GeneralMatrixQualifier.create()

    beta = None if c is None else draw(from_dtype(dtype=np.dtype(ab_type), **element_properties))
    alpha = draw(one_of(none(), from_dtype(dtype=np.dtype(ab_type), **element_properties)))

    assume(np.all(np.isfinite(a)))
    assume(np.all(np.isfinite(b)))
    assume(c is None or np.all(np.isfinite(c)))
    assert c is None or c.shape in [c_batch + (m, n), (m, n), (m,), (n,), ()]
    assert a.shape in [a_batch + (m, k), (m, k), (k,)]
    assert b.shape in [b_batch + (k, n), (k, n), (k,)]
    return MatmulInputs(
        a=a,
        b=b,
        c=c,
        m=m,
        n=n,
        k=k,
        ab_type=ab_type,
        beta=beta,
        alpha=alpha,
        qualifiers=qualifiers,
        batches=(a_batch, b_batch, c_batch),
    )


@composite
def options_strategy(draw):
    return MatmulOptions(
        blocking=draw(sampled_from(options_blocking_values)),
        allocator=draw(sampled_from(options_allocator_values)),
        inplace=draw(booleans()),
    )


@nvmath_seed()
@given(
    input_arrays=matrix_multiply_arrays(),
    order=tuples(
        sampled_from(["F", "C"]),
        sampled_from(["F", "C"]),
        sampled_from(["F", "C"]),
    ),
    options=one_of(
        none(),
        options_strategy(),
    ),
    execution=sampled_from(
        [
            # None,  # Cannot test None because not all test envs have CPU deps
            *((ExecutionCUDA(),) if CUBLAS_AVAILABLE else ()),
            *((ExecutionCPU(),) if NVPL_AVAILABLE else ()),
        ]
    ),
    preferences=one_of(
        none(),
    ),
    tensor_library=sampled_from(AVAILABLE_TENSOR_LIBRARIES),
)
def test_matmul(input_arrays, order, options, execution, preferences, tensor_library):
    """Call nvmath.linalg.generic.matmul() with valid inputs."""
    a, b, c, m, n, k, ab_type, beta, alpha, qualifiers, batches = input_arrays

    if c is None and options is not None and options.inplace:
        # Cannot have inplace operation when c is None
        return

    ax = destroy_unreferenced_matrix(a, qualifiers[0])
    bx = destroy_unreferenced_matrix(b, qualifiers[1])

    ax = np.array(ax, order=order[0])
    bx = np.array(bx, order=order[1])
    c = None if c is None else np.array(c, order=order[2])

    match tensor_library:
        case "cupy":
            d_a = cp.asarray(ax, order=order[0])
            d_b = cp.asarray(bx, order=order[1])
            d_c = None if c is None else cp.asarray(c, order=order[2])
        case "torch-cpu":
            d_a = torch.tensor(ax, device="cpu")
            d_b = torch.tensor(bx, device="cpu")
            d_c = None if c is None else torch.tensor(c, device="cpu")
        case "torch-gpu":
            d_a = torch.tensor(ax, device="cuda")
            d_b = torch.tensor(bx, device="cuda")
            d_c = None if c is None else torch.tensor(c, device="cuda")
        case _:
            d_a = np.copy(ax, order=order[0])
            d_b = np.copy(bx, order=order[1])
            d_c = None if c is None else np.copy(c, order=order[2])

    try:
        result_c = matmul(
            d_a,
            d_b,
            c=d_c,
            alpha=alpha,
            beta=beta,
            execution=execution,
            options=options,
            qualifiers=qualifiers,
        )
    except ValueError as error:
        message = str(error)
        if "No available generic matrix multiplication matches the provided matrices" in message and (
            not GeneralMatrixQualifier.is_valid(qualifiers[0]) and not GeneralMatrixQualifier.is_valid(qualifiers[1])
        ):
            logging.warning("Hypothesis ignored the following error: %s", message)
            return
        if (
            "No BLAS compatible view of the operands was found" in message
            or "Operations on the non-triangular operand" in message
            or "Operations on the non-hermitian/non-symmetric operands" in message
            or "Transpose on operand A is not supported" in message
        ):
            logging.warning("Hypothesis ignored the following error: %s", message)
            return
        if "Operations on the non-diagonal operands A,C are not supported" in message and (
            order[2] == "C"
            or (
                (DiagonalMatrixQualifier.is_valid(qualifiers[0]) and order[1] == "C")
                or (DiagonalMatrixQualifier.is_valid(qualifiers[1]) and order[0] == "C")
            )
        ):
            return
        if "Conjugate-Transpose on operand X is not supported" in message and (
            (DiagonalMatrixQualifier.is_valid(qualifiers[0]) and qualifiers[0]["conjugate"])
            or (DiagonalMatrixQualifier.is_valid(qualifiers[1]) and qualifiers[1]["conjugate"])
        ):
            return
        if "is not valid for batching" in message and (order[0] == "F" or order[1] == "F" or order[2] == "F"):
            return
        if "is unsupported" in message:
            logging.warning("Hypothesis ignored the following error: %s", message)
            return
        if "dgmm() is an unknown NVPL BLAS function" in message and (
            DiagonalMatrixQualifier.is_valid(qualifiers[0]) or DiagonalMatrixQualifier.is_valid(qualifiers[1])
        ):
            return
        if "Unsupported layout" in message and (
            ax.dtype.itemsize not in ax.strides[-2:]
            or bx.dtype.itemsize not in bx.strides[-2:]
            or (c is not None and c.dtype.itemsize not in c.strides[-2:])
        ):
            # The provided matrices are probably batched,
            # and the layout is incompatible with batching
            return
        raise error
    except NotImplementedError as error:
        message = str(error)
        logging.warning("Hypothesis ignored the following error: %s", message)
        return

    assert result_c.dtype is d_a.dtype, f"Result ({result_c}) and input ({a.dtype}) types should match!"
    if options is not None and options.inplace:
        assert result_c is d_c, "For inplace operations, the result should be the same object as operand c."

    match tensor_library:
        case "cupy":
            result_c = cp.asnumpy(result_c)
        case "torch-cpu" | "torch-gpu":
            result_c = result_c.cpu().detach().numpy()
        case _:
            pass

    verify_result(a, b, c, result_c, alpha, beta, qualifiers)
