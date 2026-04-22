# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "Solver",
    "CholeskySolver",
    "LUSolver",
    "LUPivotSolver",
    "TriangularSolver",
    "QRFactorize",
    "LQFactorize",
    "QRMultiply",
    "LQMultiply",
    "LeastSquaresSolver",
    "compile_solver_execute",
]

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal

import numpy as np

from nvmath._utils import get_nvrtc_version
from nvmath.bindings import mathdx
from nvmath.internal.utils import docstring_decorator

from .common import SHARED_DEVICE_DOCSTRINGS, check_code_type, parse_code_type, parse_sm
from .common_backend import get_isa_version, get_lto
from .common_cuda import (
    Code,
    ComputeCapability,
)
from .cusolverdx_backend import (
    _ENABLE_CUSOLVERDX_0_3,
    ALLOWED_ARRANGEMENT,
    ALLOWED_CUSOLVERDX_FUNCTIONS,
    ALLOWED_DATA_TYPE,
    ALLOWED_DIAG,
    ALLOWED_EXECUTION,
    ALLOWED_FILL_MODE,
    ALLOWED_REAL_NP_TYPES,
    ALLOWED_SIDE,
    ALLOWED_TRANSPOSE_MODE,
    CUSOLVERDX_0_3_ALLOWED_FUNCTIONS,
    DIAG_SUPPORTED_FUNCTIONS,
    FILL_MODE_SUPPORTED_FUNCTIONS,
    JOB_SUPPORT_MAP,
    JOB_SUPPORTED_FUNCTIONS,
    SIDE_SUPPORTED_FUNCTIONS,
    generate_code,
    generate_SOLVER,
    get_int_traits,
    get_str_trait,
    validate,
)
from .types import Complex, complex64, complex128

# ==========================
# docs
# ==========================

SOLVER_DOCSTRING_SIZE_BASE = """Problem size specified as a sequence of 1 to 3 elements:
``(M,)`` (treated as ``(M, M, 1)``), ``(M, N)`` (treated as ``(M, N, 1)``), or ``(M, N, K)``.""".replace("\n", " ")

SOLVER_DOCSTRING = {
    "function": f"""Solver function to be executed on execute() method. List of available options:
{", ".join(f"``'{k}'`` ({v})" for k, v in ALLOWED_CUSOLVERDX_FUNCTIONS.items())}.
Functions {", ".join(f"``'{k}'``" for k in CUSOLVERDX_0_3_ALLOWED_FUNCTIONS)} require libmathdx 0.3.2 or later.""".replace(
        "\n", " "
    ),
    #
    "size": f"""{SOLVER_DOCSTRING_SIZE_BASE} Please refer to cuSOLVERDx functionalities for detailed meaning.
""".replace("\n", " "),
    #
    "arrangement": f"""Storage layout for matrices A and B, specified as a sequence of 2 elements ``(arr_A, arr_B)``.
Each element can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_ARRANGEMENT)}.
Defaults to ``("col_major", "col_major")``.""".replace("\n", " "),
    #
    "transpose_mode": f"""Transpose mode of matrix A. Refer to cuSOLVERDx documentation,
as some functions do not support this option.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_TRANSPOSE_MODE)}.
Defaults to ``'non_transposed'``.""".replace("\n", " "),
    #
    "side": f"""Side of matrix A in a multiplication operation.
Required and supported only by functions: {", ".join(f"``'{v}'``" for v in SIDE_SUPPORTED_FUNCTIONS)}.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_SIDE)}.""".replace("\n", " "),
    #
    "diag": f"""Indicates whether the diagonal elements of matrix A are ones or not.
Required and supported only by {", ".join(f"``'{v}'``" for v in DIAG_SUPPORTED_FUNCTIONS)} functions.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_DIAG)}.""".replace("\n", " "),
    #
    "fill_mode": f"""Indicates which part of matrix A is filled and should be used by function.
Required and supported only by functions: {", ".join(f"``'{v}'``" for v in FILL_MODE_SUPPORTED_FUNCTIONS)}.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_FILL_MODE)}.""".replace("\n", " "),
    #
    "job": f"""Job type for eigenvalue computation.
Required and supported only by functions: {", ".join(f"``'{v}'``" for v in JOB_SUPPORTED_FUNCTIONS)}.
For ``'htev'``: {", ".join(f"``'{v}'``" for v in JOB_SUPPORT_MAP["htev"])}.
For ``'heev'``: {", ".join(f"``'{v}'``" for v in JOB_SUPPORT_MAP["heev"])}.
Requires libmathdx 0.3.2 or later.""".replace("\n", " "),
    #
    "batches_per_block": """Number of batches to compute in parallel in a single CUDA block.
Can be a non-zero integer or the string ``'suggested'`` for automatic selection of an optimal value.
We recommend using 1 for matrix A size larger than or equal to 16 x 16,
and using ``'suggested'`` for smaller sizes to achieve optimal performance.
Defaults to 1.""".replace("\n", " "),
    #
    "data_type": f"""The data type of the input matrices,
can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_DATA_TYPE)}.
Defaults to ``'real'``.""".replace("\n", " "),
    #
    "leading_dimensions": """The leading dimensions for input matrices A and B,
specified as a sequence of 2 elements (``lda``, ``ldb``) or ``None``.
If not provided, it will be automatically deduced from ``size`` and ``arrangement``.
""".replace("\n", " "),
    #
    "block_dim": """The block dimension for launching the CUDA kernel,
specified as a 1 to 3 integer sequence (x, y, z) where missing dimensions are assumed to be 1.
Can be a sequence of 1 to 3 positive integers, the string ``'suggested'``
for optimal value selection, or ``None`` for the default value.""".replace("\n", " "),
    #
    "execution": f"""A string specifying the execution method.
Supported values: {", ".join(f"``'{v}'``" for v in ALLOWED_EXECUTION)}.""".replace("\n", " "),
    #
    "precision": f"""The computation precision specified as a numpy float dtype.
Currently supports: {", ".join(f"``numpy.{v.__name__}``" for v in ALLOWED_REAL_NP_TYPES)}.""".replace("\n", " "),
    #
    "sm": SHARED_DEVICE_DOCSTRINGS["sm"],
}

# ==========================
# Solver class
# ==========================


@docstring_decorator(SOLVER_DOCSTRING, skip_missing=False)
class Solver:
    """
    A class that encapsulates a partial dense matrix factorization and solve
    device function.

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, rows, cols)``
        with strides ``(ld * cols, 1, ld)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, rows, cols)``
        with strides ``(ld * rows, ld, 1)``

    Args:
        function (str): {function}

        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        arrangement (Sequence[str], optional): {arrangement}

        transpose_mode (str, optional): {transpose_mode}

        side (str, optional): {side}

        diag (str, optional): {diag}

        fill_mode (str, optional): {fill_mode}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

        job (str, optional): {job}

    See Also:
        The attributes of this class provide a 1:1
        mapping with the CUDA C++ cuSOLVERDx APIs.
        For further details, please refer to :cusolverdx_doc:`cuSOLVERDx documentation
        <index.html>`.
    """

    def __init__(
        self,
        function: str,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        *,
        sm=None,
        arrangement: Sequence[str] | None = None,
        transpose_mode: str | None = None,
        side: str | None = None,
        diag: str | None = None,
        fill_mode: str | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
        job: str | None = None,
    ):
        if get_nvrtc_version() < (12, 6, 85):
            raise RuntimeError("cuSOLVERDx requires CUDA Toolkit 12.6 Update 3 or later.")

        sm = parse_sm(sm)

        validate(
            function=function,
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            arrangement=arrangement,
            transpose_mode=transpose_mode,
            side=side,
            diag=diag,
            fill_mode=fill_mode,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=leading_dimensions,
            block_dim=block_dim,
            job=job,
        )

        if len(size) == 1:
            size = (size[0], size[0], 1)
        elif len(size) == 2:
            size = (size[0], size[1], 1)
        else:
            size = (size[0], size[1], size[2])

        if block_dim is not None and block_dim != "suggested":
            if len(block_dim) == 1:
                block_dim = (block_dim[0], 1, 1)
            elif len(block_dim) == 2:
                block_dim = (block_dim[0], block_dim[1], 1)
            else:
                block_dim = (block_dim[0], block_dim[1], block_dim[2])

        self._function = function
        self._size = size
        self._precision = precision
        self._execution = execution
        self._sm = sm
        self._arrangement = tuple(arrangement) if arrangement is not None else None
        self._transpose_mode = transpose_mode
        self._side = side
        self._diag = diag
        self._fill_mode = fill_mode
        self._data_type = data_type
        self._job = job

        self._batches_per_block = batches_per_block if batches_per_block != "suggested" else None
        self._leading_dimensions = tuple(leading_dimensions) if leading_dimensions is not None else None
        self._block_dim = block_dim if block_dim != "suggested" and block_dim is not None else None

        if batches_per_block == "suggested" or block_dim == "suggested":
            # Update suggested fields
            traits = _SolverTraits(self)

            if batches_per_block == "suggested":
                self._batches_per_block = traits.suggested_batches_per_block

            if block_dim == "suggested":
                self._block_dim = traits.suggested_block_dim

    # ==========================
    # Operator properties
    # ==========================

    @property
    def function(self) -> str:
        return self._function

    @property
    def size(self) -> tuple[int, int, int]:
        return self._size

    @property
    def m(self) -> int:
        m, _, _ = self.size
        return m

    @property
    def n(self) -> int:
        _, n, _ = self.size
        return n

    @property
    def k(self) -> int:
        _, _, k = self.size
        return k

    @property
    def precision(self) -> type[np.floating]:
        return self._precision

    @property
    def execution(self) -> str:
        return self._execution

    @property
    def sm(self) -> ComputeCapability:
        return self._sm

    @property
    def arrangement(self) -> Sequence[str]:
        if self._arrangement is None:
            # TODO: replace with libmathdx trait when supported
            return ("col_major", "col_major")
        if len(self._arrangement) == 1:
            return (self._arrangement[0], "col_major")
        return self._arrangement

    @property
    def a_arrangement(self) -> str:
        arr_a, _ = self.arrangement
        return arr_a

    @property
    def b_arrangement(self) -> str:
        _, arr_b = self.arrangement
        return arr_b

    @property
    def transpose_mode(self) -> str | None:
        return self._transpose_mode

    @property
    def side(self) -> str | None:
        return self._side

    @property
    def diag(self) -> str | None:
        return self._diag

    @property
    def fill_mode(self) -> str | None:
        return self._fill_mode

    @property
    def batches_per_block(self) -> int:
        if self._batches_per_block is None:
            # TODO: replace with libmathdx trait when supported
            return 1
        return self._batches_per_block

    @property
    def data_type(self) -> str:
        if self._data_type is None:
            # TODO: replace with libmathdx trait when supported
            return "real"
        return self._data_type

    @property
    def leading_dimensions(self) -> tuple:
        if self._leading_dimensions is None:
            # TODO: replace with libmathdx trait when supported
            return self._calculate_default_leading_dimensions()

        if len(self._leading_dimensions) == 1:
            return (self._leading_dimensions[0], self._calculate_default_leading_dimensions()[1])
        return self._leading_dimensions

    @property
    def lda(self) -> int:
        lda, _ = self.leading_dimensions
        return lda

    @property
    def ldb(self) -> int:
        _, ldb = self.leading_dimensions
        return ldb

    @property
    def block_dim(self) -> tuple:
        if self._block_dim is None:
            return self._traits.block_dim
        return self._block_dim

    @property
    def block_size(self) -> int:
        return self.block_dim[0] * self.block_dim[1] * self.block_dim[2]

    @property
    def job(self) -> str | None:
        return self._job

    # ==========================
    # Trait properties
    # ==========================

    @cached_property
    def _traits(self):
        return _SolverTraits(self)

    @property
    def workspace_size(self) -> int:
        if not _ENABLE_CUSOLVERDX_0_3:
            raise RuntimeError("workspace size requires libmathdx 0.3.2")

        return self._traits.workspace_size

    @property
    def value_type(self) -> type[np.floating] | Complex:
        if self.data_type == "complex":
            return complex64 if self.precision == np.float32 else complex128
        return self.precision

    @property
    def info_type(self) -> type[np.signedinteger]:
        return np.int32

    @property
    def ipiv_type(self) -> type[np.signedinteger]:
        return np.int32

    @property
    def tau_type(self) -> type[np.floating] | Complex:
        return self.value_type

    # ==========================
    # execute()
    # ==========================

    def execute(*args):
        raise RuntimeError("execute is a device function and cannot be called on the host.")

    # ==========================
    # Private methods
    # ==========================

    def _generate_SOLVER(self, execution_api):
        return generate_SOLVER(
            function=self._function,
            size=self._size,
            precision=self._precision,
            execution=self._execution,
            sm=self._sm,
            arrangement=self.arrangement,  # TODO: remove property when libmathdx supports traits
            transpose_mode=self._transpose_mode,
            side=self._side,
            diag=self._diag,
            fill_mode=self._fill_mode,
            batches_per_block=self.batches_per_block,  # TODO: remove property when libmathdx supports traits
            data_type=self.data_type,  # TODO: remove property when libmathdx supports traits
            leading_dimensions=self.leading_dimensions,  # TODO: remove property when libmathdx supports traits
            block_dim=self._block_dim,
            execution_api=execution_api,
            job=self._job,
        )

    def _calculate_default_leading_dimensions(self) -> tuple:
        """
        Calculates default leading dimensions based on problem
        size (m, n, k) for different solver functions.
        """

        def calculate_ld(rows, cols, arr):
            return rows if arr == "col_major" else cols

        arr_a, arr_b = self.arrangement
        m, n, k = self.size

        match self.function:
            case (
                "potrf"
                | "potrs"
                | "posv"
                | "getrf_no_pivot"
                | "getrs_no_pivot"
                | "gesv_no_pivot"
                | "getrf_partial_pivot"
                | "getrs_partial_pivot"
                | "gesv_partial_pivot"
            ):
                return (calculate_ld(m, n, arr_a), calculate_ld(n, k, arr_b))
            case "trsm":
                assert self.side is not None
                return (m if self.side == "left" else n, calculate_ld(m, n, arr_b))
            case "geqrf" | "gelqf" | "htev" | "unglq" | "ungqr" | "heev":
                return (calculate_ld(m, n, arr_a), 0)
            case "unmqr":
                return (calculate_ld(m if self.side == "left" else n, k, arr_a), calculate_ld(m, n, arr_b))
            case "unmlq":
                return (calculate_ld(k, m if self.side == "left" else n, arr_a), calculate_ld(m, n, arr_b))
            case "gels":
                return (calculate_ld(m, n, arr_a), calculate_ld(max(m, n), k, arr_b))  # mirrors cusolverdx behaviour
            case "gtsv_no_pivot":
                return (0, calculate_ld(n, k, arr_b))
            case _:
                raise NotImplementedError(f"Function {self.function} is not supported for leading dimension calculation")


# ==========================
# Internal traits class
# ==========================


class _SolverTraits:
    def __init__(self, solver: Solver):
        h = solver._generate_SOLVER("compiled_leading_dim")

        try:
            self.suggested_batches_per_block = int(
                mathdx.cusolverdx_get_trait_int64(h.descriptor, mathdx.CusolverdxTraitType.SUGGESTED_BATCHES_PER_BLOCK)
            )

            self.suggested_block_dim = tuple(get_int_traits(h.descriptor, mathdx.CusolverdxTraitType.SUGGESTED_BLOCK_DIM, 3))
            self.block_dim = tuple(get_int_traits(h.descriptor, mathdx.CusolverdxTraitType.BLOCK_DIM, 3))

            if _ENABLE_CUSOLVERDX_0_3:
                self.workspace_size = int(
                    mathdx.cusolverdx_get_trait_int64(h.descriptor, mathdx.CusolverdxTraitType.WORKSPACE_SIZE)
                )

        except mathdx.LibMathDxError as e:
            raise RuntimeError(
                "Failed to compile the solver. This may indicate incompatible "
                f"parameter combination. Please refer to the cuSOLVERDx documentation. Details: {e}"
            ) from e


# ==========================
# Compile function
# ==========================


def compile_solver_execute(
    solver: Solver,
    code_type: Any,
    execution_api: str,
) -> tuple[Code, str]:
    code_type = parse_code_type(code_type)
    check_code_type(code_type, "cuSOLVERdx")

    h = solver._generate_SOLVER(execution_api).descriptor

    try:
        code = generate_code(h, code_type.cc)
    except mathdx.LibMathDxError as e:
        raise RuntimeError(
            "Failed to compile the solver. This may indicate incompatible "
            f"parameter combination. Please refer to the cuSOLVERDx documentation. Details: {e}"
        ) from e

    lto = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)
    symbol = get_str_trait(h, mathdx.CusolverdxTraitType.SYMBOL_NAME)

    return Code(code_type, isa_version, lto), symbol


# ==========================
# Pythonic Adapters
# ==========================


def _calculate_strides(shape: tuple[int, int], ld: int, arrangement: str) -> tuple[int, int, int]:
    return (ld * shape[1], 1, ld) if arrangement == "col_major" else (ld * shape[0], ld, 1)


class _SolverProperties:
    def __init__(self, source: Solver):
        self._properties_source = source

    @property
    def size(self) -> tuple[int, int, int]:
        return self._properties_source.size

    @property
    def m(self) -> int:
        return self._properties_source.m

    @property
    def n(self) -> int:
        return self._properties_source.n

    @property
    def k(self) -> int:
        return self._properties_source.k

    @property
    def precision(self) -> type[np.floating]:
        return self._properties_source.precision

    @property
    def execution(self) -> str:
        return self._properties_source.execution

    @property
    def sm(self) -> ComputeCapability:
        return self._properties_source.sm

    @property
    def arrangement(self) -> Sequence[str]:
        return self._properties_source.arrangement

    @property
    def a_arrangement(self) -> str:
        return self._properties_source.a_arrangement

    @property
    def b_arrangement(self) -> str:
        return self._properties_source.b_arrangement

    @property
    def batches_per_block(self) -> int:
        return self._properties_source.batches_per_block

    @property
    def data_type(self) -> str:
        return self._properties_source.data_type

    @property
    def leading_dimensions(self) -> tuple:
        return self._properties_source.leading_dimensions

    @property
    def lda(self) -> int:
        return self._properties_source.lda

    @property
    def ldb(self) -> int:
        return self._properties_source.ldb

    @property
    def block_dim(self) -> tuple:
        return self._properties_source.block_dim

    @property
    def block_size(self) -> int:
        return self._properties_source.block_size

    @property
    def value_type(self) -> type[np.floating] | Complex:
        return self._properties_source.value_type


class _LinearSolverProperties(_SolverProperties):
    def __init__(self, source: Solver):
        super().__init__(source)

    @property
    def info_type(self) -> type[np.signedinteger]:
        return self._properties_source.info_type

    @property
    def info_shape(self) -> tuple[int]:
        return (self.batches_per_block,)

    @property
    def info_strides(self) -> tuple[int]:
        return (1,)

    @property
    def a_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m, self.n)

    @property
    def b_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.n, self.k)

    def a_strides(self, *, lda: int | None = None) -> tuple[int, int, int]:
        lda = self.lda if lda is None else lda
        return _calculate_strides(self.a_shape[1:], lda, self.a_arrangement)

    def b_strides(self, *, ldb: int | None = None) -> tuple[int, int, int]:
        ldb = self.ldb if ldb is None else ldb
        return _calculate_strides(self.b_shape[1:], ldb, self.b_arrangement)

    def a_size(self, *, lda: int | None = None) -> int:
        return self.a_strides(lda=lda)[0] * self.a_shape[0]

    def b_size(self, *, ldb: int | None = None) -> int:
        return self.b_strides(ldb=ldb)[0] * self.b_shape[0]


# ==========================
# Cholesky Solver
# ==========================

ADAPTERS_API_LD_DOCSTRING = """
Note: When provided in the constructor, leading dimensions are set at compile-time.
To use runtime leading dimensions (avoiding recompilation for different leading dimensions),
provide the leading dimension parameters directly to the device methods instead.
""".replace("\n", " ")

CHOLESKY_SOLVER_DOCSTRING = SOLVER_DOCSTRING.copy()
del CHOLESKY_SOLVER_DOCSTRING["function"]
del CHOLESKY_SOLVER_DOCSTRING["transpose_mode"]
del CHOLESKY_SOLVER_DOCSTRING["side"]
del CHOLESKY_SOLVER_DOCSTRING["diag"]
del CHOLESKY_SOLVER_DOCSTRING["job"]

CHOLESKY_SOLVER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` represents the dimension of the square matrix A (``M`` x ``M``) used in factorization, ``N`` must be equal to ``M``.
``K`` represents the number of columns in the right-hand side matrix B
(dimensions ``M`` x ``K``) for the solve operation.""".replace("\n", " ")
CHOLESKY_SOLVER_DOCSTRING["leading_dimensions"] = SOLVER_DOCSTRING["leading_dimensions"] + ADAPTERS_API_LD_DOCSTRING
CHOLESKY_SOLVER_DOCSTRING["fill_mode"] = f"""Indicates which part of matrix A is filled and should be used by function.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_FILL_MODE)}.""".replace("\n", " ")


@docstring_decorator(CHOLESKY_SOLVER_DOCSTRING, skip_missing=False)
class CholeskySolver(_LinearSolverProperties):
    """
    A class that encapsulates Cholesky factorization and solve device functions
    for symmetric positive definite matrices.

    **Available operations:**

    * factorize: Computes the Cholesky factorization
      A = L @ L^H (lower) or A = U^H @ U (upper),
      where L is a lower triangular matrix and U is an upper triangular matrix.
      The choice depends on the fill_mode parameter.
    * solve: Solves the system Ax = B using a previously computed Cholesky factorization

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * M, lda, 1)``

    **For matrix B (N x K):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, N, K)``
      with strides ``(ldb * K, 1, ldb)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, N, K)``
      with strides ``(ldb * N, ldb, 1)``

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        fill_mode (str): {fill_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`factorize <get_started/functions/potrf.html>`
        * :cusolverdx_doc:`solve <get_started/functions/potrs.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        fill_mode: str,
        *,
        sm=None,
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        if fill_mode is None:
            raise ValueError("fill_mode must be provided for CholeskySolver")

        def construct(function):
            return Solver(
                function=function,
                size=size,
                precision=precision,
                execution=execution,
                sm=sm,
                arrangement=arrangement,
                fill_mode=fill_mode,
                batches_per_block=batches_per_block,
                data_type=data_type,
                leading_dimensions=leading_dimensions,
                block_dim=block_dim,
            )

        self._factorize = construct("potrf")
        self._solve = construct("potrs")

        super().__init__(self._solve)

        if self.m != self.n:
            raise ValueError("A must be a square matrix for CholeskySolver.")

    # ==========================
    # Property methods
    # ==========================

    @property
    def fill_mode(self) -> str:
        return self._solve.fill_mode

    # ==========================
    # Device function methods
    # ==========================

    def factorize(self, a, info, lda=None) -> None:
        """
        Computes the Cholesky factorization of a symmetric positive definite matrix A.

        This device function computes A = L @ L^H (if fill_mode = ``'lower'``)
        or A = U^H @ U (if fill_mode = ``'upper'``). Uses cuSOLVERDx ``'potrf'``.

        If ``lda`` is provided, uses runtime version with the specified
        leading dimension. If ``lda`` is not provided (``None``),
        uses compile-time version with
        default or constructor-provided leading dimensions.

        For more details, see: :cusolverdx_doc:`get_started/functions/potrf.html`

        Args:
            a: Pointer to an array in shared memory, storing
               the matrix according to the specified
               arrangement and leading dimension (see :meth:`__init__`).
               On entry, contains the symmetric positive definite matrix.
               On exit, contains the triangular factor L (lower) or U (upper).
            info: Pointer to a 1D array of ``int32``. On exit, ``info[batch_id] = 0``
                  indicates success for that batch, ``info[batch_id] != 0``
                  indicates the matrix is not positive definite.
            lda: Optional runtime leading dimension of matrix A.
                 If not specified, the compile-time ``lda`` is used.
        """
        raise RuntimeError("factorize is a device function and cannot be called on the host.")

    def solve(self, a, b, lda=None, ldb=None) -> None:
        """
        Solves a system of linear equations Ax = B using the Cholesky factorization.

        This device function uses the previously computed
        factorization A = L @ L^H (lower) or A = U^H @ U (upper)
        to solve the system. Uses cuSOLVERDx ``'potrs'``.

        If ``lda`` and ``ldb`` are provided, uses
        runtime version with the specified leading dimensions.
        If not provided (``None``), uses
        compile-time version with default or constructor-provided leading dimensions.

        For more details, see: :cusolverdx_doc:`get_started/functions/potrs.html`

        Args:
            a: Pointer to an array in shared memory, storing
               the triangular factor L (lower) or U (upper)
               from the Cholesky factorization, according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
            b: Pointer to an array in shared memory, storing the
               matrix according to the specified
               arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place with the solution matrix x.
            lda: Optional runtime leading dimension of matrix A.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldb: Optional runtime leading dimension of matrix B.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``ldb`` is used.
        """
        raise RuntimeError("solve is a device function and cannot be called on the host.")


# ==========================
# LU Solver
# ==========================

LU_SOLVER_DOCSTRING = SOLVER_DOCSTRING.copy()
del LU_SOLVER_DOCSTRING["function"]
del LU_SOLVER_DOCSTRING["fill_mode"]
del LU_SOLVER_DOCSTRING["side"]
del LU_SOLVER_DOCSTRING["diag"]
del LU_SOLVER_DOCSTRING["job"]

LU_SOLVER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` and ``N`` represent the dimensions of the matrix A used in factorization.
``K`` represents the number of columns in the right-hand side matrix B (dimensions ``N`` x ``K``) for the ``solve`` operation.
To use :meth:`solve`, ``N`` must be equal to ``M``,
otherwise an exception will be thrown when ``solver.solve()`` is used.""".replace("\n", " ")
LU_SOLVER_DOCSTRING["leading_dimensions"] = SOLVER_DOCSTRING["leading_dimensions"] + ADAPTERS_API_LD_DOCSTRING
LU_SOLVER_DOCSTRING["transpose_mode"] = f"""Transpose mode of matrix A for the solve operation.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_TRANSPOSE_MODE)}.
Defaults to ``'non_transposed'``.""".replace("\n", " ")


@docstring_decorator(LU_SOLVER_DOCSTRING, skip_missing=False)
class LUSolver(_LinearSolverProperties):
    """
    A class that encapsulates cuSOLVERDx LU factorization without pivoting
    and linear solver for general matrices.

    **Available operations:**

    * factorize: Computes the LU factorization A = L @ U,
      where L is a unit lower triangular matrix and U is an upper triangular matrix.
    * solve: Solves the system Ax = B using a previously computed LU factorization.

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * M, lda, 1)``

    **For matrix B (N x K):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, N, K)``
      with strides ``(ldb * K, 1, ldb)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, N, K)``
      with strides ``(ldb * N, ldb, 1)``

    .. note::
        If a nonsingular matrix A is diagonal dominant, then it is safe to factorize
        without pivoting. If a matrix is not diagonal dominant, then pivoting is usually
        required to ensure numerical stability (see :class:`LUPivotSolver`).

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        transpose_mode (str, optional): {transpose_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

          * :cusolverdx_doc:`factorize <get_started/functions/getrf.html>`
          * :cusolverdx_doc:`solve <get_started/functions/getrs.html>`

    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        *,
        sm=None,
        transpose_mode: str = "non_transposed",
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        def construct(function, transpose_mode):
            return Solver(
                function=function,
                size=size,
                precision=precision,
                execution=execution,
                sm=sm,
                arrangement=arrangement,
                batches_per_block=batches_per_block,
                data_type=data_type,
                leading_dimensions=leading_dimensions,
                block_dim=block_dim,
                transpose_mode=transpose_mode,
            )

        if transpose_mode is None:
            raise ValueError("transpose_mode must be provided for LUSolver")

        self._factorize = construct("getrf_no_pivot", "non_transposed")
        self._transpose_mode = transpose_mode
        self._solve = construct("getrs_no_pivot", transpose_mode) if self._factorize.m == self._factorize.n else None

        super().__init__(self._factorize)

    # ==========================
    # Property methods
    # ==========================

    @property
    def transpose_mode(self) -> str:
        return self._transpose_mode

    # ==========================
    # Device function methods
    # ==========================

    def factorize(self, a, info, lda=None) -> None:
        """
        Computes the LU factorization of a general matrix A without pivoting.

        This device function computes A = L @ U, where L is a unit lower triangular matrix
        and U is an upper triangular matrix. This variant is suitable for diagonally
        dominant matrices or when pivoting is not required.
        Uses cuSOLVERDx ``'getrf_no_pivot'``.

        If ``lda`` is provided, uses runtime version with the specified leading dimension.
        If ``lda`` is not provided (``None``), uses compile-time version with default
        or constructor-provided leading dimensions.

        .. note::
            The ``transpose_mode`` parameter does not affect factorization.
            This operation always treats the input matrix as-is (non-transposed).

        For more details, see: :cusolverdx_doc:`get_started/functions/getrf.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched matrix according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place.
               On exit, contains the factors L and U from the factorization A = L @ U.
               The unit diagonal elements of L are not stored.
            info: Pointer to a 1D array of ``int32``.
                  On exit, ``info[batch_id] = 0`` indicates success for that batch,
                  ``info[batch_id] = i > 0`` indicates ``U(i,i)`` is exactly zero,
                  meaning the factorization has been completed but the factor U
                  is singular and division by zero will occur if it is used to solve
                  a system of equations.
            lda: Optional runtime leading dimension of matrix A.
                 If not specified, the compile-time ``lda`` is used.
        """
        raise RuntimeError("factorize is a device function and cannot be called on the host.")

    def solve(self, a, b, lda=None, ldb=None) -> None:
        """
        Solves a system of linear equations Ax = B
        using the LU factorization without pivoting.
        The ``a`` operand must be a square matrix (``M == N``),
        otherwise this function will throw an exception.

        This device function uses the previously computed factorization A = L @ U
        to solve the system. Uses cuSOLVERDx ``'getrs_no_pivot'``.

        If ``lda`` and ``ldb`` are provided, uses runtime version with
        the specified leading dimensions. If not provided (``None``),
        uses compile-time version with default or constructor-provided
        leading dimensions.

        .. note::
            The ``transpose_mode`` parameter (set in constructor) determines which
            system is solved: A*x=B (``'non_transposed'``), A^T*x=B (``'transposed'``),
            or A^H*x=B (``'conj_transposed'`` for complex matrices).

        For more details, see: :cusolverdx_doc:`get_started/functions/getrs.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched factors L and U
               from the LU factorization, according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The unit diagonal elements of L are not stored.
               See the :meth:`factorize` documentation for details.
            b: Pointer to an array in shared memory, storing the batched matrix according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place with the solution matrix x.
            lda: Optional runtime leading dimension of matrix A.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldb: Optional runtime leading dimension of matrix B.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``ldb`` is used.
        """
        raise RuntimeError("solve is a device function and cannot be called on the host.")


# ==========================
# Triangular Solver
# ==========================

TRIANGULAR_SOLVER_DOCSTRING = SOLVER_DOCSTRING.copy()
del TRIANGULAR_SOLVER_DOCSTRING["function"]
del TRIANGULAR_SOLVER_DOCSTRING["job"]

TRIANGULAR_SOLVER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` and ``N`` represent the dimensions of matrices A and B.
When ``side='left'``, A is ``M`` x ``M``, otherwise when ``side='right'``, A is ``N`` x ``N``.
B is always ``M`` x ``N``.""".replace("\n", " ")

TRIANGULAR_SOLVER_DOCSTRING["side"] = f"""Side of matrix A in the triangular solve operation (required for TRSM).
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_SIDE)}.
If ``side='left'``, solves op(A) * X = B where A is ``M`` x ``M``.
If ``side='right'``, solves X * op(A) = B where A is ``N`` x ``N``.""".replace("\n", " ")

TRIANGULAR_SOLVER_DOCSTRING["fill_mode"] = f"""Indicates which part of triangular matrix A is filled and should be used.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_FILL_MODE)}.
For lower fill mode, only the diagonal and lower triangular part of A is processed, the upper part is untouched.
For upper fill mode, only the diagonal and upper triangular part of A is processed, the lower part is untouched.""".replace(
    "\n", " "
)

TRIANGULAR_SOLVER_DOCSTRING["diag"] = f"""Indicates whether the diagonal elements of matrix A are unity or not.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_DIAG)}.
For unit diagonal mode, the diagonal elements of A are unity and are not accessed.
For non-unit diagonal mode, the diagonal elements of A are used in the computation.""".replace("\n", " ")

TRIANGULAR_SOLVER_DOCSTRING["transpose_mode"] = f"""Transpose mode for operation op(A) applied to matrix A.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_TRANSPOSE_MODE)}.
Defaults to ``'non_transposed'``.""".replace("\n", " ")

TRIANGULAR_SOLVER_DOCSTRING["leading_dimensions"] = SOLVER_DOCSTRING["leading_dimensions"] + ADAPTERS_API_LD_DOCSTRING


@docstring_decorator(TRIANGULAR_SOLVER_DOCSTRING, skip_missing=False)
class TriangularSolver(_SolverProperties):
    """
    A class that encapsulates triangular matrix-matrix solve device function (``'trsm'``).

    TRSM (TRiangular Solve for Matrix) solves a triangular linear system
    with multiple right-hand sides:

    * op(A) * X = B (if ``side='left'``)
    * X * op(A) = B (if ``side='right'``)

    where:

    * A is the input batched triangular matrix stored in lower or upper mode
    * B is the batched right-hand side matrix, overwritten by the result X on exit
    * Operation op(A) indicates if matrix A is ``'non_transposed'``,
      ``'transposed'`` (for real data type),
      or ``'conj_transposed'`` (for complex data type)

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x M) with ``side='left'``:**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, M)``
      with strides ``(lda * M, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, M)``
      with strides ``(lda * M, lda, 1)``

    **For matrix A (N x N) with ``side='right'``:**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, N, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, N, N)``
      with strides ``(lda * N, lda, 1)``

    **For matrix B (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(ldb * N, 1, ldb)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(ldb * M, ldb, 1)``

    .. note::
        The TRSM function is temporarily exposed in cuSolverDx library
        and will be moved to cuBLASDx library in a future release.

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        side (str): {side}

        fill_mode (str): {fill_mode}

        diag (str): {diag}

        transpose_mode (str, optional): {transpose_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`trsm <get_started/functions/trsm.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        side: str,
        fill_mode: str,
        diag: str,
        transpose_mode: str = "non_transposed",
        *,
        sm=None,
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        self._solve: Solver = Solver(
            function="trsm",
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            side=side,
            fill_mode=fill_mode,
            diag=diag,
            transpose_mode=transpose_mode,
            arrangement=arrangement,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=leading_dimensions,
            block_dim=block_dim,
        )

        super().__init__(self._solve)

        if not isinstance(self._solve.side, str):
            raise ValueError("side must be provided for TriangularSolver")

        if not isinstance(self._solve.fill_mode, str):
            raise ValueError("fill_mode must be provided for TriangularSolver")

        if not isinstance(self._solve.diag, str):
            raise ValueError("diag must be provided for TriangularSolver")

    # ==========================
    # Property methods
    # ==========================

    @property
    def side(self) -> str:
        assert self._solve.side is not None
        return self._solve.side

    @property
    def fill_mode(self) -> str:
        assert self._solve.fill_mode is not None
        return self._solve.fill_mode

    @property
    def diag(self) -> str:
        assert self._solve.diag is not None
        return self._solve.diag

    @property
    def transpose_mode(self) -> str:
        assert self._solve.transpose_mode is not None
        return self._solve.transpose_mode

    @property
    def a_shape(self) -> tuple[int, int, int]:
        dim = self.m if self.side == "left" else self.n
        return (self.batches_per_block, dim, dim)

    @property
    def b_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m, self.n)

    def a_strides(self, *, lda: int | None = None) -> tuple[int, int, int]:
        lda = self.lda if lda is None else lda
        return _calculate_strides(self.a_shape[1:], lda, self.a_arrangement)

    def b_strides(self, *, ldb: int | None = None) -> tuple[int, int, int]:
        ldb = self.ldb if ldb is None else ldb
        return _calculate_strides(self.b_shape[1:], ldb, self.b_arrangement)

    def a_size(self, *, lda: int | None = None) -> int:
        return self.a_strides(lda=lda)[0] * self.a_shape[0]

    def b_size(self, *, ldb: int | None = None) -> int:
        return self.b_strides(ldb=ldb)[0] * self.b_shape[0]

    # ==========================
    # Device function methods
    # ==========================

    def solve(self, a, b, lda=None, ldb=None) -> None:
        """
        Solves a triangular linear system with multiple right-hand sides:
            ``op(A) * X = B`` (if ``side='left'``)
            ``X * op(A) = B`` (if ``side='right'``)

        This device function solves a triangular system where A is a triangular matrix.
        Uses cuSOLVERDx ``'trsm'``. The operation is in-place: result X overwrites B.

        If ``lda`` and ``ldb`` are provided, uses runtime version with the
        specified leading dimensions. If not provided (``None``), uses compile-time
        version with default or constructor-provided leading dimensions.

        For more details, see: :cusolverdx_doc:`get_started/functions/trsm.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched triangular matrix
               according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The ``fill_mode`` parameter denotes which
               part of the matrix is used (the other part is ignored).
               For unit diagonal mode (``diag='unit'``),
               diagonal elements are unity and not accessed.
            b: Pointer to an array in shared memory,
               storing the batched ``M`` x ``N`` right-hand side
               matrix according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The operation is in-place: result X overwrites B.
            lda: Optional runtime leading dimension for matrix A.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldb: Optional runtime leading dimension for matrix B.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``ldb`` is used.
        """
        raise RuntimeError("solve is a device function and cannot be called on the host.")


# ==========================
# LU Pivot Solver
# ==========================


@docstring_decorator(LU_SOLVER_DOCSTRING, skip_missing=False)
class LUPivotSolver(_LinearSolverProperties):
    """
    A class that encapsulates cuSOLVERDx LU factorization with partial pivoting
    and linear solver for general matrices.

    **Available operations:**

    * factorize: Computes the LU factorization P @ A = L @ U with partial pivoting,
      where P is a permutation matrix, L is a
      lower triangular matrix and U is an upper triangular matrix.
    * solve: Solves the system Ax = B using a previously
      computed LU factorization with partial pivoting

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * M, lda, 1)``

    **For matrix B (N x K):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, N, K)``
      with strides ``(ldb * K, 1, ldb)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, N, K)``
      with strides ``(ldb * N, ldb, 1)``

    .. note::
        This solver uses partial pivoting for improved numerical stability and is
        suitable for general matrices. If your matrix is diagonally dominant,
        you may consider using :class:`LUSolver` which does not use pivoting
        and may be faster.

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        transpose_mode (str, optional): {transpose_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

          * :cusolverdx_doc:`factorize <get_started/functions/getrf.html>`
          * :cusolverdx_doc:`solve <get_started/functions/getrs.html>`

    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        *,
        sm=None,
        transpose_mode: str = "non_transposed",
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        def construct(function, transpose_mode):
            return Solver(
                function=function,
                size=size,
                precision=precision,
                execution=execution,
                sm=sm,
                arrangement=arrangement,
                batches_per_block=batches_per_block,
                data_type=data_type,
                leading_dimensions=leading_dimensions,
                block_dim=block_dim,
                transpose_mode=transpose_mode,
            )

        if transpose_mode is None:
            raise ValueError("transpose_mode must be provided for LUPivotSolver")

        self._factorize = construct("getrf_partial_pivot", "non_transposed")
        self._solve = construct("getrs_partial_pivot", transpose_mode) if self._factorize.m == self._factorize.n else None
        self._transpose_mode = transpose_mode

        super().__init__(self._factorize)

    # ==========================
    # Property methods
    # ==========================

    @property
    def transpose_mode(self) -> str:
        return self._transpose_mode

    @property
    def ipiv_type(self) -> type[np.signedinteger]:
        return self._factorize.ipiv_type

    @property
    def ipiv_shape(self) -> tuple[int, int]:
        return (self.batches_per_block, min(self.m, self.n))

    @property
    def ipiv_strides(self) -> tuple[int, int]:
        return (self.ipiv_shape[1], 1)

    @property
    def ipiv_size(self) -> int:
        return self.ipiv_shape[0] * self.ipiv_shape[1]

    # ==========================
    # Device function methods
    # ==========================

    def factorize(self, a, ipiv, info, lda=None) -> None:
        """
        Computes the LU factorization of a general matrix A with partial pivoting.

        This device function computes P @ A = L @ U, where P is a permutation matrix,
        L is a unit lower triangular matrix and U is an upper triangular matrix.
        This variant uses partial pivoting for improved numerical stability
        and is suitable for general matrices. Uses cuSOLVERDx ``'getrf_partial_pivot'``.

        If ``lda`` is provided, uses runtime version with the specified leading dimension.
        If ``lda`` is not provided (``None``), uses compile-time version with default
        or constructor-provided leading dimensions.

        .. note::
            The ``transpose_mode`` parameter does not affect factorization.
            This operation always treats the input matrix as-is (non-transposed).

        For more details, see: :cusolverdx_doc:`get_started/functions/getrf.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched matrix according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place.
               On exit, contains the factors L and U from the factorization P @ A = L @ U.
               The unit diagonal elements of L are not stored.
            ipiv: Pointer to a 1D array of ``int32``, storing pivot indices.
                  The array has size min(M, N) for each batch.
                  On exit, ``ipiv[batch_id * min(M, N) + i]`` indicates that row i
                  was interchanged with row ``ipiv[batch_id * min(M, N) + i] - 1``
                  in the batch_id-th batch of A.
            info: Pointer to a 1D array of ``int32``. On exit, ``info[batch_id] = 0``
                  indicates success for that batch, ``info[batch_id] = i > 0``
                  indicates ``U(i,i)`` is exactly zero,
                  meaning the factorization has been completed but the factor U
                  is singular and division by zero will occur if it is used to solve
                  a system of equations.
            lda: Optional runtime leading dimension of matrix A.
                 If not specified, the compile-time ``lda`` is used.
        """
        raise RuntimeError("factorize is a device function and cannot be called on the host.")

    def solve(self, a, ipiv, b, lda=None, ldb=None) -> None:
        """
        Solves a system of linear equations Ax = B
        using the LU factorization with partial pivoting.
        The ``a`` operand must be a square matrix (``M == N``),
        otherwise this function will throw an exception.

        This device function uses the previously computed factorization P @ A = L @ U
        to solve the system. Uses cuSOLVERDx ``'getrs_partial_pivot'``.

        If ``lda`` and ``ldb`` are provided, uses runtime version with
        the specified leading dimensions. If not provided (``None``),
        uses compile-time version with default or constructor-provided
        leading dimensions.

        .. note::
            The ``transpose_mode`` parameter (set in constructor) determines which
            system is solved: A*x=B (``'non_transposed'``), A^T*x=B (``'transposed'``),
            or A^H*x=B (``'conj_transposed'`` for complex matrices).

        For more details, see: :cusolverdx_doc:`get_started/functions/getrs.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched factors L and U
               from the LU factorization with partial pivoting, according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The unit diagonal elements of L are not stored.
               See the :meth:`factorize` documentation for details.
            ipiv: Pointer to a 1D array of ``int32`` in shared or global memory
                  storing pivot indices.
                  The array has size min(M, N) for each batch. The ipiv array should contain
                  the pivot information from the :meth:`factorize` call.
                  ``ipiv[batch_id * min(M, N) + i]`` indicates that row i was interchanged
                  with row ``ipiv[batch_id * min(M, N) + i]`` in the batch_id-th batch of A.
            b: Pointer to an array in shared memory, storing the batched matrix according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place with the solution matrix x.
            lda: Optional runtime leading dimension of matrix A.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldb: Optional runtime leading dimension of matrix B.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``ldb`` is used.
        """
        raise RuntimeError("solve is a device function and cannot be called on the host.")


# ==========================
# QR/LQ Factorizers Base
# ==========================

ORTOGHONAL_FACTORIZER_DOCSTRING = SOLVER_DOCSTRING.copy()
del ORTOGHONAL_FACTORIZER_DOCSTRING["function"]
del ORTOGHONAL_FACTORIZER_DOCSTRING["fill_mode"]
del ORTOGHONAL_FACTORIZER_DOCSTRING["side"]
del ORTOGHONAL_FACTORIZER_DOCSTRING["diag"]
del ORTOGHONAL_FACTORIZER_DOCSTRING["transpose_mode"]
del ORTOGHONAL_FACTORIZER_DOCSTRING["job"]

ORTOGHONAL_FACTORIZER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` and ``N`` represent the dimensions of the matrix A used in factorization.
``K`` is ignored if specified.""".replace("\n", " ")

ORTOGHONAL_FACTORIZER_DOCSTRING["arrangement"] = (
    """Storage layout for matrix A.
Can be one of: ``'col_major'``, ``'row_major'``. Defaults to ``'col_major'``.""".replace("\n", " ")
    + " "
    + ADAPTERS_API_LD_DOCSTRING
)

ORTOGHONAL_FACTORIZER_DOCSTRING["leading_dimension"] = (
    """The leading dimension for input matrix A, or ``None``.
If not provided, it will be automatically deduced from ``size`` and ``arrangement``.""".replace("\n", " ")
    + " "
    + ADAPTERS_API_LD_DOCSTRING
)


class _OrthogonalFactorizerProperties:
    def __init__(self, source: Solver):
        self._properties_source = source

    @property
    def size(self) -> tuple[int, int, int]:
        return self._properties_source.size

    @property
    def m(self) -> int:
        return self._properties_source.m

    @property
    def n(self) -> int:
        return self._properties_source.n

    @property
    def precision(self) -> type[np.floating]:
        return self._properties_source.precision

    @property
    def execution(self) -> str:
        return self._properties_source.execution

    @property
    def sm(self) -> ComputeCapability:
        return self._properties_source.sm

    @property
    def a_arrangement(self) -> str:
        return self._properties_source.a_arrangement

    @property
    def batches_per_block(self) -> int:
        return self._properties_source.batches_per_block

    @property
    def data_type(self) -> str:
        return self._properties_source.data_type

    @property
    def lda(self) -> int:
        return self._properties_source.lda

    @property
    def block_dim(self) -> tuple:
        return self._properties_source.block_dim

    @property
    def block_size(self) -> int:
        return self._properties_source.block_size

    @property
    def value_type(self) -> type[np.floating] | Complex:
        return self._properties_source.value_type

    @property
    def tau_type(self) -> type[np.floating] | Complex:
        return self._properties_source.tau_type

    @property
    def tau_shape(self) -> tuple[int, int]:
        return (self.batches_per_block, min(self.m, self.n))

    @property
    def tau_strides(self) -> tuple[int, int]:
        return (self.tau_shape[1], 1)

    @property
    def a_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m, self.n)

    def a_strides(self, *, lda: int | None = None) -> tuple[int, int, int]:
        lda = self.lda if lda is None else lda
        return _calculate_strides(self.a_shape[1:], lda, self.a_arrangement)

    def a_size(self, *, lda: int | None = None) -> int:
        return self.a_strides(lda=lda)[0] * self.a_shape[0]

    @property
    def tau_size(self) -> int:
        return self.tau_shape[0] * self.tau_shape[1]


# ==========================
# QR Factorize
# ==========================


@docstring_decorator(ORTOGHONAL_FACTORIZER_DOCSTRING, skip_missing=False)
class QRFactorize(_OrthogonalFactorizerProperties):
    """
    A class that encapsulates QR orthogonal factorization device function
    for general matrices using Householder reflections.

    **Available operation:**

    * factorize: Computes the QR factorization A = Q @ R,
      where Q is a unitary M x M matrix
      and R is an upper triangular matrix (if M >= N)
      or upper trapezoidal matrix (if M < N).

    The factorization uses Householder reflection transformations and does not explicitly
    form the unitary matrix Q. Instead, Q is represented as a product of Householder vectors
    stored in the input matrix A along with the tau array.

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * M, lda, 1)``

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        arrangement (str, optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimension (int, optional): {leading_dimension}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`factorize (geqrf) <get_started/functions/geqrf.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        *,
        sm=None,
        arrangement: str | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimension: int | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        self._factorize = Solver(
            function="geqrf",
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            arrangement=(arrangement,) if arrangement is not None else None,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=(leading_dimension,) if leading_dimension is not None else None,
            block_dim=block_dim,
        )
        super().__init__(self._factorize)

    # ==========================
    # Device function methods
    # ==========================

    def factorize(self, a, tau, lda=None) -> None:
        """
        Computes the QR factorization of a general matrix A using Householder reflections.

        This device function computes A = Q @ R, where Q is a unitary M x M matrix
        and R is an upper triangular matrix (if M >= N)
        or upper trapezoidal matrix (if M < N).
        Uses cuSOLVERDx ``'geqrf'``.

        If ``lda`` is provided, uses runtime version with the
        specified leading dimension. If ``lda`` is not provided (``None``),
        uses compile-time version with
        default or constructor-provided leading dimensions.

        Matrix Q is not explicitly formed. Instead, Q is represented as a product of
        min(M, N) Householder vectors: Q = H(0) * H(1) * ... * H(min(M, N) - 1).

        Each Householder vector has the form H(i) = I - tau[i] * v * v^H, where:

        * v is a vector of size M for each batch
        * v[0:i-1] = 0, v[i] = 1
        * v[i+1:M] is stored on exit in A[i+1:M, i]

        For more details, see: :cusolverdx_doc:`get_started/functions/geqrf.html`

        Args:
            a: Pointer to an array in shared memory, storing
               the batched matrix according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place.
               On exit, the upper triangular or upper trapezoidal part (including diagonal)
               contains the matrix R. The elements below the diagonal, with the array tau,
               represent the unitary matrix Q as a product of Householder vectors.
            tau: Pointer to a 1D array of size min(M, N) for each batch.
                 Contains the scalar factors of the Householder reflections.
                 The tau array, together with the Householder vectors stored in A,
                 defines the unitary matrix Q.
            lda: Optional runtime leading dimension of matrix A.
                 If not specified, the compile-time ``lda`` is used.
        """
        raise RuntimeError("factorize is a device function and cannot be called on the host.")


# ==========================
# LQ Factorize
# ==========================


@docstring_decorator(ORTOGHONAL_FACTORIZER_DOCSTRING, skip_missing=False)
class LQFactorize(_OrthogonalFactorizerProperties):
    """
    A class that encapsulates LQ orthogonal factorization device function
    for general matrices using Householder reflections.

    **Available operation:**

    * factorize: Computes the LQ factorization A = L @ Q,
      where L is a lower triangular matrix (if M <= N)
      or lower trapezoidal matrix (if M > N)
      and Q is a unitary N x N matrix.

    The factorization uses Householder reflection transformations and does not explicitly
    form the unitary matrix Q. Instead, Q is represented as a product of Householder vectors
    stored in the input matrix A along with the tau array.

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * M, lda, 1)``

    .. note::
        The LQ factorization is essentially equivalent to the
        QR factorization of A^T (or A^H for complex types).

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        arrangement (str, optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimension (int, optional): {leading_dimension}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`factorize (gelqf) <get_started/functions/gelqf.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        *,
        sm=None,
        arrangement: str | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimension: int | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        self._factorize = Solver(
            function="gelqf",
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            arrangement=(arrangement,) if arrangement is not None else None,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=(leading_dimension,) if leading_dimension is not None else None,
            block_dim=block_dim,
        )
        super().__init__(self._factorize)

    # ==========================
    # Device function methods
    # ==========================

    def factorize(self, a, tau, lda=None) -> None:
        """
        Computes the LQ factorization of a general matrix A using Householder reflections.

        This device function computes A = L @ Q, where L is a lower triangular matrix
        (if M <= N) or lower trapezoidal matrix (if M > N),
        and Q is a unitary N x N matrix.
        Uses cuSOLVERDx ``'gelqf'``.

        If ``lda`` is provided, uses runtime version with the
        specified leading dimension. If ``lda`` is not provided (``None``),
        uses compile-time version with
        default or constructor-provided leading dimensions.

        The LQ factorization is essentially the same as the QR factorization of A^T
        (or A^H for complex data types).

        Matrix Q is not explicitly formed. Instead, Q is represented as a product of
        min(M, N) Householder vectors: Q = H(min(M, N) - 1)^H * ... * H(1)^H * H(0)^H.

        Each Householder vector has the form H(i) = I - tau[i] * v * v^H, where:

        * v is a vector of size N for each batch
        * v[0:i-1] = 0, v[i] = 1
        * conjugate(v[i+1:N]) is stored on exit in A[i, i+1:N]

        For more details, see: :cusolverdx_doc:`get_started/functions/gelqf.html`

        Args:
            a: Pointer to an array in shared memory, storing
               the batched M x N matrix according
               to the specified arrangement and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place.
               On exit, the lower triangular or lower trapezoidal part (including diagonal)
               contains the matrix L. The elements above the diagonal, with the array tau,
               represent the unitary matrix Q as a product of Householder vectors.
            tau: Pointer to a 1D array of size min(M, N) for each batch.
                 Contains the scalar factors of the Householder reflections.
                 The tau array, together with the Householder vectors stored in A,
                 defines the unitary matrix Q.
            lda: Optional runtime leading dimension of matrix A.
                 If not specified, the compile-time ``lda`` is used.
        """
        raise RuntimeError("factorize is a device function and cannot be called on the host.")


# ==========================
# QR Multiplier (UNMQR)
# ==========================

QR_MULTIPLIER_DOCSTRING = SOLVER_DOCSTRING.copy()
del QR_MULTIPLIER_DOCSTRING["function"]
del QR_MULTIPLIER_DOCSTRING["fill_mode"]
del QR_MULTIPLIER_DOCSTRING["diag"]
del QR_MULTIPLIER_DOCSTRING["job"]

QR_MULTIPLIER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` and ``N`` represent the dimensions of matrix C.
``K`` represents the number of Householder reflections from the QR factorization.
If ``side='left'``, then ``K <= M`` and A is ``M`` x ``K``.
If ``side='right'``, then ``K <= N`` and A is ``N`` x ``K``.""".replace("\n", " ")

QR_MULTIPLIER_DOCSTRING["side"] = f"""Side of matrix Q in the multiplication operation.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_SIDE)}.
If ``side='left'``, computes op(Q) * C where Q is ``M`` x ``M``.
If ``side='right'``, computes C * op(Q) where Q is ``N`` x ``N``.""".replace("\n", " ")

QR_MULTIPLIER_DOCSTRING["transpose_mode"] = f"""Transpose mode for operation op(Q) applied to matrix Q.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_TRANSPOSE_MODE)}.
Defaults to ``'non_transposed'``.""".replace("\n", " ")

QR_MULTIPLIER_DOCSTRING["leading_dimensions"] = SOLVER_DOCSTRING["leading_dimensions"] + ADAPTERS_API_LD_DOCSTRING


@docstring_decorator(QR_MULTIPLIER_DOCSTRING, skip_missing=False)
class QRMultiply(_SolverProperties):
    """
    A class that encapsulates the multiplication of a matrix C by the unitary matrix Q
    from a QR factorization (UNMQR operation).

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (containing Householder vectors):**

    * If ``side='left'``: A is ``M`` x ``K``
    * If ``side='right'``: A is ``N`` x ``K``
    * **Column-major arrangement**: Matrix shape ``(batches_per_block, rows, K)``
      with strides ``(lda * K, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, rows, K)``
      with strides ``(lda * rows, lda, 1)``

    **For matrix C (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(ldb * N, 1, ldb)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(ldb * M, ldb, 1)``

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        side (str): {side}

        transpose_mode (str, optional): {transpose_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`unmqr <get_started/functions/unmqr.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        side: str,
        *,
        sm=None,
        transpose_mode: str = "non_transposed",
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        self._multiply: Solver = Solver(
            function="unmqr",
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            side=side,
            transpose_mode=transpose_mode,
            arrangement=arrangement,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=leading_dimensions,
            block_dim=block_dim,
        )

        super().__init__(self._multiply)

        if not isinstance(self._multiply.transpose_mode, str):
            raise ValueError("transpose_mode must be provided for QRMultiply")

        if not isinstance(self._multiply.side, str):
            raise ValueError("side must be provided for QRMultiply")

    # ==========================
    # Property methods
    # ==========================

    @property
    def side(self) -> str:
        assert self._multiply.side is not None
        return self._multiply.side

    @property
    def transpose_mode(self) -> str:
        assert self._multiply.transpose_mode is not None
        return self._multiply.transpose_mode

    @property
    def tau_type(self) -> type[np.floating] | Complex:
        return self.value_type

    @property
    def tau_shape(self) -> tuple[int, int]:
        return (self.batches_per_block, self.k)

    @property
    def tau_strides(self) -> tuple[int, int]:
        return (self.tau_shape[1], 1)

    @property
    def a_shape(self) -> tuple[int, int, int]:
        rows = self.m if self.side == "left" else self.n
        return (self.batches_per_block, rows, self.k)

    @property
    def c_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m, self.n)

    def a_strides(self, *, lda: int | None = None) -> tuple[int, int, int]:
        lda = self.lda if lda is None else lda
        return _calculate_strides(self.a_shape[1:], lda, self.a_arrangement)

    def c_strides(self, *, ldc: int | None = None) -> tuple[int, int, int]:
        ldc = self.ldb if ldc is None else ldc
        return _calculate_strides(self.c_shape[1:], ldc, self.b_arrangement)

    def a_size(self, *, lda: int | None = None) -> int:
        return self.a_strides(lda=lda)[0] * self.a_shape[0]

    def c_size(self, *, ldc: int | None = None) -> int:
        return self.c_strides(ldc=ldc)[0] * self.c_shape[0]

    @property
    def tau_size(self) -> int:
        return self.tau_shape[0] * self.tau_shape[1]

    # ==========================
    # Device function methods
    # ==========================

    def multiply(self, a, tau, c, lda=None, ldc=None) -> None:
        """
        Multiplies matrix C by the unitary matrix Q from a QR factorization.

        This device function computes:
            ``op(Q) * C`` (if ``side='left'``)
            ``C * op(Q)`` (if ``side='right'``)

        where Q is the unitary matrix from the QR factorization, represented
        by Householder vectors stored in A and the tau array.
        Uses cuSOLVERDx ``'unmqr'``. The result overwrites matrix C.

        If ``lda`` and ``ldc`` are provided, uses runtime version with the
        specified leading dimensions. If not provided (``None``), uses compile-time
        version with default or constructor-provided leading dimensions.

        For more details, see: :cusolverdx_doc:`get_started/functions/unmqr.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched matrix containing
               Householder vectors from the QR factorization, according to the specified
               arrangement and leading dimension (see :meth:`__init__`).
               The elements below the diagonal of A, with the array tau, represent the
               unitary matrix Q as a product of Householder reflections.
               If ``side='left'``, A is ``M`` x ``K``.
               If ``side='right'``, A is ``N`` x ``K``.
            tau: Pointer to a 1D array of size K for each batch, containing the scalar
                 factors of the Householder reflections from the QR factorization.
                 The tau array, together with the Householder vectors in A, defines Q.
            c: Pointer to an array in shared memory,
               storing the batched ``M`` x ``N`` matrix
               according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The operation is in-place: result overwrites C.
            lda: Optional runtime leading dimension for matrix A.
                 The ``lda`` and ``ldc`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldc: Optional runtime leading dimension for matrix C.
                 The ``lda`` and ``ldc`` must be specified together.
                 If not specified, the compile-time ``ldc`` is used.
        """
        raise RuntimeError("multiply is a device function and cannot be called on the host.")


# ==========================
# LQ Multiplier (UNMLQ)
# ==========================

LQ_MULTIPLIER_DOCSTRING = SOLVER_DOCSTRING.copy()
del LQ_MULTIPLIER_DOCSTRING["function"]
del LQ_MULTIPLIER_DOCSTRING["fill_mode"]
del LQ_MULTIPLIER_DOCSTRING["diag"]
del LQ_MULTIPLIER_DOCSTRING["job"]

LQ_MULTIPLIER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` and ``N`` represent the dimensions of matrix C.
``K`` represents the number of Householder reflections from the LQ factorization.
If ``side='left'``, then ``K <= M`` and A is ``K`` x ``M``.
If ``side='right'``, then ``K <= N`` and A is ``K`` x ``N``.""".replace("\n", " ")

LQ_MULTIPLIER_DOCSTRING["side"] = f"""Side of matrix Q in the multiplication operation.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_SIDE)}.
If ``side='left'``, computes op(Q) * C where Q is ``M`` x ``M``.
If ``side='right'``, computes C * op(Q) where Q is ``N`` x ``N``.""".replace("\n", " ")

LQ_MULTIPLIER_DOCSTRING["transpose_mode"] = f"""Transpose mode for operation op(Q) applied to matrix Q.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_TRANSPOSE_MODE)}.
Defaults to ``'non_transposed'``.""".replace("\n", " ")

LQ_MULTIPLIER_DOCSTRING["leading_dimensions"] = SOLVER_DOCSTRING["leading_dimensions"] + ADAPTERS_API_LD_DOCSTRING


@docstring_decorator(LQ_MULTIPLIER_DOCSTRING, skip_missing=False)
class LQMultiply(_SolverProperties):
    """
    A class that encapsulates the multiplication of a matrix C by the unitary matrix Q
    from an LQ factorization (UNMLQ operation).

    **Memory Layout Requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (containing Householder vectors):**

    * If ``side='left'``: A is ``K`` x ``M``
    * If ``side='right'``: A is ``K`` x ``N``
    * **Column-major arrangement**: Matrix shape ``(batches_per_block, K, cols)``
      with strides ``(lda * cols, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, K, cols)``
      with strides ``(lda * K, lda, 1)``

    **For matrix C (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(ldc * N, 1, ldc)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(ldc * M, ldc, 1)``

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        side (str): {side}

        transpose_mode (str, optional): {transpose_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`unmlq <get_started/functions/unmlq.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        side: str,
        *,
        sm=None,
        transpose_mode: str = "non_transposed",
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        self._multiply: Solver = Solver(
            function="unmlq",
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            side=side,
            transpose_mode=transpose_mode,
            arrangement=arrangement,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=leading_dimensions,
            block_dim=block_dim,
        )

        super().__init__(self._multiply)

        if not isinstance(self._multiply.transpose_mode, str):
            raise ValueError("transpose_mode must be provided for LQMultiply")

        if not isinstance(self._multiply.side, str):
            raise ValueError("side must be provided for LQMultiply")

    # ==========================
    # Property methods
    # ==========================

    @property
    def side(self) -> str:
        assert self._multiply.side is not None
        return self._multiply.side

    @property
    def transpose_mode(self) -> str:
        assert self._multiply.transpose_mode is not None
        return self._multiply.transpose_mode

    @property
    def tau_type(self) -> type[np.floating] | Complex:
        return self.value_type

    @property
    def tau_shape(self) -> tuple[int, int]:
        return (self.batches_per_block, self.k)

    @property
    def tau_strides(self) -> tuple[int, int]:
        return (self.tau_shape[1], 1)

    @property
    def a_shape(self) -> tuple[int, int, int]:
        cols = self.m if self.side == "left" else self.n
        return (self.batches_per_block, self.k, cols)

    @property
    def c_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m, self.n)

    def a_strides(self, *, lda: int | None = None) -> tuple[int, int, int]:
        lda = self.lda if lda is None else lda
        return _calculate_strides(self.a_shape[1:], lda, self.a_arrangement)

    def c_strides(self, *, ldc: int | None = None) -> tuple[int, int, int]:
        ldc = self.ldb if ldc is None else ldc
        return _calculate_strides(self.c_shape[1:], ldc, self.b_arrangement)

    def a_size(self, *, lda: int | None = None) -> int:
        return self.a_strides(lda=lda)[0] * self.a_shape[0]

    def c_size(self, *, ldc: int | None = None) -> int:
        return self.c_strides(ldc=ldc)[0] * self.c_shape[0]

    @property
    def tau_size(self) -> int:
        return self.tau_shape[0] * self.tau_shape[1]

    # ==========================
    # Device function methods
    # ==========================

    def multiply(self, a, tau, c, lda=None, ldc=None) -> None:
        """
        Multiplies matrix C by the unitary matrix Q from an LQ factorization.

        This device function computes:
            ``op(Q) * C`` (if ``side='left'``)
            ``C * op(Q)`` (if ``side='right'``)

        where Q is the unitary matrix from the LQ factorization, represented
        by Householder vectors stored in A and the tau array.
        Uses cuSOLVERDx ``'unmlq'``. The result overwrites matrix C.

        If ``lda`` and ``ldc`` are provided, uses runtime version with the
        specified leading dimensions. If not provided (``None``), uses compile-time
        version with default or constructor-provided leading dimensions.

        For more details, see: :cusolverdx_doc:`get_started/functions/unmlq.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched matrix containing
               Householder vectors from the LQ factorization, according to the specified
               arrangement and leading dimension (see :meth:`__init__`).
               The elements above the diagonal of A, with the array tau, represent the
               unitary matrix Q as a product of Householder reflections.
               If ``side='left'``, A is ``K`` x ``M``.
               If ``side='right'``, A is ``K`` x ``N``.
            tau: Pointer to a 1D array of size K for each batch, containing the scalar
                 factors of the Householder reflections from the LQ factorization.
                 The tau array, together with the Householder vectors in A, defines Q.
            c: Pointer to an array in shared memory,
               storing the batched ``M`` x ``N`` matrix
               according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The operation is in-place: result overwrites C.
            lda: Optional runtime leading dimension for matrix A.
                 The ``lda`` and ``ldc`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldc: Optional runtime leading dimension for matrix C.
                 The ``lda`` and ``ldc`` must be specified together.
                 If not specified, the compile-time ``ldc`` is used.
        """
        raise RuntimeError("multiply is a device function and cannot be called on the host.")


# ==========================
# Least Squares Solver
# ==========================

LEAST_SQUARES_SOLVER_DOCSTRING = SOLVER_DOCSTRING.copy()
del LEAST_SQUARES_SOLVER_DOCSTRING["function"]
del LEAST_SQUARES_SOLVER_DOCSTRING["fill_mode"]
del LEAST_SQUARES_SOLVER_DOCSTRING["side"]
del LEAST_SQUARES_SOLVER_DOCSTRING["diag"]

LEAST_SQUARES_SOLVER_DOCSTRING["size"] = f"""{SOLVER_DOCSTRING_SIZE_BASE}
``M`` and ``N`` represent the dimensions of matrix A (``M`` x ``N``).
``K`` represents the number of columns in the right-hand side matrix B.""".replace("\n", " ")

LEAST_SQUARES_SOLVER_DOCSTRING["transpose_mode"] = f"""Transpose mode for operation op(A) applied to matrix A.
Can be one of: {", ".join(f"``'{v}'``" for v in ALLOWED_TRANSPOSE_MODE)}.
Defaults to ``'non_transposed'``.""".replace("\n", " ")

LEAST_SQUARES_SOLVER_DOCSTRING["leading_dimensions"] = SOLVER_DOCSTRING["leading_dimensions"] + ADAPTERS_API_LD_DOCSTRING


@docstring_decorator(LEAST_SQUARES_SOLVER_DOCSTRING, skip_missing=False)
class LeastSquaresSolver(_SolverProperties):
    """
    A class that encapsulates least squares solver device function (``'gels'``).
    GELS (GEneral Least Square) solves overdetermined
    or underdetermined least squares problems:

    .. math::
       \\min \\| op(A) * X - B \\|_2

    using the QR or LQ factorization of A, and overwriting B with the solution X.

    The configurations supported by GELS are:

    **Memory layout requirements:**

    Matrices must be stored in shared memory according
    to their arrangement and leading dimension (ld):

    **For matrix A (M x N):**

    * **Column-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * N, 1, lda)``
    * **Row-major arrangement**: Matrix shape ``(batches_per_block, M, N)``
      with strides ``(lda * M, lda, 1)``

    **Matrix B and X** are stored in the same buffer (B is overwritten by X in place).
    The buffer has shape ``(batches_per_block, max(M, N), K)``. The second leading
    dimension ``ldb`` refers to this shared B/X buffer
    and must satisfy ``ldb >= max(M, N)``.
    Logical shapes differ by :attr:`transpose_mode`.
    Use :attr:`b_shape` and :attr:`x_shape` for B and X.

    **Matrix B** (right-hand side):

    * **Logical shape**: :attr:`b_shape` - ``(batches_per_block, M, K)`` if non-transposed,
      ``(batches_per_block, N, K)`` if transposed.
    * **Column-major arrangement**: strides ``(ldb * K, 1, ldb)``.
    * **Row-major arrangement**: strides ``(ldb * max(M, N), ldb, 1)``.

    **Matrix X** (solution):

    * **Logical shape**: :attr:`x_shape` - ``(batches_per_block, N, K)`` if non-transposed,
      ``(batches_per_block, M, K)`` if transposed.
    * **Column-major arrangement**: strides ``(ldb * K, 1, ldb)``.
    * **Row-major arrangement**: strides ``(ldb * max(M, N), ldb, 1)``.

    .. note::
        GELS is an in-place function. Matrix A is overwritten by the QR or LQ factorization,
        and matrix B is overwritten by the solution X. Both B and X use the single buffer
        of shape ``(max(M, N), K)`` per batch.

    Args:
        size (Sequence[int]): {size}

        precision (type[np.floating]): {precision}

        execution (str): {execution}

        sm (ComputeCapability): {sm}

        transpose_mode (str, optional): {transpose_mode}

        arrangement (Sequence[str], optional): {arrangement}

        batches_per_block (int | Literal["suggested"], optional): {batches_per_block}

        data_type (str, optional): {data_type}

        leading_dimensions (Sequence[int], optional): {leading_dimensions}

        block_dim (Sequence[int] | Literal["suggested"], optional): {block_dim}

    See Also:
        For further details, please refer to the cuSOLVERDx documentation:

        * :cusolverdx_doc:`gels <get_started/functions/gels.html>`
    """

    # ==========================
    # Constructor
    # ==========================

    def __init__(
        self,
        size: Sequence[int],
        precision: type[np.floating],
        execution: str,
        *,
        sm=None,
        transpose_mode: str = "non_transposed",
        arrangement: Sequence[str] | None = None,
        batches_per_block: int | Literal["suggested"] | None = None,
        data_type: str | None = None,
        leading_dimensions: Sequence[int] | None = None,
        block_dim: Sequence[int] | Literal["suggested"] | None = None,
    ):
        self._solve: Solver = Solver(
            function="gels",
            size=size,
            precision=precision,
            execution=execution,
            sm=sm,
            transpose_mode=transpose_mode,
            arrangement=arrangement,
            batches_per_block=batches_per_block,
            data_type=data_type,
            leading_dimensions=leading_dimensions,
            block_dim=block_dim,
        )

        super().__init__(self._solve)

    # ==========================
    # Property methods
    # ==========================

    @property
    def transpose_mode(self) -> str:
        assert self._solve.transpose_mode is not None
        return self._solve.transpose_mode

    @property
    def tau_type(self) -> type[np.floating] | Complex:
        return self._solve.tau_type

    @property
    def tau_shape(self) -> tuple[int, int]:
        return (self.batches_per_block, min(self.m, self.n))

    @property
    def tau_strides(self) -> tuple[int, int]:
        return (self.tau_shape[1], 1)

    @property
    def a_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m, self.n)

    @property
    def b_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.m if self.transpose_mode == "non_transposed" else self.n, self.k)

    @property
    def x_shape(self) -> tuple[int, int, int]:
        return (self.batches_per_block, self.n if self.transpose_mode == "non_transposed" else self.m, self.k)

    def a_strides(self, *, lda: int | None = None) -> tuple[int, int, int]:
        lda = self.lda if lda is None else lda
        return _calculate_strides(self.a_shape[1:], lda, self.a_arrangement)

    def bx_strides(self, *, ldb: int | None = None) -> tuple[int, int, int]:
        ldb = self.ldb if ldb is None else ldb
        return _calculate_strides((max(self.m, self.n), self.k), ldb, self.b_arrangement)

    def a_size(self, *, lda: int | None = None) -> int:
        return self.a_strides(lda=lda)[0] * self.a_shape[0]

    def bx_size(self, *, ldb: int | None = None) -> int:
        return self.bx_strides(ldb=ldb)[0] * self.batches_per_block

    @property
    def tau_size(self) -> int:
        return self.tau_shape[0] * self.tau_shape[1]

    # ==========================
    # Device function methods
    # ==========================

    def solve(self, a, tau, b, lda=None, ldb=None) -> None:
        """
        Solves a least squares problem using QR or LQ factorization.

        This device function solves:

        .. math::
           \\min \\| op(A) * X - B \\|_2

        using the QR or LQ factorization of A, and overwrites B with the solution X.
        Uses cuSOLVERDx ``'gels'``. The operation is in-place: matrix A is overwritten
        by the factorization, and matrix B is overwritten by the solution X.

        If ``lda`` and ``ldb`` are provided, uses runtime version with the
        specified leading dimensions. If not provided (``None``), uses compile-time
        version with default or constructor-provided leading dimensions.

        .. note::
            The choice between QR and LQ factorization depends on the problem dimensions
            and transpose mode:

            * If ``op(A)`` is ``'non_transposed'`` and ``M >= N``: uses QR factorization
            * If ``op(A)`` is ``'non_transposed'`` and ``M < N``: uses LQ factorization
            * If ``op(A)`` is ``'transposed'`` or ``'conj_transposed'``
              and ``M >= N``: uses LQ factorization
            * If ``op(A)`` is ``'transposed'`` or ``'conj_transposed'``
              and ``M < N``: uses QR factorization

        For more details, see: :cusolverdx_doc:`get_started/functions/gels.html`

        Args:
            a: Pointer to an array in shared memory, storing the batched matrix
               according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The matrix is overwritten in place by the QR or LQ factorization.
            tau: Pointer to a 1D array of size min(M, N) for each batch.
                 Contains the scalar factors of the Householder reflections.
                 The tau array, together with the Householder vectors stored in A,
                 defines the unitary matrix Q.
            b: Pointer to an array in shared memory,
               storing the batched right-hand side matrix
               according to the specified arrangement
               and leading dimension (see :meth:`__init__`).
               The storage size is ``max(M, N) x K`` per batch.
               The operation is in-place: result X overwrites B.
            lda: Optional runtime leading dimension for matrix A.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``lda`` is used.
            ldb: Optional runtime leading dimension for matrix B.
                 The ``lda`` and ``ldb`` must be specified together.
                 If not specified, the compile-time ``ldb`` is used.
        """
        raise RuntimeError("solve is a device function and cannot be called on the host.")
