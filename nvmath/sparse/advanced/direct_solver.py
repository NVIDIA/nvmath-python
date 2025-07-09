# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


__all__ = [
    "direct_solver",
    "DirectSolver",
    "DirectSolverAlgType",
    "DirectSolverFactorizationConfig",
    "DirectSolverFactorizationInfo",
    "DirectSolverPlanConfig",
    "DirectSolverPlanInfo",
    "DirectSolverSolutionConfig",
]

from collections.abc import Sequence
import itertools
import math
import logging
import operator
import os
from typing import Any, TypeAlias

from nvmath.bindings import cudss

from nvmath.internal import formatters, utils
from nvmath.internal.package_wrapper import StreamHolder
from nvmath.internal import tensor_wrapper
from nvmath.sparse.advanced._configuration import DirectSolverOptions, ExecutionCUDA, ExecutionHybrid, HybridMemoryModeOptions
from nvmath.sparse._internal import common_utils as sp_utils
from nvmath.sparse._internal import cudss_config_ifc, cudss_data_ifc, cudss_utils


from nvmath.internal.typemaps import NAME_TO_DATA_TYPE


VALID_INDEX_TYPES = ("int32",)

VALID_DTYPES = ("float32", "float64", "complex64", "complex128")

# Qualified names for public export.
DirectSolverPlanConfig: TypeAlias = cudss_config_ifc.PlanConfig
DirectSolverFactorizationConfig: TypeAlias = cudss_config_ifc.FactorizationConfig
DirectSolverSolutionConfig: TypeAlias = cudss_config_ifc.SolutionConfig
DirectSolverPlanInfo: TypeAlias = cudss_data_ifc.PlanInfo
DirectSolverFactorizationInfo: TypeAlias = cudss_data_ifc.FactorizationInfo
DirectSolverAlgType: TypeAlias = cudss.AlgType


def get_threading_lib(library=None):
    """
    Return the name of the threading library, if defined using an environment variable and
    the path is valid, or None.
    """
    if library is None:
        library = os.getenv("CUDSS_THREADING_LIB")
    if library is None or not os.path.isfile(library):
        return
    return library


def check_dense_tensor_layout(shape: Sequence[int], strides: Sequence[int], *, explicitly_batched=None):
    """
    Check that the dense vector or matrix (each sample matrix in a N-D tensor) is
    in col-major format.
    """

    num_dimensions = len(shape)
    assert explicitly_batched is not None and len(strides) == num_dimensions, "Internal Error."

    # For explicitly batched matrices, each sample must be a matrix or vector.
    if explicitly_batched and num_dimensions > 2:
        return False

    first = int(num_dimensions > 1)

    # Only column-major matrices are currently supported.
    is_col_major = strides[-1 - first] == 1

    # Check that the LD doesn't lead to overlapping memory.
    # For batched matrices, LD has to be equal to the first matrix dimension.
    comparison_op_for_ld = operator.eq if num_dimensions > 2 else operator.ge

    if num_dimensions > 1:
        return is_col_major and comparison_op_for_ld(strides[-1], shape[-2])

    return is_col_major


def check_rhs_sequence_layout(
    shape: Sequence[Sequence[int]],
    strides: Sequence[Sequence[int]],
):
    return all(check_dense_tensor_layout(s, d, explicitly_batched=True) for s, d in zip(shape, strides, strict=True))


# TODO: unify to tensor_wrapper.to() and the axis utilities below.
def copy_single_or_sequence(operands, device_id, stream_holder):
    if isinstance(operands, Sequence):
        return tuple(o.to(device_id, stream_holder) for o in operands)

    return operands.to(device_id, stream_holder)


def axis_order_in_memory(shape, strides):
    """
    Compute the order in which the axes appear in memory.
    """
    # The shape is used to resolve cases like (1, 2, 1) : (2, 1, 1) in CuTe notation.
    _, _, axis_order = zip(*sorted(zip(strides, shape, range(len(strides)), strict=True)), strict=True)

    return axis_order


def calculate_strides(shape, axis_order):
    """
    Calculate the strides for the provided shape and axis order.
    """
    strides = [None] * len(shape)

    stride = 1
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return strides


SHARED_DSS_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_DSS_DOCUMENTATION.update(
    {
        "a": """\
The sparse operand (or sequence of operands) representing the left-hand side (LHS) of the system of equations. The LHS
operand may be a (sequence of) :class:`scipy.sparse.csr_matrix`, :class:`scipy.sparse.csr_array`,
:class:`cupyx.scipy.sparse.csr_matrix`, or :py:func:`torch.sparse_csr_tensor`. That is, the LHS is a sparse matrix or
tensor in Compressed Sparse Row (CSR) format from one of the supported packages: SciPy, CuPy, PyTorch. Refer to the
:ref:`semantics <Semantics>` section for details.
""".replace("\n", " "),
        #
        "b": """\
The ndarray/tensor or (sequence of ndarray/tensors) representing the dense right-hand side (RHS) of the system of equations.
The RHS operand may be a (sequence of) :class:`numpy.ndarray`, :class:`cupy.ndarray`, and :class:`torch.Tensor`. Refer to the
:ref:`semantics <Semantics>` section for details.
""".replace("\n", " "),
        #
        "options": """\
Specify options for the direct sparse solver as a :class:`DirectSolverOptions` object. Alternatively, a `dict` containing
the parameters for the ``DirectSolverOptions`` constructor can also be provided. If not specified, the value will be set
to the default-constructed ``DirectSolverOptions`` object.""".replace("\n", " "),
        #
        "execution": """\
Specify execution space options for the direct solver as a :class:`ExecutionCUDA` or :class:`ExecutionHybrid` object.
Alternatively, a string ('cuda' or 'hybrid'), or a `dict` with the 'name' key set to 'cuda' or 'hybrid' and optional
parameters relevant to the given execution space. The default execution space is 'cuda' and the corresponding
:class:`ExecutionCUDA` object will be default-constructed.""".replace("\n", " "),
        #
        "result": """\
The result of the specified sparse direct solve, which has the same shape, remains on the same device, and belongs to the
same package as the RHS ``b``. If ``b`` is a sequence, the result ``x`` is also a sequence of ndarray/tensor, each of which
has the same shape as the corresponding tensor in ``b``.
""".replace("\n", " "),
        #
        "semantics": r"""\
        The sparse direct solver solves :math:`a @ x = b` for ``x`` given the left-hand side (LHS) ``a`` and the right-hand side
        (RHS) ``b``.

        * In the simplest version with no batching, ``a`` is sparse (square) matrix of size ``n`` in Compressed Sparse Row (CSR)
          format, and ``b`` is a dense vector or matrix. A matrix (2D ndarray or tensor) in the RHS is treated as multiple
          column vectors corresponding to multiple solution vectors (i.e. ``x`` is the same shape as the RHS) to solve for.

          .. important:: Currently, only column-major (Fortran) layout is supported for the RHS.

        * Batching can be specified either **explicitly** or **implicitly**.

          * An **explicitly-batched** LHS is provided as a Python sequence of sparse CSR matrices :math:`[a_0, a_1, ..., a_n]`.
            Likewise an explicitly-batched RHS is provided as a sequence of vectors or matrices :math:`[b_0, b_1, ..., b_n]`.
            The solver will solve all :math:`n` systems :math:`a_i @ x_i = b_i` for the solution sequence :math:`x_i`.
            Each sample in explicit batching can be of a different size, with the only constraint being that a given
            LHS size is consistent with that of its corresponding RHS.

          * An **implicitly-batched** LHS is provided as a higher-dimensional :math:`N \geq 3D` sparse tensor in CSR format,
            where the leading :math:`N - 2` dimensions are the batch dimensions, and the last two dimensions correspond to that
            of the :math:`n \times n` sparse system. Currently, only PyTorch supports higher-dimensional sparse CSR tensors.
            Likewise, an implicitly-batched RHS is provided as a :math:`N \geq 3D` dense ndarray/tensor, where the leading
            :math:`N - 2` dimensions are the batch dimensions, and the last two dimensions correspond to the :math:`n \times 1`
            vector or :math:`n \times m` matrix for each sample. The solver solves :math:`a_i @ x_i = b_i` for each sample
            :math:`i` in the batch, and the solution ``x`` has the same shape as the RHS ``b``.

          * Each sample :math:`a_i` and :math:`b_i` in the (explicitly- or implicitly-specified) batch are essentially
            treated as, and subject to, the same rules as the case with no batching.

          * The LHS and RHS batch specification is independent: for example, the LHS can be explicitly-batched while the
            RHS is implicitly-batched (or vice-versa). The same batch specification can be used for both as well.

          * The solution ``x`` always has the same form as the RHS ``b``. It is a sequence of matrices or vectors if
            ``b`` is explicitly-batched, or a higher-dimensional ndarray/tensor if ``b`` is implicitly-batched.
""".strip(),
    }
)


class InvalidDirectSolverState(Exception):
    pass


@utils.docstring_decorator(SHARED_DSS_DOCUMENTATION, skip_missing=False)
class DirectSolver:
    """
    Create a stateful object that encapsulates the specified sparse direct solver
    computations and required resources. This object ensures the validity of resources
    during use and releases them when they are no longer needed to prevent misuse.

    This object encompasses all functionalities of function-form API :func:`direct_solver`,
    which is a convenience wrapper around it.  The stateful object also allows for the
    amortization of preparatory costs when the same solve operation is to be performed
    with different left-hand side (LHS) and right-hand side (RHS) with the same problem
    specification (see :meth:`reset_operands` for more details).

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with the defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` for reordering to minimize fill-in and
       symbolic factorization for this specific direct sparse solver operation.
    3. **Execution**: Factorize and solve the system.
    4. **Resource Management**: Ensure all resources are released either by explicitly
       calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on each step described above can be obtained by passing in a
    :class:`logging.Logger` object to :class:`DirectSolverOptions` or by setting the
    appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    Args:
        a: {a}

        b: {b}

        options: {options}

        execution: {execution}

        stream: {stream}

    See Also:
        :class:`DirectSolverPlanConfig`, :class:`DirectSolverFactorizationConfig`,
        :class:`DirectSolverSolutionConfig`, :class:`DirectSolverPlanInfo`,
        :class:`DirectSolverFactorizationInfo`, :class:`DirectSolverOptions`,
        :class:`ExecutionCUDA`, :class:`ExecutionHybrid`, :meth:`plan`,
        :meth:`reset_operands`, :meth:`factorize`, :meth:`solve`.

    Examples:

        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> import nvmath

        Create a sparse float64 ndarray in CSR format on the CPU for the LHS.

        >>> n = 16
        >>> a = sp.random_array((n, n), density=0.5, format="csr", dtype="float64")

        Ensure that the randomly-generated LHS is not singular.

        >>> a += sp.diags_array([2.0] * n, format="csr", dtype="float64")

        The RHS can be a vector or matrix. Here we create a random vector.

        >>> b = np.random.rand(n).astype(dtype="float64")

        We will define a sparse direct solver operation for solving the system a @ x = b
        using the specialized sparse direct solver interface.

        >>> solver = nvmath.sparse.advanced.DirectSolver(a, b)

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`DirectSolverOptions`).

        Next, plan the operation. The planning operation can be configured through
        the :class:`DirectSolverPlanConfig` interface, which is accessed through
        :attr:`plan_config`.

        >>> plan_config = solver.plan_config

        Here we set the reordering algorithm to choice 1.

        >>> AlgType = nvmath.sparse.advanced.DirectSolverAlgType
        >>> plan_config.reordering_algorithm = AlgType.ALG_1

        Plan the operation, which reorders the system to minimize fill-in and performs the
        symbolic factorization. Planning returns a :class:`DirectSolverPlanInfo` object,
        whose attributes (such as row or column permutation) can be queried.

        >>> plan_info = solver.plan()
        >>> plan_info.col_permutation
        array([ 0,  1,  8,  9,  2,  3,  4, 11, 15,  5, 10,  6, 12, 13,  7, 14],
              dtype=int32)

        The next step is to perform the numerical factorization of the system using
        :meth:`factorize`. Similar to planning, the numerical factorization step can
        also be configured if desired.

        >>> fac_config = solver.factorization_config

        Here we set the pivot epsilon value to ``1e-14``, instead of the default ``1e-13``.

        >>> fac_config.pivot_eps = 1e-14

        Factorize the system, which returns a :class:`DirectSolverFactorizationInfo` object,
        whose attributes can be inspected. We print the Sylvester inertia here.

        >>> fac_info = solver.factorize()
        >>> fac_info.inertia
        array([0, 0], dtype=int32)

        Now solve the factorized system, and obtain the result `x` as a NumPy ndarray on
        the CPU.

        >>> x = solver.solve()

        Finally, free the object's resources. To avoid having to explicitly making this
        call, it's recommended to use the DirectSolver object as a context manager as
        shown below, if possible.

        >>> solver.free()

        .. note:: All :class:`DirectSolver` methods execute on the package current stream
            by default. Alternatively, the `stream` argument can be used to run a method on
            a specified stream.

        Let's now look at a batched solve with PyTorch operands on the GPU.

        Create a 3D complex128 PyTorch sparse tensor on the GPU representing the LHS, along
        with the corresponding RHS:

        >>> import torch
        >>> n = 8
        >>> batch = 2
        >>> device_id = 0

        Prepare sample input data. Create a diagonally-dominant random CSR matrix.

        >>> a = torch.rand(n, n, dtype=torch.complex128) + torch.diag(torch.tensor([2.0] * n))
        >>> a = torch.stack([a] * batch, dim=0)
        >>> a = a.to_sparse_csr()

        .. important:: PyTorch uses int64 for index buffers, whereas cuDSS currently
            requires int32. So we'll have to convert the indices.

        >>> a = torch.sparse_csr_tensor(
        ...     a.crow_indices().to(dtype=torch.int32),
        ...     a.col_indices().to(dtype=torch.int32),
        ...     a.values(),
        ...     size=a.size(),
        ...     device=device_id,
        ... )

        Create the RHS, which can be a matrix or vector in column-major layout.

        >>> b = torch.ones(batch, 3, n, dtype=torch.complex128, device=device_id)
        >>> b = b.permute(0, 2, 1)

        Create a :class:`DirectSolver` object encapsulating the problem specification
        described earlier and use it as a context manager.

        >>> with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
        ...     plan_info = solver.plan()
        ...
        ...     # Factorize the system.
        ...     fac_info = solver.factorize()
        ...
        ...     # Solve the factorized system.
        ...     x1 = solver.solve()
        ...
        ...     # Update the RHS b in-place (see reset_operands() for an alternative).
        ...     b[...] = torch.rand(*b.shape, dtype=torch.complex128, device=device_id)
        ...
        ...     # Solve again to get the new result.
        ...     x2 = solver.solve()

        All the resources used by the object are released at the end of the block.

        Batching can be implicitly-specified as shown above, or explicitly-specified as
        a sequence for both the LHS and the RHS. This, as well as other options and
        usage patterns, are illustrated in the examples found in the
        `nvmath/examples/sparse/advanced/direct_solver
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver>`_
        directory.
    """  # noqa: W505

    def __init__(
        self,
        a,
        b,
        *,
        options: DirectSolverOptions | None = None,
        execution: ExecutionCUDA | ExecutionHybrid | None = None,
        stream: utils.AnyStream | int | None = None,
    ):
        # Process options.
        self.options: Any = utils.check_or_create_options(DirectSolverOptions, options, "sparse direct solver options")

        # Process execution options. The default execution space is CUDA.
        self.execution_options = utils.check_or_create_one_of_options(
            (ExecutionCUDA, ExecutionHybrid), execution, "execution options", default_name="cuda"
        )
        if self.execution_options.name == "cuda":
            self.execution_options.hybrid_memory_mode_options = utils.check_or_create_options(
                HybridMemoryModeOptions,
                self.execution_options.hybrid_memory_mode_options,
                "hybrid memory mode options",
            )

        self.logger = self.options.logger if self.options.logger is not None else logging.getLogger()
        self.logger.info("= SPECIFICATION PHASE =")

        # Wrap the LHS.
        try:
            self.a = sp_utils.wrap_sparse_operands(a)
        except Exception as e:
            raise TypeError(
                """The LHS must be an N-D sparse CSR array/tensor or a sequence of 2D sparse CSR array/tensor from one
of the supported packages: CuPy, PyTorch, or SciPy."""
            ) from e

        # The LHS can be implicitly (N-D CSR tensor) or explicitly batched (provided as a
        # sequence of CSR matrices).
        self.implicitly_batched_lhs = False
        self.explicitly_batched_lhs = isinstance(self.a, Sequence)
        if not self.explicitly_batched_lhs:
            self.implicitly_batched_lhs = self.a.num_dimensions > 2

        # For explicitly batched LHS, check that all operands are matrices.
        if self.explicitly_batched_lhs and any(a.num_dimensions > 2 for a in self.a):
            raise TypeError(
                f"""Every operator in the batched LHS provided as a sequence (explicit batching) must be a CSR matrix.
The specified LHS sequence = {a}."""
            )

        # The determination of a batched solve is purely based on the LHS. The RHS for each
        # sample in the batch for explicit batching can be a vector or matrix, with the
        # latter being considered multiple-RHS vectors as opposed to a batch. An
        # implicitly-batched RHS is always 3D or higher dimension (implicitly-batched
        # vectors are batched matrices with the last dimension of unit extent).
        self.batched = self.explicitly_batched_lhs or self.implicitly_batched_lhs

        # The LHS batch shape should be empty for explicit batching.
        self.lhs_batch_shape: Any = None if self.explicitly_batched_lhs else tuple(self.a.shape[:-2])

        # Set the LHS package.
        if self.explicitly_batched_lhs:
            # The package is the same for explicitly batched LHS since we've have
            # successfully wrapped them.
            self.lhs_package = utils.infer_object_package(self.a[0].tensor)
        else:  # Single or implicitly-batched LHS.
            self.lhs_package = utils.infer_object_package(self.a.tensor)

        # Determine the batch count and batch indices.
        self.batch_count = 0
        self.batch_indices = None  # Needed only for implicit batching.
        if self.explicitly_batched_lhs:
            self.batch_count = len(self.a)
        elif self.implicitly_batched_lhs:
            self.batch_count = math.prod(self.lhs_batch_shape)
            # Create the sequence of batch coordinates to use for creating batched CSR
            # matrix type.
            self.batch_indices = tuple(itertools.product(*list(map(range, self.lhs_batch_shape))))  # type: ignore

        self.logger.info(f"The LHS package is {self.lhs_package}.")
        if self.explicitly_batched_lhs:
            self.logger.info(f"The LHS is explicitly batched, with a batch count of {self.batch_count}.")
        elif self.implicitly_batched_lhs:
            self.logger.info(
                f"The LHS is implicitly batched with shape = {self.a.shape}, dtype = {self.a.dtype}, \
and index type = {self.a.index_type}."
            )
            self.logger.info(f"The LHS batch shape is {self.lhs_batch_shape}, with a batch count of  {self.batch_count}.")
            self.logger.debug(f"The batch indices (generated from the LHS) are: {self.batch_indices}.")

        # Wrap the RHS. It can be a N-D tensor or a sequence of matrices or vectors.
        self.rhs_batch_shape = ()
        self.implicitly_batched_rhs = self.explicitly_batched_rhs = False
        if isinstance(b, Sequence):
            self.explicitly_batched_rhs = True
            self.b: Any = tensor_wrapper.wrap_operands(b)
            # For explicitly batched RHS, check that all operands are vectors or matrices.
            if any(len(r.shape) > 2 for r in self.b):
                raise TypeError(
                    f"""Every RHS object in the batched RHS provided as a sequence (explicit batching) must be a dense matrix
or vector. The specified RHS sequence = {b}."""
                )
            rhs_batch_count = len(self.b)
        else:
            self.b = tensor_wrapper.wrap_operand(b)  # type:ignore
            self.implicitly_batched_rhs = len(self.b.shape) > 2
            self.rhs_batch_shape = tuple(self.b.shape[:-2])  # type: ignore
            if self.implicitly_batched_lhs and self.lhs_batch_shape != self.rhs_batch_shape:
                raise TypeError(
                    f"The batch shapes for the LHS {self.lhs_batch_shape} and RHS {self.rhs_batch_shape} must match."
                )
            rhs_batch_count = math.prod(self.rhs_batch_shape) if self.rhs_batch_shape else 0
            if self.batch_indices is None:
                # Create the sequence of batch coordinates to use for creating batched dense
                # matrix type.
                self.batch_indices = tuple(itertools.product(*list(map(range, self.rhs_batch_shape))))
                self.logger.debug(f"The batch indices (generated from the RHS) are: {self.batch_indices}.")

        # The LHS can be implicitly or explicitly batched, independent from how the RHS is
        # batched. So we need to check that the batch counts match.
        if rhs_batch_count != self.batch_count:
            raise TypeError(f"The batch count for the LHS {self.batch_count} and RHS {rhs_batch_count} must match.")

        # Set the RHS package.
        if self.explicitly_batched_rhs:
            # The package is the same for explicitly batched RHS since we've have
            # successfully wrapped them.
            self.rhs_package = utils.infer_object_package(self.b[0].tensor)
        else:  # Single or implicitly-batched RHS
            self.rhs_package = utils.infer_object_package(self.b.tensor)

        self.logger.info(f"The RHS package is {self.rhs_package}.")
        if self.explicitly_batched_rhs:
            self.logger.info(f"The RHS is explicitly batched, with a batch count of {rhs_batch_count}.")
        elif self.implicitly_batched_rhs:
            self.logger.info(f"The RHS is implicitly batched with shape = {self.b.shape} and dtype = {self.b.dtype}")
            self.logger.info(f"The RHS batch shape is {self.rhs_batch_shape}, with a batch count of  {rhs_batch_count}.")

        # Note that while the LHS and RHS packages can be different they must be compatible
        # (such as scipy-numpy, cupyx-cupy).
        if (self.lhs_package, self.rhs_package) not in cudss_utils.COMPATIBLE_LHS_RHS_PACKAGES:
            raise TypeError(
                f"""The LHS package {self.lhs_package} and RHS package {self.rhs_package} are not part of the
compatible choices: {cudss_utils.COMPATIBLE_LHS_RHS_PACKAGES}."""
            )

        # Get key LHS attributes and perform more basic checks.
        if self.explicitly_batched_lhs:
            # For sequence, the get_[attribute]() functions check for consistency within
            # the sequence.
            self.device_id = utils.get_operands_device_id(self.a)
            self.value_type = utils.get_operands_dtype(self.a)
            self.index_type = sp_utils.get_operands_index_type(self.a)
            self.lhs_shape = shapes = tuple(o.shape for o in self.a)
            if any(len(s) != 2 or s[0] != s[1] for s in shapes):
                raise TypeError("Each object in an explicitly-batched LHS must be a CSR matrix of shape (N, N).")
            self._N = tuple(s[0] for s in shapes)
            self.lhs_nnz = tuple(o.values.size for o in self.a)
        else:  # Single or implicitly-batched LHS
            self.device_id = self.a.device_id
            self.value_type = self.a.dtype
            self.index_type = self.a.index_type
            message = f"The LHS {self.a.tensor} must be a CSR matrix of shape (N, N) or CSR tensor with shape (..., N, N)."
            if self.a.num_dimensions < 2:
                raise TypeError(message)
            n, m = self.a.shape[-2:]
            if n != m:
                raise TypeError(message)
            self.lhs_shape = self.a.shape
            self._N = n
            self.lhs_nnz = self.a.values.size

        # Note that torch by default uses int64 which is not supported. SciPy and CuPy adapt
        # the index type based on the dimension.
        if self.index_type not in VALID_INDEX_TYPES:
            raise TypeError(
                f"The index type {self.index_type} is not supported. The supported index types are {VALID_INDEX_TYPES}."
            )

        if self.value_type not in VALID_DTYPES:
            raise TypeError(
                f"The dtype (value type) {self.value_type} is not supported. The supported dtypes are {VALID_DTYPES}."
            )

        # Get key RHS attributes and perform more basic checks.
        if self.explicitly_batched_rhs:
            # For a sequence, the get_attribute functions check for consistency within
            # the sequence.
            rhs_device_id = utils.get_operands_device_id(self.b)
            rhs_value_type = utils.get_operands_dtype(self.b)
            self.rhs_shape = tuple(o.shape for o in self.b)
            self.rhs_strides = tuple(o.strides for o in self.b)
            if not check_rhs_sequence_layout(self.rhs_shape, self.rhs_strides):
                raise TypeError(
                    f"Each object in an explicitly-batched RHS {[o.tensor for o in self.b]} must be a dense matrix or vector \
with compact col-major layout."
                )
            # For explicitly-batched RHS, the matrix or vector is compact so we can use
            # the RHS strides for the result.
            self.result_strides = self.rhs_strides
            # Capture the RHS solution extent for compatibility check with the LHS.
            self.rhs_n = tuple(s[0] for s in self.rhs_shape)  # sample is not batched.
        else:  # Single or implicitly-batched RHS
            rhs_device_id = self.b.device_id
            rhs_value_type = self.b.dtype
            self.rhs_shape = self.b.shape
            self.rhs_strides = self.b.strides
            if not check_dense_tensor_layout(self.rhs_shape, self.rhs_strides, explicitly_batched=False):
                raise TypeError(
                    f"The RHS {self.b.tensor} must be a matrix or vector with col-major layout, and for implicitly-batched \
RHS (N-D >= 3), each matrix sample must have col-major layout (the second dimension from the end must have unit stride."
                )
            # For single or implicitly-batched RHS, the matrix may not be compact so we use
            # the axis ordering to determine the strides.
            axis_order = axis_order_in_memory(self.rhs_shape, self.rhs_strides)
            self.result_strides = calculate_strides(self.rhs_shape, axis_order)
            self.rhs_n = self.rhs_shape[-2] if len(self.rhs_shape) > 1 else self.rhs_shape[-1]

        # Consistency within LHS and RHS sequence has been checked at this point.
        # Now check that the extents match between a and b in Ax = b.
        message = "The extent N corresponding to the number of equations is not consistent between the LHS and RHS."
        if self.explicitly_batched_lhs:
            if self.explicitly_batched_rhs:
                if not all(x == y for (x, y) in zip(self._N, self.rhs_n, strict=True)):
                    raise TypeError(message)
            elif self.implicitly_batched_rhs:
                if not all(x == self.rhs_n for x in self._N):
                    raise TypeError(message)
            else:
                raise TypeError("The RHS is not batched, but the LHS is.")
        elif self.implicitly_batched_lhs:
            if self.explicitly_batched_rhs:
                if not all(x == self._N for x in self.rhs_n):
                    raise TypeError(message)
            elif self.implicitly_batched_rhs:
                if self.rhs_n != self._N:
                    raise TypeError(message)
            else:
                raise TypeError("The RHS is not batched, but the LHS is.")
        else:
            if self.rhs_n != self._N:
                raise TypeError(message)

        if self.device_id != rhs_device_id:
            raise TypeError(f"The LHS device ID {self.device_id} does not match the RHS device ID {rhs_device_id}.")

        if self.value_type != rhs_value_type:
            raise TypeError(f"The LHS dtype {self.value_type} does not match the RHS dtype {rhs_value_type}.")

        # The RHS index type is currently always set to int32.

        # The value types must currently be the same between a, x, and b.
        self.result_data_type = self.value_type

        self.logger.info(f"The device_id={self.device_id}, dtype = {self.value_type}, index type = {self.index_type}.")
        self.logger.info(f"The number of equations = {self._N}.")
        self.logger.debug(f"The LHS shape = {self.lhs_shape}.")
        self.logger.debug(f"The RHS shape = {self.lhs_shape}, strides = {self.rhs_strides}.")

        # Currently the value and index types must match between the LHS and RHS.
        self.cuda_value_type = NAME_TO_DATA_TYPE[self.value_type]
        self.cuda_index_type = NAME_TO_DATA_TYPE[self.index_type]

        # For batched LHS and RHS check package, device_id, dtype, index_type is the same.
        # Check consistency between LHS and RHS (same attributes, numpy -> scipy)
        # Check that the size (_N) of each sample is consistent between LHS and RHS.

        # Set the memory space "cuda" or "cpu".
        self.memory_space = self.a[0].device if self.explicitly_batched_lhs else self.a.device

        # Set the execution space "cuda" or "hybrid".
        self.execution_space = self.execution_options.name

        # Note #1: The device ID of the result is the same as that of the operands for
        # hybrid execution since we don't copy them. Capture the original device here
        # before it's potentially changed below. Also see note #3.
        self.result_device_id = self.device_id

        # Track potential issue with copying NumPy ndarray to GPU.
        # TODO: This should be fixed when we use cuda.core for copying across memspace.
        rhs_layout_flag = self.rhs_package == "numpy" and len(self.rhs_shape) > 2 and self.implicitly_batched_rhs

        if self.device_id == "cpu":
            # Note #2: We always set the RHS package to one that supports GPU execution,
            # irrespective of whether it's hybrid or CUDA execution since we need the
            # current stream.
            if self.rhs_package == "numpy":
                self.rhs_package = "cupy"
                # TODO: remove this call after cupy is dropped.
                tensor_wrapper.maybe_register_package("cupy")
            # For CPU operands, set the device ID based on the execution options.
            self.device_id = self.execution_options.device_id
        self.logger.info(
            f"The operands' memory space is {self.memory_space}, and the execution space is on device {self.device_id}."
        )

        self.copy_across_memspace = False
        if self.execution_space == "hybrid":
            # No need to copy CPU operands.
            if self.batched:
                raise TypeError(f"Batching is not supported for hybrid execution: {self.execution_options}.")

            # For non-batched b, matrix (multiple-RHS) is not supported for CPU memory
            # space (seems to be ignored for CUDA as well).
            if len(self.b.shape) > 1:
                raise TypeError(f"Matrix RHS (multiple RHS) is not supported for hybrid execution: {self.execution_options}.")
        else:  # execute in the CUDA space.
            # The operands must be on the GPU even for *hybrid memory* mode for CUDA
            # execution.
            self.copy_across_memspace = self.memory_space != "cuda"
            # Note #3: For CUDA execution, the result's device ID should be that of the
            # execution device (self.device_id), where the operands also reside (may or
            # may not be copied, depending on their original memspace). Also see note #1.
            self.result_device_id = self.device_id

        # Flag whether to copy CPU operands to GPU or not, based on execution option.
        #   - if hybrid execution, check not batched and nrhs==1. Accept CPU or GPU
        #     operands.
        #   - if hybrid memory and CUDA execution, need to copy to GPU.

        # Create the stream holder.
        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.rhs_package)
        self.logger.info(f"The specified stream for the DirectSolver ctor is {stream_holder.obj}.")

        # cupy.asarray() doesn't preserve layout for > 2D arrays when copying from CPU
        # to GPU.
        if self.copy_across_memspace and rhs_layout_flag:
            raise TypeError(
                f"Implicit RHS batching for NumPy ndarrays is currently not supported with CUDA execution \
since the layout cannot be preserved when copying to the GPU (shape = {self.rhs_shape}, strides = {self.rhs_strides})."
            )

        # Copy operands to device if needed.
        if self.copy_across_memspace:
            self.a = copy_single_or_sequence(self.a, self.device_id, stream_holder)
            self.b = copy_single_or_sequence(self.b, self.device_id, stream_holder)

        # Create (batched or not, CSR or dense) matrix pointers for the LHS and RHS.
        self.resources_a, self.a_ptr = cudss_utils.create_cudss_csr_wrapper(
            self.cuda_index_type,
            self.cuda_value_type,
            self.index_type,
            self.options.sparse_system_type,
            self.options.sparse_system_view,
            self.batch_indices,
            self.a,
            stream_holder,
        )
        self.resources_b, self.b_ptr = cudss_utils.create_cudss_dense_wrapper(
            self.cuda_index_type, self.cuda_value_type, self.index_type, self.batch_indices, self.b, stream_holder
        )

        # Use `b` for creating the (potentially explicitly or implicitly batched) solution
        # matrix or vector. The pointers will be updated later in execute.
        self.resources_x, self.x_ptr = cudss_utils.create_cudss_dense_wrapper(
            self.cuda_index_type, self.cuda_value_type, self.index_type, self.batch_indices, self.b, stream_holder
        )

        # Create or set handle, and create config and data pointers.
        with utils.device_ctx(self.device_id):
            if self.options.handle is not None:
                self.own_handle = False
                self.handle = self.options.handle
                self.logger.info(f"The library handle has been set to the specified value: {self.handle}.")
            else:
                self.own_handle = True
                self.handle = cudss.create()
                self.logger.info(f"The library handle has been created: {self.handle}.")

            self.config_ptr = cudss.config_create()
            self.data_ptr = cudss.data_create(self.handle)

        # Create the config interfaces for the various phases.
        self._plan_config = cudss_config_ifc.PlanConfig(self)
        self._factorization_config = cudss_config_ifc.FactorizationConfig(self)
        self._solution_config = cudss_config_ifc.SolutionConfig(self)

        # Create the data interfaces for the various phases.
        self._plan_info = cudss_data_ifc.PlanInfo(self)
        self._factorization_info = cudss_data_ifc.FactorizationInfo(self)

        # Set the threading layer, if available.
        threading_lib = get_threading_lib(self.options.multithreading_lib)
        if threading_lib is not None:
            cudss.set_threading_layer(self.handle, threading_lib)
        else:
            self.logger.warning(
                "No multithreading interface library was specified using the \
DirectSolverOptions. The performance of CPU operations like planning will \
be significantly lower than if you provide a multithreading library."
            )

        # This doesn't guarantee that the library is usable, just that it is present.
        self.multithreading = threading_lib is not None

        # Set private attributes based on the options.
        self._internal_config = cudss_config_ifc.InternalConfig(self)

        # Set hybrid execution options. We have already checked that for batching and
        # matrix RHS, which are not currently supported.
        if self.execution_space == "hybrid":
            self._internal_config.hybrid_execute_mode = 1
            num_threads = self.execution_options.num_threads
            if num_threads is not None:
                if num_threads > 1 and threading_lib is None:
                    raise ValueError(f"""The threading library must be specified if the number of threads is more
than 1 (num_threads = {num_threads}.""")
                self._internal_config.host_nthreads = num_threads

        # Set CUDA execution options (including hybrid memory mode).
        if self.execution_space == "cuda":
            hmo = self.execution_options.hybrid_memory_mode_options
            self._internal_config.hybrid_mode = hmo.hybrid_memory_mode
            # Set device memory limit for hybrid memory.
            if hmo.hybrid_device_memory_limit is not None:
                memory_limit = utils.get_memory_limit_from_device_id(hmo.hybrid_device_memory_limit, self.device_id)
                self.logger.info(f"The hybrid memory limit is {formatters.MemoryStr(memory_limit)}.")
                self._internal_config.hybrid_device_memory_limit = memory_limit
            self._internal_config.use_cuda_register_memory = hmo.register_cuda_memory

        # State tracking attributes.
        self.solver_planned = False
        self.solver_factorized = False

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == "cpu"
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )

        # The result (solution) class is that of the wrapped RHS.
        self.result_class = self.b[0].__class__ if self.explicitly_batched_rhs else self.b.__class__

        # The result shape is a single value or a sequence, depending on whether the RHS
        # is explicitly or implicitly batched as set above.
        self.result_shape = self.rhs_shape

        # Tracking attributes.
        self.solver_planned = False
        self.solver_factorized = False

        self.valid_state = True
        self.logger.info("The sparse direct solver operation has been created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    def _check_valid_solver(self, *args, **kwargs):
        """
        Check if the DirectSolver object is alive and well.
        """
        if not self.valid_state:
            raise InvalidDirectSolverState("The DirectSolver object cannot be used after resources are free'd")

    def _check_valid_operands(self, *args, **kwargs):
        """
        Check if the operands are available for the operation.
        """
        what = kwargs["what"]
        if self.a is None or self.b is None:
            raise RuntimeError(
                f"{what} cannot be performed if the operands have been set to None. Use reset_operands() to set the "
                f"desired input before using performing the {what.lower()}."
            )

    def _check_planned(self, *args, **kwargs):
        what = kwargs["what"]
        if not self.solver_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _check_factorized(self, *args, **kwargs):
        what = kwargs["what"]
        if not self.solver_factorized:
            raise RuntimeError(f"{what} cannot be performed before factorize() has been called.")

    def _allocate_single_result(self, stream_holder: StreamHolder | None, log_debug):
        if log_debug:
            self.logger.debug("Beginning output (empty) tensor creation...")
            self.logger.debug(
                f"""The output tensor shape = {self.result_shape}, with strides = {self.result_strides}
                and data type '{self.result_data_type}'."""
            )
        result = utils.create_empty_tensor(
            self.result_class,
            self.result_shape,
            self.result_data_type,
            self.result_device_id,  # see notes #1 and #2.
            stream_holder=None if self.result_device_id == "cpu" else stream_holder,
            verify_strides=False,  # the strides are computed so that they are contiguous
            strides=self.result_strides,
        )
        if log_debug:
            self.logger.debug("The output (empty) tensor has been created.")
        return result

    def _allocate_batched_result(self, stream_holder: StreamHolder | None, log_debug):
        if log_debug:
            self.logger.debug("Beginning output tensor sequence creation...")
            self.logger.debug(
                f"""The output tensor sequence shape = {self.result_shape}, with strides = {self.result_strides}
                and data type '{self.result_data_type}'."""
            )
        result = tuple(
            utils.create_empty_tensor(
                self.result_class,
                shape,
                self.result_data_type,
                self.result_device_id,  # see notes #1 and #2.
                stream_holder=None if self.result_device_id == "cpu" else stream_holder,
                verify_strides=False,  # the strides are computed so that they are contiguous
                strides=strides,
            )
            for (shape, strides) in zip(self.result_shape, self.result_strides, strict=True)
        )

        if log_debug:
            self.logger.debug("The output tensor sequence has been created.")
        return result

    @property
    def plan_config(self):
        """
        An accessor to configure or query the solver planning phase attributes.

        Returns:
            A :class:`DirectSolverPlanConfig` object, whose attributes can be set (or
            queried) to configure the planning phase.

        See Also:
            :class:`DirectSolverPlanConfig`, :meth:`plan`.
        """
        return self._plan_config

    @property
    def factorization_config(self):
        """
        An accessor to configure or query the solver factorization phase attributes.

        Returns:
            A :class:`DirectSolverFactorizationConfig` object, whose attributes can be set
            (or queried) to configure the factorization phase.

        See Also:
            :class:`DirectSolverFactorizationConfig`, :meth:`factorize`.
        """
        return self._factorization_config

    @property
    def solution_config(self):
        """
        An accessor to configure or query the solver solution phase attributes.

        Returns:
            A :class:`DirectSolverSolutionConfig` object, whose attributes can be set
            (or queried) to configure the factorization phase.

        See Also:
            :class:`DirectSolverSolutionConfig`, :meth:`solve`.
        """
        return self._solution_config

    @property
    def plan_info(self):
        """
        An accessor to get information about the solver planning phase.

        Returns:
            A :class:`DirectSolverPlanInfo` object, whose attributes can be queried for
            information regarding the planning phase.

        See Also:
            :class:`DirectSolverPlanInfo`, :meth:`plan`.
        """
        return self._plan_info

    @property
    def factorization_info(self):
        """
        Query solver factorization information
        (see :class:`nvmath.sparse.advanced.DirectSolverFactorizationInfo`).
        An accessor to get information about the solver factorization phase.

        Returns:
            A :class:`DirectSolverFactorizationInfo` object, whose attributes can be
            queried for information regarding the factorization phase.

        See Also:
            :class:`DirectSolverFactorizationInfo`, :meth:`factorize`.
        """
        return self._factorization_info

    @utils.precondition(_check_valid_solver)
    def reset_operands(
        self,
        a=None,
        b=None,
        *,
        stream: utils.AnyStream | int | None = None,
    ):
        """
        Reset the operands held by this :class:`DirectSolver` instance.

        This method has two use cases:

        (1) it can be used to provide new operands for execution when the original
            operands are on the CPU

        (2) it can be used to release the internal reference to the previous operands
            and make their memory available for other use by passing ``None`` for *all*
            arguments. In this case, this method must be called again to provide the
            desired operands before another call to planning and execution APIs like
            :meth:`plan`, :meth:`factorize`, or :meth:`solve`.

        This method will perform various checks on the new operands to make sure:

        - The shapes, index and data types match those of the old ones.

        - The packages that the operands belong to match those of the old ones.

        - If input tensors are on GPU, the device must match.

        Args:
            a: {a}

            b: {b}

            stream: {stream}

        Examples:

            >>> import cupy as cp
            >>> import cupyx.scipy.sparse as sp
            >>> import nvmath

            Prepare sample input data.

            >>> n = 8
            >>> a = sp.random(n, n, density=0.15, format="csr", dtype="float64")
            >>> a += sp.diags([2.0] * n, format="csr", dtype="float64")

            Create the RHS, which can be a matrix or vector in column-major layout.

            >>> b = cp.ones((n,), dtype="float64")

            Specify, plan, factorize and solve a @ x = b. Use the stateful object as a
            context manager to automatically release resources.

            >>> with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
            ...     # Plan the operation.
            ...     plan_info = solver.plan()
            ...
            ...     # Factorize the system.
            ...     fac_info = solver.factorize()
            ...
            ...     # Solve the factorized system for the first result.
            ...     x1 = solver.solve()
            ...
            ...     # Reset the RHS to a new CuPy ndarray.
            ...     c = cp.random.rand(n, dtype="float64")
            ...     solver.reset_operands(b=c)
            ...
            ...     # Solve for the second result corresponding to the updated operands.
            ...     x2 = solver.solve()

            .. tip:: If only a subset of operands are reset, the operands that are not
                reset hold their original values.

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands` is
            equivalent to updating the operand `b` in-place, i.e, replacing
            ``solver.reset_operands(b=c)`` with ``b[:]=c``.

            .. danger:: Updating the operand in-place can only yield the expected result
                under the additional constraints below:

                - The operand is on the GPU (more precisely, the operand memory space should
                  be accessible from the execution space).

                - The user has called :meth:`factorize` if needed (they are not relying on
                  iterative refinement for example)

                Assuming that the constraint above is satisfied, updating an operand
                in-place is preferred to avoid the extra checks incurred in
                :meth:`reset_operands`.

            For more details, please refer to `the reset operand example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver/example05_reset_operands.py>`_.
        """

        if a is None and b is None:
            self.a = self.b = None
            self.logger.info("The operands have been reset to None.")
            return

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.rhs_package)

        # Update LHS.
        if a is not None:
            if isinstance(a, Sequence) and not self.explicitly_batched_lhs:
                raise TypeError("The specified type for 'a` is a sequence while the original type is {type.self.a.tensor}.")

            # Wrap A.
            try:
                a = sp_utils.wrap_sparse_operands(a)
            except Exception as e:
                raise TypeError(
                    """The LHS 'a' must be an N-D sparse CSR array/tensor or a sequence of 2D sparse CSR array/tensor from one
of the supported packages: CuPy, PyTorch, or SciPy."""
                ) from e

            explicitly_batched = isinstance(a, Sequence)
            if explicitly_batched:
                # Successfully wrapping A means that these package is the same for all
                # sparse operands.
                lhs_package = utils.infer_object_package(a[0].tensor)
                # The get_() helpers ensure that the attributes are the same for all
                # operands.
                device_id = utils.get_operands_device_id(a)
                memory_space = a[0].device
                value_type = utils.get_operands_dtype(a)
                index_type = sp_utils.get_operands_index_type(a)

                shape = tuple(o.shape for o in a)
                nnz = tuple(o.values.size for o in a)

            else:  # Single or implicitly-batched LHS
                lhs_package = utils.infer_object_package(a.tensor)
                device_id = a.device_id
                memory_space = a.device
                value_type = a.dtype
                index_type = a.index_type

                shape = a.shape
                nnz = a.values.size

            # Check package, device ID, dtype, and index type.
            if lhs_package != self.lhs_package:
                raise TypeError("The package for 'a' ({lhs_package}) doesn't match the original one ({self.lhs_package}).")

            if memory_space != self.memory_space:
                raise TypeError(
                    "The memory space for 'a' ({memory_space}) doesn't match the original one ({self.memory_space})."
                )

            if device_id != "cpu" and device_id != self.device_id:
                raise TypeError("The device id for 'a' ({device_id}) doesn't match the original one ({self.device_id}).")

            if value_type != self.value_type:
                raise TypeError("The dtype for 'a' ({value_type}) doesn't match the original one ({self.value_type}).")

            if index_type != self.index_type:
                raise TypeError("The index type for 'a' ({index_type}) doesn't match the original one ({self.index_type}).")

            # Checking that the shape is consistent also checks the batch count for both
            # implicit and explicit batching.
            if shape != self.lhs_shape:
                raise TypeError(f"The shape of 'a' ({shape}) doesn't match the original one ({self.lhs_shape}).")

            if nnz != self.lhs_nnz:
                raise TypeError(f"The number of non-zeros of 'a' ({nnz}) doesn't match the original one ({self.lhs_nnz}).")

            # Copy operand if needed, and replace object reference.
            if self.copy_across_memspace:
                # Copy operand into original buffer if it exists or create new ones.
                log_warning = False
                if explicitly_batched:
                    if self.a is not None:
                        for x, y in zip(self.a, a, strict=True):
                            x.copy_(y, stream_holder)
                    else:
                        self.a = [x.to(self.device_id, stream_holder) for x in a]
                        log_warning = True
                else:
                    if self.a is not None:
                        self.a.copy_(a, stream_holder)
                    else:
                        self.a = a.to(self.device_id, stream_holder)
                        log_warning = True
                if log_warning:
                    self.logger.warning(
                        "The LHS buffer pointers have changed when copying between CPU-GPU, which requires calling \
plan() and factorize() again even if the sparsity structure is identical."
                    )
            else:
                # Invalidate the plan, since the buffer pointers could have changed.
                self.solver_planned = self.solver_factorized = False
                self.logger.warning(
                    "The specified LHS may have different buffers for the compressed row or column indices \
or values, which requires calling plan() and factorize() again even if the sparsity structure is identical. To avoid this, it \
is recommended to update the values in place and refactorize if needed."
                )
                self.a = a

            # Update the pointer references, and keep reference to the internal buffers.
            self.resources_ra = cudss_utils.update_cudss_csr_ptr_wrapper(
                self.a_ptr, batch_indices=self.batch_indices, new_lhs=self.a, stream_holder=stream_holder
            )

            self.logger.warning(
                "Resetting the LHS 'a' typically requires calling factorize() again. An exception is the use of \
iterative refinement during solve(), but it's the user's responsibility to check that the solution has converged."
            )

        # Update RHS.
        if b is not None:
            # Wrap b.
            explicitly_batched = isinstance(b, Sequence)
            if explicitly_batched:
                b = tensor_wrapper.wrap_operands(b)
                rhs_package = utils.get_operands_package(b)

                # The get_() helpers ensure that the attributes are the same for all
                # operands.
                device_id = utils.get_operands_device_id(b)
                memory_space = b[0].device
                value_type = utils.get_operands_dtype(b)
                shape = tuple(o.shape for o in b)
                strides = tuple(o.strides for o in b)
            else:  # Single or implicitly-batched RHS
                b = tensor_wrapper.wrap_operand(b)
                rhs_package = utils.infer_object_package(b.tensor)

                device_id = b.device_id
                memory_space = b.device
                value_type = b.dtype
                shape = b.shape
                strides = b.strides

            # Handle cupy <> numpy asymmetry. See note #2.
            if rhs_package == "numpy":
                rhs_package = "cupy"

            # Check package, device ID, shape, strides, and dtype.
            if rhs_package != self.rhs_package:
                raise TypeError(f"The package for 'b' ({rhs_package}) doesn't match the original one ({self.rhs_package}).")

            if memory_space != self.memory_space:
                raise TypeError(
                    f"The memory space for 'b' ({memory_space}) doesn't match the original one ({self.memory_space})."
                )

            if device_id != "cpu" and device_id != self.device_id:
                raise TypeError(f"The device id for 'b' ({device_id}) doesn't match the original one ({self.device_id}).")

            if value_type != self.value_type:
                raise TypeError(f"The dtype for 'b' ({value_type}) doesn't match the original one ({self.value_type}).")

            # Checking that the shape is consistent also checks the batch count for both
            # implicit and explicit batching.
            if shape != self.rhs_shape:
                raise TypeError(f"The shape of 'b' ({shape}) doesn't match the original one ({self.rhs_shape}).")

            if strides != self.rhs_strides:
                raise TypeError(f"The strides of 'b' ({strides}) don't match the original one ({self.rhs_strides}).")

            # Copy operand if needed, and replace object reference.
            # Copy operand if needed, and replace object reference.
            if self.copy_across_memspace:
                # Copy operand into original buffer if it exists or create new ones.
                if explicitly_batched:
                    if self.b is not None:
                        for x, y in zip(self.b, b, strict=True):
                            x.copy_(y, stream_holder)
                    else:
                        self.b = [x.to(self.device_id, stream_holder) for x in b]
                else:
                    if self.b is not None:
                        self.b.copy_(b, stream_holder)
                    else:
                        self.b = b.to(self.device_id, stream_holder)
            else:
                self.b = b

            # Update the pointer references, and keep reference to the internal buffers.
            self.resources_rb = cudss_utils.update_cudss_dense_ptr_wrapper(
                self.b_ptr, batch_indices=self.batch_indices, new_rhs=self.b, stream_holder=stream_holder
            )

    @utils.precondition(_check_valid_solver)
    @utils.precondition(_check_valid_operands, "Planning")
    def plan(self, *, stream: utils.AnyStream | None = None):
        """
        Plan the sparse direct solve (reordering to minimize fill-in, and symbolic
        factorization). The planning phase can be optionally configured through
        the property :attr:`plan_config` (an object of type
        :class:`DirectSolverPlanConfig`). Planning returns a :class:`DirectSolverPlanInfo`
        object, which can also be accessed through the property :attr:`plan_info`.

        Args:
            stream: {stream}

        Returns:
            A :class:`DirectSolverPlanInfo` object, whose attributes can be queried for
            information regarding the plan.

        See Also:
            :attr:`plan_config`, :class:`DirectSolverPlanConfig`,
            :class:`DirectSolverPlanInfo`.

        Examples:

            >>> import cupy as cp
            >>> import cupyx.scipy.sparse as sp
            >>> import nvmath

            Prepare sample input data.

            >>> n = 8
            >>> a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
            >>> a += sp.diags([2.0] * n, format="csr", dtype="float64")

            Create the RHS, which can be a matrix or vector in column-major layout.

            >>> b = cp.ones((n,), dtype="float64")

            Specify, configure, and plan a @ x = b. Use the stateful object as a context
            manager to automatically release resources.

            >>> with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
            ...     # Configure the reordering algorithm for the plan.
            ...     plan_config = solver.plan_config
            ...     plan_config.algorithm = nvmath.sparse.advanced.DirectSolverAlgType.ALG_1
            ...     # Plan the operation using the specified plan configuration, which
            ...     # returns a DirectSolverPlanInfo object.
            ...     plan_info = solver.plan()
            ...     # Query the column permutation, memory estimates, ...
            ...     plan_info.col_permutation
            array([6, 1, 4, 0, 3, 2, 5, 7], dtype=int32)

        Further examples can be found in the `nvmath/examples/sparse/advanced/direct_solver
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver>`_
        directory.
        """
        r = self._execute(phase=cudss.Phase.ANALYSIS, stream=stream)
        self.solver_planned = True
        return r

    @utils.precondition(_check_valid_solver)
    @utils.precondition(_check_planned, "Factorization")
    @utils.precondition(_check_valid_operands, "Factorization")
    def factorize(self, *, stream: utils.AnyStream | None = None):
        """
        Factorize the system of equations. Numerical factorization is required each time
        the values in the LHS change, unless (for example) iterative refinement is used
        during the solve and it converges (see :attr:`solution_config`).

        This phase can be optionally configured through the property
        :attr:`factorization_config` (an object of type
        :class:`DirectSolverFactorizationConfig`). Factorization returns a
        :class:`DirectSolverFactorizationInfo` object, which can also be accessed through
        the property :attr:`factorization_info`.

        Args:
            stream: {stream}

        Returns:
            A :class:`DirectSolverFactorizationInfo` object, whose attributes can be
            queried for information regarding the numerical factorization.

        See Also:
            :attr:`factorization_config`, :class:`DirectSolverFactorizationConfig`,
            :class:`DirectSolverFactorizationInfo`.

        Examples:

            >>> import cupy as cp
            >>> import cupyx.scipy.sparse as sp
            >>> import nvmath

            Prepare sample input data.

            >>> n = 8
            >>> a = sp.random(n, n, density=0.25, format="csr", dtype="float64")
            >>> a += sp.diags([2.0] * n, format="csr", dtype="float64")

            Create the RHS, which can be a matrix or vector in column-major layout.

            >>> b = cp.ones((n, 2), dtype="float64", order="F")

            Specify, plan, and factorize a @ x = b. Use the stateful object as a context
            manager to automatically release resources.

            >>> with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
            ...     # Plan the operation (configure if desired).
            ...     plan_info = solver.plan()
            ...     # (Optionally) configure the factorization.
            ...     fac_config = solver.factorization_config
            ...     fac_config.pivot_eps = 1e-14
            ...     # Factorize using the specified factorization configuration, which
            ...     # returns a DirectSolverFactorizationInfo object.
            ...     fac_info = solver.factorize()
            ...     # Query the number of non-zeros, inertia, ...
            ...     fac_info.lu_nnz
            40

        Further examples can be found in the `nvmath/examples/sparse/advanced/direct_solver
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver>`_
        directory.
        """
        r = self._execute(phase=cudss.Phase.FACTORIZATION, stream=stream)
        self.solver_factorized = True
        return r

    @utils.precondition(_check_valid_solver)
    @utils.precondition(_check_planned, "Solver")
    @utils.precondition(_check_factorized, "Solver")
    @utils.precondition(_check_valid_operands, "Solver")
    def solve(self, *, stream: utils.AnyStream | None = None):
        """
        Solve the factorized system of equations.

        This phase can be optionally configured through the property
        :attr:`solution_config` (an object of type
        :class:`DirectSolverSolutionConfig`).

        Args:
            stream: {stream}

        Returns:
           {result}

        See Also:
            :attr:`solution_config`, :class:`DirectSolverSolutionConfig`.

        Examples:

            >>> import cupy as cp
            >>> import cupyx.scipy.sparse as sp
            >>> import nvmath

            Prepare sample input data.

            >>> n = 8
            >>> a = sp.random(n, n, density=0.25, format="csr", dtype="float64")
            >>> a += sp.diags([2.0] * n, format="csr", dtype="float64")

            Create the RHS, which can be a matrix or vector in column-major layout.

            >>> b = cp.ones((n, 2), dtype="float64", order="F")

            Specify, plan, factorize, and solve a @ x = b for x. Use the stateful
            object as a context manager to automatically release resources.

            >>> with nvmath.sparse.advanced.DirectSolver(a, b) as solver:
            ...     # Plan the operation (configure if desired).
            ...     plan_info = solver.plan()
            ...     # Factorize the system (configure if desired).
            ...     fac_info = solver.factorize()
            ...     # (Optionally) configure the solve.
            ...     solution_config = solver.solution_config
            ...     solution_config.ir_num_steps = 10
            ...     # Solve the system based on the solution configuration set above.
            ...     x = solver.solve()

        Further examples can be found in the `nvmath/examples/sparse/advanced/direct_solver
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver>`_
        directory.
        """
        return self._execute(phase=cudss.Phase.SOLVE, stream=stream)

    def _execute(self, *, phase=None, stream=None):
        """
        For internal use only. Execute the specified operation (reordering, factorization,
        and solve).

        Args:
            phase: The operation phase (as Python `int` or enumeration of type Phase).

            stream: {stream}

        Returns:
            {plan|factorization]_info object for reordering and factorization, and the
            solution in the expected memory space for solve.
        """

        assert phase is not None, "Internal error."

        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.rhs_package)

        cudss.set_stream(self.handle, stream_holder.ptr)

        # Allocate result for only the solve phase and update the pointer.
        result = None
        if phase == cudss.Phase.SOLVE:
            # The result can be either a single tensor or a sequence of tensors, depending
            # on whether the RHS is explicitly batched.
            result_allocator = self._allocate_batched_result if self.explicitly_batched_rhs else self._allocate_single_result
            result = result_allocator(stream_holder, log_debug)
            cudss_utils.update_cudss_dense_ptr_wrapper(
                self.x_ptr, batch_indices=self.batch_indices, new_rhs=result, stream_holder=stream_holder
            )

        if log_info:
            self.logger.info(f"Starting solver phase {phase.name}...")
            self.logger.info(f"{self.call_prologue}")

        with utils.cuda_call_ctx(stream_holder, self.blocking, timing=log_info) as (
            self.last_compute_event,
            elapsed,
        ):
            cudss.execute(self.handle, phase, self.config_ptr, self.data_ptr, self.a_ptr, self.x_ptr, self.b_ptr)

        if log_info and elapsed.data is not None:
            self.logger.info(f"The solver phase {phase.name} took {elapsed.data:.3f} ms to complete.")

        if phase == cudss.Phase.ANALYSIS:
            return self.plan_info

        if phase == cudss.Phase.FACTORIZATION:
            return self.factorization_info

        # Ideally, we should set the x_ptr to 0, but this adds overhead for the batched
        # case.

        # Copy result to required memory spaces if needed.
        if self.copy_across_memspace:
            result = copy_single_or_sequence(result, "cpu", stream_holder)

        # Extract the result tensor from the wrapped result for the solution phase.
        if isinstance(result, Sequence):
            out = tuple(r.tensor for r in result)
        else:
            out = result.tensor

        return out

    def free(self):
        """Free DirectSolver resources.

        It is recommended that the :class:`DirectSolver` object be used within a context,
        but if it is not possible then this method must be called explicitly to ensure
        that the sparse direct solver resources (especially internal library objects) are
        properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Currently, workspace is allocated internally by the library.

            # Release internal resource references.
            self.resources_a = self.resources_b = self.resources_x = None
            self.resources_ra = self.resources_rb = None

            # Free matrix pointers.
            cudss.matrix_destroy(self.x_ptr)
            cudss.matrix_destroy(self.b_ptr)
            cudss.matrix_destroy(self.a_ptr)
            self.x_ptr = self.b_ptr = self.a_ptr = None

            # Free config and data pointers.
            cudss.data_destroy(self.handle, self.data_ptr)
            cudss.config_destroy(self.config_ptr)
            self.data_ptr = self.config_ptr = None

            # Free handle if we own it.
            if self.handle is not None and self.own_handle:
                cudss.destroy(self.handle)
                self.handle, self.own_handle = None, False

        except Exception as e:
            self.logger.critical("Internal error: only part of the DirectSolver object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The DirectSolver object's resources have been released.")


@utils.docstring_decorator(SHARED_DSS_DOCUMENTATION, skip_missing=False)
def direct_solver(
    a,
    b,
    /,
    *,
    options: DirectSolverOptions | None = None,
    execution: ExecutionCUDA | ExecutionHybrid | None = None,
    stream: utils.AnyStream | int | None = None,
):
    """
    Solve :math:`a @ x = b` for :math:`x`. This function-form API is a wrapper around the
    stateful :class:`DirectSolver` object APIs and is meant for *single* use (the user needs
    to perform just one sparse direct solve, for example), in which case there is no
    possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing
    in a :class:`logging.Logger` object to :class:`DirectSolverOptions` or by setting the
    appropriate options in the root logger object, which is used by default:

    >>> import logging
    >>> logging.basicConfig(
    ...     level=logging.INFO,
    ...     format="%(asctime)s %(levelname)-8s %(message)s",
    ...     datefmt="%m-%d %H:%M:%S",
    ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python :mod:`logging` module.

    Args:
        a: {a}

        b: {b}

        options: {options}

        execution: {execution}

        stream: {stream}

    Returns:
        {result}

    .. _Semantics:

    Semantics:
        {semantics}

    See Also:
        :class:`DirectSolver`, :class:`DirectSolverOptions`, :class:`ExecutionCUDA`,
        :class:`ExecutionHybrid`.

    Examples:

        >>> import cupy as cp
        >>> import cupyx.scipy.sparse as sp
        >>> import nvmath

        Create a sparse float32 ndarray in CSR format on the CPU for the LHS.

        >>> n = 16
        >>> a = sp.random(n, n, density=0.5, format="csr", dtype="float32")

        Ensure that the randomly-generated LHS is not singular.

        >>> a += sp.diags([2.0] * n, format="csr", dtype="float32")

        The RHS can be a vector or matrix. Here we create a random matrix with 4 columns
        (indicating 4 vectors to be solved for) in column-major format.

        >>> b = cp.random.rand(4, n, dtype="float32").T

        Solve a @ x = b for x.

        >>> x = nvmath.sparse.advanced.direct_solver(a, b)

        Batching can be specified, explicitly or implicitly, following the semantics
        described above. Here we explicitly batch the LHS since CuPy doesn't support 3D CSR,
        while we implicitly batch the RHS.

        Create an explicit batch of two CSR matrices:

        >>> batch = 2
        >>> a = [a] * batch
        >>> a[1] *= 10.0

        Create a 3D ndarray, with each sample in the batch having column-major layout.

        >>> b = cp.random.rand(batch, 4, n, dtype="float32").transpose(0, 2, 1)

        Solve the batched system a @ x = b for x, where x has the same shape as b.

        >>> x = nvmath.sparse.advanced.direct_solver(a, b)

        Options can be provided to the sparse direct solver using
        :class:`DirectSolverOptions`, and the execution space can be specified using the
        ``execution`` option. Refer to :class:`DirectSolver` and the GitHub link below for
        examples.

    Notes:

        - This function is a convenience wrapper around :class:`DirectSolver` and is
          specifically meant for *single* use.

    Further examples can be found in the `nvmath/examples/sparse/advanced/direct_solver
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/sparse/advanced/direct_solver>`_
    directory.
    """

    with DirectSolver(
        a,
        b,
        options=options,
        execution=execution,
        stream=stream,
    ) as solver:
        # Planning and factorization information cannot be returned without making copies
        # because of scope (the interfaces need the solver object, which is released when
        # the function returns).
        solver.plan(stream=stream)

        solver.factorize(stream=stream)

        r = solver.solve(stream=stream)

    return r
