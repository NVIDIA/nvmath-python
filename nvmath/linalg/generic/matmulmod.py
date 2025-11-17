# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "Matmul",
    "matmul",
]

import dataclasses
import logging
from typing import TypeAlias
from collections.abc import Sequence

import numpy as np
import cuda.core.experimental as ccx

from nvmath.bindings import cublas
from nvmath._internal import templates
from nvmath.internal import utils, tensor_wrapper, typemaps, formatters
from nvmath.linalg._internal.batch import BatchTraits
from nvmath.linalg._internal.layout import BLASMMTraits, BLASMatrixTraits, check_extents, check_strides
from nvmath.linalg.generic._configuration import (
    GeneralMatrixQualifier,
    MatrixQualifier,
    matrix_qualifiers_dtype,
    MatmulOptions,
    select_blas_mm_function,
    vector_to_square,
)
from nvmath.linalg.advanced.matmulmod import SHARED_MM_DOCUMENTATION
from nvmath.linalg.generic._dtype import check_dtype

AnyTensor: TypeAlias = tensor_wrapper.AnyTensor
SideMode: TypeAlias = cublas.SideMode
FillMode: TypeAlias = cublas.FillMode
DiagType: TypeAlias = cublas.DiagType


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCUDA(templates.ExecutionCUDA):
    """
    A data class for providing GPU execution options.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.

    .. seealso::
        :class:`ExecutionCPU`, :class:`Matmul`, :func:`matmul`
    """

    pass


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCPU(templates.ExecutionCPU):
    """
    A data class for providing CPU execution options.

    Attributes:
        num_threads: The number of CPU threads used to execute the operation.
                     If not specified, defaults to the number of CPU cores available to the
                     process.

    .. seealso::
       :class:`ExecutionCUDA`, :class:`Matmul`, :func:`matmul`,
       :func:`~nvmath.bindings.nvpl.blas.set_num_threads_local`
    """

    pass


class InvalidMatmulState(Exception):
    pass


GENERIC_MM_DOCUMENTATION = SHARED_MM_DOCUMENTATION.copy()
GENERIC_MM_DOCUMENTATION.update(
    {
        "qualifiers": """\
If desired, specify the matrix qualifiers as a :class:`numpy.ndarray` of
:class:`~nvmath.linalg.generic.matrix_qualifiers_dtype` objects of length <= 3 corresponding to the operands `a`, `b`, and
`c`. By default, :class:`GeneralMatrixQualifier` is assumed for each tensor. See
:ref:`matrix-tensor-qualifiers` for the motivation behind qualifiers.""".replace("\n", " "),
        #
        "execution": """\
Specify execution space options for the Matmul as a :class:`ExecutionCUDA` or :class:`ExecutionCPU` object. If not specified,
the execution space will be selected to match operand's storage (in GPU or host memory), and the corresponding
:class:`ExecutionCUDA` or :class:`ExecutionCPU` object will be default-constructed.""".replace("\n", " "),
        #
        "options": """\
Specify options for the matrix multiplication as a :class:`MatmulOptions` object. If not specified, the
value will be set to the default-constructed ``MatmulOptions`` object.""".replace("\n", " "),
        #
        "result": """\
The result of the specified matrix multiplication, which remains on the same device and belong to the
same package as the input operands.""".replace("\n", " "),
        #
        "semantics": """\
        .. _semantics:

        The semantics of the matrix multiplication follows :func:`numpy.matmul` semantics, with some restrictions.

        * Batching is not supported in this API, but is planned for a future release. See the advanced API
          (:func:`nvmath.linalg.advanced.matmul`) for an API that supports batching.
        * Broadcasting `c` is not supported in this API, but may be supported in the future. See the advanced API
          (:func:`nvmath.linalg.advanced.matmul`) for an API that supports broadcasting `c`.

        In addition, the semantics for the fused matrix addition are described below:

        * If arguments `a` and `b` are matrices, they are multiplied according to the rules of matrix multiplication.
        * If argument `a` is 1-D, it is promoted to a matrix by prefixing ``1`` to its dimensions. After matrix
          multiplication, the prefixed ``1`` is removed from the result's dimensions.
        * If argument `b` is 1-D, it is promoted to a matrix by appending ``1`` to its dimensions. After matrix
          multiplication, the appended ``1`` is removed from the result's dimensions.
        * The operand for the matrix addition `c` must be the expected shape of the result of the matrix multiplication.

""".strip(),
    }
)


@utils.docstring_decorator(GENERIC_MM_DOCUMENTATION, skip_missing=False)
class Matmul(templates.StatefulAPI[MatmulOptions]):
    """
    Create a stateful object encapsulating the specified matrix multiplication computation
    :math:`\\alpha a @ b + \\beta c` and the required resources to perform the operation. A
    stateful object can be used to amortize the cost of preparation (planning in the case of
    matrix multiplication) across multiple executions (also see the :ref:`Stateful APIs
    <host api types>` section).

    The function-form API :func:`matmul` is a convenient alternative to using stateful
    objects for *single* use (the user needs to perform just one matrix multiplication, for
    example), in which case there is no possibility of amortizing preparatory costs. The
    function-form APIs are just convenience wrappers around the stateful object APIs.

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation
       for this specific matrix multiplication operation.
    3. **Execution**: Perform the matrix multiplication computation with :meth:`execute`.

    Detailed information on what's happening in the various phases described above can be
    obtained by passing in a :class:`logging.Logger` object to :class:`MatmulOptions` or by
    setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python `logging` module.

    Args:
        a: {a}

        b: {b}

        c: {c}

        alpha: {alpha}

        beta: {beta}

        qualifiers: {qualifiers}

        options: {options}

        execution: {execution}

        stream: {stream}

    Semantics:
        {semantics}

    .. seealso::
        :meth:`reset_operands`, :meth:`execute`

    Examples:

        >>> import numpy as np
        >>> import nvmath

        Create two 2-D float64 ndarrays on the CPU:

        >>> M, N, K = 1024, 1024, 1024
        >>> a = np.random.rand(M, K)
        >>> b = np.random.rand(K, N)

        We will define a matrix multiplication operation using the generic matrix
        multiplication interface.

        Create a Matmul object encapsulating the problem specification above:

        >>> mm = nvmath.linalg.Matmul(a, b)

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`MatmulOptions`).

        Next, plan the operation. The operands' layouts, qualifiers, and dtypes will be
        considered to select an appropriate matrix multiplication:

        >>> mm.plan()

        Now execute the matrix multiplication, and obtain the result `r1` as a NumPy
        ndarray.

        >>> r1 = mm.execute()

        Note that all :class:`Matmul` methods execute on the current stream by default.
        Alternatively, the `stream` argument can be used to run a method on a specified
        stream.

        Let's now look at the same problem with CuPy ndarrays on the GPU.

        Create a 3-D complex128 CuPy ndarray on the GPU:

        >>> import cupy as cp
        >>> a = cp.random.rand(M, K)
        >>> b = cp.random.rand(K, N)

        Create an Matmul object encapsulating the problem specification described earlier
        and use it as a context manager.

        >>> with nvmath.linalg.Matmul(a, b) as mm:
        ...     # Plan the operation.
        ...     mm.plan()
        ...
        ...     # Execute the operation to get the first result.
        ...     r1 = mm.execute()
        ...
        ...     # Update operands A and B in-place (see reset_operands() for an
        ...     # alternative).
        ...     a[:] = cp.random.rand(M, K)
        ...     b[:] = cp.random.rand(K, N)
        ...
        ...     # Execute the operation to get the new result.
        ...     r2 = mm.execute()


        All the resources used by the object are released at the end of the block.

        Further examples can be found in the `nvmath/examples/linalg/generic/matmul
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/generic/matmul>`_
        directory.
    """

    _input_traits: BLASMMTraits
    _batch_traits: tuple[BatchTraits, BatchTraits, BatchTraits]
    _qualifiers: MatrixQualifier

    def __init__(
        self,
        a: AnyTensor,
        b: AnyTensor,
        /,
        c: AnyTensor | None = None,
        *,
        alpha: float | complex | None = None,
        beta: float | complex | None = None,
        qualifiers: MatrixQualifier | None = None,
        options: MatmulOptions | None = None,
        execution: ExecutionCPU | ExecutionCUDA | None = None,
        stream: utils.AnyStream | int | None = None,
    ):
        options = utils.check_or_create_options(MatmulOptions, options, "Matrix multiplication options")
        assert options is not None

        if c is None and options.inplace:
            raise ValueError("Operation cannot be inplace if operand C is not provided.")

        a = tensor_wrapper.wrap_operand(a)
        check_dtype(a.dtype, "A")
        check_extents(a.shape, "A")
        check_strides(a.strides, "A")

        b = tensor_wrapper.wrap_operand(b)
        check_dtype(b.dtype, "B")
        check_extents(b.shape, "B")
        check_strides(b.strides, "B")
        operands = [a, b]

        self.num_operands = 2
        if c is not None:
            self.num_operands = 3
            c = tensor_wrapper.wrap_operand(c)
            check_dtype(c.dtype, "C")
            check_extents(c.shape, "C")
            check_strides(c.strides, "C")
            operands.append(c)

        super().__init__(operands, options=options, execution=execution, stream=stream)

        self._logger.info(f"The data type of operand A is '{a.dtype}', and that of operand B is '{b.dtype}'.")
        if c is not None:
            self._logger.info(f"The data type of operand C is '{c.dtype}'.")

        if self.options.inplace:
            self._logger.info("The operation will be performed inplace with operand C.")

        if c is not None and beta is None:
            raise ValueError("A value for beta must be provided if operand C is provided.")

        assert self.num_operands == 2 or self.num_operands == 3, "Internal Error."

        if a.dtype != b.dtype or (c is not None and a.dtype != c.dtype):
            raise ValueError(
                "Unsupported combination of dtypes. "
                f"A ({a.dtype}), B ({b.dtype}), and C ({getattr(c, 'dtype', None)}) must all have the same dtype."
            )
        # Determine the data types for a and b.
        self.a_dtype = typemaps.NAME_TO_DATA_TYPE[a.dtype]
        self.b_dtype = typemaps.NAME_TO_DATA_TYPE[b.dtype]
        self.a_dtype_name = a.dtype
        self.b_dtype_name = b.dtype

        self.is_complex = "complex" in self.a_dtype_name or "complex" in self.b_dtype_name

        # Determine the data types for c.
        if c is None:
            self.c_dtype = self.a_dtype
        else:
            self.c_dtype = typemaps.NAME_TO_DATA_TYPE[c.dtype]
        self.c_dtype_name = typemaps.DATA_TYPE_TO_NAME[self.c_dtype]

        self._logger.info(f"The data type for the result C is '{self.c_dtype_name}'.")

        self.scale_type_name = self.a_dtype_name

        # Set alpha and beta.
        self.alpha = np.zeros((1,), dtype=self.scale_type_name)
        try:
            self.alpha[0] = alpha if alpha is not None else 1
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'.") from e

        self.beta = np.zeros((1,), dtype=self.scale_type_name)
        if beta is not None and c is None:
            self._logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
        try:
            self.beta[0] = beta if beta is not None and c is not None else 0
        except (ValueError, TypeError) as e:
            raise ValueError(f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'.") from e

        if qualifiers is None:
            self._qualifiers = np.empty(3, dtype=matrix_qualifiers_dtype)
            self._qualifiers[:] = GeneralMatrixQualifier.create()
        else:
            if not ((len(qualifiers) == 3) or (len(qualifiers) == 2 and c is None)):
                raise ValueError("The number of MatrixQualifiers must match the number of operands.")
            new_qualifiers = np.empty(3, dtype=matrix_qualifiers_dtype)
            new_qualifiers[:2] = qualifiers[:2]
            new_qualifiers[2] = GeneralMatrixQualifier.create() if len(qualifiers) < 3 else qualifiers[2]
            self._qualifiers = new_qualifiers
        self._logger.info(
            f"The matrix multiplication qualifiers are "
            f"A = {GeneralMatrixQualifier.to_string(self._qualifiers[0])}, "
            f"B = {GeneralMatrixQualifier.to_string(self._qualifiers[1])}, and "
            f"C = {GeneralMatrixQualifier.to_string(self._qualifiers[2])}."
        )

        # Set qualifiers based on torch lazy conjugation flag if not provided.
        self._qualifiers[0]["conjugate"] = self._qualifiers[0]["conjugate"] ^ self._operands[0].is_conjugate
        self._qualifiers[1]["conjugate"] = self._qualifiers[1]["conjugate"] ^ self._operands[1].is_conjugate
        self.lazy_conjugation = (self._operands[0].is_conjugate, self._operands[1].is_conjugate, False)
        if c is not None:
            self._qualifiers[2]["conjugate"] = self._qualifiers[2]["conjugate"] ^ self._operands[2].is_conjugate
        if self._qualifiers[2]["conjugate"]:
            raise ValueError("The conjugate flag is currently not supported for operand C.")

        # Capture operand extents and strides for consistency check when resetting operands.
        self.operand_extents = tuple(o.shape for o in self._operands)
        self.operand_strides = tuple(o.strides for o in self._operands)

        # Create operand layouts.
        a_layout = BLASMatrixTraits(
            self.a_dtype,
            *vector_to_square(
                self._operands[0].shape[-2:],
                self._operands[0].strides[-2:],
                self._qualifiers[0],
            ),
            is_conjugate=bool(self._qualifiers[0]["conjugate"]),
            is_transpose=bool(self._qualifiers[0]["transpose"]),
            is_lower=self._qualifiers[0]["uplo"] == FillMode.LOWER,
        )
        b_layout = BLASMatrixTraits(
            self.b_dtype,
            *vector_to_square(
                self._operands[1].shape[-2:],
                self._operands[1].strides[-2:],
                self._qualifiers[1],
            ),
            is_conjugate=bool(self._qualifiers[1]["conjugate"]),
            is_transpose=bool(self._qualifiers[1]["transpose"]),
            is_lower=self._qualifiers[1]["uplo"] == FillMode.LOWER,
        )
        c_layout = (
            None
            if c is None
            else BLASMatrixTraits(
                self.c_dtype,
                *vector_to_square(
                    self._operands[2].shape[-2:],
                    self._operands[2].strides[-2:],
                    self._qualifiers[2],
                ),
                is_conjugate=bool(self._qualifiers[2]["conjugate"]),
                is_transpose=bool(self._qualifiers[2]["transpose"]),
                is_lower=self._qualifiers[2]["uplo"] == FillMode.LOWER,
            )
        )

        # Get the operation traits.
        self._input_traits = BLASMMTraits.from_layouts(a_layout, b_layout, c_layout, self._logger)
        self._logger.info(
            f"The matrix multiplication attributes are M = {self._input_traits.M}, N = {self._input_traits.N}, and "
            f"K = {self._input_traits.K}."
        )

        a_batch = BatchTraits.from_full_shape_and_strides(
            self._operands[0].shape,
            self._operands[0].strides,
            num_trailing_dims=2,
            overlap_allowed=True,
        )
        b_batch = BatchTraits.from_full_shape_and_strides(
            self._operands[1].shape,
            self._operands[1].strides,
            num_trailing_dims=2,
            overlap_allowed=True,
        )
        c_batch = (
            BatchTraits.from_full_shape_only(
                (*(a_batch * b_batch), *self._input_traits.c_layout_traits.shape),
                num_trailing_dims=2,
            )
            if c is None
            else BatchTraits.from_full_shape_and_strides(
                self._operands[2].shape,
                self._operands[2].strides,
                num_trailing_dims=2,
                overlap_allowed=False,
            )
        )
        self._logger.debug("Operand A has %s.", a_batch)
        self._logger.debug("Operand B has %s.", b_batch)
        self._logger.debug("Operand C has %s.", c_batch)
        if a_batch.shape != () or b_batch.shape != () or c_batch.shape != ():
            raise ValueError("Batched inputs are unsupported by the generic matmul API at this time.")
        if (a_batch * b_batch) != c_batch.shape:
            raise ValueError(
                f"The batch dimensions of operand C are invalid. {a_batch * b_batch} does not match {c_batch.shape}."
            )
        self._batch_traits = (a_batch, b_batch, c_batch)

        self._logger.info(
            f"The batch count is {self._batch_traits[2].count}, and the batch shape is {self._batch_traits[2].shape}."
        )

        # Attributes to establish stream ordering.
        self.workspace_stream: ccx.Stream | None = None
        self.last_compute_event: ccx.Event | None = None

        self.valid_state = True
        self._logger.info("The Matmul operation has been created.")

    def _check_valid_matmul(self, *args, **kwargs):
        """
        Check if the Matmul object is alive and well.
        """
        if not self.valid_state:
            raise InvalidMatmulState("The Matmul object cannot be used after resources are free'd")

    @utils.precondition(_check_valid_matmul)
    def plan(self) -> None:
        """
        Plan the matrix multiplication operation.

        Unlike :py:meth:`nvmath.linalg.advanced.Matmul.plan`, this method takes no
        tuning parameters. Its primary function is to find the correct matrix multiplication
        implementation based on the operands and options provided to the constructor.

        Args:
            stream: {stream}

        Returns:
            Nothing.
        """
        self._logger.info("= PLANNING PHASE =")

        mm_traits = self._input_traits.blas_compatible(self._logger, self.options.inplace)
        if self.options.inplace:
            self.result_layout_traits = self._input_traits.c_layout_traits
            self.result_batch_traits = self._batch_traits[2]
        else:
            self.result_layout_traits = self._input_traits.c_layout_traits.trim_strides()
            self.result_batch_traits = BatchTraits.from_full_shape_only(
                shape=(*self._batch_traits[2].shape, *self.result_layout_traits.shape),
                num_trailing_dims=len(self.result_layout_traits.shape),
            )

        # Base FLOP count.
        self.flop_count = 2 * mm_traits.M * mm_traits.N * mm_traits.K
        self._logger.info(f"The base matrix multiplication FLOP count is {formatters.FLOPSStr(self.flop_count, 'FLOP')}.")

        self._function = select_blas_mm_function(
            (*self._batch_traits[:2], self.result_batch_traits),
            mm_traits,
            self._qualifiers,
            self._logger,
            self.execution,
        )

        self._has_plan = True

        return

    def _check_and_set_operand(
        self,
        new_operand: utils.TensorHolder,
        operand_name: str,
        stream_holder: utils.StreamHolder | None,
        *,
        operand_index: int,
        dtype: str | None = None,
        extents: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
    ):
        """
        Check to make sure that the provided operand is consistent with the one it's
        updating, and update it.
        """
        # Make sure that the data type and extents match.
        utils.check_attribute_match(dtype, new_operand.dtype, "data type")
        utils.check_attribute_match(extents, new_operand.shape, "extents")

        # Package must be the same to preserve stream ordering
        if self._operands_package != new_operand.name:
            raise TypeError(
                f"Library package mismatch: The operand {operand_name} must from the same package ({new_operand.name}) "
                f"as the original operand ({self._operands_package})."
            )

        if self._operands_device_id != new_operand.device_id:
            raise ValueError(
                f"The operand {operand_name} must be on the same device ({new_operand.device_id}) as the original operand "
                f"({self._operands_device_id})."
            )

        # Conjugate flag of the provided operands must match the original qualifiers
        if self.lazy_conjugation[operand_index] != new_operand.is_conjugate:
            raise ValueError(f"The provided operand {operand_name} has different conjugate flag than the original operand")

        self._operands[operand_index], self._operands_backup[operand_index] = templates.copy_operand_perhaps(
            internal_operand=self._operands[operand_index],
            operand=new_operand,
            stream_holder=stream_holder,
            execution_device_id=getattr(self.execution, "device_id", "cpu"),
            operands_device_id=new_operand.device_id,
        )

        # Check strides after copying because copy could affect data layout?
        # FIXME: Could end up with a broken operand state if user catches error raised here?
        # But if we don't use copy_operand_perhaps, we can't do inplace operand reset.
        utils.check_attribute_match(strides, self._operands[operand_index].strides, "strides")

        self._logger.info(f"Operand '{operand_name}' has been reset to the new value.")

        return

    @utils.precondition(_check_valid_matmul)
    def reset_operands(
        self,
        a=None,
        b=None,
        c=None,
        *,
        alpha=None,
        beta=None,
        stream: utils.AnyStream | int | None = None,
    ):
        """
        Reset the operands held by this :class:`Matmul` instance.

        This method has two use cases:
            (1) it can be used to provide new operands for execution when the original
                operands are on the CPU
            (2) it can be used to release the internal reference to the previous operands
                and make their memory available for other use by passing ``None`` for *all*
                arguments. In this case, this method must be called again to provide the
                desired operands before another call to execution APIs like :meth:`autotune`
                or :meth:`execute`.

        This method is not needed when the operands reside on the GPU and in-place
        operations are used to update the operand values.

        This method will perform various checks on the new operands to make sure:

        - The shapes, strides, datatypes match those of the old ones.
        - The packages that the operands belong to match those of the old ones.
        - If input tensors are on GPU, the device must match.

        Args:
            a: {a}

            b: {b}

            c: {c}

            alpha: {alpha}

            beta: {beta}

            stream: {stream}

        Examples:

            >>> import cupy as cp
            >>> import nvmath

            Create two 3-D float64 ndarrays on the GPU:

            >>> M, N, K = 128, 128, 256
            >>> a = cp.random.rand(M, K)
            >>> b = cp.random.rand(K, N)

            Create an matrix multiplication object as a context manager

            >>> with nvmath.linalg.Matmul(a, b) as mm:
            ...     # Plan the operation.
            ...     mm.plan()
            ...
            ...     # Execute the MM to get the first result.
            ...     r1 = mm.execute()
            ...
            ...     # Reset the operands to new CuPy ndarrays.
            ...     c = cp.random.rand(M, K)
            ...     d = cp.random.rand(K, N)
            ...     mm.reset_operands(c, d)
            ...
            ...     # Execute to get the new result corresponding to the updated operands.
            ...     r2 = mm.execute()

            Note that if only a subset of operands are reset, the operands that are not
            reset hold their original values.

            With :meth:`reset_operands`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operands` is
            equivalent to updating the operands in-place, i.e, replacing
            ``mm.reset_operand(c, d)`` with ``a[:]=c`` and ``b[:]=d``. Note that updating
            the operand in-place should be adopted with caution as it can only yield the
            expected result under the additional constraint below:

                - The operand is on the GPU (more precisely, the operand memory space should
                  be accessible from the execution space).

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/advanced/matmul/example05_stateful_inplace.py>`_.
        """

        if c is not None and self.num_operands == 2:
            raise ValueError(
                "The matrix multiplication problem specification does not include operand C, so it cannot be reset."
            )

        if a is None and b is None and c is None and alpha is None and beta is None:
            self._operands = None  # type: ignore[assignment]
            self._logger.info("The operands have been reset to None.")
            return

        # If the operands have been reset to None, then all required operands (a, b, c, and
        # epilog_inputs need to be provided).
        if not self._operands:
            if a is None or b is None or (c is None and self.num_operands == 3):
                op_names = "A, B"
                if c is None and self.num_operands == 3:
                    op_names += ", C"
                raise ValueError(f"Operands {op_names} must be provided.")
            self._operands = [None] * self.num_operands  # type: ignore[list-item]

        # Future operations on the workspace stream should be ordered after the computation.
        if self.last_compute_event is not None:
            # FIMXE: What if result is in-place? Then don't we need to wait for copy
            # from result to out?
            assert self.workspace_stream is not None
            self.workspace_stream.wait(self.last_compute_event)
            self.last_compute_event = None

        # Update alpha.
        if alpha is not None:
            try:
                self.alpha[0] = alpha
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"The value provided for alpha {alpha} is not convertible to dtype '{self.alpha.dtype}'."
                ) from e

        # Update beta.
        if beta is not None:
            if self.num_operands == 2:
                self._logger.warning(f"Matmul: The provided beta value {beta} is ignored since operand C is not specified.")
            else:
                try:
                    self.beta[0] = beta
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"The value provided for beta {beta} is not convertible to dtype '{self.beta.dtype}'."
                    ) from e

        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)

        # Reset the provided operands.
        if a is not None:
            a = tensor_wrapper.wrap_operand(a)
            index = 0
            self._check_and_set_operand(
                a,
                "A",
                operand_stream_holder,
                operand_index=index,
                dtype=self.a_dtype_name,
                extents=self.operand_extents[index],
                strides=self.operand_strides[index],
            )

        if b is not None:
            b = tensor_wrapper.wrap_operand(b)
            index = 1
            self._check_and_set_operand(
                b,
                "B",
                operand_stream_holder,
                operand_index=index,
                dtype=self.b_dtype_name,
                extents=self.operand_extents[index],
                strides=self.operand_strides[index],
            )

        if c is not None:  # If we get here, we know that C is one of the operands in the problem specification.
            c = tensor_wrapper.wrap_operand(c)
            index = 2
            self._check_and_set_operand(
                c,
                "C",
                operand_stream_holder,
                operand_index=index,
                dtype=self.c_dtype_name,
                extents=self.operand_extents[index],
                strides=self.operand_strides[index],
            )

    @utils.precondition(_check_valid_matmul)
    @utils.precondition(templates.StatefulAPI._check_planned, "Execution")
    @utils.precondition(templates.StatefulAPI._check_valid_operands, "Execution")
    def execute(self, *, stream: utils.AnyStream | int | None = None) -> utils.AnyTensor:
        """
        Execute a prepared (planned) matrix multiplication.

        This method is a wrapper around :py:meth:`_execute`, which takes the same arguments,
        but skips as many correctness and safety checks as possible.

        Args:
            stream: {stream}

        Returns:
           {result}
        """
        return self._execute(stream=stream)

    def _execute(self, *, stream: utils.AnyStream | int | None = None) -> utils.AnyTensor:
        log_info = self._logger.isEnabledFor(logging.INFO)
        log_debug = self._logger.isEnabledFor(logging.DEBUG)
        if log_info:
            self._logger.info("= EXECUTION PHASE =")
        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)
        if log_info:
            self._logger.info(
                "The specified stream for execute() is "
                f"{getattr(exec_stream_holder or operand_stream_holder, 'obj', 'no stream')}."
            )

        # We must handle all valid combinations of:
        # - c-provided and c-not-provided
        # - results in-place and out-of-place

        # Create empty tensor for the result.
        if self.num_operands == 2 or not self.options.inplace:
            if log_debug:
                self._logger.debug("Beginning output (empty) tensor creation...")
                self._logger.debug(
                    f"The output tensor shape = {self.result_layout_traits.shape} with strides = "
                    f"{self.result_layout_traits.strides} and data type '{self.c_dtype_name}'."
                )
                self._logger.debug(
                    f"The output tensor has batch dimensions with shape {self.result_batch_traits.shape} "
                    f"and strides {self.result_batch_traits.strides}."
                )
            result = utils.create_empty_tensor(
                self._result_class,
                (*self.result_batch_traits.shape, *self.result_layout_traits.shape),
                self.c_dtype_name,
                getattr(self.execution, "device_id", "cpu"),
                exec_stream_holder,
                # verify_strides=False because we need strides to be exactly what we
                # request; not arbitrary if the strides aren't contiguous and dense.
                # Otherwise, the layout parameters will mismatch what we pass to the matmul
                # implementation.
                verify_strides=False,
                strides=(*self.result_batch_traits.strides, *self.result_layout_traits.strides),
            )
            if log_debug:
                self._logger.debug("The output (empty) tensor has been created.")
        else:  # num_operands == 3 and self.options.inplace
            result = self._operands[2]
            self._logger.debug("The output tensor is C (in-place execution).")

        if self.num_operands == 3 and not self.options.inplace:
            result.copy_(self._operands[2], exec_stream_holder)
            self._logger.debug("Operand C copied to result tensor (out-of-place execution).")

        a, b = self._operands[0], self._operands[1]
        if log_info:
            self._logger.info("Starting matrix multiplication...")

        if self.execution.name == "cuda":
            assert exec_stream_holder is not None
            self.workspace_stream = exec_stream_holder.obj
            with utils.cuda_call_ctx(exec_stream_holder, self._blocking, timing=log_info) as (
                self.last_compute_event,
                elapsed,
            ):
                self._function(
                    a,
                    b,
                    result,
                    self.alpha,
                    self.beta,
                    exec_stream_holder,
                )
        else:
            with utils.host_call_ctx(timing=log_info) as elapsed:
                self._function(
                    a,
                    b,
                    result,
                    self.alpha,
                    self.beta,
                    exec_stream_holder,
                )

        if log_info and elapsed.data is not None:
            self._logger.info(f"The matrix multiplication calculation took {elapsed.data:.3f} ms to complete.")

        # Return the result and auxiliary outputs, if present.
        if self._operands_device_id != getattr(self.execution, "device_id", "cpu"):
            if self.options.inplace:
                c = self._operands_backup[2]
                assert c is not None, (
                    "Internal Error. "
                    "Inplace operation was requested, but the execution space was different from the input space, "
                    "and we didn't keep a reference to the input tensor."
                )
                c.copy_(result, stream_holder=operand_stream_holder)
                out = c.tensor
            else:
                out = result.to(self._operands_device_id, stream_holder=operand_stream_holder).tensor
        else:
            out = result.tensor

        return out

    def __exit__(self, *args, **kwargs) -> bool | None:
        pass


@utils.docstring_decorator(GENERIC_MM_DOCUMENTATION, skip_missing=False)
def matmul(
    a: AnyTensor,
    b: AnyTensor,
    /,
    c: AnyTensor | None = None,
    *,
    alpha: float | complex | None = None,
    beta: float | complex | None = None,
    qualifiers: MatrixQualifier | None = None,
    options: MatmulOptions | None = None,
    execution: ExecutionCPU | ExecutionCUDA | None = None,
    stream: utils.AnyStream | int | None = None,
):
    """
    Perform the specified matrix multiplication computation :math:`\\alpha a @ b + \\beta
    c`. This function-form is a wrapper around the stateful :class:`Matmul` object APIs and
    is meant for *single* use (the user needs to perform just one matrix multiplication, for
    example), in which case there is no possibility of amortizing preparatory costs.

    Detailed information on what's happening within this function can be obtained by passing
    in a :class:`logging.Logger` object to :class:`MatmulOptions` or by setting the
    appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    A user can select the desired logging level and, in general, take advantage of all of
    the functionality offered by the Python `logging` module.

    Args:
        a: {a}

        b: {b}

        c: {c}

        alpha: {alpha}

        beta: {beta} from a previously planned and autotuned matrix multiplication.

        qualifiers: {qualifiers}

        options: {options}

        execution: {execution}

        stream: {stream}

    Returns:
        {result}

    Semantics:
        {semantics}

    .. seealso::
        :class:`Matmul`, :class:`MatmulOptions`, :class:`matrix_qualifiers_dtype`,
        :class:`MatrixQualifier`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create three float32 ndarrays on the GPU:

        >>> M, N, K = 128, 64, 256
        >>> a = cp.random.rand(M, K, dtype=cp.float32)
        >>> b = cp.random.rand(K, N, dtype=cp.float32)
        >>> c = cp.random.rand(M, N, dtype=cp.float32)

        Perform the operation :math:`\\alpha A @ B + \\beta C` using :func:`matmul`. The
        result `r` is also a CuPy float64 ndarray:

        >>> r = nvmath.linalg.matmul(a, b, c, alpha=1.23, beta=0.74)

        The package current stream is used by default, but a stream can be explicitly
        provided to the Matmul operation. This can be done if the operands are computed on a
        different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...     a = cp.random.rand(M, K)
        ...     b = cp.random.rand(K, N)
        >>> r = nvmath.linalg.matmul(a, b, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input
        computation.

        Create  NumPy ndarrays on the CPU.

        >>> import numpy as np
        >>> a = np.random.rand(M, K)
        >>> b = np.random.rand(K, N)

        Provide the NumPy ndarrays to :func:`matmul`, with the result also being a NumPy
        ndarray:

        >>> r = nvmath.linalg.matmul(a, b)

    Notes:
        - This function is a convenience wrapper around :class:`Matmul` and and is
          specifically meant for *single* use.

    Further examples can be found in the `nvmath/examples/linalg/generic/matmul
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/linalg/generic/matmul>`_
    directory.
    """

    with Matmul(
        a,
        b,
        c=c,
        alpha=alpha,
        beta=beta,
        qualifiers=qualifiers,
        options=options,
        execution=execution,
        stream=stream,
    ) as mm:
        mm.plan()

        r = mm.execute(stream=stream)

    return r
