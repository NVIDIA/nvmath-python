"""
Defines dataclasses for tracking matrix layout and traits specifically for wranging inputs
into a form accepted by matrix multiplication operations in BLAS APIs.

The module defines two classes: BLASMatrixTraits which tracks a single matrix, and
BLASMMTraits which tracks a collection of BLASMatrixTraits to be used together in a matrix
multiplication.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import nvmath.bindings.cublasLt as cublaslt
import nvmath.bindings.cublas as cublas
from nvmath.internal import typemaps


def check_extents(shape: Sequence[int], name: str):
    """Raises an error if any element in shape is non-positive.

    0D tensors are expected to have shape () and stride ().
    """
    if any(e <= 0 for e in shape):
        message = f"The specified extents {shape} for operand {name} are not valid. The extents must be strictly positive."
        raise ValueError(message)


def check_strides(strides: Sequence[int], name: str):
    """Raises an error if any element in strides is negative.

    0D tensors are expected to have shape () and stride().
    Strides may be 0 or positive.
    BLAS disallows negative strides, but the cuBLAS dgmm extension allows negative strides.
    """
    if any(s < 0 for s in strides):
        message = f"The specified strides {strides} for operand {name} are not valid. The strides must be non-negative."
        raise ValueError(message)


@dataclass(slots=True)
class BLASMatrixTraits:
    """Manages a tensor's layout and its compatibility with BLAS API mamtuls.

    :class:`BLASMatrixTraits` encapsulates attributes and methods to handle single matrix
    layouts, including memory order, operations (e.g., transpose, conjugation), and
    dimensions. It provides functionality to transform matrices into compatible formats for
    BLAS matrix multiplication.

    Diagonal matrices should be described as the equivalent square matrix, with equal
    strides in both dimensions.

    Attributes:
        dtype : The CUDA data type representing the matrix element.

        shape: The number of elements along each dimension.

        strides: The number of elements in memory to move between elements along the
            corresponding dimension. We expect a stride of 0 for singleton dimensions.

        is_conjugate : A flag indicating if the matrix elements should be conjugated.

        is_transpose : A flag indicating if the matrix should be transposed.

        is_lower : A flag indicating if the matrix is a lower (true) or upper (false)
            triangular matrix. We track this by default even if the matrix is full because
            we can always ignore this parameter.
    """

    dtype: typemaps.cudaDataType
    shape: Sequence[int]
    strides: Sequence[int]
    is_conjugate: bool
    is_transpose: bool
    is_lower: bool

    def __post_init__(self):
        assert len(self.shape) < 3, "Internal Error: BLASMatrixTraits supports only 0..2D matrices."
        assert len(self.strides) == len(self.shape), "Internal Error: BLASMatrixTraits strides and shape must have same length."
        assert all(extent > 0 for extent in self.shape), "Internal Error: BLASMatrixTraits supports only positive shapes."
        assert all(stride >= 0 for stride in self.strides), (
            "Internal Error: BLASMatrixTraits supports only non-negative strides."
        )
        # Set strides for singleton dimensions to zero. The stride length of singleton
        # dimensions is unused because we don't travel along it, but non-zero strides will
        # confuse us when we try to guess the ordering of the matrix.
        self.strides = tuple(0 if extent == 1 else stride for extent, stride in zip(self.shape, self.strides, strict=True))

    @property
    def order(self) -> cublaslt.Order:
        """The indexing order of the matrix in memory."""
        if len(self.shape) < 2:
            # For vectors and scalars we return COL-order because COL is the default.
            return cublaslt.Order.COL
        msg = f"Unsupported layout for shape: {self.shape} strides: {self.strides}. At least one dimension must be contiguous."
        if 1 not in self.strides and 0 not in self.strides:
            raise ValueError(msg)
        if self.shape[0] * self.strides[0] <= self.strides[1]:
            return cublaslt.Order.COL
        if self.shape[1] * self.strides[1] <= self.strides[0]:
            return cublaslt.Order.ROW
        raise ValueError(msg)

    @property
    def ld(self) -> int:
        """The leading dimension of the matrix without operations applied."""
        assert self.order == cublaslt.Order.COL, (
            f"Internal Error: {self.__class__.__name__}.ld should only be accessed if the matrix is COL-order."
        )
        assert len(self.shape) == 2, (
            f"Internal Error: {self.__class__.__name__}.ld should only be accessed if the matrix is 2D."
        )
        if any(stride == 0 and extent != 1 for extent, stride in zip(self.shape, self.strides, strict=True)):
            raise ValueError(
                f"Unsupported layout for shape: {self.shape} strides: {self.strides}. "
                "Only singleton dimensions may have zero stride."
            )
        if self.shape[1] == 1:
            return self.shape[0]  # extent = 1 strides cannot be trusted
        return self.strides[1]

    @property
    def operation(self) -> cublas.Operation:
        """The operation to be applied to the matrix before multiplication."""
        match (self.is_conjugate, self.is_transpose):
            case (False, False):
                return cublas.Operation.N
            case (False, True):
                return cublas.Operation.T
            case (True, True):
                return cublas.Operation.C
            case (True, False):
                raise NotImplementedError("Conjugate non-transpose operation is not supported.")
            case _:
                raise NotImplementedError("Conjugate and transpose flags must be python booleans.")

    def mm_shape(self) -> Sequence[int]:
        """The shape of the matrix after applying operations."""
        if len(self.shape) == 2:
            return self.shape[::-1] if self.is_transpose else self.shape
        raise NotImplementedError("mm_shape only implemented for 2D matrices.")

    def transpose_and_reorder(self, logger: logging.Logger):
        """Return a new :class:`BLASMatrixTraits` that has been transposed and reordered.

        Simultaneous transpose and reorder is a non-operation because the data does not need
        to move.
        """
        new = BLASMatrixTraits(
            dtype=self.dtype,
            shape=self.shape[::-1],
            strides=self.strides[::-1],
            is_conjugate=self.is_conjugate,
            is_transpose=not self.is_transpose,
            is_lower=not self.is_lower,
        )
        logger.debug("The matrix was transposed and reordered from %s to %s.", self.order.name, new.order.name)
        return new

    def promote_left(self, logger: logging.Logger, ndim: int = 2):
        """Return a new :class:`BLASMatrixTraits` with new singleton dimensions added to
        the left side of `shape` until the matrix has at least `ndim` dimensions."""
        add = max(ndim - len(self.shape), 0)
        promoted = BLASMatrixTraits(
            dtype=self.dtype,
            shape=(*[1] * add, *self.shape),
            strides=(*[0] * add, *self.strides),
            is_conjugate=self.is_conjugate,
            is_transpose=self.is_transpose,
            is_lower=self.is_lower,
        )
        if add > 0:
            logger.debug("The matrix was promoted from shape %s to shape %s", self.shape, promoted.shape)
        return promoted

    def blas_A_compatible(self, logger: logging.Logger):
        """Return ``self`` or a new :class:`BLASMatrixTraits` that is compatible with the
        BLAS API's A matrices."""
        if len(self.shape) < 2:
            return self.promote_left(logger).blas_A_compatible(logger)
        if self.order == cublaslt.Order.ROW:
            return self.transpose_and_reorder(logger).blas_A_compatible(logger)
        # else self.order is COL
        if self.is_conjugate and not self.is_transpose:
            # We can only perform conjugate transpose; conjugate non-tranpose is not an
            # option.
            msg = f"BLAS APIs only accept COL-order matrix for A. {self} was not convertible to a valid A."
            raise ValueError(msg)
        return self

    def promote_right(self, logger: logging.Logger, ndim: int = 2):
        """Return a new :class:`BLASMatrixTraits` with new singleton dimensions added to
        the right side of `shape` until the matrix has at least `ndim` dimensions."""
        add = max(ndim - len(self.shape), 0)
        promoted = BLASMatrixTraits(
            dtype=self.dtype,
            shape=(*self.shape, *[1] * add),
            strides=(*self.strides, *[0] * add),
            is_conjugate=self.is_conjugate,
            is_transpose=self.is_transpose,
            is_lower=self.is_lower,
        )
        if add > 0:
            logger.debug("The matrix was promoted from shape %s to shape %s", self.shape, promoted.shape)
        return promoted

    def blas_B_compatible(self, logger: logging.Logger):
        """Return ``self`` or a new :class:`BLASMatrixTraits` that is compatible with the
        BLAS API's B matrices."""
        if len(self.shape) < 2:
            return self.promote_right(logger).blas_B_compatible(logger)
        if self.order == cublaslt.Order.ROW:
            return self.transpose_and_reorder(logger).blas_B_compatible(logger)
        # else self.order is COL
        if self.is_conjugate and not self.is_transpose:
            # We can only perform conjugate transpose; conjugate non-tranpose is not an
            # option.
            msg = f"BLAS APIs only accept COL-order matrix for B. {self} was not convertible to a valid B."
            raise ValueError(msg)
        return self

    def blas_C_compatible(self, logger: logging.Logger):
        """Return ``self`` or a new :class:`BLASMatrixTraits` that is compatible with the
        BLAS API's C matrix."""
        if len(self.shape) < 2:
            return self.promote_right(logger).blas_C_compatible(logger)
        match (self.order, self.is_conjugate, self.is_transpose):
            case (cublaslt.Order.COL, False, False):
                return self
            case (cublaslt.Order.ROW, False, True):
                return self.transpose_and_reorder(logger)
            case _:
                msg = (
                    "BLAS APIs only accept COL-order, non-tranpose, non-conjugate matrices for C. "
                    f"{self} was not convertible to a valid C."
                )
                raise ValueError(msg)

    def trim_strides(self):
        """Return ``self`` or a new :class:`BLASMatrixTraits` with strides adjusted so
        the matrix is contiguous and dense along all dimensions."""
        match len(self.shape):
            case 0:
                new_strides = ()
            case 1:
                new_strides = (1,)
            case 2:
                if self.order == cublaslt.Order.COL:
                    new_strides = (1, self.shape[0])
                else:  # self.order == cublaslt.Order.ROW
                    new_strides = (self.shape[1], 1)
        if self.strides == new_strides:
            return self
        return BLASMatrixTraits(
            dtype=self.dtype,
            shape=self.shape,
            strides=new_strides,
            is_conjugate=self.is_conjugate,
            is_transpose=self.is_transpose,
            is_lower=self.is_lower,
        )


@dataclass(slots=True)
class BLASMMTraits:
    """
    BLASMMTraits represents the traits required for describing BLAS-compatible matrix
    multiplications, including operand A, operand B, and an optional operand C. It ensures
    that the operands comply with specific BLAS API compatibility rules, such as
    broadcasting, promotion of dimensions, and swapping of A and B when required.

    Attributes:
        M : The number of rows in the resulting matrix multiplication.

        N : The number of columns in the resulting matrix multiplication.

        K : The shared dimension between operands A and B for matrix multiplication.

        a_layout_traits : Layout traits for operand A.

        b_layout_traits : Layout traits for operand B.

        c_layout_traits : Layout traits for operand C.

        is_swapped_AB : Indicates if operands A and B were swapped in order to ensure
            compatibility.
    """

    M: int | None
    N: int | None
    K: int | None
    a_layout_traits: BLASMatrixTraits
    b_layout_traits: BLASMatrixTraits
    c_layout_traits: BLASMatrixTraits
    is_swapped_AB: bool

    @staticmethod
    def from_layouts(
        a_layout: BLASMatrixTraits,
        b_layout: BLASMatrixTraits,
        c_layout: BLASMatrixTraits | None,
        logger: logging.Logger,
    ):
        """Create a `BLASMMTraits` from 2 or 3 `BLASMatrixTraits`.

        See nvmath.linalg.advanced.matmulmod semantics docstring for matrix promotion and
        broadcasting rules.
        """
        logger.debug("Constructing a BLASMMTraits.")
        logger.debug(f"Operand A is shape {a_layout.shape} with strides {a_layout.strides} and order {a_layout.order.name}.")
        logger.debug(f"Operand B is shape {b_layout.shape} with strides {b_layout.strides} and order {b_layout.order.name}.")
        match len(a_layout.shape):
            case 0:
                M0, K0 = None, None
            case 1:
                M0, K0 = None, a_layout.shape[0]
            case _:
                M0, K0 = a_layout.mm_shape()
        match len(b_layout.shape):
            case 0:
                K1, N0 = None, None
            case 1:
                K1, N0 = b_layout.shape[0], None
            case _:
                K1, N0 = b_layout.mm_shape()
        if c_layout is None:
            logger.debug("Operand C was not provided.")
            shape: Sequence[int]
            strides: Sequence[int]
            if M0 is None and N0 is not None:
                shape = (N0,)
                strides = (1,)
            elif M0 is not None and N0 is None:
                shape = (M0,)
                strides = (1,)
            elif M0 is not None and N0 is not None:
                shape = (M0, N0)
                strides = (N0, 1) if a_layout.order == cublaslt.Order.ROW and b_layout.order == cublaslt.Order.ROW else (1, M0)
            else:  # both are None
                shape = ()
                strides = ()
            c_layout_ = BLASMatrixTraits(
                dtype=a_layout.dtype,
                shape=shape,
                # Create COL-order tensor by default, but match orders if all ROW
                strides=strides,
                is_transpose=False,
                is_conjugate=False,
                is_lower=True,
            )
        else:
            c_layout_ = c_layout
        logger.debug(f"Operand C is shape {c_layout_.shape} with strides {c_layout_.strides} and order {c_layout_.order.name}.")

        match len(c_layout_.shape):
            case 0:
                M1, N1 = None, None
            case 1:
                if len(a_layout.shape) <= 1:
                    M1, N1 = c_layout_.shape[0], None
                else:
                    M1, N1 = None, c_layout_.shape[0]
            case _:
                M1, N1 = c_layout_.mm_shape()
        if K0 != K1:
            raise ValueError(
                f"The 'K' extent must match for the operands: K={K0} in operand A is not equal to K={K1} in operand B."
            )
        if M0 is None and N0 is None:
            # Both a,b are vectors; c must be a scalar
            # NOTE: Since BLAS does not support broadcasting c, the shape of c should match
            # a@b exactly not shape (1, 1) or (1,)
            if (M1, N1) != (None, None):
                raise ValueError(f"When both operands A,B are vectors, operand C must be scalar-like, not shape {(M1, N1)}.")
        elif (M0 is not None and N0 is None) or (M0 is None and N0 is not None):
            # One of a,b is a vector; c must be a vector.
            if (M1 or N1) not in [M0, N0]:
                raise ValueError(
                    f"When one of operands A,B is a vector, operand C must be a vector with shape ({M0 or N0},), "
                    f"not shape ({M1}, {N1})."
                )
        else:
            # Both a,b are matrices; c must have shape (M0, N0)
            if M0 != M1:
                raise ValueError(
                    f"The 'M' extent must match for the operands: M={M0} in operand A is not equal to M={M1} in operand C."
                )
            if N0 != N1:
                raise ValueError(
                    f"The 'N' extent must match for the operands: N={N0} in operand B is not equal to N={N1} in operand C."
                )

        return BLASMMTraits(
            M=M0,
            N=N0,
            K=K0,
            a_layout_traits=a_layout,
            b_layout_traits=b_layout,
            c_layout_traits=c_layout_,
            is_swapped_AB=False,
        )

    def blas_compatible(self, logger: logging.Logger, inplace: bool):
        """Return ``self`` or a new :class:`BLASMMTraits` that is compatible with the
        BLAS API.

        Args
        ----
        inplace: Whether C will be inplace or copied into a memory-compact array of the same
            shape, but potentially different strides.
        """
        logger.debug("Making a BLAS compatible BLASMMTraits.")
        a_layout = self.a_layout_traits
        b_layout = self.b_layout_traits
        c_layout = self.c_layout_traits
        is_swapped_AB = False

        logger.debug("Making a BLAS compatible view of operand C.")
        if len(a_layout.shape) < 2 and len(c_layout.shape) < 2:
            c_layout = c_layout.promote_left(logger)
        if c_layout.order == cublaslt.Order.ROW:
            c_layout = c_layout.transpose_and_reorder(logger)
        if not inplace:
            c_layout = c_layout.trim_strides()
        if c_layout.is_transpose:
            # We can use property of transpose that (A @ B).T = B.T @ A.T to remove
            # transpose operation from C.
            a_layout, b_layout = b_layout, a_layout
            is_swapped_AB = True
            a_layout.is_transpose = not a_layout.is_transpose
            b_layout.is_transpose = not b_layout.is_transpose
            c_layout.is_transpose = not c_layout.is_transpose
            logger.debug("Operands A, B will be swapped and transposed in order to transpose C.")
        c_layout = c_layout.blas_C_compatible(logger)
        logger.debug(
            f"The BLAS operand C is shape {c_layout.shape} with strides {c_layout.strides} and order {c_layout.order.name}."
        )
        logger.debug("The matrix multiplication will be performed with %s for operand C.", c_layout.operation.name)

        logger.debug("Making a BLAS compatible view of operand A.")
        a_layout = a_layout.blas_A_compatible(logger)
        logger.debug(
            f"The BLAS operand A is shape {a_layout.shape} with strides {a_layout.strides} and order {a_layout.order.name}."
        )
        logger.debug("The matrix multiplication will be performed with %s for operand A.", a_layout.operation.name)

        logger.debug("Making a BLAS compatible view of operand B.")
        b_layout = b_layout.blas_B_compatible(logger)
        logger.debug(
            f"The BLAS operand B is shape {b_layout.shape} with strides {b_layout.strides} and order {b_layout.order.name}."
        )
        logger.debug("The matrix multiplication will be performed with %s for operand B.", b_layout.operation.name)

        M0, K0 = a_layout.mm_shape()
        K1, N0 = b_layout.mm_shape()
        M1, N1 = c_layout.mm_shape()
        assert M0 is not None
        assert K0 is not None
        assert N0 is not None
        if K0 != K1:
            raise ValueError(
                f"The 'K' extent must match for the operands: K={K0} in operand A is not equal to K={K1} in operand B."
            )
        if M0 != M1:
            raise ValueError(
                f"The 'M' extent must match for the operands: M={M0} in operand A is not equal to M={M1} in operand C."
            )
        if N0 != N1:
            raise ValueError(
                f"The 'N' extent must match for the operands: N={N0} in operand B is not equal to N={N1} in operand C."
            )

        return BLASMMTraits(
            M=M0,
            N=N0,
            K=K0,
            a_layout_traits=a_layout,
            b_layout_traits=b_layout,
            c_layout_traits=c_layout,
            is_swapped_AB=is_swapped_AB,
        )
