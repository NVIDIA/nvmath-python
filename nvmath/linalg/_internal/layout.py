# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Defines dataclasses for tracking matrix layout and traits specifically for wrangling inputs
into a form accepted by matrix multiplication operations in BLAS APIs.

The module defines three classes: BLASMatrixTraits which represents a single matrix,
InputMMTraits which represents a collection of BLASMatrixTraits to be used together in a
matrix multiplication, and BLASMMTraitsView which represents a view of the InputMMTraits
that is compatible with the BLAS API.
"""

from __future__ import annotations

import itertools
import logging
import typing
from collections.abc import Sequence
from dataclasses import dataclass

import nvmath.bindings.cublas as cublas
import nvmath.bindings.cublasLt as cublaslt
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
    """Manages a tensor's layout and its compatibility with BLAS API matmuls.

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

    @property
    def order(self) -> cublaslt.Order:
        """The indexing order of the matrix in memory."""
        if len(self.shape) < 2:
            # For vectors and scalars we return COL-order because COL is the default.
            return cublaslt.Order.COL
        msg = f"Unsupported layout for shape: {self.shape} strides: {self.strides}. At least one dimension must be contiguous."
        if (strides_contiguous_perhaps := self.strides_contiguous_perhaps) is None:
            raise ValueError(msg)
        if self.shape[0] * strides_contiguous_perhaps[0] <= strides_contiguous_perhaps[1]:
            return cublaslt.Order.COL
        if self.shape[1] * strides_contiguous_perhaps[1] <= strides_contiguous_perhaps[0]:
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
        if self.shape == (1, 1):
            return 1  # A special case where the strides don't matter
        if self.strides[0] == 0 and self.shape[0] != 1 or self.strides[1] == 0 and self.shape[1] != 1:
            raise ValueError("The leading dimension cannot be 0 for a non-singleton dimension.")
        if (strides_contiguous_perhaps := self.strides_contiguous_perhaps) is None:
            msg = (
                f"Unsupported layout for shape: {self.shape} strides: {self.strides}. "
                "At least one dimension must be contiguous."
            )
            raise ValueError(msg)
        return strides_contiguous_perhaps[1]

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

    @property
    def is_contiguous(self) -> bool:
        """Whether the matrix is contiguous in memory."""
        return 1 in self.strides or 0 in self.strides or len(self.shape) == 0

    @property
    def has_contiguous_view(self) -> bool:
        """Whether the matrix can be viewed as contiguous in memory.

        A view/slice of a non-contiguous matrix can be contiguous if the matrix is a vector
        or a scalar.
        """
        return self.is_contiguous or 1 in self.shape or 0 in self.shape

    @property
    def strides_contiguous_perhaps(self) -> Sequence[int] | None:
        """A contiguous view of the matrix's strides if the matrix can be viewed as
        contiguous in memory, otherwise None.
        """
        if self.is_contiguous:
            return self.strides
        if self.has_contiguous_view:
            new_strides = tuple(
                0 if shape in (0, 1) else stride for shape, stride in zip(self.shape, self.strides, strict=True)
            )
            return new_strides
        return None

    @property
    def has_a_contiguous_dim(self) -> bool:
        """Whether the matrix has at least one contiguous dimension in memory."""
        return 1 in self.strides or 0 in self.strides or len(self.shape) == 0

    @property
    def has_singleton_dim(self) -> bool:
        """Whether the matrix has a dimension that can be viewed as contiguous in memory.

        A view/slice of a non-contiguous matrix can be contiguous if the matrix is a vector
        or a scalar.
        """
        return 1 in self.shape or 0 in self.shape

    def transpose_and_reorder(self, logger: logging.Logger) -> BLASMatrixTraits:
        """Return a new :class:`BLASMatrixTraits` that has been transposed, reordered,
        and the transpose flag flipped.

        Simultaneous transpose and reorder is a useful operation because the data does not
        need to move, but we can change the layout order of the matrix. Flipping the
        transpose flag makes the matrix the same shape as the original.
        """
        assert self.has_contiguous_view, "Internal Error: transpose_and_reorder only supports for contiguous matrices."
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

    def flip_transpose_flag(self, logger: logging.Logger) -> BLASMatrixTraits:
        """Return a new :class:`BLASMatrixTraits` with the transpose flag flipped."""
        new = BLASMatrixTraits(
            dtype=self.dtype,
            shape=self.shape,
            strides=self.strides,
            is_conjugate=self.is_conjugate,
            is_transpose=not self.is_transpose,
            is_lower=self.is_lower,
        )
        logger.debug("The matrix was transposed.")
        return new

    def promote_left(self, logger: logging.Logger, ndim: int = 2) -> BLASMatrixTraits:
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

    def promote_right(self, logger: logging.Logger, ndim: int = 2) -> BLASMatrixTraits:
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

    def __str__(self):
        match self.is_conjugate, self.is_transpose:
            case (True, True):
                operation = "CONJ-TRAN"
            case (True, False):
                operation = "CONJ"
            case (False, True):
                operation = "TRAN"
            case (False, False):
                operation = "NONE"
        return f"shape {self.shape} strides {self.strides} order {self.order.name} and operation {operation}"


@dataclass(slots=True, frozen=True)
class InputMMTraits:
    """
    InputMMTraits represents the traits of the input and output matrices for BLAS-compatible
    matrix multiplications, including operand A, operand B, an optional operand C, and the
    result matrix, D. It ensures that the operands are compatible with each other.

    Attributes:

        M : The number of rows in the resulting matrix multiplication.

        N : The number of columns in the resulting matrix multiplication.

        K : The shared dimension between operands A and B for matrix multiplication.

        a_layout_traits : Layout traits of operand A.

        b_layout_traits : Layout traits of operand B.

        c_layout_traits : Layout traits of operand C.

        d_layout_traits : Layout traits of the result tensor or operand C if inplace.

        inplace: whether the result will overwrite operand C or the result will be copied
            into a new memory-compact array, D, the same shape as C, but potentially
            different strides.
    """

    M: int | None
    N: int | None
    K: int | None
    a_layout_traits: BLASMatrixTraits
    b_layout_traits: BLASMatrixTraits
    c_layout_traits: BLASMatrixTraits
    d_layout_traits: BLASMatrixTraits
    inplace: bool

    @staticmethod
    def from_layouts(
        a_layout: BLASMatrixTraits,
        b_layout: BLASMatrixTraits,
        c_layout: BLASMatrixTraits | None,
        inplace: bool,
        logger: logging.Logger,
    ):
        """Create a `InputMMTraits` from 2 or 3 `BLASMatrixTraits`.

        See nvmath.linalg.advanced.matmulmod semantics docstring for matrix promotion and
        broadcasting rules.
        """
        logger.debug("Checking Matmul operands for layout compatibility.")
        match len(a_layout.shape):
            case 0:
                M0, K0 = None, None
            case 1:
                M0, K0 = None, a_layout.shape[0]
            case _:
                M0, K0 = a_layout.shape[::-1] if a_layout.is_transpose else a_layout.shape
        match len(b_layout.shape):
            case 0:
                K1, N0 = None, None
            case 1:
                K1, N0 = b_layout.shape[0], None
            case _:
                K1, N0 = b_layout.shape[::-1] if b_layout.is_transpose else b_layout.shape
        if inplace:
            assert c_layout is not None, "Internal Error: Cannot have inplace operation without C."
            d_layout = c_layout
        else:
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
                # Match order of input C by default.
                # If input C is None, ROW if all ROW else COL
                order = (
                    (a_layout.order == cublaslt.Order.ROW and b_layout.order == cublaslt.Order.ROW)
                    if c_layout is None
                    else c_layout.order
                )
                strides = (N0, 1) if order == cublaslt.Order.ROW else (1, M0)
            else:  # both are None
                shape = ()
                strides = ()
            d_layout = BLASMatrixTraits(
                dtype=a_layout.dtype,
                shape=shape,
                strides=strides,
                is_transpose=False,
                is_conjugate=False,
                is_lower=True,
            )
            logger.debug("Out-of-place result has traits of %s", d_layout)
        if c_layout is None:
            c_layout = d_layout
        match len(c_layout.shape):
            case 0:
                M1, N1 = None, None
            case 1:
                if len(a_layout.shape) <= 1:
                    M1, N1 = c_layout.shape[0], None
                else:
                    M1, N1 = None, c_layout.shape[0]
            case _:
                M1, N1 = c_layout.shape[::-1] if c_layout.is_transpose else c_layout.shape
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
        logger.debug("Matmul operands are layout compatible.")

        if not (a_layout.has_singleton_dim or a_layout.has_a_contiguous_dim):
            raise ValueError("Unsupported layout: Operand A is not contiguous.")
        if not (b_layout.has_singleton_dim or b_layout.has_a_contiguous_dim):
            raise ValueError("Unsupported layout: Operand B is not contiguous.")
        if not (c_layout.has_singleton_dim or c_layout.has_a_contiguous_dim):
            raise ValueError("Unsupported layout: Operand C is not contiguous.")

        return InputMMTraits(
            M=M0,
            N=N0,
            K=K0,
            a_layout_traits=a_layout,
            b_layout_traits=b_layout,
            c_layout_traits=c_layout,
            d_layout_traits=d_layout,
            inplace=inplace,
        )


@dataclass(slots=True, frozen=True)
class BLASMMTraitsView:
    """
    BLASMMTraitsView represents a view of the InputMMTraits that is compatible with the BLAS
    API. i.e. one where all operands are COL order.

    A series of no-op (no data movement required) transformations are applied to the
    InputMMTraits to create a BLASMMTraitsView.

    Attributes:

        M : The number of rows in the resulting matrix multiplication.

        N : The number of columns in the resulting matrix multiplication.

        K : The shared dimension between operands A and B for matrix multiplication.

        a_layout_traits : A BLAS compatible view of operand A.

        b_layout_traits : A BLAS compatible view of operand B.

        c_layout_traits : A BLAS compatible view of operand C.

        is_swapped_AB : Indicates if operands A and B were swapped in order to ensure
            compatibility.
    """

    M: int
    N: int
    K: int
    a_layout_traits: BLASMatrixTraits
    b_layout_traits: BLASMatrixTraits
    c_layout_traits: BLASMatrixTraits
    is_swapped_AB: bool

    @staticmethod
    def from_input_traits(
        input_traits: InputMMTraits,
        layout_checker: MMLayoutChecker,
        logger: logging.Logger,
        lookup_table_table: dict[MMLayoutChecker, MMLayoutCheckerLookupTable] | None = None,
    ) -> BLASMMTraitsView:
        logger.debug("Creating a BLAS compatible view of Matmul operands.")

        a_layout_traits = input_traits.a_layout_traits.promote_left(logger)
        b_layout_traits = input_traits.b_layout_traits.promote_right(logger)
        if input_traits.M is None:
            c_layout_traits = input_traits.d_layout_traits.promote_left(logger)
        else:
            c_layout_traits = input_traits.d_layout_traits.promote_right(logger)
        M0, K0 = a_layout_traits.shape[::-1] if a_layout_traits.is_transpose else a_layout_traits.shape
        K1, N0 = b_layout_traits.shape[::-1] if b_layout_traits.is_transpose else b_layout_traits.shape
        M1, N1 = c_layout_traits.shape[::-1] if c_layout_traits.is_transpose else c_layout_traits.shape
        assert M0 == M1, "Internal Error: M must match."
        assert K0 == K1, "Internal Error: K must match."
        assert N0 == N1, "Internal Error: N must match."

        supported_layout = BLASMMTraitsView(
            M=M0,
            N=N0,
            K=K0,
            a_layout_traits=a_layout_traits,
            b_layout_traits=b_layout_traits,
            c_layout_traits=c_layout_traits,
            is_swapped_AB=False,
        ).lookup_supported_layout(layout_checker, lookup_table_table, logger)
        logger.debug("A BLAS compatible view of the operands was created.")
        return supported_layout  # type: ignore[return-value]

    def swap_AB_and_transpose_ABC(self, logger: logging.Logger) -> BLASMMTraitsView:
        """Return a new :class:`BLASMMTraits` with operands A and B swapped."""
        logger.debug("Operands A, B will be swapped and transposed in order to transpose C.")
        return BLASMMTraitsView(
            M=self.N,
            N=self.M,
            K=self.K,
            a_layout_traits=self.b_layout_traits.flip_transpose_flag(logger),
            b_layout_traits=self.a_layout_traits.flip_transpose_flag(logger),
            c_layout_traits=self.c_layout_traits.flip_transpose_flag(logger),
            is_swapped_AB=not self.is_swapped_AB,
        )

    def transpose_and_reorder_A(self, logger: logging.Logger) -> BLASMMTraitsView:
        """Return a new :class:`BLASMMTraits` with operand A transposed and reordered."""
        logger.debug("Operand A was transposed and reordered.")
        return BLASMMTraitsView(
            M=self.M,
            N=self.N,
            K=self.K,
            a_layout_traits=self.a_layout_traits.transpose_and_reorder(logger),
            b_layout_traits=self.b_layout_traits,
            c_layout_traits=self.c_layout_traits,
            is_swapped_AB=self.is_swapped_AB,
        )

    def transpose_and_reorder_B(self, logger: logging.Logger) -> BLASMMTraitsView:
        """Return a new :class:`BLASMMTraits` with operand B transposed and reordered."""
        logger.debug("Operand B was transposed and reordered.")
        return BLASMMTraitsView(
            M=self.M,
            N=self.N,
            K=self.K,
            a_layout_traits=self.a_layout_traits,
            b_layout_traits=self.b_layout_traits.transpose_and_reorder(logger),
            c_layout_traits=self.c_layout_traits,
            is_swapped_AB=self.is_swapped_AB,
        )

    def transpose_and_reorder_C(self, logger: logging.Logger) -> BLASMMTraitsView:
        """Return a new :class:`BLASMMTraits` with operand C transposed and reordered."""
        logger.debug("Operand C was transposed and reordered.")
        return BLASMMTraitsView(
            M=self.M,
            N=self.N,
            K=self.K,
            a_layout_traits=self.a_layout_traits,
            b_layout_traits=self.b_layout_traits,
            c_layout_traits=self.c_layout_traits.transpose_and_reorder(logger),
            is_swapped_AB=self.is_swapped_AB,
        )

    def find_supported_layout(self, layout_checker: MMLayoutChecker, logger: logging.Logger) -> BLASMMTraitsView:
        # If we assume that A, B, C are all non-transposed, but may be conjugate and may be
        # either ROW or COL, then the input space for the matmul is 2 * 2 = 4 for EACH
        # operand (two layouts and either conjugate or not). That makes the total input
        # space 4 * 4 * 4 = 64 possible input combinations which is too large to enumerate.
        #
        # The operations we are allowed to apply to the operands are:
        # - Transpose and reorder A
        # - Transpose and reorder B
        # - Transpose and reorder C
        # - Swap A, B and transpose A, B, C
        #
        # The order in which the operations are applied does not matter, and applying each
        # operation twice is a no-op. Therefore we only care about whether to apply each
        # operation or not.
        #
        # That makes the possible search space of applying each of these operations 2
        # * 2 * 2 * 2 = 16 which is much smaller than 64. We just need check whether the
        # result of applying each of the 16 combinations results in a supported layout.
        allowed_operations = [
            "transpose_and_reorder_A",
            "transpose_and_reorder_B",
            "transpose_and_reorder_C",
            "swap_AB_and_transpose_ABC",
        ]
        # Prioritize the simplest cases: when operands are all COL (apply no operations) or
        # all are ROW (apply all operations)
        for search_depth in [0, 4, 1, 2, 3]:
            for combination in itertools.combinations(allowed_operations, search_depth):
                new_traits = self
                logger.debug("Trying combination: %s", combination)
                for operation in combination:
                    new_traits = getattr(new_traits, operation)(logger)
                if layout_checker(new_traits):
                    return new_traits  # type: ignore[return-value]
        raise ValueError("No BLAS compatible view of the operands was found.")

    def lookup_supported_layout(
        self,
        layout_checker: MMLayoutChecker,
        lookup_table_table: dict[MMLayoutChecker, MMLayoutCheckerLookupTable] | None,
        logger: logging.Logger,
    ) -> BLASMMTraitsView:
        if lookup_table_table is None or layout_checker not in lookup_table_table:
            logger.debug("No cached layout lookup table found for layout checker; performing exhaustive search.")
            return self.find_supported_layout(layout_checker, logger)
        lookup_table = lookup_table_table[layout_checker]
        key = (
            self.a_layout_traits.order,
            self.b_layout_traits.order,
            self.c_layout_traits.order,
            self.a_layout_traits.is_conjugate,
            self.b_layout_traits.is_conjugate,
            self.c_layout_traits.is_conjugate,
        )
        assert not self.a_layout_traits.is_transpose, "Internal Error: A must be non-transpose."
        assert not self.b_layout_traits.is_transpose, "Internal Error: B must be non-transpose."
        assert not self.c_layout_traits.is_transpose, "Internal Error: C must be non-transpose."
        try:
            operations = lookup_table[key]
            logger.debug("Layout lookup table contains a mapping.")
        except KeyError:
            # We have to fall back to the exhaustive search because ambigously ordered
            # matrices (vectors/scalars) miss the cache
            logger.debug("Layout lookup table does not contain a mapping; performing exhaustive search.")
            return self.find_supported_layout(layout_checker, logger)
        new_traits = self
        for operation in operations:
            new_traits = getattr(new_traits, operation)(logger)
        assert layout_checker(new_traits), "Internal Error: Layout checker returned False for a supported layout."
        return new_traits  # type: ignore[return-value]


"""A function which checks if a layout is supported for a BLAS matrix multiplication."""
MMLayoutChecker: typing.TypeAlias = typing.Callable[[BLASMMTraitsView], bool]

"""Maps from input traits to the operations needed to get a supported layout."""
MMLayoutCheckerLookupTable: typing.TypeAlias = dict[
    tuple[cublaslt.Order, cublaslt.Order, cublaslt.Order, bool, bool, bool], list[str]
]
