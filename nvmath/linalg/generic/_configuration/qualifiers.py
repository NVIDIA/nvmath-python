"""
Matrix qualifier dataclasses for describing structured matrix types and their properties in
linear algebra operations. Provides qualifiers for general, symmetric, hermitian,
triangular, and diagonal matrices with associated metadata like fill modes, transpose flags,
and BLAS function abbreviations.
"""

import abc
import dataclasses
import typing

import numpy as np
import numpy.typing as npt

import nvmath.bindings.cublas as cublas

from nvmath._internal.templates import StatefulAPIOptions
from nvmath.internal import utils


FillMode: typing.TypeAlias = cublas.FillMode
DiagType: typing.TypeAlias = cublas.DiagType


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MatmulOptions(StatefulAPIOptions):
    """A dataclass for providing options to a :class:`Matmul` object.

    Attributes:
        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used
            to draw device memory. If an allocator is not provided, a memory allocator from
            the library package will be used (:func:`torch.cuda.caching_allocator_alloc` for
            PyTorch operands, :func:`cupy.cuda.alloc` otherwise).

        blocking: A flag specifying the behavior of the stream-ordered functions and
            methods. When ``blocking`` is `True`, the stream-ordered methods do not return
            until the operation is complete. When ``blocking`` is ``"auto"``, the methods
            return immediately when the inputs are on the GPU. The stream-ordered methods
            always block when the operands are on the CPU to ensure that the user doesn't
            inadvertently use the result before it becomes available. The default is
            ``"auto"``.

        inplace: Whether the matrix multiplication is performed in-place (operand C is
            overwritten).

        logger: Python Logger object. The root logger will be used if a
            logger object is not provided.

    .. seealso::
       :class:`StatefulAPI`
    """

    inplace: bool = False


MM_QUALIFIERS_DOCUMENTATION = {
    #
    "abbreviation": """\
The two character abbreviation of the matrix qualifier.""".replace("\n", " "),
    #
    "conjugate": """\
Whether the matrix is conjugate.""".replace("\n", " "),
    #
    "transpose": """\
Whether the matrix is transpose.""".replace("\n", " "),
    #
    "uplo": """\
The :py:class:`~nvmath.bindings.cublas.FillMode` of the matrix. e.g. upper, lower...""".replace("\n", " "),
    #
    "diag": """\
The :py:class:`~nvmath.bindings.cublas.DiagType` of the matrix. e.g. unit, non-unit...""".replace("\n", " "),
    #
    "incx": """\
The direction to read the diagonal. +1 for forward; -1 for reverse.""".replace("\n", " "),
    #
}


# We name this variable as foo_bar_dtype, so that the docstrings format correctly
# Docstrings for custom dtypes are defined in docs/sphinx/conf.py
matrix_qualifiers_dtype = np.dtype(
    [
        ("abbreviation", "U2"),
        ("conjugate", np.bool_),
        ("transpose", np.bool_),
        ("uplo", np.int_),
        ("diag", np.int_),
        ("incx", np.int_),
    ]
)

"""A NumPy array of type :class:`matrix_qualifiers_dtype`.

    .. seealso::
        :class:`GeneralMatrixQualifier`, :class:`HermitianMatrixQualifier`,
        :class:`SymmetricMatrixQualifier`, :class:`TriangularMatrixQualifier`
        :class:`DiagonalMatrixQualifier`, :class:`matrix_qualifiers_dtype`
"""
MatrixQualifier: typing.TypeAlias = npt.NDArray


class MatrixQualifierConstructor(abc.ABC):
    abbreviation: typing.ClassVar[str]

    def __init__(self):
        msg = f"The {self.__class__.__name__} constructor should not be called. Use {self.__class__.__name__}.create() instead."
        raise RuntimeError(msg)

    @classmethod
    def create(
        cls,
        conjugate: bool = False,
        transpose: bool = False,
        uplo: FillMode = FillMode.FULL,
        diag: DiagType = DiagType.NON_UNIT,
        incx: typing.Literal[-1, 1, 0] = 0,
    ):
        return np.array((cls.abbreviation, conjugate, transpose, uplo, diag, incx), dtype=matrix_qualifiers_dtype)

    @classmethod
    def is_valid(cls, other: MatrixQualifier) -> np.bool_:
        """Return ``True`` if all elements of `other` are valid examples of the
        :class:`matrix_qualifiers_dtype` constructed by this class."""
        return np.all(other["abbreviation"] == cls.abbreviation) and np.bool_(
            all(
                n in other.dtype.names  # type: ignore[operator]
                for n in ("conjugate", "transpose")
            )
        )

    @classmethod
    def to_string(cls, other: MatrixQualifier) -> str:
        """Return a pretty string representation of `other`."""
        return (
            f"({other['abbreviation']}, conjugate={other['conjugate']}, "
            f"transpose={other['transpose']}, uplo={FillMode(other['uplo']).name}, "
            f"diag={DiagType(other['diag']).name}, incx={other['incx']})"
        )


@utils.docstring_decorator(MM_QUALIFIERS_DOCUMENTATION, skip_missing=False)
class GeneralMatrixQualifier(MatrixQualifierConstructor):
    """A class which constructs and validates :class:`matrix_qualifiers_dtype` for a general
    rectangular matrix.

    Examples:

        >>> import numpy as np
        >>> from nvmath.linalg import GeneralMatrixQualifier, matrix_qualifiers_dtype

        Create a general matrix qualifier:

        >>> GeneralMatrixQualifier.create()  # doctest: +ELLIPSIS
        array(('ge', False, False, 2, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create a conjugate general matrix qualifier:

        >>> GeneralMatrixQualifier.create(conjugate=True)  # doctest: +ELLIPSIS
        array(('ge', True, False, 2, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create an array of general matrix qualifiers:

        >>> np.full(
        ...     2, GeneralMatrixQualifier.create(), dtype=matrix_qualifiers_dtype
        ... )  # doctest: +ELLIPSIS
        array([('ge', False, False, 2, 0, 0), ('ge', False, False, 2, 0, 0)],
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

    .. seealso::
        :class:`GeneralMatrixQualifier`, :class:`HermitianMatrixQualifier`,
        :class:`SymmetricMatrixQualifier`, :class:`TriangularMatrixQualifier`
        :class:`DiagonalMatrixQualifier`, :class:`matrix_qualifiers_dtype`
    """

    abbreviation: typing.ClassVar[str] = "ge"

    @classmethod
    def create(  # type: ignore[override]
        cls,
        conjugate: bool = False,
        transpose: bool = False,
    ):
        """Return a :class:`np.ndarray` of type :class:`matrix_qualifiers_dtype` whose
        element describes a general matrix.

        Args:
            conjugate: {conjugate}

            transpose: {transpose}
        """
        return super().create(conjugate=conjugate, transpose=transpose)


@utils.docstring_decorator(MM_QUALIFIERS_DOCUMENTATION, skip_missing=False)
class DiagonalMatrixQualifier(MatrixQualifierConstructor):
    """A class which constructs and validates :class:`matrix_qualifiers_dtype` for a
    diagonal matrix.

    Examples:

        >>> import numpy as np
        >>> from nvmath.linalg import (
        ...     DiagonalMatrixQualifier,
        ...     GeneralMatrixQualifier,
        ...     matrix_qualifiers_dtype,
        ... )

        Create a diagonal matrix qualifier:

        >>> DiagonalMatrixQualifier.create()  # doctest: +ELLIPSIS
        array(('dg', False, False, 2, 0, 1),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create a conjugate diagonal matrix qualifier:

        >>> DiagonalMatrixQualifier.create(conjugate=True)  # doctest: +ELLIPSIS
        array(('dg', True, False, 2, 0, 1),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create an array of matrix qualifiers with one general and one diagonal matrix:

        >>> qualifiers = np.full(
        ...     2,
        ...     GeneralMatrixQualifier.create(),
        ...     dtype=matrix_qualifiers_dtype,
        ... )
        >>> qualifiers[1] = DiagonalMatrixQualifier.create()
        >>> qualifiers  # doctest: +ELLIPSIS
        array([('ge', False, False, 2, 0, 0), ('dg', False, False, 2, 0, 1)],
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

    .. seealso::
        :class:`GeneralMatrixQualifier`, :class:`HermitianMatrixQualifier`,
        :class:`SymmetricMatrixQualifier`, :class:`TriangularMatrixQualifier`
        :class:`DiagonalMatrixQualifier`, :class:`matrix_qualifiers_dtype`
    """

    abbreviation: typing.ClassVar[str] = "dg"

    @classmethod
    def create(  # type: ignore[override]
        cls,
        conjugate: bool = False,
        transpose: bool = False,
        incx: typing.Literal[-1, 1] = 1,
    ):
        """Return a :class:`np.ndarray` of type :class:`matrix_qualifiers_dtype` whose
        element describes a diagonal matrix.

        Args:
            conjugate: {conjugate}

            transpose: {transpose}

            incx: {incx}
        """
        if incx not in (-1, 1):
            raise ValueError(f"The 'incx' parameter must be '-1' or '1' not {incx}")
        return super().create(conjugate, transpose, incx=incx)

    @classmethod
    def is_valid(cls, other):
        return super().is_valid(other) and np.all(np.logical_or(other["incx"] == -1, other["incx"] == 1))


@utils.docstring_decorator(MM_QUALIFIERS_DOCUMENTATION, skip_missing=False)
class SquareMatrixQualifier(MatrixQualifierConstructor, abc.ABC):
    @classmethod
    def create(  # type: ignore[override]
        cls,
        conjugate: bool = False,
        transpose: bool = False,
        uplo: FillMode = FillMode.LOWER,
        **kwargs,
    ):
        if uplo not in (FillMode.UPPER, FillMode.LOWER):
            raise ValueError(f"The 'uplo' parameter must be 'UPPER' or 'LOWER', not {uplo}.")
        return super().create(conjugate, transpose, uplo=uplo, **kwargs)

    @classmethod
    def is_valid(cls, other):
        return super().is_valid(other) and np.all(
            np.logical_or(other["uplo"] == FillMode.UPPER, other["uplo"] == FillMode.LOWER)
        )


@utils.docstring_decorator(MM_QUALIFIERS_DOCUMENTATION, skip_missing=False)
class HermitianMatrixQualifier(SquareMatrixQualifier):
    """A class which constructs and validates :class:`matrix_qualifiers_dtype` for a
    hermitian matrix.

    Examples:

        >>> import numpy as np
        >>> from nvmath.linalg import (
        ...     HermitianMatrixQualifier,
        ...     GeneralMatrixQualifier,
        ...     matrix_qualifiers_dtype,
        ... )

        Create a hermitian matrix qualifier:

        >>> HermitianMatrixQualifier.create()  # doctest: +ELLIPSIS
        array(('he', False, False, 0, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create a conjugate hermitian matrix qualifier:

        >>> HermitianMatrixQualifier.create(conjugate=True)  # doctest: +ELLIPSIS
        array(('he', True, False, 0, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create an array of matrix qualifiers with one general and one hermitian matrix:

        >>> qualifiers = np.full(
        ...     2,
        ...     GeneralMatrixQualifier.create(),
        ...     dtype=matrix_qualifiers_dtype,
        ... )
        >>> qualifiers[1] = HermitianMatrixQualifier.create()
        >>> qualifiers  # doctest: +ELLIPSIS
        array([('ge', False, False, 2, 0, 0), ('he', False, False, 0, 0, 0)],
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

    .. seealso::
        :class:`GeneralMatrixQualifier`, :class:`HermitianMatrixQualifier`,
        :class:`SymmetricMatrixQualifier`, :class:`TriangularMatrixQualifier`
        :class:`DiagonalMatrixQualifier`, :class:`matrix_qualifiers_dtype`
    """

    abbreviation: typing.ClassVar[str] = "he"

    @classmethod
    def create(  # type: ignore[override]
        cls,
        conjugate: bool = False,
        transpose: bool = False,
        uplo: FillMode = FillMode.LOWER,
    ):
        """Return a :class:`np.ndarray` of type :class:`matrix_qualifiers_dtype` whose
        element describes a hermitian matrix.

        Args:
            conjugate: {conjugate}

            transpose: {transpose}

            uplo: {uplo}
        """
        return super().create(conjugate, transpose, uplo)


@utils.docstring_decorator(MM_QUALIFIERS_DOCUMENTATION, skip_missing=False)
class SymmetricMatrixQualifier(SquareMatrixQualifier):
    """A class which constructs and validates :class:`matrix_qualifiers_dtype` for a
    symmetric matrix.

    Examples:

        >>> import numpy as np
        >>> from nvmath.linalg import (
        ...     SymmetricMatrixQualifier,
        ...     GeneralMatrixQualifier,
        ...     matrix_qualifiers_dtype,
        ... )

        Create a symmetric matrix qualifier:

        >>> SymmetricMatrixQualifier.create()  # doctest: +ELLIPSIS
        array(('sy', False, False, 0, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create a conjugate symmetric matrix qualifier:

        >>> SymmetricMatrixQualifier.create(conjugate=True)  # doctest: +ELLIPSIS
        array(('sy', True, False, 0, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create an array of matrix qualifiers with one general and one symmetric matrix:

        >>> qualifiers = np.full(
        ...     2,
        ...     GeneralMatrixQualifier.create(),
        ...     dtype=matrix_qualifiers_dtype,
        ... )
        >>> qualifiers[1] = SymmetricMatrixQualifier.create()
        >>> qualifiers  # doctest: +ELLIPSIS
        array([('ge', False, False, 2, 0, 0), ('sy', False, False, 0, 0, 0)],
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

    .. seealso::
        :class:`GeneralMatrixQualifier`, :class:`HermitianMatrixQualifier`,
        :class:`SymmetricMatrixQualifier`, :class:`TriangularMatrixQualifier`
        :class:`DiagonalMatrixQualifier`, :class:`matrix_qualifiers_dtype`
    """

    abbreviation: typing.ClassVar[str] = "sy"

    @classmethod
    def create(  # type: ignore[override]
        cls,
        conjugate: bool = False,
        transpose: bool = False,
        uplo: FillMode = FillMode.LOWER,
    ):
        """Return a :class:`np.ndarray` of type :class:`matrix_qualifiers_dtype` whose
        element describes a symmetric matrix.

        Args:
            conjugate: {conjugate}

            transpose: {transpose}

            uplo: {uplo}
        """
        return super().create(conjugate, transpose, uplo)


@utils.docstring_decorator(MM_QUALIFIERS_DOCUMENTATION, skip_missing=False)
class TriangularMatrixQualifier(SquareMatrixQualifier):
    """A class which constructs and validates :class:`matrix_qualifiers_dtype` for a
    triangular matrix.

    Examples:

        >>> import numpy as np
        >>> from nvmath.linalg import (
        ...     TriangularMatrixQualifier,
        ...     GeneralMatrixQualifier,
        ...     matrix_qualifiers_dtype,
        ... )

        Create a triangular matrix qualifier:

        >>> TriangularMatrixQualifier.create()  # doctest: +ELLIPSIS
        array(('tr', False, False, 0, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create a conjugate triangular matrix qualifier:

        >>> TriangularMatrixQualifier.create(conjugate=True)  # doctest: +ELLIPSIS
        array(('tr', True, False, 0, 0, 0),
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

        Create an array of matrix qualifiers with one general and one triangular matrix:

        >>> qualifiers = np.full(
        ...     2,
        ...     GeneralMatrixQualifier.create(),
        ...     dtype=matrix_qualifiers_dtype,
        ... )
        >>> qualifiers[1] = TriangularMatrixQualifier.create()
        >>> qualifiers  # doctest: +ELLIPSIS
        array([('ge', False, False, 2, 0, 0), ('tr', False, False, 0, 0, 0)],
              dtype=[('abbreviation', '<U2'), ('conjugate', '?'), ...

    .. seealso::
        :class:`GeneralMatrixQualifier`, :class:`HermitianMatrixQualifier`,
        :class:`SymmetricMatrixQualifier`, :class:`TriangularMatrixQualifier`
        :class:`DiagonalMatrixQualifier`, :class:`matrix_qualifiers_dtype`
    """

    abbreviation: typing.ClassVar[str] = "tr"

    @classmethod
    def create(  # type: ignore[override]
        cls,
        conjugate: bool = False,
        transpose: bool = False,
        uplo: FillMode = FillMode.LOWER,
        diag: DiagType = DiagType.NON_UNIT,
    ):
        """Return a :class:`np.ndarray` of type :class:`matrix_qualifiers_dtype` whose
        element describes a triangular matrix.

        Args:
            conjugate: {conjugate}

            transpose: {transpose}

            uplo: {uplo}

            diag: {diag}
        """
        if diag not in (DiagType.UNIT, DiagType.NON_UNIT):
            raise ValueError(f"The 'diag' parameter must be 'UNIT' or 'NON_UNIT', not {diag}.")
        return super().create(conjugate, transpose, diag=diag, uplo=uplo)

    @classmethod
    def is_valid(cls, other):
        return super().is_valid(other) and np.all(
            np.logical_or(other["diag"] == DiagType.UNIT, other["diag"] == DiagType.NON_UNIT)
        )


def vector_to_square(
    shape: typing.Sequence[int], strides: typing.Sequence[int], qualifier: MatrixQualifier
) -> tuple[typing.Sequence[int], typing.Sequence[int]]:
    """If `qualifier` is a DiagonalMatrixQualifier, convert `shape` and `stride` from a
    vector to the equivalent square matrix."""
    if DiagonalMatrixQualifier.is_valid(qualifier):
        if len(shape) != 1:
            msg = f"The shape of a diagonal matrix must be 1D; not {shape}."
            raise ValueError(msg)
        shape = tuple(shape) * 2
        if len(strides) != 1:
            msg = f"The strides of a diagonal matrix must be 1D; not {strides}."
            raise ValueError(msg)
        strides = (0, *strides)
    return shape, strides
