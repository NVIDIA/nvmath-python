# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["matmul", "TransposeMode", "Matmul", "SharedStorageCalc"]

from functools import cached_property
import itertools
from collections.abc import Sequence
import re
from typing import Any, overload
from warnings import warn

from .common import (
    Layout,
    Partitioner,
    check_code_type,
    check_in,
    SHARED_DEVICE_DOCSTRINGS,
    pad_or_truncate,
    parse_sm,
)
from .common_backend import MATHDX_TYPES_TO_NP, get_isa_version, get_lto
from .common_cuda import (
    Code,
    CodeType,
    Dim3,
)
from .cublasdx_backend import (
    Alignment,
    Arrangement,
    Precision,
    generate_MM,
    generate_code,
    generate_function_code,
    generate_tensor,
    generate_tensors,
    get_function_code,
    get_str_trait,
    get_int_traits,
    get_tensor_traits,
    validate,
    LeadingDimension,
    TransposeMode,
    validate_alignment,
    validate_execute_api,
    validate_tensor_types,
    MAX_ALIGNMENT,  # noqa: F401
)
from ._deprecated import deprecated
from nvmath.internal.utils import docstring_decorator

from nvmath.bindings import mathdx
import numpy

CUBLASDX_DOCSTRING = SHARED_DEVICE_DOCSTRINGS.copy()
CUBLASDX_DOCSTRING.update(
    {
        "size": """\
A sequence of integers denoting the three dimensions ``(m, n, k)`` for the matrix multiplication
problem.""".replace("\n", " "),
        #
        "data_type": """\
The data type of the input matrices, can be either ``'real'`` or ``'complex'``.""".replace("\n", " "),
        #
        "block_size": """\
The total block size, optional. If not provided or set to ``'suggested'``, will be set to a suggested value for 1D block
dim. """.replace("\n", " "),
        #
        "block_dim": """\
The block dimension for launching the CUDA kernel, optional. If not provided or set to ``'suggested'``, will be set to a
suggested value. Cannot be used when `block_size` is explicitly specified.""".replace("\n", " "),
        #
        "leading_dimension": """\
The leading dimensions for the input matrices, optional. If not provided, will be set to match the matrix row/column
dimension. Alternatively, if provided as ``'suggested'``, will be set to a suggested value for optimal performance.
""".replace("\n", " "),
        #
        "transpose_mode": """\
The transpose mode for all input matrices ;
transpose_mode or arrangement must be provided.""".replace("\n", " "),
        #
        "arrangement": """\
The arrangement for all input matrices ;
transpose_mode or arrangement must be provided.""".replace("\n", " "),
        #
        "alignment": """\
The alignment for the input matrices in shared memory.
Defines the alignments (in bytes) of the input matrices A, B, and C
(either arrays or wrapped in opaque tensors) that are passed to the
execute(...) method. Default alignment is equal to an element size of the
matrix unless used suggested layout. In that case alignment is greater or equal
than the element size.""".replace("\n", " "),
        #
        "global_memory_alignment": """\
Same as alignment, but for the global memory. Used to optimize copying between
shared and global memory.
""".replace("\n", " "),
        #
        "function": """\
A string specifying the name of the function. Currently supports ``'MM'`` (default) for matrix
multiplication.""".replace("\n", " "),
        #
        "execute_api": """\
A string specifying the signature of the function that handles problems with default or custom/dynamic leading dimensions.
Could be ``'static_leading_dimensions'`` or ``'dynamic_leading_dimensions'``.""".replace("\n", " "),
        "tensor_types": """\
A list of strings specifying the tensors being used at execute signature.""".replace("\n", " "),
    }
)

#
# A set of knobs, potentially in-complete (ie not sufficient to generate a device functions)
#


class SharedStorageCalc:
    """
    Helper class to calculate shared storage size.

    For further details, please refer to `cuBLASDx documentation
    <https://docs.nvidia.com/cuda/cublasdx/>`_.
    """

    _memory: int = 0

    @overload
    def add(self, alignment: int, matrix_size_bytes: int) -> None: ...
    @overload
    def add(self, alignment: int, elem_size: int, num_elements: int) -> None: ...
    @overload
    def add(self, alignment: int, elem_size: int, layout: Layout) -> None: ...
    def add(self, *args):
        assert len(args) in {2, 3}

        if len(args) == 2:
            [alignment, matrix_size_bytes] = args

            assert matrix_size_bytes > 0
        else:
            [alignment, elem_size, num_elements] = args

            if isinstance(num_elements, Layout):
                num_elements = num_elements.cosize

            assert elem_size > 0
            assert num_elements > 0

            matrix_size_bytes = elem_size * num_elements

        assert alignment > 0

        self._memory = ((self._memory + alignment - 1) // alignment) * alignment + matrix_size_bytes

    def get(self):
        return self._memory


@docstring_decorator(CUBLASDX_DOCSTRING, skip_missing=False)
class Matmul:
    """
    A class that encapsulates a partial Matmul device function. A partial device function
    can be queried for available or optimal values for some knobs (such as
    `leading_dimension` or `block_dim`).

    .. versionchanged:: 0.7.0
        `Matmul` has replaced `BlasOptions` and `BlasOptionsComplete`.

    Args:
        size: {size}

        precision: {precision}

        data_type: {data_type}

        sm (ComputeCapability): {sm}

        block_size (int): {block_size}

        block_dim (Dim3): {block_dim}

        leading_dimension (LeadingDimension): {leading_dimension}

        transpose_mode (TransposeMode): {transpose_mode}

        arrangement (Arrangement): {arrangement}

        alignment (Alignment): {alignment}

        function (str): {function}

        execution (str): {execution}

        execute_api (str): {execute_api}

            .. versionchanged:: 0.5.0
                execute_api is not part of the Matmul (ex. Blas) type. Pass this
                argument to :py:func:`nvmath.device.matmul` instead.

        tensor_types (Sequence[str]): {tensor_types}

            .. versionchanged:: 0.5.0
                tensor_types is not part of the Matmul (ex. Blas) type. Pass
                this argument to :py:func:`nvmath.device.matmul` instead.

    .. seealso::
        The attributes of this class provide a 1:1 mapping with the CUDA C++ cuBLASDx APIs.
        For further details, please refer to `cuBLASDx documentation
        <https://docs.nvidia.com/cuda/cublasdx/>`_.
    """

    def __init__(
        self,
        size,
        precision,
        data_type,
        *,
        sm=None,
        block_size=None,
        block_dim=None,
        leading_dimension=None,
        transpose_mode=None,
        arrangement=None,
        alignment=None,
        function="MM",
        static_block_dim=False,
        execution="Block",
    ):
        sm = parse_sm(sm)

        if transpose_mode is not None:
            warn(
                "transpose_mode is deprecated and may be removed in future versions. User arrangement instead",
                category=DeprecationWarning,
            )
            if not isinstance(transpose_mode, Sequence) or len(transpose_mode) != 2:
                raise ValueError(
                    "transpose_mode should be an instance of TransposeMode or a 2-tuple ; "
                    f"got transpose_mode = {transpose_mode}"
                )
            transpose_mode = TransposeMode(*transpose_mode)
        if arrangement is not None:
            if not isinstance(arrangement, Sequence) or len(arrangement) != 3:
                raise ValueError(
                    f"arrangement should be an instance of Arrangement or a 3-tuple ; got arrangement = {arrangement}"
                )
            arrangement = Arrangement(*arrangement)

        if alignment is not None:
            if not isinstance(alignment, Sequence) or len(alignment) != 3:
                raise ValueError(f"alignment should be an instance of Alignment or a 3-tuple ; got alignment = {alignment}")
            alignment = Alignment(*alignment)

        if leading_dimension is not None and leading_dimension != "suggested":
            if not isinstance(leading_dimension, Sequence) or len(leading_dimension) != 3:
                raise ValueError(
                    "leading_dimension should be a 3-tuple, an instance of LeadingDimension, 'suggested' or None ; "
                    f"got leading_dimension = {leading_dimension}"
                )
            else:
                leading_dimension = LeadingDimension(*leading_dimension)

        if isinstance(precision, Sequence):
            if len(precision) != 3:
                raise ValueError(
                    "precision should be a 3-len sequence, an instance of Precision, or a single value; "
                    f"got precision = {precision}"
                )
        else:
            precision = (precision, precision, precision)
        precision = Precision(*precision)

        #
        # Check that the knobs are, individually, valid
        #

        if block_size is not None and block_dim is not None:
            raise ValueError(f"Both block_size ({block_size}) and block_dim ({block_dim}) cannot be specified.")
        if block_size is not None:
            check_in("block_dim", block_dim, [None])
            if block_size == "suggested":
                block_dim = "suggested"
            else:
                block_dim = Dim3(block_size, 1, 1)
        if block_dim is not None and isinstance(block_dim, Sequence) and block_dim != "suggested":
            if len(block_dim) != 3:
                raise ValueError(
                    f"block_dim should be a 3-tuple, an instance of Dim3, 'suggested' or None ; got block_dim = {block_dim}"
                )
            else:
                block_dim = Dim3(*block_dim)

        validate(
            size=size,
            precision=precision,
            data_type=data_type,
            transpose_mode=transpose_mode,
            arrangement=arrangement,
            alignment=alignment,
            sm=sm,
            leading_dimension=leading_dimension,
            block_dim=block_dim,
            function=function,
            execution=execution,
            static_block_dim=static_block_dim,
        )

        #
        # Traits set by input
        #

        self._size = size
        self._precision = precision
        self._data_type = data_type
        self._transpose_mode = transpose_mode
        self._arrangement = arrangement
        self._alignment = alignment
        self._sm = sm
        self._block_dim = block_dim
        self._function = function
        self._execution = execution
        self._leading_dimension = leading_dimension
        self._static_block_dim = static_block_dim

        #
        # Update suggested traits
        #

        if leading_dimension == "suggested":
            self._leading_dimension = self._suggested_leading_dimension

        if block_dim == "suggested":
            self._block_dim = self._suggested_block_dim

    @cached_property
    def _traits(self):
        return _MatmulTraits(self)

    @property
    def precision(self) -> Precision:
        return self._precision

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def size(self) -> tuple[int, int, int]:
        return self._size

    @property
    def execution(self) -> str:
        return self._execution

    @property
    @deprecated("transpose_mode trait is deprecated and may be removed in future versions. Use arrangement instead")
    def transpose_mode(self) -> TransposeMode:
        return self._transpose_mode

    @property
    def arrangement(self) -> Arrangement:
        return self._arrangement

    @property
    def alignment(self) -> Alignment:
        if self._alignment is None:
            return self._traits.alignment
        return self._alignment

    @property
    def sm(self):
        return self._sm

    @property
    def function(self) -> str:
        return self._function

    @property
    def block_size(self) -> int:
        return self.block_dim[0] * self.block_dim[1] * self.block_dim[2]

    @property
    def block_dim(self) -> Dim3:
        if self._block_dim is None:
            return self._traits.block_dim
        return self._block_dim

    @property
    def static_block_dim(self) -> bool:
        return self._static_block_dim

    @property
    def leading_dimension(self) -> LeadingDimension:
        if self._leading_dimension is None:
            return self._traits.leading_dimension
        return self._leading_dimension

    #
    # Extensions
    #

    def valid(self, *knobs):
        return itertools.product(*[self._valid(knob) for knob in knobs])

    @deprecated("definition is deprecated and may be removed in future versions")
    def definition(self):
        """
        .. deprecated:: 0.7.0
        """
        dd = {
            "size": self.size,
            "precision": self.precision,
            "data_type": self.data_type,
            "transpose_mode": self.transpose_mode,
            "arrangement": self.arrangement,
            "alignment": self.alignment,
            "sm": self.sm,
            "block_dim": self.block_dim,
            "static_block_dim": self.static_block_dim,
            "function": self.function,
            "execution": self.execution,
            "leading_dimension": self.leading_dimension,
        }
        return dd

    @deprecated("create is deprecated and may be removed in future versions. Use `functools.partial` instead")
    def create(
        self, code_type=None, compiler=None, execute_api=None, tensor_types=None, global_memory_alignment=None, **kwargs
    ):
        """
        Creates a copy of the instance with provided arguments updated.

        .. deprecated:: 0.7.0
            Please use :py:func:`functools.partial` instead.
        """
        if code_type is not None:
            DeprecationWarning("code_type is deprecated and will be removed in future releases. It is no longer needed.")
        if compiler is not None:
            DeprecationWarning("compiler is deprecated and will be removed in future releases. It is no longer needed.")
        if execute_api is not None:
            DeprecationWarning("execute_api is deprecated and will be removed in future releases. It is no longer needed.")
        if tensor_types is not None:
            DeprecationWarning("tensor_types is deprecated and will be removed in future releases. It is no longer needed.")
        if global_memory_alignment is not None:
            DeprecationWarning(
                "global_memory_alignment is deprecated and will be removed in future releases. It is no longer needed."
            )
        dd = self.definition()
        dd.update(**kwargs)
        return Matmul(**dd)

    #
    # Private implementations
    #

    def _valid(self, knob):
        if knob == "block_dim":
            return [self._suggested_block_dim]
        else:
            raise ValueError("Unsupported knob")

    @cached_property
    def _suggested_leading_dimension(self):
        # Generate special PTX for suggested_leading_dimension_of
        descriptor = generate_MM(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            sm=self.sm,
            transpose_mode=self._transpose_mode,
            arrangement=self._arrangement,
            alignment=self._alignment,
            block_dim=None,
            static_block_dim=self._static_block_dim,
            leading_dimension=None,
            execution=self.execution,
        )

        return LeadingDimension(*get_int_traits(descriptor.descriptor, mathdx.CublasdxTraitType.SUGGESTED_LEADING_DIMENSION, 3))

    @cached_property
    def _suggested_block_dim(self):
        # Generate full PTX
        descriptor = generate_MM(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            sm=self.sm,
            transpose_mode=self._transpose_mode,
            arrangement=self._arrangement,
            alignment=self._alignment,
            block_dim=None,
            static_block_dim=self._static_block_dim,
            leading_dimension=None,
            execution=self.execution,
        )

        return Dim3(*get_int_traits(descriptor.descriptor, mathdx.CublasdxTraitType.SUGGESTED_BLOCK_DIM, 3))

    @property
    def a_value_type(self):
        return self._traits.value_types[0]

    @property
    def b_value_type(self):
        return self._traits.value_types[1]

    @property
    def c_value_type(self):
        return self._traits.value_types[2]

    @property
    @deprecated("value_type trait is deprecated. Please use {a|b|c}_value_type instead")
    def value_type(self):
        if not all(vt == self._traits.value_types[0] for vt in self._traits.value_types):
            raise RuntimeError("value_type may be used only if all {a|b|c}_value_type have the same type")
        return self.a_value_type

    @property
    @deprecated("input_type trait is deprecated. Please use {a|b}_value_type instead")
    def input_type(self):
        if self.a_value_type != self.b_value_type:
            raise RuntimeError("input_type may be used only if A and B input matrix have the same type")
        return self.a_value_type

    @property
    @deprecated("output_type trait is deprecated. Please use c_value_type instead")
    def output_type(self):
        return self.c_value_type

    @cached_property
    def a_dim(self):
        (m, _, k) = self.size

        dim = (m, k)
        if self._transpose_mode is not None and self._transpose_mode.a in {"transposed", "conj_transposed"}:
            dim = dim[::-1]

        return dim

    @cached_property
    def b_dim(self):
        (_, n, k) = self.size

        dim = (k, n)
        if self._transpose_mode is not None and self._transpose_mode.b in {"transposed", "conj_transposed"}:
            dim = dim[::-1]

        return dim

    @cached_property
    def c_dim(self):
        (m, n, _) = self.size
        return (m, n)

    def _calculate_abc_sizes(self, ld: LeadingDimension) -> tuple[int, int, int]:
        if self._transpose_mode:
            non_ld = (self.a_dim[1], self.b_dim[1], self.c_dim[1])
        elif self._arrangement:
            non_ld = (
                self.a_dim[1 if self._arrangement.a == "col_major" else 0],
                self.b_dim[1 if self._arrangement.b == "col_major" else 0],
                self.c_dim[1 if self._arrangement.c == "col_major" else 0],
            )

        return tuple(x * y for x, y in zip(ld, non_ld, strict=True))

    @cached_property
    def _abc_sizes(self):
        return self._calculate_abc_sizes(self.leading_dimension)

    @property
    def a_size(self):
        return self._abc_sizes[0]

    @property
    def b_size(self):
        return self._abc_sizes[1]

    @property
    def c_size(self):
        return self._abc_sizes[2]

    @property
    @deprecated(
        "shared_memory_size trait is deprecated and will be removed in "
        "future versions. Use get_shared_storage_size instead. Don't "
        "use with Opaque Tensors. Use get_shared_storage_size(...) or"
        "SharedStorageCalc instead"
    )
    def shared_memory_size(self):
        return self.get_shared_storage_size()

    @property
    def max_threads_per_block(self):
        return self.block_dim.x * self.block_dim.y * self.block_dim.z

    def _get_shared_storage_size(self, *args, ab=False) -> int | None:  # type: ignore
        # Complex will be over-aligned (eg: f32x2 complex is aligned on 8B) with
        # this logic (which is what we want - for performance and vectorization)
        item_sizes = tuple(numpy.dtype(vt).itemsize for vt in self._traits.value_types)

        alignment = self.alignment
        sizes = None

        if len(args) == 0:
            sizes = (self.a_size, self.b_size, self.c_size)
        elif all(isinstance(arg, int) for arg in args):
            sizes = self._calculate_abc_sizes(LeadingDimension(*pad_or_truncate(list(args), 3)))
        elif all(isinstance(arg, Layout) for arg in args):
            sizes = tuple(arg.cosize for arg in args)

        if sizes is None:
            return None

        smem_calc = SharedStorageCalc()
        smem_calc.add(alignment[0], item_sizes[0], sizes[0])
        smem_calc.add(alignment[1], item_sizes[1], sizes[1])
        if not ab:
            smem_calc.add(alignment[2], item_sizes[2], sizes[2])
        return smem_calc.get()

    @overload
    def get_shared_storage_size(self) -> int: ...
    @overload
    def get_shared_storage_size(self, lda: int, ldb: int, ldc: int) -> int: ...
    @overload
    def get_shared_storage_size(self, matrix_a_layout: Layout, matrix_b_layout: Layout, matrix_c_layout: Layout) -> int: ...
    def get_shared_storage_size(self, *args) -> int:  # type: ignore
        value_error = ValueError(
            "get_shared_storage_size() takes either 0 or 3 arguments. If 3 "
            "arguments are provided, they must be either all integers or "
            "all Layout objects."
        )
        if len(args) not in {0, 3}:
            raise value_error
        if any(not isinstance(arg, Layout) for arg in args) and any(not isinstance(arg, int) for arg in args):
            raise value_error
        size = self._get_shared_storage_size(*args, ab=False)
        if size is None:
            raise value_error
        return size

    @overload
    def get_shared_storage_size_ab(self) -> int: ...
    @overload
    def get_shared_storage_size_ab(self, lda: int, ldb: int) -> int: ...
    @overload
    def get_shared_storage_size_ab(self, matrix_a_layout: Layout, matrix_b_layout: Layout) -> int: ...
    def get_shared_storage_size_ab(self, *args) -> int:  # type: ignore
        value_error = ValueError(
            "get_shared_storage_size_ab() takes either 0 or 2 arguments. "
            "If 2 arguments are provided, they must be either all integers "
            "or all Layout objects."
        )
        if len(args) not in {0, 2}:
            raise value_error
        if any(not isinstance(arg, Layout) for arg in args) and any(not isinstance(arg, int) for arg in args):
            raise value_error
        size = self._get_shared_storage_size(*args, ab=True)
        if size is None:
            raise value_error
        return size

    def get_layout_gmem_a(self, leading_dimension: int | None = None) -> Layout:
        return _BlasLayout(self, "get_layout_gmem_a", leading_dimension)

    def get_layout_gmem_b(self, leading_dimension: int | None = None) -> Layout:
        return _BlasLayout(self, "get_layout_gmem_b", leading_dimension)

    def get_layout_gmem_c(self, leading_dimension: int | None = None) -> Layout:
        return _BlasLayout(self, "get_layout_gmem_c", leading_dimension)

    def get_layout_smem_a(self) -> Layout:
        return _BlasLayout(self, "get_layout_smem_a")

    def get_layout_smem_b(self) -> Layout:
        return _BlasLayout(self, "get_layout_smem_b")

    def get_layout_smem_c(self) -> Layout:
        return _BlasLayout(self, "get_layout_smem_c")

    def suggest_layout_smem_a(self) -> Layout:
        return _BlasLayout(self, "suggest_layout_smem_a")

    def suggest_layout_smem_b(self) -> Layout:
        return _BlasLayout(self, "suggest_layout_smem_b")

    def suggest_layout_smem_c(self) -> Layout:
        return _BlasLayout(self, "suggest_layout_smem_c")

    def suggest_layout_rmem_c(self) -> Layout:
        return _BlasLayout(self, "suggest_layout_rmem_c")

    def suggest_partitioner(self) -> Partitioner:
        raise RuntimeError("suggest_partitioner should not be called directly outside of a numba.cuda.jit(...) kernel.")

    @deprecated("Calling MM(...) directly is deprecated, please use MM.execute(...) method instead.")
    def __call__(self, *args):
        raise RuntimeError("__call__ should not be called directly outside of a numba.cuda.jit(...) kernel.")

    def execute(self, *args):
        raise RuntimeError("execute should not be called directly outside of a numba.cuda.jit(...) kernel.")

    @property
    @deprecated("files is deprecated and is no longer required and will be removed in future releases.")
    def files(self) -> list[str]:
        """The list of binary files for the lto functions."""
        return []

    @property
    @deprecated("codes is deprecated and is no longer required and will be removed in future releases.")
    def codes(self) -> list[Code]:
        """A list of :class:`Code` objects for all lto functions."""
        return []


class _MatmulTraits:
    def __init__(self, mm: Matmul):
        h = generate_MM(
            size=mm._size,
            function=mm._function,
            precision=mm._precision,
            data_type=mm._data_type,
            sm=mm._sm,
            transpose_mode=mm._transpose_mode,
            arrangement=mm._arrangement,
            alignment=mm._alignment,
            block_dim=mm._block_dim,
            static_block_dim=mm._static_block_dim,
            leading_dimension=mm._leading_dimension,
            execution=mm._execution,
        ).descriptor

        self.value_types = tuple(MATHDX_TYPES_TO_NP[vt] for vt in get_int_traits(h, mathdx.CublasdxTraitType.VALUE_TYPE, 3))
        self.leading_dimension = LeadingDimension(*get_int_traits(h, mathdx.CublasdxTraitType.LEADING_DIMENSION, 3))
        self.block_dim = Dim3(*get_int_traits(h, mathdx.CublasdxTraitType.BLOCK_DIM, 3))
        self.alignment = Alignment(*get_int_traits(h, mathdx.CublasdxTraitType.ALIGNMENT, 3))


#
# A compiled BLAS device function, with knobs and device function
#


def compile_blas_execute(
    blas: Matmul,
    code_type: Any,
    execute_api: str | None = None,
    tensor_types: Sequence[str] | None = None,
    global_memory_alignment: Sequence[int] | None = None,
) -> tuple[Code, str]:
    if global_memory_alignment is not None:
        if not isinstance(global_memory_alignment, Sequence) or len(global_memory_alignment) != 3:
            raise ValueError(
                "global_memory_alignment should be an instance of Alignment"
                "or a 3-tuple ; "
                "got global_memory_alignment = {global_memory_alignment}"
            )
        global_memory_alignment = Alignment(*global_memory_alignment)

    check_code_type(code_type, "cuBLASDx")
    validate_execute_api(execute_api)
    tensors_api = execute_api == "tensors"
    if tensors_api:
        validate_tensor_types(tensor_types)

    if global_memory_alignment is not None:
        # Perform validation only after initialization since we need to
        # know precision and data_type
        validate_alignment(
            global_memory_alignment,
            blas.precision,
            blas.data_type,
            gmem=True,
        )

    handle = generate_MM(
        size=blas.size,
        function=blas.function,
        precision=blas.precision,
        data_type=blas.data_type,
        sm=blas.sm,
        transpose_mode=blas.transpose_mode,
        arrangement=blas.arrangement,
        alignment=blas.alignment,
        block_dim=blas.block_dim,
        static_block_dim=blas._static_block_dim,
        # TODO: find better way to exclude ld operator for dynamic_leading_dimensions
        leading_dimension=blas._leading_dimension if execute_api == "static_leading_dimensions" else None,
        execution=blas._execution,
        execute_api=execute_api,
        tensor_types=tensor_types,
    )

    # Now compile the LTO device function
    h = handle.descriptor

    if tensors_api:
        resp = generate_tensors(h, tensor_types, global_memory_alignment)
        _, target_tensors = resp.gmem, resp.target
        code, symbol = generate_function_code(h, mathdx.CublasdxDeviceFunctionType.EXECUTE, target_tensors, code_type.cc)
    else:
        code = generate_code(h, code_type.cc)

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    ltos = [Code(code_type, isa_version, lto_fn)]

    if tensor_types:
        symbol = symbol
    else:
        symbol = get_str_trait(h, mathdx.CublasdxTraitType.SYMBOL_NAME)

    return ltos[0], symbol


def _blas_tensors_handle(MM: Matmul):
    handle = generate_MM(
        size=MM.size,
        function=MM.function,
        precision=MM.precision,
        data_type=MM.data_type,
        sm=MM.sm,
        transpose_mode=MM.transpose_mode,
        arrangement=MM.arrangement,
        alignment=MM.alignment,
        block_dim=MM.block_dim,
        static_block_dim=MM._static_block_dim,
        execution=MM._execution,
        execute_api="tensors",
    )
    return handle.descriptor


@docstring_decorator(CUBLASDX_DOCSTRING, skip_missing=False)
def matmul(*, compiler=None, code_type=None, execute_api=None, tensor_types=None, global_memory_alignment=None, **kwargs):
    """
    Create an :class:`Matmul` object that encapsulates a compiled and ready-to-use
    device function for matrix multiplication.

    .. deprecated:: 0.7.0

    Args:
        size: {size}

        precision: {precision}

        data_type: {data_type}

        compiler: {compiler}

            .. versionchanged:: 0.7.0
                compiler is no longer needed and does not take effect. Use
                :py:func:`nvmath.device.compile_blas_execute` to get device
                function code.

        code_type (CodeType): {code_type}

            .. versionchanged:: 0.7.0
                code_type should be used by
                :py:func:`nvmath.device.compile_blas_execute` and no longer
                needed for numba-cuda usage.

        block_size (int): {block_size}

        block_dim (Dim3): {block_dim}

        leading_dimension (LeadingDimension): {leading_dimension}

        transpose_mode (TransposeMode): {transpose_mode}

        arrangement (Arrangement): {arrangement}

        alignment (Alignment): {alignment}

        function (str): {function}

        execution (str): {execution}

        execute_api (str): {execute_api}

            .. versionchanged:: 0.7.0
                execute_api should be used by
                :py:func:`nvmath.device.compile_blas_execute` and no longer
                needed for numba-cuda usage.

        tensor_types (str): {tensor_types}

            .. versionchanged:: 0.7.0
                tensor_types should be used by
                :py:func:`nvmath.device.compile_blas_execute` and no longer
                needed for numba-cuda usage.

        global_memory_alignment (Alignment): {global_memory_alignment}

            .. versionchanged:: 0.7.0
                alignment should be set at :py:func:`nvmath.device.copy`
                global_memory_alignment should be used by
                :py:func:`nvmath.device.compile_blas_execute` for non numba-cuda
                usage. Alignment should be set

    .. seealso::
        The attributes of :class:`Matmul` provide a 1:1 mapping with the CUDA C++
        cuBLASDx APIs. For further details, please refer to `cuBLASDx documentation
        <https://docs.nvidia.com/cuda/cublasdx/>`_.

    Examples:

        >>> from numba import cuda
        >>> from nvmath.device import matmul
        >>> import numpy as np
        >>> m, n, k = 32, 16, 64
        >>> block_size = 256

        Use :func:`nvmath.device.matmul` to create the compiled matrix multiplication
        object:

        >>> MM = matmul(
        ...     size=(m, n, k),
        ...     precision=np.float32,
        ...     data_type="real",
        ...     transpose_mode=("non_transposed", "transposed"),
        ...     execution="Block",
        ...     block_size=block_size,
        ...     compiler="numba",
        ... )

        Pass ``link=MM.files`` to the :func:`numba.cuda.jit` decorator when defining your
        kernel to link with the compiled code.

        cuBLASDx works on shared memory arrays. It requires column-major (F order) arrays
        but :class:`cuda.shared.array` creates row-major (C order) arrays only. You can
        emulate a column-major array by flipping dimensions. With your shared memory arrays
        ready and filled with actual data, you can run the matrix multiplication by calling
        `MM`

        >>> a_dim, b_dim, c_dim = MM.a_dim, MM.b_dim, MM.c_dim
        >>> @cuda.jit(link=MM.files)
        ... def f():
        ...     a = cuda.shared.array(shape=(a_dim[1], a_dim[0]), dtype=np.float32)
        ...     b = cuda.shared.array(shape=(b_dim[1], b_dim[0]), dtype=np.float32)
        ...     c = cuda.shared.array(shape=(c_dim[1], c_dim[0]), dtype=np.float32)
        ...     # TODO: Populate the arrays with actual data.
        ...     alpha, beta = 1.0, 0.0
        ...     MM(alpha, a, b, beta, c)
        ...     cuda.syncthreads()
        ...     # TODO: Copy the result (c) from the shared memory
        >>> f[1, block_size]()

        Further examples can be found in the `nvmath/examples/device
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/device>`_ directory.
    """
    DeprecationWarning("matmul is deprecated and will be removed in future releases. Please use Matmul class directly.")
    if code_type is not None:
        DeprecationWarning("code_type is deprecated and will be removed in future releases. It is no longer needed.")
    if compiler is not None:
        DeprecationWarning("compiler is deprecated and will be removed in future releases. It is no longer needed.")
    if execute_api is not None:
        DeprecationWarning("execute_api is deprecated and will be removed in future releases. It is no longer needed.")
    if tensor_types is not None:
        DeprecationWarning("tensor_types is deprecated and will be removed in future releases. It is no longer needed.")
    if global_memory_alignment is not None:
        DeprecationWarning(
            "global_memory_alignment is deprecated and will be removed in "
            "future releases. It is no longer needed. Please set alignment "
            "at copy()"
        )
    return Matmul(**kwargs)


def _parse_layout(layout: str) -> tuple[bool, str, str]:
    """Parse layout string to extract tensor type and memory type.

    Returns: tuple of (suggest, memory, tensor)
        suggest: bool, True if the layout is a suggested layout
        memory: str, memory type ('s' for shared, 'g' for global, 'r' for register)
        tensor: str, tensor type ('a', 'b', 'c')
    """
    # extracting tensor type from layout
    pattern = re.compile(r"^(?:(suggest|get)_)?layout_([srg])mem_([abc])$")

    match = pattern.match(layout)

    assert match is not None

    suggest, memory, tensor = match.group(1, 2, 3)

    return suggest == "suggest", memory, tensor


class _BlasLayout(Layout):
    """BlasLayout for the OpaqueTensor"""

    _size: int
    _cosize: int

    # Runtime fields for the opaque tensor
    _uid: int
    _leading_dimension: int | None

    # Internal fields to recreate the numba layout type
    _MM: Matmul | None
    _layout: str

    # Cached fields to avoid recomputing
    _tensor_index: int

    def __init__(self, MM: Matmul, layout: str, leading_dimension: int | None = None):
        if not isinstance(MM, Matmul):
            raise ValueError("MM should be an instance of Matmul")

        suggested, memory, tensor = _parse_layout(layout)
        self._tensor_index = ["a", "b", "c"].index(tensor)

        tensor_type = f"{memory}mem_{tensor}"

        if suggested:
            tensor_type = "suggested_" + tensor_type

        self._dtype = MM._traits.value_types[self._tensor_index]
        itemsize = numpy.dtype(self._dtype).itemsize

        if memory == "g":
            self._uid = -1  # gmem tensors at this stage do not have a uid
            self._size = (MM.a_size, MM.b_size, MM.c_size)[self._tensor_index]
            storage_size = self._size * itemsize
        else:
            th = generate_tensor(_blas_tensors_handle(MM), tensor_type)
            self._uid, self._size, storage_size = get_tensor_traits(th.descriptor)
        assert storage_size % itemsize == 0
        self._cosize = storage_size // itemsize
        if mathdx.get_version_ex() < (0, 3, 0):
            self._size = (MM.a_size, MM.b_size, MM.c_size)[self._tensor_index]
            if memory == "r":
                self._size = self._cosize

        self._dynamic_ld = memory == "g"  # dynamic ld only global memory
        self._MM = MM
        self._layout = layout
        self._tensor_type = tensor_type
        self._leading_dimension = leading_dimension

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def cosize(self) -> int:
        return self._cosize


def compile_blas_copy(
    src_tensor: _BlasLayout,
    dst_tensor: _BlasLayout,
    code_type: CodeType,
    alignment: int | None = None,
):
    check_code_type(code_type, "cuBLASDx")
    assert src_tensor._MM == dst_tensor._MM
    assert src_tensor._MM is not None

    MM = src_tensor._MM

    handle = _blas_tensors_handle(MM)
    src_handler = generate_tensor(
        handle, src_tensor._tensor_type, gmem_alignment=alignment if "gmem" in src_tensor._tensor_type else None
    )
    dst_handler = generate_tensor(
        handle, dst_tensor._tensor_type, gmem_alignment=alignment if "gmem" in dst_tensor._tensor_type else None
    )

    return get_function_code(
        handle, mathdx.CublasdxDeviceFunctionType.COPY, [src_handler.descriptor, dst_handler.descriptor], code_type
    )


def compile_blas_clear(
    tensor: _BlasLayout,
    code_type: CodeType,
):
    check_code_type(code_type, "cuBLASDx")
    assert tensor._MM is not None

    MM = tensor._MM

    handle = _blas_tensors_handle(MM)
    tensor_handler = generate_tensor(handle, tensor._tensor_type)

    return get_function_code(handle, mathdx.CublasdxDeviceFunctionType.CLEAR, [tensor_handler.descriptor], code_type)


def compile_blas_axpby(
    x_tensor: _BlasLayout,
    y_tensor: _BlasLayout,
    code_type: CodeType,
):
    check_code_type(code_type, "cuBLASDx")
    assert x_tensor._MM == y_tensor._MM
    assert x_tensor._MM is not None

    MM = x_tensor._MM

    handle = _blas_tensors_handle(MM)
    x_handler = generate_tensor(handle, x_tensor._tensor_type)
    y_handler = generate_tensor(handle, y_tensor._tensor_type)

    return get_function_code(
        handle, mathdx.CublasdxDeviceFunctionType.AXPBY, [x_handler.descriptor, y_handler.descriptor], code_type
    )


def _compile_blas_partitioner_function(
    MM: Matmul,
    code_type: CodeType,
    function: mathdx.CublasdxDeviceFunctionType,
):
    check_code_type(code_type, "cuBLASDx")

    handle = _blas_tensors_handle(MM)
    tensor_handle = generate_tensor(handle, "suggested_rmem_c")

    return get_function_code(handle, function, [tensor_handle.descriptor], code_type)


def compile_blas_map_idx2crd_partitioner(
    MM: Matmul,
    code_type: CodeType,
):
    return _compile_blas_partitioner_function(
        MM,
        code_type,
        mathdx.CublasdxDeviceFunctionType.MAP_IDX2CRD_PARTITIONER,
    )


def compile_blas_is_thread_active(
    MM: Matmul,
    code_type: CodeType,
):
    return _compile_blas_partitioner_function(
        MM,
        code_type,
        mathdx.CublasdxDeviceFunctionType.IS_THREAD_ACTIVE,
    )


def compile_blas_is_predicated(
    MM: Matmul,
    code_type: CodeType,
):
    return _compile_blas_partitioner_function(
        MM,
        code_type,
        mathdx.CublasdxDeviceFunctionType.IS_PREDICATED,
    )


def compile_blas_is_index_in_bounds(
    MM: Matmul,
    code_type: CodeType,
):
    return _compile_blas_partitioner_function(
        MM,
        code_type,
        mathdx.CublasdxDeviceFunctionType.IS_INDEX_IN_BOUNDS,
    )
