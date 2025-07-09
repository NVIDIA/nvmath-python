# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["matmul", "TransposeMode", "BlasOptions"]

from functools import cached_property
import itertools
from collections.abc import Sequence
import math
import re
from typing import overload
from warnings import warn

from .common import (
    Layout,
    make_binary_tempfile,
    check_in,
    SHARED_DEVICE_DOCSTRINGS,
    pad_or_truncate,
)
from .common_backend import MATHDX_TYPES_TO_NP, get_isa_version, get_lto
from .common_cuda import MAX_SUPPORTED_CC, get_default_code_type, ComputeCapability, Code, CodeType, Dim3
from .common_numba import NP_TYPES_TO_NUMBA_FE_TYPES
from .cublasdx_backend import (
    Alignment,
    Arrangement,
    Precision,
    generate_MM,
    generate_code,
    generate_code_tensors,
    generate_copy_wait_lto,
    generate_tensors,
    get_str_trait,
    get_int_traits,
    get_tensor_int_traits,
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
class BlasOptions:
    """
    A class that encapsulates a partial BLAS device function. A partial device function can
    be queried for available or optimal values for some knobs (such as `leading_dimension`
    or `block_dim`). It does not contain a compiled, ready-to-use, device function until
    finalized using :meth:`create`.

    Args:
        size: {size}

        precision: {precision}

        data_type: {data_type}

        code_type (CodeType): {code_type}

        block_size (int): {block_size}

        block_dim (Dim3): {block_dim}

        leading_dimension (LeadingDimension): {leading_dimension}

        transpose_mode (TransposeMode): {transpose_mode}

        arrangement (Arrangement): {arrangement}

        alignment (Alignment): {alignment}

        function (str): {function}

        execution (str): {execution}

        execute_api:
            .. versionchanged:: 0.5.0
                execute_api is not part of the Blas type. Pass this argument to
                :py:func:`nvmath.device.matmul` instead.

        tensor_types:
            .. versionchanged:: 0.5.0
                tensor_types is not part of the Blas type. Pass this argument to
                :py:func:`nvmath.device.matmul` instead.

    See Also:
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
        code_type=None,
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
        if not isinstance(code_type, Sequence) or len(code_type) != 2:
            raise ValueError(f"code_type should be an instance of CodeType or a 2-tuple ; got code_type = {code_type}")
        code_type = CodeType(code_type[0], ComputeCapability(*code_type[1]))
        if code_type.cc.major < 7:
            raise RuntimeError(
                f"Minimal compute capability 7.0 is required by cuBLASDx, got {code_type.cc.major}.{code_type.cc.minor}"
            )
        if (code_type.cc.major, code_type.cc.minor) > MAX_SUPPORTED_CC:
            raise RuntimeError(
                "The maximum compute capability currently supported by device "
                f"APIs is {MAX_SUPPORTED_CC}, "
                f"got {code_type.cc.major}.{code_type.cc.minor}"
            )

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
            code_type=code_type,
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
        self._code_type = code_type
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
        return self._alignment

    @property
    def code_type(self):
        return self._code_type

    @property
    def function(self) -> str:
        return self._function

    @property
    def block_size(self) -> int:
        return self._block_dim[0] * self._block_dim[1] * self._block_dim[2]

    @property
    def block_dim(self) -> Dim3:
        return self._block_dim

    @property
    def static_block_dim(self) -> bool:
        return self._static_block_dim

    @property
    def leading_dimension(self) -> LeadingDimension:
        return self._leading_dimension

    #
    # Extensions
    #

    def valid(self, *knobs):
        return itertools.product(*[self._valid(knob) for knob in knobs])

    def definition(self):
        dd = {
            "size": self.size,
            "precision": self.precision,
            "data_type": self.data_type,
            "transpose_mode": self.transpose_mode,
            "arrangement": self.arrangement,
            "alignment": self.alignment,
            "code_type": self.code_type,
            "block_dim": self.block_dim,
            "static_block_dim": self.static_block_dim,
            "function": self.function,
            "execution": self.execution,
            "leading_dimension": self.leading_dimension,
        }
        return dd

    def create(self, **kwargs):
        dd = self.definition()
        dd.update(**kwargs)
        return matmul(**dd)

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
        if self.code_type is None:
            raise ValueError("leading_dimension='suggested' require code_type to be set.")
        if self.execution != "Block":
            raise ValueError("leading_dimension='suggested' require execution to be 'Block'.")
        # Generate special PTX for suggested_leading_dimension_of
        descriptor = generate_MM(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
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
        if self.code_type is None:
            raise ValueError("block_dim='suggested' require code_type to be set.")
        if self.execution != "Block":
            raise ValueError("block_dim='suggested' require execution to be 'Block'.")
        # Generate full PTX
        descriptor = generate_MM(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
            transpose_mode=self._transpose_mode,
            arrangement=self._arrangement,
            alignment=self._alignment,
            block_dim=None,
            static_block_dim=self._static_block_dim,
            leading_dimension=None,
            execution=self.execution,
        )

        return Dim3(*get_int_traits(descriptor.descriptor, mathdx.CublasdxTraitType.SUGGESTED_BLOCK_DIM, 3))


#
# A complete set of knobs, ie sufficient to generate a device functions and query all traits
# Not exposed to end users
#
class BlasOptionsComplete(BlasOptions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.code_type is None:
            raise NotImplementedError(f"code_type should be set, but got code_type = {self.code_type}")
        if self.execution != "Block":
            raise NotImplementedError(f"Only execution=Block is implemented ; got execution = {self.execution}")

        (m, n, k) = self.size

        h = generate_MM(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
            transpose_mode=self._transpose_mode,
            arrangement=self.arrangement,
            alignment=self._alignment,
            block_dim=self.block_dim,
            static_block_dim=self._static_block_dim,
            leading_dimension=self._leading_dimension,
            execution=self.execution,
        ).descriptor

        self._value_types = tuple(MATHDX_TYPES_TO_NP[vt] for vt in get_int_traits(h, mathdx.CublasdxTraitType.VALUE_TYPE, 3))
        self._leading_dimension = LeadingDimension(*get_int_traits(h, mathdx.CublasdxTraitType.LEADING_DIMENSION, 3))
        self._block_dim = Dim3(*get_int_traits(h, mathdx.CublasdxTraitType.BLOCK_DIM, 3))
        self._alignment = Alignment(*get_int_traits(h, mathdx.CublasdxTraitType.ALIGNMENT, 3))

        self._a_dim = (m, k)
        self._b_dim = (k, n)
        self._c_dim = (m, n)

        if self._transpose_mode is not None:
            if self._transpose_mode.a in {"transposed", "conj_transposed"}:
                self._a_dim = self._a_dim[::-1]
            if self._transpose_mode.b in {"transposed", "conj_transposed"}:
                self._b_dim = self._b_dim[::-1]

        [self._a_size, self._b_size, self._c_size] = self._calculate_abc_sizes(self._leading_dimension)

        self._max_threads_per_block = self._block_dim.x * self._block_dim.y * self._block_dim.z

    def _calculate_abc_sizes(self, ld: LeadingDimension) -> tuple[int, int, int]:
        assert isinstance(ld, LeadingDimension)
        if self._transpose_mode:
            non_ld = (self._a_dim[1], self._b_dim[1], self._c_dim[1])
        elif self._arrangement:
            non_ld = (
                self._a_dim[1 if self._arrangement.a == "col_major" else 0],
                self._b_dim[1 if self._arrangement.b == "col_major" else 0],
                self._c_dim[1 if self._arrangement.c == "col_major" else 0],
            )

        return tuple(x * y for x, y in zip(ld, non_ld, strict=True))

    @property
    def a_value_type(self):
        return self._value_types[0]

    @property
    def b_value_type(self):
        return self._value_types[1]

    @property
    def c_value_type(self):
        return self._value_types[2]

    @property
    @deprecated("value_type trait is deprecated. Please use {a|b|c}_value_type instead")
    def value_type(self):
        if not all(vt == self.a_value_type for vt in self._value_types):
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

    @property
    def a_dim(self):
        return self._a_dim

    @property
    def b_dim(self):
        return self._b_dim

    @property
    def c_dim(self):
        return self._c_dim

    @property
    def leading_dimension(self):
        return self._leading_dimension

    @property
    def a_size(self):
        return self._a_size

    @property
    def b_size(self):
        return self._b_size

    @property
    def c_size(self):
        return self._c_size

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
        return self._max_threads_per_block

    def _get_shared_storage_size(self, *args, ab=False) -> int | None:  # type: ignore
        # Complex will be over-aligned (eg: f32x2 complex is aligned on 8B) with
        # this logic (which is what we want - for performance and vectorization)
        item_sizes = tuple(numpy.dtype(vt).itemsize for vt in self._value_types)

        alignment = self.alignment
        sizes = None

        if len(args) == 0:
            sizes = (self._a_size, self._b_size, self._c_size)
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


#
# A compiled BLAS device function, with knobs and device function
#
class BlasCompiled(BlasOptionsComplete):
    def __init__(self, **kwargs):
        execute_api = kwargs.pop("execute_api", "static_leading_dimensions")
        tensor_types = kwargs.pop("tensor_types", None)
        global_memory_alignment = kwargs.pop("global_memory_alignment", None)

        # Build set of knobs
        super().__init__(**kwargs)

        if global_memory_alignment is not None:
            if not isinstance(global_memory_alignment, Sequence) or len(global_memory_alignment) != 3:
                raise ValueError(
                    "global_memory_alignment should be an instance of Alignment"
                    "or a 3-tuple ; "
                    "got global_memory_alignment = {global_memory_alignment}"
                )
            global_memory_alignment = Alignment(*global_memory_alignment)

            validate_alignment(
                global_memory_alignment,
                self.precision,
                self.data_type,
                gmem=True,
            )

        validate_execute_api(execute_api)
        tensors_api = execute_api == "tensors"
        if tensors_api:
            validate_tensor_types(tensor_types)

        self._execute_api = execute_api
        self._tensor_types = tensor_types
        self._global_memory_alignment = global_memory_alignment

        handle = generate_MM(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
            transpose_mode=self._transpose_mode,
            arrangement=self.arrangement,
            alignment=self._alignment,
            block_dim=self.block_dim,
            static_block_dim=self._static_block_dim,
            # TODO: find better way to exclude ld operator for dynamic_leading_dimensions
            leading_dimension=self._leading_dimension if self._execute_api == "static_leading_dimensions" else None,
            execution=self._execution,
            execute_api=self._execute_api,
            tensor_types=self._tensor_types,
        )

        # TODO: remove once MM.files is deprecated
        self._handle = handle

        # Now compile the LTO device function
        h = handle.descriptor

        if tensors_api:
            self._declare_tensors(h)

            code, self._tensor_api_symbols = generate_code_tensors(
                h, self.code_type.cc, self._gmem_tensors, self._target_tensors, rmem_c="rmem" in self._tensor_types[2]
            )
        else:
            code = generate_code(h, self.code_type.cc)

        # Compile
        lto_fn = get_lto(code.descriptor)
        isa_version = get_isa_version(code.descriptor)

        self._ltos = [Code(self.code_type, isa_version, lto_fn)]
        self._symbol = get_str_trait(h, mathdx.CublasdxTraitType.SYMBOL_NAME)

        if self._tensor_types:
            _, copy_wait_lto = generate_copy_wait_lto(self.code_type.cc)
            self._ltos += [Code(self.code_type, isa_version, copy_wait_lto)]

    def _declare_tensors(self, h):
        # Complex will be over-aligned (eg: f32x2 complex is aligned on 8B) with
        # this logic (which is what we want - for performance and vectorization)
        item_sizes = tuple(numpy.dtype(vt).itemsize for vt in self._value_types)

        self._gmem_tensors, self._target_tensors = generate_tensors(h, self._tensor_types, self._global_memory_alignment)
        self._target_tensor_sizes = get_tensor_int_traits(self._target_tensors, mathdx.CublasdxTensorTrait.STORAGE_BYTES)
        for ts, _is in zip(self._target_tensor_sizes, item_sizes, strict=True):
            assert ts % _is == 0
        self._target_tensor_sizes = tuple(ts // item_sizes[i] for i, ts in enumerate(self._target_tensor_sizes))
        self._gmem_tensor_uids = get_tensor_int_traits(self._gmem_tensors, mathdx.CublasdxTensorTrait.UID)
        self._target_tensor_uids = get_tensor_int_traits(self._target_tensors, mathdx.CublasdxTensorTrait.UID)

    def definition(self):
        dd = super().definition()
        dd.update(execute_api=self.execute_api)
        if self.execute_api == "tensors":
            dd.update(tensor_types=self.tensor_types)
        return dd

    @cached_property
    def _tempfiles(self):
        """
        Create temporary files for the LTO functions.
        """
        return [make_binary_tempfile(lto.data, ".ltoir") for lto in self._ltos]

    @property
    def files(self):
        """The list of binary files for the lto functions."""
        return [v.name for v in self._tempfiles]

    @property
    def codes(self):
        """A list of :class:`Code` objects for all lto functions."""
        return self._ltos

    @property
    def symbol(self):
        """The name of the device function."""
        return self._symbol

    @property
    def execute_api(self) -> str:
        """
        The API used to execute the function. It defines the signature of the
        LTO function.
        """
        return self._execute_api

    @property
    def tensor_types(self) -> tuple[str, str, str]:
        """
        The tensor types used in the function. Defines types of the tensors for
        the tensors API.
        """
        if self.execute_api != "tensors":
            raise RuntimeError("tensor_types is only available when execute_api is 'tensors'")
        return self._tensor_types


#
# A compiled BLAS device function, with knobs and device function, to be used with Numba
#
class BlasNumba(BlasCompiled):
    """
    A class that encapsulates a compiled BLAS device function compatible with Numba.


    """

    def __init__(self, **kwargs):
        if "code_type" not in kwargs:
            kwargs["code_type"] = get_default_code_type()

        # Build LTO device functions
        super().__init__(**kwargs)

        self._numba_value_types = tuple(NP_TYPES_TO_NUMBA_FE_TYPES[vt] for vt in self._value_types)

    @property
    def a_value_type(self):
        return self._numba_value_types[0]

    @property
    def b_value_type(self):
        return self._numba_value_types[1]

    @property
    def c_value_type(self):
        return self._numba_value_types[2]

    @deprecated("Calling MM(...) directly is deprecated, please use MM.execute(...) method instead.")
    def __call__(self, *args):
        raise RuntimeError("__call__ should not be called directly outside of a numba.cuda.jit(...) kernel.")

    def execute(self, *args):
        raise RuntimeError("execute should not be called directly outside of a numba.cuda.jit(...) kernel.")

    @cached_property
    def _copy_symbols_map(self):
        if self.execute_api != "tensors":
            return {}

        return {
            (self._gmem_tensor_uids[0], self._target_tensor_uids[0]): self._tensor_api_symbols.copy_a,
            (self._gmem_tensor_uids[1], self._target_tensor_uids[1]): self._tensor_api_symbols.copy_b,
            (self._gmem_tensor_uids[2], self._target_tensor_uids[2]): self._tensor_api_symbols.copy_c,
            (self._target_tensor_uids[2], self._gmem_tensor_uids[2]): self._tensor_api_symbols.copy_c_back,
        }


@docstring_decorator(CUBLASDX_DOCSTRING, skip_missing=False)
def matmul(*, compiler=None, **kwargs):
    """
    Create an :class:`BlasOptions` object that encapsulates a compiled and ready-to-use
    device function for matrix multiplication.

    Args:
        size: {size}

        precision: {precision}

        data_type: {data_type}

        compiler: {compiler}

        code_type (CodeType): {code_type}

        block_size (int): {block_size}

        block_dim (Dim3): {block_dim}

        leading_dimension (LeadingDimension): {leading_dimension}

        transpose_mode (TransposeMode): {transpose_mode}

        arrangement (Arrangement): {arrangement}

        alignment (Alignment): {alignment}

        function (str): {function}

        execution (str): {execution}

        execute_api (str): {execute_api}

        tensor_types (str): {tensor_types}

        global_memory_alignment (Alignment): {global_memory_alignment}

    See Also:
        The attributes of :class:`BlasOptions` provide a 1:1 mapping with the CUDA C++
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
    check_in("compiler", compiler, [None, "numba"])
    if compiler is None:
        return BlasCompiled(**kwargs)
    elif compiler == "numba":
        return BlasNumba(**kwargs)


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
    _MM: BlasNumba | None
    _layout: str

    # Cached fields to avoid recomputing
    _is_register: bool
    _tensor_index: int

    def __init__(self, MM: BlasOptionsComplete, layout: str, leading_dimension: int | None = None):
        if not isinstance(MM, BlasCompiled):
            raise ValueError("MM should be an instance of BlasCompiled, support for BlasOptionsComplete is in progress")
        if MM.execute_api != "tensors":
            raise ValueError(f"{layout} is only available for execute_api='tensors'")

        assert MM._tensor_types is not None

        suggested, memory, tensor = _parse_layout(layout)
        self._tensor_index = ["a", "b", "c"].index(tensor)

        self._size = math.prod((MM.a_dim, MM.b_dim, MM.c_dim)[self._tensor_index])

        if memory == "g":
            self._uid = MM._gmem_tensor_uids[self._tensor_index]
            self._cosize = self._size
        else:
            tensor_type = f"{memory}mem_{tensor}"

            if suggested:
                tensor_type = "suggested_" + tensor_type

            if tensor_type not in set(MM._tensor_types):
                raise ValueError(f"Invalid layout {layout} for tensor {tensor_type}. Available layouts are {MM._tensor_types}")
            self._uid = MM._target_tensor_uids[self._tensor_index]
            self._cosize = MM._target_tensor_sizes[self._tensor_index]

        if memory == "r":
            # for register memory, we are using fragment so it does not have
            # any gaps and contain only small chank, so dimension production
            # does not apply
            self._size = self._cosize

        self._is_register = memory == "r"
        self._dynamic_ld = memory == "g"  # dynamic ld only global memory
        self._MM = MM if isinstance(MM, BlasNumba) else None
        self._layout = layout
        self._leading_dimension = leading_dimension

    @property
    def size(self) -> int:
        return self._size

    @property
    def cosize(self) -> int:
        return self._cosize
