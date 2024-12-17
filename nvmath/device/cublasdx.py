# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["matmul", "TransposeMode", "BlasOptions"]

from functools import cached_property
import itertools

from .common import (
    make_binary_tempfile,
    find_dim3,
    find_unsigned,
    check_in,
    find_dim2,
    find_mangled_name,
    SHARED_DEVICE_DOCSTRINGS,
)
from .common_cpp import enum_to_np
from .common_cuda import get_default_code_type, ComputeCapability, Code, CodeType, Symbol, Dim3
from .common_numba import NP_TYPES_TO_NUMBA_FE_TYPES
from .cublasdx_backend import generate_block, generate_block_ld, validate, LeadingDimension, TransposeMode
from .cublasdx_numba import codegen
from .nvrtc import compile
from .._internal.utils import docstring_decorator

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
suggested value. Can't not be used when `block_size` is explicitly specified.""".replace("\n", " "),
        #
        "leading_dimension": """\
The leading dimensions for the input matrices, optional. If not provided, will be set to match the matrix row/column
dimension. Alternatively, if provided as ``'suggested'``, will be set to a suggested value for optimal performance.
""".replace("\n", " "),
        #
        "transpose_mode": """\
The transpose mode for all input matrices. If not provided, no transposition by default.""".replace("\n", " "),
        #
        "function": """\
A string specifying the name of the function. Currently supports ``'MM'`` (default) for matrix
multiplication.""".replace("\n", " "),
    }
)

#
# A set of knobs, potentially in-complete (ie not sufficient to generate a device functions)
#


@docstring_decorator(CUBLASDX_DOCSTRING, skip_missing=False)
class BlasOptions:
    """
    A class that encapsulates a partial BLAS device function. A partial device function can
    be queried for available or optimal values for the some knobs (such as
    `leading_dimension` or `block_dim`). It does not contain a compiled, ready-to-use,
    device function until finalized using :meth:`create`.

    Args:
        size: {size}

        precision: {precision}

        data_type: {data_type}

        code_type (CodeType): {code_type}

        block_size (int): {block_size}

        block_dim (Dim3): {block_dim}

        leading_dimension (LeadingDimension): {leading_dimension}

        transpose_mode (TransposeMode): {transpose_mode}

        function (str): {function}

        execution (str): {execution}

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
        transpose_mode=TransposeMode("non_transposed", "non_transposed"),
        function="MM",
        execution="Block",
    ):
        if len(code_type) != 2:
            raise ValueError(f"code_type should be an instance of CodeType or a 2-tuple ; got code_type = {code_type}")
        code_type = CodeType(code_type[0], ComputeCapability(*code_type[1]))
        if code_type.cc.major < 7:
            raise RuntimeError(
                "Minimal compute capability 7.0 is required by cuBLASDx, got " f"{code_type.cc.major}.{code_type.cc.minor}"
            )

        if len(transpose_mode) != 2:
            raise ValueError(
                "transpose_mode should be an instance of TransposeMode or a 2-tuple ; " f"got transpose_mode = {transpose_mode}"
            )
        transpose_mode = TransposeMode(*transpose_mode)

        if isinstance(leading_dimension, tuple):
            if len(leading_dimension) != 3:
                raise ValueError(
                    "leading_dimension should be a 3-tuple, an instance of LeadingDimension, 'suggested' or None ; "
                    f"got leading_dimension = {leading_dimension}"
                )
            else:
                leading_dimension = LeadingDimension(*leading_dimension)

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
        if block_dim is not None and isinstance(block_dim, tuple):
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
            code_type=code_type,
            leading_dimension=leading_dimension,
            block_dim=block_dim,
            function=function,
            execution=execution,
        )

        #
        # Traits set by input
        #

        self._size = size
        self._precision = precision
        self._data_type = data_type
        self._transpose_mode = transpose_mode
        self._code_type = code_type
        self._block_dim = block_dim
        self._function = function
        self._execution = execution
        self._leading_dimension = leading_dimension

        #
        # Update suggested traits
        #

        if leading_dimension == "suggested":
            self._leading_dimension = self._suggested_leading_dimension

        if block_dim == "suggested":
            self._block_dim = self._suggested_block_dim

    @property
    def precision(self):
        return self._precision

    @property
    def data_type(self):
        return self._data_type

    @property
    def size(self):
        return self._size

    @property
    def execution(self):
        return self._execution

    @property
    def transpose_mode(self):
        return self._transpose_mode

    @property
    def code_type(self):
        return self._code_type

    @property
    def function(self):
        return self._function

    @property
    def block_size(self):
        return self._block_dim[0] * self._block_dim[1] * self._block_dim[2]

    @property
    def block_dim(self):
        return self._block_dim

    @property
    def leading_dimension(self):
        return self._leading_dimension

    #
    # Extensions
    #

    def valid(self, *knobs):
        vals = []
        for knob in knobs:
            vals.append(self._valid(knob))
        return itertools.product(*vals)

    def create(self, **kwargs):
        dd = {
            "size": self.size,
            "precision": self.precision,
            "data_type": self.data_type,
            "transpose_mode": self.transpose_mode,
            "code_type": self.code_type,
            "block_dim": self.block_dim,
            "function": self.function,
            "execution": self.execution,
            "leading_dimension": self.leading_dimension,
        }
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
        cpp = generate_block_ld(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
            transpose_mode=self._transpose_mode,
            block_dim=None,
            leading_dimension=None,
            execution=self.execution,
        )
        _, ptx = compile(cpp=cpp["cpp"], cc=self.code_type.cc, rdc=True, code="ptx")
        ld = find_dim3("suggested_leading_dimension", ptx)
        return LeadingDimension(ld[0], ld[1], ld[2])

    @cached_property
    def _suggested_block_dim(self):
        if self.code_type is None:
            raise ValueError("block_dim='suggested' require code_type to be set.")
        if self.execution != "Block":
            raise ValueError("block_dim='suggested' require execution to be 'Block'.")
        # Generate full PTX
        cpp = generate_block(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
            transpose_mode=self._transpose_mode,
            block_dim=None,
            leading_dimension=None,
            execution=self.execution,
        )
        _, ptx = compile(cpp=cpp["cpp"], cc=self.code_type.cc, rdc=True, code="ptx")
        return Dim3(*find_dim3("suggested_block_dim", ptx))


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

        self._cpp = generate_block(
            size=self.size,
            function=self.function,
            precision=self.precision,
            data_type=self.data_type,
            code_type=self.code_type,
            transpose_mode=self._transpose_mode,
            block_dim=self.block_dim,
            leading_dimension=self._leading_dimension,
            execution=self.execution,
        )
        _, self._ptx = compile(cpp=self._cpp["cpp"], cc=self.code_type.cc, rdc=True, code="ptx")

        # Look into PTX for traits
        self._value_type = enum_to_np(find_unsigned("value_type", self._ptx))
        self._input_type = enum_to_np(find_unsigned("input_type", self._ptx))
        self._output_type = enum_to_np(find_unsigned("output_type", self._ptx))
        self._a_dim = find_dim2("a_dim", self._ptx)
        self._b_dim = find_dim2("b_dim", self._ptx)
        self._c_dim = find_dim2("c_dim", self._ptx)
        self._leading_dimension = LeadingDimension(
            find_unsigned("lda", self._ptx), find_unsigned("ldb", self._ptx), find_unsigned("ldc", self._ptx)
        )
        self._a_size = find_unsigned("a_size", self._ptx)
        self._b_size = find_unsigned("b_size", self._ptx)
        self._c_size = find_unsigned("c_size", self._ptx)
        self._shared_memory_size = find_unsigned("shared_memory_size", self._ptx)
        self._block_dim = Dim3(*find_dim3("block_dim", self._ptx))
        self._max_threads_per_block = find_unsigned("max_threads_per_block", self._ptx)

    @property
    def value_type(self):
        return self._value_type

    @property
    def input_type(self):
        return self._input_type

    @property
    def output_type(self):
        return self._output_type

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
    def shared_memory_size(self):
        return self._shared_memory_size

    @property
    def max_threads_per_block(self):
        return self._max_threads_per_block


#
# A compiled BLAS device function, with knobs and device function
#
class BlasCompiled(BlasOptionsComplete):
    def __init__(self, **kwargs):
        # Build set of knobs
        super().__init__(**kwargs)

        # Now compile the LTO device function
        version, lto_fn = compile(cpp=self._cpp["cpp"], cc=self.code_type.cc, rdc=True, code="lto")
        self._ltos = [Code(self.code_type, version, lto_fn)]
        apis = ["smem_basic", "smem_ldabc"]
        self._symbols = {api: find_mangled_name(self._cpp["names"][api], self._ptx) for api in apis}
        self._tempfiles = [make_binary_tempfile(lto_fn, ".ltoir")]

    @property
    def files(self):
        """The list of binary files for the lto functions."""
        return list(v.name for v in self._tempfiles)

    @property
    def codes(self):
        """A list of :class:`Code` objects for all lto functions."""
        return self._ltos

    @property
    def symbols(self):
        """A list of :class:`Symbol` objects for all lto symbols."""
        return [Symbol(k, v) for k, v in self._symbols.items()]


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

        # Add Numba logic
        self._codegened = codegen({"value_type": self.value_type, "symbols": self._symbols, "execution": self.execution}, self)

    @property
    def value_type(self):
        return NP_TYPES_TO_NUMBA_FE_TYPES[super(BlasCompiled, self).value_type]

    @property
    def input_type(self):
        return NP_TYPES_TO_NUMBA_FE_TYPES[super(BlasCompiled, self).input_type]

    @property
    def output_type(self):
        return NP_TYPES_TO_NUMBA_FE_TYPES[super(BlasCompiled, self).output_type]

    def __call__(self, *args):
        raise Exception("__call__ should not be called directly outside of a numba.cuda.jit(...) kernel.")


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

        function (str): {function}

        execution (str): {execution}

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
