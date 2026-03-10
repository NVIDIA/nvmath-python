# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["TransposeMode", "Matmul", "SharedStorageCalc", "Accumulator", "DevicePipeline", "TilePipeline"]

from abc import abstractmethod
from functools import cached_property
import itertools
from collections.abc import Sequence
import re
from typing import Any, overload
from warnings import warn
import weakref

from nvmath._utils import get_nvrtc_version
from nvmath.device.common_opaque_tensor import _LIBMATHDX_RUNTIME, OpaqueLayout

from .common import (
    Layout,
    OpaqueTensor,
    check_code_type,
    check_in,
    SHARED_DEVICE_DOCSTRINGS,
    pad_or_truncate,
    parse_sm,
)
from .common_backend import MATHDX_TYPES_TO_NP, NP_TYPES_TO_MATHDX_TYPES, DescriptorWrapper, get_isa_version, get_lto
from .common_cuda import (
    Code,
    CodeType,
    ComputeCapability,
    Dim3,
    get_current_device,
    get_current_device_cc,
    get_default_code_type,
)
from .cublasdx_backend import (
    Alignment,
    Arrangement,
    Precision,
    _compile_blas_device_pipeline_destroy_kernel,
    _compile_blas_device_pipeline_init_kernel,
    generate_MM,
    generate_code,
    generate_device_pipeline,
    generate_tensor_like,
    generate_tile_pipeline,
    generate_function_code,
    generate_function_with_pipelines_code,
    generate_tensor,
    generate_tensors,
    get_function_code,
    get_str_trait,
    get_int_traits,
    get_tensor_traits,
    validate,
    LeadingDimension,
    TransposeMode,
    validate_execute_api,
    validate_tensor_types,
    MAX_ALIGNMENT,  # noqa: F401
)
from ._deprecated import deprecated
from nvmath.internal.utils import docstring_decorator

from nvmath.bindings import mathdx
import numpy

try:
    from cuda.core import (
        Device,
        Buffer,
        LaunchConfig,
        launch,
    )
except ImportError:
    from cuda.core.experimental import (
        Device,
        Buffer,
        LaunchConfig,
        launch,
    )

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
        #
        "function": """\
A string specifying the name of the function. Currently supports ``'MM'`` (default) for matrix
multiplication.""".replace("\n", " "),
        #
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


class Partitioner:
    """
    Partitioner is an abstraction for partitioning a global memory tensor into a
    partitioned tensor.

    .. note:: Do not create directly, use
        :py:func:`nvmath.device.Matmul.suggest_partitioner`.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#partitioner-register-tensor-other-label
    """

    def __init__(self, *args):
        raise RuntimeError("Partitioner should not be called directly")

    def partition_like_C(self, gmem_c: OpaqueTensor) -> OpaqueTensor:
        """
        Partitions the given global memory tensor `gmem_c` into a partitioned tensor.
        The partitioned tensor is used for accessing the C matrix when working
        with register fragment.
        """
        raise RuntimeError("partition_like_C is a device function")

    def map_fragment_index(self, fragment_index: int) -> tuple[int, int]:
        """
        Maps the given fragment index to a global memory index.
        This is used to access the correct element in the partitioned tensor.
        """
        raise RuntimeError("map_fragment_index is a device function")

    def is_thread_active(self) -> bool:
        """
        Checks if the current thread takes part in GEMM.
        """
        raise RuntimeError("is_thread_active is a device function")

    def is_predicated(self) -> bool:
        """
        Checks if the current thread is predicated.
        This is used to determine if the thread should execute the kernel.
        """
        raise RuntimeError("is_predicated is a device function")

    def is_index_in_bounds(self, index: int) -> bool:
        """
        Checks if the given index is within the bounds of the partitioned tensor.
        This is used to prevent out-of-bounds access in the kernel.
        """
        raise RuntimeError("is_index_in_bounds is a device function")

    def get_alignment(self) -> int:
        raise NotImplementedError("not implemented")

    def make_empty_fragment(self) -> OpaqueTensor:
        """Creates an empty fragment tensor in register memory. Fragment layout
        is same as accumulator layout."""
        raise RuntimeError("make_empty_fragment is a device function")

    def partition_and_copy(self, src: OpaqueTensor, dst: OpaqueTensor):
        """Partition gmem tensor and copy to rmem fragment."""
        raise RuntimeError("partition_and_copy is a device function")

    def make_partition_and_copy(self, src: OpaqueTensor) -> OpaqueTensor:
        """Same as partition_and_copy but returns the partitioned rmem tensor."""
        raise RuntimeError("make_partition_and_copy is a device function")


class Accumulator(Partitioner):
    """Accumulator is an abstraction that provides the link between the
    global memory and register layouts. It offers operations like partitioning,
    copying data, and mapping register indices to matrix coordinates.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#accumulator-and-register-fragment-tensors
    """

    def get_results(self, out=None) -> OpaqueTensor:
        raise RuntimeError("get_results is a device function")

    def partition_and_store(self, tensor: OpaqueTensor):
        raise NotImplementedError("not implemented")

    def clear(self):
        raise NotImplementedError("not implemented")

    def size(self):
        raise NotImplementedError("not implemented")

    def axpby(self):
        raise NotImplementedError("not implemented")


class DevicePipeline:
    """DevicePipeline allows users to optimally configure kernel calls for pipelined
    matrix multiplication. It also provides an access point for getting a
    :class:`TilePipeline` object within a kernel.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html
    """

    def __init__(self, mm: "Matmul", pipeline_depth: int, a: numpy.ndarray, b: numpy.ndarray):
        self.mm = mm
        self.pipeline_depth = pipeline_depth

        # TODO: assert that arrays are on device memory
        self.a = a
        self.b = b
        self.pipeline_depth = pipeline_depth

        h = _blas_device_pipeline_handle(self)

        self._storage_bytes = int(mathdx.cublasdx_get_pipeline_trait_int64(h, mathdx.CublasdxPipelineTrait.STORAGE_BYTES))
        self._storage_alignment_bytes = int(
            mathdx.cublasdx_get_pipeline_trait_int64(h, mathdx.CublasdxPipelineTrait.STORAGE_ALIGNMENT_BYTES)
        )
        self._buffer_size = int(mathdx.cublasdx_get_pipeline_trait_int64(h, mathdx.CublasdxPipelineTrait.BUFFER_SIZE))
        self._buffer_alignment_bytes = int(
            mathdx.cublasdx_get_pipeline_trait_int64(h, mathdx.CublasdxPipelineTrait.BUFFER_ALIGNMENT_BYTES)
        )
        block_dim = numpy.zeros(3, dtype=numpy.int64)
        mathdx.cublasdx_get_pipeline_trait_int64s(
            h, mathdx.CublasdxPipelineTrait.BLOCK_DIM, len(block_dim), block_dim.ctypes.data
        )
        self._block_dim = Dim3(*block_dim.tolist())

        device = Device(get_current_device())
        device.set_current()

        # We do not need _storage_alignment_bytes here as device allocated
        # memory is maximum aligned.
        self._storage: Buffer = device.allocate(self._storage_bytes)

        self._init_kernel_launch(a, b, device)

        mm_descriptor = _blas_tensors_handle(self.mm)
        pipeline_descriptor = _blas_device_pipeline_handle(self)
        self._finalizer = weakref.finalize(
            self, DevicePipeline._destruct_kernel_execute, mm_descriptor, pipeline_descriptor, self._storage, device
        )

    @property
    def buffer_alignment(self) -> int:
        return self._buffer_alignment_bytes

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def storage_bytes(self) -> int:
        return self._storage_bytes

    @property
    def storage_alignment(self) -> int:
        return self._storage_alignment_bytes

    @property
    def block_dim(self) -> Dim3:
        return self._block_dim

    def get_tile(self, smem: numpy.ndarray, blockIdx_x: int, blockIdx_y: int) -> "TilePipeline":
        raise RuntimeError("get_tile is a device function")

    def reset_tile(self, tile_pipeline: "TilePipeline", idx: int | tuple[int, int], idy: int | tuple[int, int]):
        raise RuntimeError("reset_tile is a device function")

    @cached_property
    def a_strides(self):
        a = self.a
        for s in a.strides:
            assert s % numpy.dtype(a.dtype).itemsize == 0
        return tuple(s // numpy.dtype(a.dtype).itemsize for s in a.strides)

    @cached_property
    def b_strides(self):
        b = self.b
        for s in b.strides:
            assert s % numpy.dtype(b.dtype).itemsize == 0
        return tuple(s // numpy.dtype(b.dtype).itemsize for s in b.strides)

    def _debug_print(self):
        import cupy

        vhex = numpy.vectorize(hex)
        tma_cp = cupy.from_dlpack(self._storage).view(dtype=numpy.uint8)

        print(f"A_ptr: 0x{int(self.a.gpu_data.device_pointer):x}")
        print(f"B_ptr: 0x{int(self.b.gpu_data.device_pointer):x}")
        print("Device pipeline buffer:", vhex(cupy.asnumpy(tma_cp)))

    def _init_kernel_launch(self, a, b, device: Device):
        mm_descriptor = _blas_tensors_handle(self.mm)
        pipeline_descriptor = _blas_device_pipeline_handle(self)
        kernel = _compile_blas_device_pipeline_init_kernel(
            mm_descriptor, pipeline_descriptor, code_type=get_default_code_type()
        )

        # Create the launch configuration
        config = LaunchConfig(grid=(1,), block=(1,))
        ker_args = (int(self._storage.handle), int(a.gpu_data.device_pointer), int(b.gpu_data.device_pointer))
        # TODO: add support for cupy array
        # ker_args = (int(self._storage.handle), int(a.data.ptr), int(b.data.ptr))

        # Launch the kernel
        launch(device.default_stream, config, kernel, *ker_args)
        device.default_stream.sync()

    @staticmethod
    def _destruct_kernel_execute(mm_descriptor: int, pipeline_descriptor: int, storage: Buffer, device: Device):
        kernel = _compile_blas_device_pipeline_destroy_kernel(
            mm_descriptor, pipeline_descriptor, code_type=get_default_code_type()
        )

        # Create the launch configuration
        config = LaunchConfig(grid=(1,), block=(1,))
        ker_args = (int(storage.handle),)

        # Launch the kernel
        device.set_current()
        launch(device.default_stream, config, kernel, *ker_args)
        device.default_stream.sync()


class TilePipeline:
    """TilePipeline allows users to execute an pipelined matrix multiplication
    with partial tile results accumulated into an acuumulator.

    Refer to the cuBLASDx documentation for more details on how to use this class:
    https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html
    """

    def __init__(self, device_pipeline: DevicePipeline):
        self.device_pipeline = device_pipeline

        MM_descriptor = _blas_tensors_handle(device_pipeline.mm)
        device_pipeline_descriptor = _blas_device_pipeline_handle(device_pipeline)
        h = generate_tile_pipeline(
            MM_descriptor,
            device_pipeline_descriptor,
        )
        self._storage_bytes = int(
            mathdx.cublasdx_get_pipeline_trait_int64(h.descriptor, mathdx.CublasdxPipelineTrait.STORAGE_BYTES)
        )
        self._storage_alignment_bytes = int(
            mathdx.cublasdx_get_pipeline_trait_int64(h.descriptor, mathdx.CublasdxPipelineTrait.STORAGE_ALIGNMENT_BYTES)
        )

    def _init(self, device_pipeline: DevicePipeline, smem, idx: int, idy: int):
        raise RuntimeError("_init is a device function")

    def _del(self):
        raise RuntimeError("_del is a device function")

    @property
    def storage_bytes(self) -> int:
        return self._storage_bytes

    @property
    def storage_alignment(self) -> int:
        return self._storage_alignment_bytes

    def execute(self, accumulator):
        raise RuntimeError("execute is a device function")


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
        with_pipeline: bool = False,
        enable_input_streaming: bool = False,
    ):
        sm = parse_sm(sm)
        if sm.integer not in {900, 1000, 1030, 1100}:
            # remove arch modifier
            sm = ComputeCapability(sm.major, sm.minor)

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
            with_pipeline=with_pipeline,
            enable_input_streaming=enable_input_streaming,
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
        self._with_pipeline = with_pipeline
        self._enable_input_streaming = enable_input_streaming

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

    @property
    def with_pipeline(self) -> bool:
        return self._with_pipeline

    @property
    def enable_input_streaming(self) -> bool:
        return self._enable_input_streaming

    #
    # Extensions
    #

    def valid(self, *knobs):
        return itertools.product(*[self._valid(knob) for knob in knobs])

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

    @cached_property
    def a_dim(self) -> tuple[int, int]:
        (m, _, k) = self.size

        dim = (m, k)
        if self._transpose_mode is not None and self._transpose_mode.a in {"transposed", "conj_transposed"}:
            dim = dim[::-1]

        return dim

    @cached_property
    def b_dim(self) -> tuple[int, int]:
        (_, n, k) = self.size

        dim = (k, n)
        if self._transpose_mode is not None and self._transpose_mode.b in {"transposed", "conj_transposed"}:
            dim = dim[::-1]

        return dim

    @cached_property
    def c_dim(self) -> tuple[int, int]:
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
    def _abc_sizes(self) -> tuple[int, int, int]:
        return self._calculate_abc_sizes(self.leading_dimension)

    @property
    def a_size(self) -> int:
        return self._abc_sizes[0]

    @property
    def b_size(self) -> int:
        return self._abc_sizes[1]

    @property
    def c_size(self) -> int:
        return self._abc_sizes[2]

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
        return _BlasMatmulLayout(self, "get_layout_gmem_a", leading_dimension)

    def get_layout_gmem_b(self, leading_dimension: int | None = None) -> Layout:
        return _BlasMatmulLayout(self, "get_layout_gmem_b", leading_dimension)

    def get_layout_gmem_c(self, leading_dimension: int | None = None) -> Layout:
        return _BlasMatmulLayout(self, "get_layout_gmem_c", leading_dimension)

    def get_layout_smem_a(self) -> Layout:
        return _BlasMatmulLayout(self, "get_layout_smem_a")

    def get_layout_smem_b(self) -> Layout:
        return _BlasMatmulLayout(self, "get_layout_smem_b")

    def get_layout_smem_c(self) -> Layout:
        return _BlasMatmulLayout(self, "get_layout_smem_c")

    def suggest_layout_smem_a(self) -> Layout:
        return _BlasMatmulLayout(self, "suggest_layout_smem_a")

    def suggest_layout_smem_b(self) -> Layout:
        return _BlasMatmulLayout(self, "suggest_layout_smem_b")

    def suggest_layout_smem_c(self) -> Layout:
        return _BlasMatmulLayout(self, "suggest_layout_smem_c")

    def suggest_layout_rmem_c(self) -> Layout:
        return _BlasMatmulLayout(self, "suggest_layout_rmem_c")

    def get_layout_rmem_c(self) -> Layout:
        return _BlasMatmulLayout(self, "get_layout_rmem_c")

    def _suggest_accumulator_c(self) -> Layout:
        return _BlasMatmulLayout(self, "suggest_accumulator_c")

    def _get_accumulator_c(self) -> Layout:
        return _BlasMatmulLayout(self, "get_accumulator_c")

    def get_accumulator(self) -> Accumulator:
        raise RuntimeError("get_accumulator is a device function")

    def suggest_accumulator(self) -> Accumulator:
        raise RuntimeError("suggest_accumulator is a device function")

    def suggest_device_pipeline(self, pipeline_depth: int, a: numpy.ndarray, b: numpy.ndarray) -> DevicePipeline:
        cc = get_current_device_cc()
        ctk_version = get_nvrtc_version()
        if ctk_version < (13, 0, 0):
            raise RuntimeError("DevicePipeline requires CUDA Toolkit 13.0 or higher.")
        if cc.major >= 10 and ctk_version < (13, 1, 0):
            raise RuntimeError("DevicePipeline on compute capability 10.0 and higher requires CUDA Toolkit 13.1 or higher.")

        return DevicePipeline(self, pipeline_depth, a, b)

    def execute(self, *args):
        raise RuntimeError("execute should not be called directly outside of a numba.cuda.jit(...) kernel.")


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
            with_pipeline=mm._with_pipeline,
            enable_input_streaming=mm._enable_input_streaming,
        ).descriptor

        self.value_types = tuple(MATHDX_TYPES_TO_NP[vt] for vt in get_int_traits(h, mathdx.CublasdxTraitType.VALUE_TYPE, 3))
        self.leading_dimension = LeadingDimension(*get_int_traits(h, mathdx.CublasdxTraitType.LEADING_DIMENSION, 3))
        self.block_dim = Dim3(*get_int_traits(h, mathdx.CublasdxTraitType.BLOCK_DIM, 3))
        self.alignment = Alignment(*get_int_traits(h, mathdx.CublasdxTraitType.ALIGNMENT, 3))


#
# A compiled BLAS device function, with knobs and device function
#


def compile_blas_execute(
    blas: Matmul, code_type: Any, execute_api: str = "static_leading_dimensions", tensor_types: Sequence[str] | None = None
) -> tuple[Code, str]:
    check_code_type(code_type, "cuBLASDx")
    validate_execute_api(execute_api)
    tensors_api = execute_api == "tensors"
    if tensors_api:
        validate_tensor_types(tensor_types)

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
        with_pipeline=blas.with_pipeline,
        enable_input_streaming=blas.enable_input_streaming,
    )

    # Now compile the LTO device function
    h = handle.descriptor

    if tensors_api:
        resp = generate_tensors(h, tensor_types)
        target_tensors = resp.target
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


def _blas_handle(
    MM: Matmul,
    execute_api: str = "static_leading_dimensions",
):
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
        execute_api=execute_api,
        with_pipeline=MM._with_pipeline,
        enable_input_streaming=MM._enable_input_streaming,
    )
    return handle.descriptor


def _blas_tensors_handle(MM: Matmul):
    return _blas_handle(MM, execute_api="tensors")


def _blas_device_pipeline_handle(pipeline: DevicePipeline):
    MM_descriptor = _blas_tensors_handle(pipeline.mm)
    return generate_device_pipeline(
        MM_descriptor,
        pipeline.pipeline_depth,
        NP_TYPES_TO_MATHDX_TYPES[pipeline.a.dtype.type],
        NP_TYPES_TO_MATHDX_TYPES[pipeline.b.dtype.type],
        pipeline.a.shape,
        pipeline.b.shape,
        pipeline.a_strides,
        pipeline.b_strides,
    ).descriptor


def _blas_tile_pipeline_handle(pipeline: TilePipeline):
    MM_descriptor = _blas_tensors_handle(pipeline.device_pipeline.mm)
    device_pipeline_descriptor = _blas_device_pipeline_handle(pipeline.device_pipeline)
    h = generate_tile_pipeline(MM_descriptor, device_pipeline_descriptor)
    return h.descriptor


def _parse_layout(layout: str) -> tuple[bool, bool, str, str]:
    """Parse layout string to extract tensor type and memory type.

    Returns: tuple of (suggest, accumulator, memory, tensor)
        suggest: bool, True if the layout is a suggested layout
        accumulator: bool, True if the layout is an accumulator
        memory: str, memory type ('s' for shared, 'g' for global,
            'r' for register, '' for accumulator)
        tensor: str, tensor type ('a', 'b', 'c')
    """
    # extracting tensor type from layout
    pattern = re.compile(r"^(?:(suggest|get)_)?(layout|accumulator)_(?:([srg])mem_)?([abc])$")

    match = pattern.match(layout)

    assert match is not None

    suggest, layout_type, memory, tensor = match.group(1, 2, 3, 4)

    return suggest == "suggest", layout_type == "accumulator", memory, tensor


class _BaseBlasLayout(OpaqueLayout):
    _uid: int
    _logical_size: int
    _storage_bytes: int
    _alignment_bytes: int

    _MM: Matmul

    def __init__(self, MM: Matmul, shape: tuple[int, ...], strides: tuple[int, ...], dtype: numpy.number):
        super().__init__(shape, strides, dtype)
        self._MM = MM

    @abstractmethod
    def _get_descriptor(self) -> DescriptorWrapper:
        pass

    def _init_traits(self):
        d = self._get_descriptor()
        self._uid, self._logical_size, self._storage_bytes, self._alignment_bytes = get_tensor_traits(d.descriptor)

    @property
    def MM(self) -> Matmul:
        return self._MM

    @property
    def uid(self) -> int:
        return self._uid

    @property
    def size(self) -> int:
        return self._logical_size

    @property
    def storage_bytes(self) -> int:
        return self._storage_bytes

    @cached_property
    def cosize(self) -> int:
        assert self._storage_bytes % numpy.dtype(self._dtype).itemsize == 0
        return self._storage_bytes // numpy.dtype(self._dtype).itemsize

    @property
    def alignment(self) -> int:
        return self._alignment_bytes


class _BlasMatmulLayout(_BaseBlasLayout):
    """BlasLayout for the OpaqueTensor"""

    _layout: str
    _tensor_type: str
    _tensor_index: int
    _accumulator: bool
    _memory_space: str
    _tensor: str

    _default_ld: int | None

    def _get_descriptor(self) -> DescriptorWrapper:
        return generate_tensor(_blas_tensors_handle(self._MM), self._tensor_type)

    def __init__(self, MM: Matmul, layout: str, leading_dimension: int | None = None):
        if not isinstance(MM, Matmul):
            raise ValueError("MM should be an instance of Matmul")

        self._default_ld = leading_dimension
        self._suggested, self._accumulator, memory, tensor = _parse_layout(layout)
        self._tensor_index = ["a", "b", "c"].index(tensor)

        if self._accumulator:
            tensor_type = f"accumulator_{tensor}"
        else:
            tensor_type = f"{memory}mem_{tensor}"

        if self._suggested:
            tensor_type = "suggested_" + tensor_type

        self._tensor_type = tensor_type

        # Inheritance support
        if hasattr(self, "_dtype") and self._dtype is not None:
            dtype = self._dtype
        else:
            dtype = MM._traits.value_types[self._tensor_index]
        itemsize = numpy.dtype(dtype).itemsize

        self._MM = MM

        if memory == "g":
            self._uid = -1
            self._logical_size = MM._abc_sizes[self._tensor_index]
            self._storage_bytes = self._logical_size * itemsize
            # TODO: should we take it as an argument?
            self._alignment_bytes = itemsize

            shape: tuple[int, ...] = tuple((MM.a_dim, MM.b_dim, MM.c_dim)[self._tensor_index])
            strides: tuple[int, ...] = (_LIBMATHDX_RUNTIME, 1)
        else:
            self._init_traits()
            shape, strides = (self.size,), (1,)
        assert self._storage_bytes % itemsize == 0
        self._cosize = self._storage_bytes // itemsize

        self._memory_space = memory
        self._layout = layout
        self._tensor_type = tensor_type

        super().__init__(MM, dtype=dtype, shape=shape, strides=strides)

    @property
    def suggested(self) -> bool:
        return self._suggested

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def accumulator(self) -> bool:
        return self._accumulator

    @property
    def tensor_type(self) -> str:
        return self._tensor_type

    @property
    def tensor_index(self) -> int:
        """Tensor index is 0 for A, 1 for B and 2 for C."""
        return self._tensor_index

    @property
    def memory_space(self) -> str:
        """Memory space is 's' for shared, 'g' for global, 'r' for register,
        '' for accumulator."""
        return self._memory_space

    @property
    def default_ld(self) -> int | None:
        """Default leading dimension if provided during layout creation.
        Only available for gmem layouts created on the host side.
        Strides will be set to (_LIBMATHDX_RUNTIME,1).
        """
        return self._default_ld


class _BlasMatmulLikeLayout(_BlasMatmulLayout):
    """BlasLayout for the OpaqueTensor created with make_fragment_like"""

    _dtype_orig: numpy.number

    def __init__(self, MM: Matmul, layout: str, dtype: numpy.number, leading_dimension: int | None = None):
        self._dtype = dtype
        super().__init__(MM, layout, leading_dimension)
        self._dtype_orig = MM._traits.value_types[self._tensor_index]

    def _get_descriptor(self) -> DescriptorWrapper:
        mm_handle = _blas_tensors_handle(self._MM)
        src_tensor = generate_tensor(mm_handle, self._tensor_type)
        dst_tensor = generate_tensor_like(mm_handle, src_tensor.descriptor, self._dtype)

        return dst_tensor

    @property
    def dtype_orig(self) -> numpy.number:
        """Original dtype of the tensor in the Matmul object."""
        return self._dtype_orig


def compile_blas_copy(
    src_tensor: _BlasMatmulLayout,
    dst_tensor: _BlasMatmulLayout,
    code_type: CodeType,
    alignment: int | None = None,
):
    check_code_type(code_type, "cuBLASDx")
    assert src_tensor._MM is not None
    assert dst_tensor._MM is not None

    src_MM_descriptor = _blas_tensors_handle(src_tensor._MM)
    src_tensor_descriptor = src_tensor._get_descriptor()
    dst_tensor_descriptor = dst_tensor._get_descriptor()

    return get_function_code(
        src_MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.COPY,
        [src_tensor_descriptor.descriptor, dst_tensor_descriptor.descriptor],
        code_type,
    )


def compile_blas_clear(
    tensor: _BlasMatmulLayout,
    code_type: CodeType,
):
    check_code_type(code_type, "cuBLASDx")
    assert tensor._MM is not None

    MM = tensor._MM

    handle = _blas_tensors_handle(MM)
    tensor_handler = tensor._get_descriptor()

    return get_function_code(handle, mathdx.CublasdxDeviceFunctionType.CLEAR, [tensor_handler.descriptor], code_type)


def compile_blas_axpby(
    x_tensor: _BlasMatmulLayout,
    y_tensor: _BlasMatmulLayout,
    code_type: CodeType,
):
    check_code_type(code_type, "cuBLASDx")
    assert x_tensor._MM == y_tensor._MM
    assert x_tensor._MM is not None

    MM = x_tensor._MM

    handle = _blas_tensors_handle(MM)
    x_handler = x_tensor._get_descriptor()
    y_handler = y_tensor._get_descriptor()

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


def compile_blas_device_pipeline_destroy(
    pipeline: DevicePipeline,
    code_type: CodeType,
) -> tuple[Code, str]:
    assert isinstance(pipeline, DevicePipeline)

    MM_descriptor = _blas_tensors_handle(pipeline.mm)
    pipeline_descriptor = _blas_device_pipeline_handle(pipeline)

    code, symbol = generate_function_with_pipelines_code(
        MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.DESTROY,
        (),
        (pipeline_descriptor,),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


def compile_blas_tile_pipeline_init(
    pipeline: TilePipeline,
    code_type: CodeType,
):
    assert isinstance(pipeline, TilePipeline)
    assert isinstance(code_type, CodeType)

    MM_descriptor = _blas_tensors_handle(pipeline.device_pipeline.mm)
    pipeline_descriptor = _blas_tile_pipeline_handle(pipeline)

    code, symbol = generate_function_with_pipelines_code(
        MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.CREATE,
        (),
        (pipeline_descriptor,),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


def compile_blas_device_pipeline_reset_tile(
    device_pipeline: DevicePipeline,
    tile_pipeline: TilePipeline,
    code_type: CodeType,
):
    assert isinstance(device_pipeline, DevicePipeline)
    assert isinstance(tile_pipeline, TilePipeline)
    assert isinstance(code_type, CodeType)

    MM_descriptor = _blas_tensors_handle(tile_pipeline.device_pipeline.mm)
    tile_pipeline_descriptor = _blas_tile_pipeline_handle(tile_pipeline)
    device_pipeline_descriptor = _blas_device_pipeline_handle(device_pipeline)

    code, symbol = generate_function_with_pipelines_code(
        MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.RESET,
        (),
        (device_pipeline_descriptor, tile_pipeline_descriptor),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


def compile_blas_tile_pipeline_destroy(
    pipeline: TilePipeline,
    code_type: CodeType,
):
    assert isinstance(pipeline, TilePipeline)
    assert isinstance(code_type, CodeType)

    MM_descriptor = _blas_tensors_handle(pipeline.device_pipeline.mm)
    pipeline_descriptor = _blas_tile_pipeline_handle(pipeline)

    code, symbol = generate_function_with_pipelines_code(
        MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.DESTROY,
        (),
        (pipeline_descriptor,),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


def compile_blas_accumulator_init(
    layout: Layout,
    code_type: CodeType,
):
    assert isinstance(layout, _BlasMatmulLayout)
    assert layout.accumulator
    assert isinstance(code_type, CodeType)

    MM_descriptor = _blas_tensors_handle(layout._MM)
    opaque_tensor_descriptor = layout._get_descriptor()

    code, symbol = generate_function_code(
        MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.CREATE,
        [opaque_tensor_descriptor.descriptor],
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol


def compile_blas_tile_pipeline_execute(
    pipeline: TilePipeline,
    accumulator: Layout,
    code_type: CodeType,
):
    assert isinstance(accumulator, _BlasMatmulLayout)
    assert accumulator.accumulator
    assert isinstance(code_type, CodeType)

    MM_descriptor = _blas_tensors_handle(pipeline.device_pipeline.mm)
    tile_pipeline_descriptor = _blas_tile_pipeline_handle(pipeline)
    tensor_descriptor = accumulator._get_descriptor().descriptor

    code, symbol = generate_function_with_pipelines_code(
        MM_descriptor,
        mathdx.CublasdxDeviceFunctionType.EXECUTE,
        (tensor_descriptor,),
        (tile_pipeline_descriptor,),
        code_type.cc,
    )

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    lto = Code(code_type, isa_version, lto_fn)

    return lto, symbol
