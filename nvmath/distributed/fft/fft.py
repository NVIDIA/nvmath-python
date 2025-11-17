# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["allocate_operand", "FFT", "fft", "ifft", "rfft", "irfft"]

from typing import Literal, cast
from types import ModuleType
from collections.abc import Sequence
from dataclasses import dataclass
import functools
import logging
import math
import numpy as np

from ._configuration import FFTOptions, FFTDirection

import nvmath.distributed
from nvmath.distributed.distribution import Distribution, Slab, Box
from nvmath.bindings import cufftMp as cufft  # type: ignore
from nvmath.bindings import nvshmem  # type: ignore
from nvmath import memory
from nvmath.distributed._internal.nvshmem import NvshmemMemoryManager, NvshmemNDBufferAllocator

import nvmath.internal.ndbuffer.ndbuffer as ndbuffer
from nvmath.internal import formatters
from nvmath.distributed._internal import tensor_wrapper
from nvmath.distributed._internal.tensor_ifc import DistributedTensor
from nvmath.distributed._internal.tensor_ifc_numpy import CudaDistributedTensor
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE, NAME_TO_ITEM_SIZE
from nvmath.internal import utils
from nvmath._internal.layout import is_overlapping_layout
from nvmath.internal.package_wrapper import AnyStream, StreamHolder


@dataclass
class TensorLayout:
    """An internal data class for capturing the tensor layout."""

    shape: Sequence[int]
    strides: Sequence[int]


@dataclass
class _ProblemSpec:
    """This is used in a custom MPI reduction to check that the FFT problem
    specification is consistent across processes, and to infer global information
    (e.g shape)."""

    @dataclass
    class Options:
        """
        This is used for _ProblemSpec instead of FFTOptions
        because it's going to be serialized as part of the custom reduction of the
        _ProblemSpec, and we want to control which fields are included (for example
        we don't need the logger).
        """

        def __init__(self, options: FFTOptions):
            self.fft_type = options.fft_type
            self.reshape = options.reshape
            self.blocking = options.blocking

        fft_type: Literal["C2C", "C2R", "R2C"] | None
        reshape: bool
        blocking: Literal[True, "auto"]

    shape: list[int]  # operand shape
    is_C: bool  # Is C memory layout
    operand_dtype: str  # str because TensorHolder.dtype returns str
    package: Literal["numpy", "cupy", "torch"]  # operand package
    memory_space: Literal["cuda", "cpu"]  # operand memory space
    distribution: Slab | Sequence[Box]  # distribution of FFT input/output operands
    options: Options  # FFT options

    # Global number of elements in the operand (calculated as part of the reduction).
    # NOTE: Only computed and used with box distribution.
    global_size: int = 0
    # Max number of elements of the input operand across processes.
    # NOTE: Only computed and used with box distribution.
    input_max_elements: int = 0
    # Max number of elements of the output operand across processes.
    # NOTE: Only computed and used with box distribution.
    output_max_elements: int = 0
    # is_leaf=True means that this is the _ProblemSpec of a process before reducing
    # with that of another process.
    is_leaf: bool = True


SHARED_FFT_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_FFT_DOCUMENTATION.update(
    {
        "operand": SHARED_FFT_DOCUMENTATION["operand"],
        #
        "operand_admonitions": """
            .. important::
                GPU operands must be on the symmetric heap (for example, allocated with
                ``nvmath.distributed.allocate_symmetric_memory()``).
""",
        #
        "options": """\
Specify options for the FFT as a :class:`FFTOptions` object. Alternatively, a `dict` containing the parameters for the
``FFTOptions`` constructor can also be provided. If not specified, the value will be set to the default-constructed
``FFTOptions`` object.""".replace("\n", " "),
        #
        "distribution": """\
Specifies the distribution of input and output operands across processes, which can be: (i) according to
a Slab distribution (see :class:`nvmath.distributed.distribution.Slab`), or (ii) a custom box distribution
(see :class:`nvmath.distributed.distribution.Box`). With Slab distribution, this indicates the distribution
of the input operand (the output operand will use the complementary Slab distribution).
With box distribution, this indicates the input and output boxes.""".replace("\n", " "),
        #
        "direction": """\
Specify whether forward or inverse FFT is performed (:class:`FFTDirection` object, or as a string from ['forward',
'inverse'], "or as an int from [-1, 1] denoting forward and inverse directions respectively).""".replace("\n", " "),
        #
        "sync_symmetric_memory": """\
Indicates whether to issue a symmetric memory synchronization operation on the execute stream
before the FFT. Note that before the FFT starts executing, it is required that the input operand
be ready on all processes. A symmetric memory synchronization ensures completion and visibility
by all processes of previously issued local stores to symmetric memory. Advanced users who choose
to manage the synchronization on their own using the appropriate NVSHMEM API, or who know that
GPUs are already synchronized on the source operand, can set this to False.""".replace("\n", " "),
        #
        "function_signature": """\
operand,
distribution: Distribution | Sequence[Box],
sync_symmetric_memory: bool = True,
options: FFTOptions | None = None,
stream: AnyStream | None = None
""".replace("\n", " "),
    }
)


def _calculate_slab_shape_strides(global_extents, partition_dim, rank, nranks, global_extents_padded=None):
    """Calculate the local slab shape and strides for the given rank, given the global shape
    and partition dimension. If `global_extents_padded` is provided, calculate the strides
    based on this shape.
    """
    n = nranks
    S = global_extents[partition_dim]
    partition_dim_local_size = (S // n + 1) if rank < S % n else S // n
    slab_shape = list(global_extents)
    slab_shape[partition_dim] = partition_dim_local_size
    if global_extents_padded is not None:
        _, strides = _calculate_slab_shape_strides(global_extents_padded, partition_dim, rank, nranks)
    else:
        strides = calculate_strides(slab_shape, reversed(range(len(global_extents))))
    return tuple(slab_shape), strides


def _calculate_local_box(global_shape, partition_dim, rank, nranks):
    """Given a global shape of data that is partitioned across ranks along the
    `partition_dim` dimension according to cuFFTMp's slab distribution, return
    the local box of this rank (as lower and upper coordinates in the global shape).
    """
    lower = [0] * len(global_shape)
    for i in range(rank):
        shape, _ = _calculate_slab_shape_strides(global_shape, partition_dim, i, nranks)
        lower[partition_dim] += shape[partition_dim]
    shape, _ = _calculate_slab_shape_strides(global_shape, partition_dim, rank, nranks)
    upper = list(shape)
    upper[partition_dim] += lower[partition_dim]
    return lower, upper


def _get_fft_concrete_type(dtype, fft_abstract_type):
    FFTType = cufft.Type
    if fft_abstract_type == "C2C":
        if dtype == "complex64":
            return FFTType["C2C"]
        elif dtype == "complex128":
            return FFTType["Z2Z"]
        else:
            raise ValueError(f"Incompatible dtype '{dtype}' for complex-to-complex transform.")
    elif fft_abstract_type == "R2C":
        if dtype == "float32":
            return FFTType["R2C"]
        elif dtype == "float64":
            return FFTType["D2Z"]
        else:
            raise ValueError(f"Incompatible dtype '{dtype}' for real-to-complex transform.")
    elif fft_abstract_type == "C2R":
        if dtype == "complex64":
            return FFTType["C2R"]
        elif dtype == "complex128":
            return FFTType["Z2D"]
        else:
            raise ValueError(f"Incompatible dtype '{dtype}' for complex-to-real transform.")
    else:
        raise ValueError(f"Unsupported FFT Type: '{fft_abstract_type}'")


def _get_validate_direction(direction, fft_abstract_type):
    if isinstance(direction, str) and (d := direction.upper()) in ["FORWARD", "INVERSE"]:
        direction = FFTDirection[d]
    else:
        direction = FFTDirection(direction)

    if fft_abstract_type == "C2R":
        if direction != FFTDirection.INVERSE:
            raise ValueError(
                f"The specified direction {direction.name} is not compatible with the FFT type '{fft_abstract_type}'."
            )
    elif fft_abstract_type == "R2C":  # noqa: SIM102
        if direction != FFTDirection.FORWARD:
            raise ValueError(
                f"The specified direction {direction.name} is not compatible with the FFT type '{fft_abstract_type}'."
            )
    return direction


def complex_to_real_equivalent(name):
    assert "complex" in name, f"Internal Error ({name=})"
    m = name.split("complex")
    assert len(m) in (1, 2)
    size = int(m[-1]) // 2
    if len(m) == 1:
        return f"float{size}"
    else:
        return f"{m[0]}float{size}"


def real_to_complex_equivalent(name):
    assert "float" in name, f"Internal Error ({name=})"
    m = name.split("float")
    assert len(m) in (1, 2)
    size = int(m[-1])
    if len(m) == 1:
        return f"complex{size * 2}"
    else:
        return f"{m[0]}complex{size * 2}"


def _get_default_fft_abstract_type(dtype, fft_type) -> Literal["R2C", "C2R", "C2C"]:
    if fft_type is not None:
        return fft_type

    f, c = "float", "complex"
    if dtype[: len(f)] == f:
        fft_type = "R2C"
    elif dtype[: len(c)] == c:
        fft_type = "C2C"
    else:
        raise ValueError(f"Unsupported dtype '{dtype}' for FFT.")
    return fft_type


def _get_fft_result_and_compute_types(dtype, fft_abstract_type):
    """
    Return result and compute data type given the input data type and the FFT type.
    """
    if fft_abstract_type == "C2C":
        return dtype, dtype
    elif fft_abstract_type == "C2R":
        return complex_to_real_equivalent(dtype), dtype
    elif fft_abstract_type == "R2C":
        return real_to_complex_equivalent(dtype), dtype
    else:
        raise ValueError(f"Unsupported FFT Type: '{fft_abstract_type}'")


def _get_fft_default_direction(fft_abstract_type) -> FFTDirection:
    """
    Return the default FFT direction (as object of type configuration.FFTDirection) based on
    the FFT type.
    """
    if fft_abstract_type in ["C2C", "R2C"]:
        return FFTDirection.FORWARD
    elif fft_abstract_type == "C2R":
        return FFTDirection.INVERSE
    else:
        raise ValueError(f"Unsupported FFT Type: '{fft_abstract_type}'")


def check_inplace_overlapping_layout(operand: utils.TensorHolder):
    if is_overlapping_layout(operand.shape, operand.strides):
        raise ValueError(
            f"In-place transform is not supported because the tensor with shape "
            f"{operand.shape} and strides {operand.strides} overlaps in memory."
        )


def calculate_strides(shape, axis_order):
    """
    Calculate the strides for the provided shape and axis order.
    """
    strides = [None] * len(shape)

    stride = 1
    for axis in axis_order:
        strides[axis] = stride
        stride *= shape[axis]

    return tuple(strides)


def _allocate_with_padded_buffer(
    shape: Sequence[int],
    capacity: int,
    input_dtype,
    memory_space: Literal["cpu", "cuda"],
    package: ModuleType,
):
    """Allocate distributed tensor with memory for `capacity` elements of `input_dtype`
    dtype on each rank, and return a view of shape `shape` of the first prod(shape)
    elements in the 1D array.

    Args:
        shape: Shape of the view returned. Note that this view will have a base tensor with
            possibly larger capacity than required for this shape. The shape can vary across
            ranks.

        capacity: capacity of the allocated buffer in number of elements of the specified
            dtype. **NOTE: the capacity must be the same on every rank and this is not
            checked. Non-uniform capacity across ranks can lead to undefined behavior**.

        input_dtype: dtype of the tensor elements.
    """
    size = math.prod(shape)
    assert size <= capacity, f"Internal error: requested shape {shape} exceeds specified capacity {capacity}"
    if memory_space == "cuda":
        if package is ndbuffer:
            ctx = nvmath.distributed.get_context()
            assert ctx is not None
            device_id = ctx.device_id
            itemsize = NAME_TO_ITEM_SIZE[input_dtype]
            allocator = NvshmemNDBufferAllocator(device_id, ctx, make_symmetric=False, skip_symmetric_check=True)
            with utils.device_ctx(device_id):
                buf = ndbuffer.empty((capacity,), device_id, input_dtype, itemsize, device_memory_pool=allocator)

            strides = calculate_strides(shape, reversed(range(len(shape))))
            view = ndbuffer.wrap_external(buf, buf.data_ptr, input_dtype, shape, strides, device_id, itemsize)
            return CudaDistributedTensor(view)
        else:
            a = nvmath.distributed.allocate_symmetric_memory((capacity,), package, dtype=input_dtype, skip_symmetric_check=True)
    else:
        a = package.empty((capacity,), dtype=input_dtype)
    return tensor_wrapper.wrap_operand(a[:size]).reshape(shape, copy=False)


def _calculate_capacity(
    problem_spec: _ProblemSpec,
    global_shape: Sequence[int],
    fft_type: Literal["C2C", "C2R", "R2C"],
    nranks: int,
) -> int:
    """Calculate the max number of elements that the input buffer on every rank must be able
    to hold in order to perform the specified distributed FFT. Since the memory allocation
    is on the symmetric heap, we need to use the same (max) capacity on every rank. Also
    recall that the transform is inplace, so the buffer must be able to hold both the input
    and output given the FFT type and input/output operand distribution."""

    distribution = problem_spec.distribution
    if fft_type == "C2C":
        if isinstance(distribution, Slab):
            # capacity is max of X-slab and Y-slab size on rank 0.
            s1, _ = _calculate_slab_shape_strides(global_shape, 0, 0, nranks)  # X-slab
            s2, _ = _calculate_slab_shape_strides(global_shape, 1, 0, nranks)  # Y-slab
            return max(math.prod(s1), math.prod(s2))
        else:
            # capacity is the max number of elements across ranks for both input and output.
            return max(problem_spec.input_max_elements, problem_spec.output_max_elements)
    elif fft_type == "R2C":
        if isinstance(distribution, Slab):
            # capacity is max of X-slab and Y-slab size on rank 0 for complex shape.
            global_output_shape = list(global_shape)
            global_output_shape[-1] = global_output_shape[-1] // 2 + 1  # this is the complex shape

            s1, _ = _calculate_slab_shape_strides(global_output_shape, 0, 0, nranks)  # X-slab
            s2, _ = _calculate_slab_shape_strides(global_output_shape, 1, 0, nranks)  # Y-slab

            # Capacity is returned in terms of input (real) elements.
            return max(math.prod(s1) * 2, math.prod(s2) * 2)
        else:
            # Capacity is returned in terms of input (real) elements.
            return max(problem_spec.input_max_elements, 2 * problem_spec.output_max_elements)
    elif fft_type == "C2R":
        if isinstance(distribution, Slab):
            # capacity is max of X-slab and Y-slab size on rank 0.
            s1, _ = _calculate_slab_shape_strides(global_shape, 0, 0, nranks)  # X-slab
            s2, _ = _calculate_slab_shape_strides(global_shape, 1, 0, nranks)  # Y-slab
            return max(math.prod(s1), math.prod(s2))
        else:
            # Capacity is returned in terms of input (complex) elements.
            return max(
                problem_spec.input_max_elements, problem_spec.output_max_elements // 2 + problem_spec.output_max_elements % 2
            )
    raise AssertionError(f"Internal error: Unknown FFT type {fft_type}")


def _allocate_for_fft(
    global_input_shape: Sequence[int],
    shape: Sequence[int],
    distribution: Slab | Sequence[Box],
    input_dtype,
    memory_space: Literal["cpu", "cuda"],
    package: ModuleType,
    fft_type: Literal["C2C", "C2R", "R2C"],
    capacity: int,
    rank: int,
    nranks: int,
):
    """Allocate distributed tensor for the given distributed FFT operation. The same
    capacity must be provided on every rank, and must be large enough for the specified
    transform."""
    if fft_type == "R2C" and isinstance(distribution, Slab):
        partition_dim = distribution.partition_dim

        # For input, the strides depend on the padding.
        global_output_shape = list(global_input_shape)
        global_output_shape[-1] = global_output_shape[-1] // 2 + 1  # this is the complex shape
        global_input_shape_padded = list(global_output_shape)
        global_input_shape_padded[-1] *= 2

        padded_shape, _ = _calculate_slab_shape_strides(global_input_shape_padded, partition_dim, rank, nranks)
        a = _allocate_with_padded_buffer(padded_shape, capacity, input_dtype, memory_space, package)

        # Return a view strided on the last axis.
        if a.name == "cuda":
            view = ndbuffer.wrap_external(a.tensor, a.data_ptr, a.dtype, shape, a.strides, a.device_id, a.itemsize)
            return CudaDistributedTensor(view)
        else:
            return tensor_wrapper.wrap_operand(a.tensor[..., : shape[-1]])
    else:
        # These might not be the most efficient input strides for the R2C FFT (the whole
        # input is packed at the beginning of the buffer with no strides), but to support
        # other strides we probably need the user to pass them.
        return _allocate_with_padded_buffer(shape, capacity, input_dtype, memory_space, package)


_SUPPORTED_PACKAGES = ("numpy", "cupy", "torch")


@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=True)
def allocate_operand(
    shape: Sequence[int],
    package: ModuleType,
    *,
    input_dtype=None,
    distribution: Distribution | Sequence[Box],
    memory_space: Literal["cpu", "cuda"] | None = None,
    fft_type: Literal["C2C", "C2R", "R2C"] | None = None,
    logger: logging.Logger | None = None,
):
    """Return uninitialized operand of the given shape and type, to use as input for
    distributed FFT. The resulting tensor is backed by a buffer large enough for the
    specified FFT (the buffer can hold both the input and output -distributed FFT is
    inplace-, accounting for both the input and output distribution).
    For CUDA memory space, the tensor is allocated on the symmetric heap, on the
    device on which nvmath.distributed was initialized.
    **This is a collective operation and must be called by all processes**.

    Args:
        shape: Shape of the tensor to allocate.

        package: Python package determining the tensor type (e.g. numpy, cupy, torch).

        input_dtype: Tensor dtype in a form recognized by the package. If None, will use
            the package's default dtype.

        distribution: {distribution}

        memory_space: The memory space (``'cpu'`` or ``'cuda'``) on which to allocate
            the tensor. If not provided, this is inferred for packages that support
            a single memory space like numpy and cupy. For other packages it must be
            provided.

        fft_type: The type of FFT to perform. Available options include ``'C2C'``,
            ``'C2R'``, and ``'R2C'``. The default is ``'C2C'`` for complex input and
            ``'R2C'`` for real input.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.
    """

    package_name = package.__name__
    if package_name not in _SUPPORTED_PACKAGES:
        raise ValueError(f"The package must be one of {_SUPPORTED_PACKAGES}. Got {package}.")

    if memory_space is None:
        if package_name == "cupy":
            memory_space = "cuda"
        elif package_name == "numpy":
            memory_space = "cpu"
        else:
            raise ValueError(f"You must provide memory_space for package {package}")

    if memory_space not in ("cuda", "cpu"):
        raise ValueError(f"memory_space must be 'cuda' or 'cpu'. Got {memory_space}")

    if (package_name == "cupy" and memory_space == "cpu") or (package_name == "numpy" and memory_space == "cuda"):
        raise ValueError(f"'{memory_space}' memory space is not compatible with package {package_name}")

    distributed_ctx = nvmath.distributed.get_context()
    if distributed_ctx is None:
        raise RuntimeError(
            "nvmath.distributed has not been initialized. Refer to "
            "https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html#initializing-the-distributed-runtime"
            " for more information."
        )
    if not distributed_ctx.nvshmem_available:
        raise RuntimeError("nvmath.distributed wasn't initialized with NVSHMEM backend")
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if package_name in ("numpy", "cupy"):
        if input_dtype is None:
            # This mimics numpy and cupy
            input_dtype = np.float64

        input_dtype_name = np.dtype(input_dtype).name
    elif package_name == "torch":
        if input_dtype is None:
            import torch

            input_dtype = torch.get_default_dtype()

        input_dtype_name = str(input_dtype).split(".")[-1]

    package_name = cast(Literal["numpy", "cupy", "torch"], package_name)

    if isinstance(distribution, Distribution):
        distribution = distribution.to(Slab, ndim=len(shape), copy=True)
        distribution = cast(Slab, distribution)
    else:
        # Must be a Box pair (this is checked in the ProblemSpec reducer).
        distribution = tuple(cast(Box, box.copy()) for box in distribution)

    options = FFTOptions(fft_type=fft_type)
    problem_spec = _ProblemSpec(
        distribution=distribution,
        shape=list(shape),
        is_C=True,
        operand_dtype=input_dtype_name,
        options=_ProblemSpec.Options(options),
        package=package_name,
        memory_space=memory_space,
        global_size=math.prod(shape),
    )
    if nranks > 1:
        problem_spec = comm.allreduce(problem_spec, op=_problem_spec_reducer)
    else:
        # This ensures error-checking with one rank.
        problem_spec = _problem_spec_reducer(problem_spec, problem_spec)
    if isinstance(problem_spec, Exception):
        # There is an error or inconsistency in the problem spec across processes.
        # Note that since this comes from an allreduce, all processes will have
        # received the same exception.
        raise problem_spec

    fft_type = _get_default_fft_abstract_type(input_dtype_name, fft_type)
    if (fft_type == "R2C" and "float" not in input_dtype_name) or (
        fft_type in ("C2C", "C2R") and "complex" not in input_dtype_name
    ):
        raise ValueError(f"input dtype {input_dtype_name} is not compatible with FFT type {fft_type}")

    logger = logger if logger is not None else logging.getLogger()
    logger.info(
        f"Allocating {package.__name__} operand with shape {shape} and dtype "
        f"{input_dtype_name} for FFT type {fft_type} on {memory_space}, with "
        f"distribution {distribution}."
    )

    # Infer global shape.
    operand_dim = len(shape)
    if isinstance(distribution, Slab):
        global_shape = tuple(problem_spec.shape)
    else:
        global_boxes = cast(Sequence[Box], problem_spec.distribution)
        lower, upper = global_boxes[0]
        global_shape = tuple(int(upper[i] - lower[i]) for i in range(operand_dim))

    # Calculate max capacity for this transform.
    capacity = _calculate_capacity(problem_spec, global_shape, fft_type, nranks)

    return _allocate_for_fft(
        global_shape, shape, distribution, input_dtype, memory_space, package, fft_type, capacity, rank, nranks
    ).tensor


def _get_view(
    array,
    desired_shape: Sequence[int],
    desired_dtype: str,
    comm,
    collective_error_checking: bool,
):
    """Returns view of the array of the desired shape and dtype. If the given array doesn't
    have the same dtype and number of elements, tries to return a view from the base array
    (original array that owns the memory), where elements are taken from contiguous memory
    starting from the beginning of the buffer."""
    error = None
    desired_size = math.prod(desired_shape)  # number of elements
    rank = comm.Get_rank()
    try:
        if array.dtype == desired_dtype and array.size == desired_size:
            if tuple(array.shape) != tuple(desired_shape):
                result = array.reshape(desired_shape, copy=False)
            else:
                result = array
        else:

            def error_msg(base):
                return (
                    f"[{rank}] Internal error: tensor doesn't have a base array large enough "
                    "for the required shape and dtype: base array shape and dtype is "
                    f"({base.shape}, {base.dtype}), desired shape and dtype is "
                    f"({desired_shape}, {desired_dtype}). Consider allocating the operand "
                    "for this FFT with nvmath.distributed.fft.allocate_operand()"
                )

            if array.name == "cuda":
                base: ndbuffer.NDBuffer = array.tensor
                while True:
                    if not hasattr(base, "data") or not isinstance(base.data, ndbuffer.NDBuffer):
                        break
                    base = base.data

                itemsize = NAME_TO_ITEM_SIZE[desired_dtype]
                nbytes_required = desired_size * itemsize
                if base.size_in_bytes < nbytes_required:
                    # Note: if this error occurs, it can easily happen on one process
                    # but not others.
                    raise RuntimeError(error_msg(base))

                desired_strides = calculate_strides(desired_shape, reversed(range(len(desired_shape))))
                view = ndbuffer.wrap_external(
                    base,
                    base.data_ptr,
                    desired_dtype,
                    desired_shape,
                    desired_strides,
                    base.device_id,
                    itemsize,
                )
                result = CudaDistributedTensor(view)
            else:
                try:
                    base = array.tensor.base
                except AttributeError:
                    base = array.tensor._base

                if base is None:
                    base = array.tensor

                dtype = array.name_to_dtype[desired_dtype]
                nbytes_required = desired_size * dtype.itemsize
                if base.nbytes < nbytes_required:  # type: ignore
                    # Note: if this error occurs, it can easily happen on one process
                    # but not others.
                    raise RuntimeError(error_msg(base))

                if len(base.shape) > 1:
                    # Flatten the base array.
                    base = base.reshape(-1)  # type: ignore

                v = base.view(dtype)[:desired_size]  # type: ignore
                result = tensor_wrapper.wrap_operand(v).reshape(desired_shape, copy=False)
    except Exception as e:
        error = e

    if collective_error_checking:
        error = comm.allreduce(error, _reduce_exception)
    if error:
        raise error

    return result


def _copy_operand_perhaps(
    internal_operand: DistributedTensor | None,
    operand: DistributedTensor,
    stream_holder,
    execution_space,
    memory_space,
    device_id: int | Literal["cpu"],
    fft_abstract_type,
    global_shape,
    distribution,
    capacity,
    rank,
    nranks,
    logger,
):
    if execution_space == memory_space:
        return operand, None
    else:
        # Copy the `operand` to memory that matches the exec space and keep the
        # original `operand` since distributed FFT has inplace behavior and the
        # result will overwrite the original operand.
        if internal_operand is None:
            assert execution_space == "cuda"
            package: ModuleType
            if operand.name == "numpy":
                package = ndbuffer
                dtype = operand.dtype
            elif operand.name == "torch":
                import torch as package

                dtype = operand.tensor.dtype

            # XXX: not passing the stream to allocator because nvshmem_malloc doesn't
            # take a stream.
            exec_space_copy = _allocate_for_fft(
                global_shape,
                operand.shape,
                distribution,
                dtype,
                "cuda",
                package,
                fft_abstract_type,
                capacity,
                rank,
                nranks,
            )
            assert exec_space_copy.device_id == device_id
            exec_space_copy.copy_(operand, stream_holder)
            return exec_space_copy, operand
        else:
            # In-place copy to existing pointer
            tensor_wrapper.copy_([operand], [internal_operand], stream_holder)
            return internal_operand, operand


def _problem_spec_reducer(p1: _ProblemSpec, p2: _ProblemSpec):
    try:
        if isinstance(p1, Exception):
            return p1  # propagate exception

        if isinstance(p2, Exception):
            return p2  # propagate exception

        if len(p1.shape) != len(p2.shape):
            return ValueError("The number of dimensions of the input operand is inconsistent across processes")

        # Check if rank is 2-D or 3-D.
        if len(p1.shape) not in (2, 3):
            return ValueError(
                "Distributed FFT is currently supported only for 2-D and 3-D tensors."
                f" The number of dimensions of the operand is {len(p1.shape)}."
            )

        if p1.operand_dtype != p2.operand_dtype:
            return ValueError("The operand dtype is inconsistent across processes")

        if p1.package != p2.package:
            return ValueError("operand doesn't belong to the same package on all processes")

        if p1.memory_space != p2.memory_space:
            return ValueError('operand is not on the same memory space ("cpu", "cuda") on all processes')

        if p1.options != p2.options:
            return ValueError(f"options are inconsistent across processes: {p1.options} != {p2.options}")

        # Determine the memory layout shared by all processes.
        p1.is_C &= p2.is_C
        if not p1.is_C:
            return ValueError("The input memory layout is not C on every process")

        is_box_1 = not isinstance(p1.distribution, Slab)
        is_box_2 = not isinstance(p2.distribution, Slab)
        if is_box_1 != is_box_2:
            return ValueError("distribution must be either Slab or box on all processes, not a mix of both")

        fft_abstract_type = _get_default_fft_abstract_type(p1.operand_dtype, p1.options.fft_type)

        if len(p1.shape) == 2 and not is_box_1:
            if fft_abstract_type == "R2C" and p1.distribution != Slab.X:
                return ValueError("2D FFT R2C only supports X-slab input")
            elif fft_abstract_type == "C2R" and p1.distribution != Slab.Y:
                return ValueError("2D FFT C2R only supports Y-slab input")

        if not is_box_1:
            if p1.distribution != p2.distribution:
                raise ValueError("The slab distribution is inconsistent across processes")

            slab = cast(Slab, p1.distribution)

            if slab.ndim != len(p1.shape):
                raise ValueError(
                    f"The dimensionality of {p1.distribution} doesn't match the dimensionality "
                    "of the FFT operand ({len(p1.shape)})"
                )

            # Using cuFFTMp slab distribution.
            partitioned_dim = slab.partition_dim

            if partitioned_dim not in (0, 1):
                raise ValueError("The Slab partition dimension must be X or Y")

            if any(p1.shape[i] != p2.shape[i] for i in range(len(p1.shape)) if i != partitioned_dim):
                return ValueError("The problem size is inconsistent across processes")

            if p1 is not p2:  # with nranks=1 p1 is p2
                # Reduce the partitioned dimension to get the global size.
                p1.shape[partitioned_dim] += p2.shape[partitioned_dim]
        else:
            # Custom distribution given by input and output boxes on each process.
            for distribution in (p1.distribution, p2.distribution):
                if not isinstance(distribution, Sequence) or not all(isinstance(d, Box) for d in distribution):
                    return ValueError("distribution must be a Slab or a Box pair")

            if len(p1.distribution) != 2 or len(p2.distribution) != 2:  # type: ignore
                return ValueError("Must provide a Box pair on every process")
            input_box1, output_box1 = cast(Sequence[Box], p1.distribution)
            input_box2, output_box2 = cast(Sequence[Box], p2.distribution)
            for box in (input_box1, output_box1, input_box2, output_box2):
                if box.ndim != len(p1.shape):
                    return ValueError(
                        f"The dimensionality of {box} doesn't match the dimensionality of the FFT operand ({len(p1.shape)})"
                    )

            for p_spec in (p1, p2):
                if p_spec.is_leaf:
                    # Check that the input box shape of this process matches the shape of
                    # the input operand.
                    input_lower, input_upper = p_spec.distribution[0]  # type: ignore
                    input_box_shape = tuple(input_upper[i] - input_lower[i] for i in range(len(p_spec.shape)))
                    if input_box_shape != tuple(p_spec.shape):
                        return ValueError(
                            f"The operand shape {p_spec.shape} does not match the input box shape {input_box_shape}"
                        )

                    output_lower, output_upper = p_spec.distribution[1]  # type: ignore
                    output_box_shape = tuple(output_upper[i] - output_lower[i] for i in range(len(p_spec.shape)))
                    p_spec.input_max_elements = math.prod(input_box_shape)
                    p_spec.output_max_elements = math.prod(output_box_shape)

            if p1 is not p2:  # with nranks=1 p1 is p2
                p1.global_size += p2.global_size

            p1.input_max_elements = max(p1.input_max_elements, p2.input_max_elements)
            p1.output_max_elements = max(p1.output_max_elements, p2.output_max_elements)

            def reduce_boxes(box1, box2):
                """This function returns the smallest box that encompasses `box1`
                and `box2`"""
                lower = np.minimum(np.array(box1.lower), np.array(box2.lower)).tolist()
                upper = np.maximum(np.array(box1.upper), np.array(box2.upper)).tolist()
                return Box(lower, upper)

            # Merge the boxes to get the global operand shape. Note that this is applied
            # progressively throughout the MPI reduction, starting with the local boxes.
            p1.distribution = (reduce_boxes(input_box1, input_box2), reduce_boxes(output_box1, output_box2))

    except Exception as e:
        return e
    p1.is_leaf = False
    return p1


def _reduce_exception(e1, e2):
    if e1 is not None:
        return e1
    return e2


class InvalidFFTState(Exception):
    pass


@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
class FFT:
    """
    Create a stateful object that encapsulates the specified distributed FFT computations
    and required resources. This object ensures the validity of resources during use and
    releases them when they are no longer needed to prevent misuse.

    This object encompasses all functionalities of function-form APIs :func:`fft`,
    :func:`ifft`, :func:`rfft`, and :func:`irfft`, which are convenience wrappers around it.
    The stateful object also allows for the amortization of preparatory costs when the same
    FFT operation is to be performed on multiple operands with the same problem
    specification (see :meth:`reset_operand` for more details).

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and
       options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation
       for this specific FFT operation.
    3. **Execution**: Perform the FFT computation with :meth:`execute`, which can be either
       forward or inverse FFT transformation.
    4. **Resource Management**: Ensure all resources are released either by explicitly
       calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on each step described above can be obtained by passing in a
    :class:`logging.Logger` object to :class:`FFTOptions` or by setting the appropriate
    options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(
        ...     level=logging.INFO,
        ...     format="%(asctime)s %(levelname)-8s %(message)s",
        ...     datefmt="%m-%d %H:%M:%S",
        ... )

    Args:
        operand: {operand}
            {operand_admonitions}

        distribution: {distribution}

        options: {options}

        stream: {stream}

    .. seealso::
        :meth:`plan`, :meth:`reset_operand`, :meth:`execute`

    Examples:

        >>> import cupy as cp
        >>> import nvmath.distributed

        Get MPI communicator used to initialize nvmath.distributed (for information on
        initializing nvmath.distributed, you can refer to the documentation or to the
        FFT examples in `nvmath/examples/distributed/fft
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft>`_):

        >>> comm = nvmath.distributed.get_context().communicator

        Get the number of processes:

        >>> nranks = comm.Get_size()

        Create a 3-D complex128 ndarray on GPU symmetric memory, distributed according to
        the Slab distribution on the X axis (the global shape is (128, 128, 128)):

        >>> from nvmath.distributed.distribution import Slab
        >>> shape = 128 // nranks, 128, 128

        cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
        operands to be on the symmetric heap:

        >>> a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)

        After allocating, we initialize the CuPy ndarray's memory:

        >>> a[:] = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

        We will define a 3-D C2C FFT operation, creating an FFT object encapsulating the
        above problem specification. Each process provides their own local operand (which
        is part of the PGAS space, but otherwise can be operated on as any other CuPy
        ndarray for local operations) and specifies how the operand is distributed across
        processes:

        >>> f = nvmath.distributed.fft.FFT(a, distribution=Slab.X)

        More information on distribution of operands can be found in the documentation:
        https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/fft/index.html

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`FFTOptions`).

        Next, plan the FFT:

        >>> f.plan()

        Now execute the FFT, and obtain the result `r1` as a CuPy ndarray. Note that
        distributed FFT computations are inplace, so operands a and r1 share the same
        symmetric memory buffer:

        >>> r1 = f.execute()

        Finally, free the FFT object's resources. To avoid this explicit call, it's
        recommended to use the FFT object as a context manager as shown below, if possible.

        >>> f.free()

        Any symmetric memory that is owned by the user must be deleted explicitly (this is
        a collective call and must be called by all processes). Note that because operands
        a and r1 share the same buffer, only one of them must be freed:

        >>> nvmath.distributed.free_symmetric_memory(a)

        Note that all :class:`FFT` methods execute on the current stream by default.
        Alternatively, the `stream` argument can be used to run a method on a specified
        stream.

        Let's now look at the same problem with NumPy ndarrays on the CPU.

        Create a 3-D complex128 NumPy ndarray on the CPU:

        >>> import numpy as np
        >>> shape = 128 // nranks, 128, 128
        >>> a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        Create an FFT object encapsulating the problem specification described earlier and
        use it as a context manager.

        >>> with nvmath.distributed.fft.FFT(a, distribution=Slab.X) as f:
        ...     f.plan()
        ...
        ...     # Execute the FFT to get the first result.
        ...     r1 = f.execute()

        All the resources used by the object are released at the end of the block.

        The operation was performed on the GPU, with the NumPy array temporarily copied to
        GPU symmetric memory and transformed on the GPU.

        Further examples can be found in the `nvmath/examples/distributed/fft
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft>`_
        directory.
    """

    def _free_internal_sheap(self, exception: Exception | None = None) -> bool:
        # This is a fail-safe to free NVSHMEM internal memory in case of invalid
        # state (FFT constructor fails). Since we might call nvshmem_free here, we're
        # assuming that all processes equally failed in the ctor, which might not be true,
        # but if it weren't true they would end up in deadlock most likely anyway.
        if (
            hasattr(self, "memory_space")
            and self.memory_space == "cpu"
            and self.operand is not None
            and self.operand.device == "cuda"
        ):
            with utils.device_ctx(self.device_id):
                self.operand.free_symmetric()
        return True

    @utils.atomic(_free_internal_sheap, method=True)
    def __init__(
        self,
        operand,
        *,
        distribution: Distribution | Sequence[Box],
        options: FFTOptions | None = None,
        stream: AnyStream | None = None,
    ):
        distributed_ctx = nvmath.distributed.get_context()
        if distributed_ctx is None:
            raise RuntimeError(
                "nvmath.distributed has not been initialized. Refer to "
                "https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html#initializing-the-distributed-runtime"
                " for more information."
            )
        if not distributed_ctx.nvshmem_available:
            raise RuntimeError("nvmath.distributed wasn't initialized with NVSHMEM backend")
        self.communicator = communicator = distributed_ctx.communicator
        self.rank = rank = communicator.Get_rank()
        self.nranks = nranks = communicator.Get_size()

        self.operand = operand = tensor_wrapper.wrap_operand(operand)
        self.options = options = cast(FFTOptions, utils.check_or_create_options(FFTOptions, options, "Distributed FFT options"))
        self.package = operand.name

        if isinstance(distribution, Distribution):
            distribution = distribution.to(Slab, ndim=len(operand.shape), copy=True)
            distribution = cast(Slab, distribution)
        else:
            # Must be a Box pair (this is checked in the ProblemSpec reducer).
            distribution = tuple(cast(Box, box.copy()) for box in distribution)

        is_C = sorted(operand.strides, reverse=True) == list(operand.strides)

        # Merge the problem specification across processes to make sure that there are no
        # inconsistencies and to calculate the global shape. Importantly, this also does
        # collective error checking of the FFT input parameters, to ensure that all
        # processes fail on error of any one process, thus preventing deadlock.
        problem_spec = _ProblemSpec(
            distribution=distribution,
            shape=list(operand.shape),
            is_C=is_C,
            operand_dtype=operand.dtype,
            options=_ProblemSpec.Options(options),
            package=self.package,
            memory_space=operand.device,
            global_size=math.prod(operand.shape),
        )
        if nranks > 1:
            problem_spec = communicator.allreduce(problem_spec, op=_problem_spec_reducer)
        else:
            # Ensure we error-check with one rank.
            problem_spec = _problem_spec_reducer(problem_spec, problem_spec)
        if isinstance(problem_spec, Exception):
            # There is an error or inconsistency in the problem spec across processes.
            # Note that since this comes from an allreduce, all processes will have
            # received the same exception.
            raise problem_spec

        self.operand_dim = len(operand.shape)

        self.operand_data_type = operand.dtype
        self.fft_abstract_type = _get_default_fft_abstract_type(self.operand_data_type, options.fft_type)

        self.result_data_type, self.compute_data_type = _get_fft_result_and_compute_types(operand.dtype, self.fft_abstract_type)

        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info(f"The FFT type is {self.fft_abstract_type}.")
        self.logger.info(
            f"The input data type is {self.operand_data_type}, and the result data type is {self.result_data_type}."
        )

        # cuFFTMp doesn't support complex32.
        if self.result_data_type == "complex32":
            raise TypeError(f"The result data type {self.result_data_type} is not supported.")

        # Infer operand package, execution space, and memory space.
        execution_device_id: int = distributed_ctx.device_id
        if operand.device_id != "cpu":  # exec space matches the mem space
            self.memory_space = "cuda"
            self.device_id = operand.device_id
            assert operand.device_id == execution_device_id
        else:  # we need to move inputs cpu -> gpu and outputs gpu -> cpu
            self.memory_space = "cpu"
            self.device_id = execution_device_id
        self.execution_space = "cuda"
        self.operand_device_id = operand.device_id
        self.internal_op_package = self._internal_operand_package(self.package)
        stream_holder: StreamHolder = utils.get_or_create_stream(self.device_id, stream, self.internal_op_package)

        if self.memory_space == "cuda" and not operand.is_symmetric_memory:
            raise TypeError("Distributed FFT requires GPU operand to be on symmetric memory")

        self.logger.info(
            f"The input tensor's memory space is {self.memory_space}, and the execution space "
            f"is {self.execution_space}, with device {self.device_id}."
        )

        self.logger.info(f"The specified stream for the FFT ctor is {stream_holder and stream_holder.obj}")

        # Infer the global extents.
        if isinstance(distribution, Slab):
            self.global_extents = tuple(problem_spec.shape)
            # Check that this process has the correct slab shape.
            error = None
            try:
                distribution._bind(self.global_extents, shape=self.operand.shape)
            except Exception as e:
                error = e
            error = communicator.allreduce(error, _reduce_exception)
            if error:
                raise error
        else:
            # Infer the global shape from the global input box. Note that cuFFTMp doesn't
            # require lower coordinates for the merged (global) boxes to be 0.
            lower, upper = problem_spec.distribution[0]  # type: ignore
            self.global_extents = tuple(int(upper[i] - lower[i]) for i in range(self.operand_dim))

            # This can't throw error since the local operand shape was already checked
            # against the box shape in the ProblemSpec reducer.
            distribution[0]._bind(self.global_extents, shape=self.operand.shape)

            # The global number of elements must be compatible with the global shape.
            if problem_spec.global_size != math.prod(self.global_extents):
                raise ValueError(
                    f"The global number of elements is incompatible with the inferred global shape {self.global_extents}"
                )

        for i in (0, 1):
            if self.global_extents[i] < nranks:
                raise ValueError(
                    f"The FFT dimension {i} has global length {self.global_extents[i]} which "
                    f"is smaller than the number of processes ({nranks})"
                )

        self.logger.info(f"The global FFT extents are {self.global_extents}.")

        # Calculate the required buffer capacity (in number of elements) for this transform.
        self.capacity: int = _calculate_capacity(problem_spec, self.global_extents, self.fft_abstract_type, nranks)

        # Copy the operand to execution_space's device if needed.
        self.operand, self.operand_backup = _copy_operand_perhaps(
            None,
            operand,
            stream_holder,
            self.execution_space,
            self.memory_space,
            self.device_id,
            self.fft_abstract_type,
            self.global_extents,
            distribution,
            self.capacity,
            rank,
            nranks,
            self.logger,
        )

        operand = self.operand
        # Capture operand layout for consistency checks when resetting operands.
        self.operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)

        self.logger.info("The FFT will be performed in-place, with the result overwriting the input.")

        # The result's package and device.
        self.result_class: DistributedTensor = operand.__class__

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == "cpu"
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )

        if not isinstance(distribution, Slab):
            # Reshape only applies to cuFFTMp's default slab distribution.
            self.options.reshape = False
            self.logger.info("Reshape option is ignored when using box distribution.")

        # Set memory allocator.
        self.allocator = NvshmemMemoryManager(self.device_id, self.logger)

        self.distribution: Slab | Sequence[Box] = distribution
        # Map possible distributions to the corresponding operand TensorLayout.
        self.distribution_layout: dict[Slab | Box, TensorLayout] = {}
        # The subformat is an identifier that cuFFTMp uses to refer to an operand
        # distribution. It can be one of:
        # - cufftMp.XtSubFormat.FORMAT_INPLACE (refers to Slab.X)
        # - cufftMp.XtSubFormat.FORMAT_INPLACE_SHUFFLED (refers to Slab.Y)
        # - cufft.XtSubFormat.FORMAT_DISTRIBUTED_INPUT (the input box at FFT plan time)
        # - cufft.XtSubFormat.FORMAT_DISTRIBUTED_OUTPUT (the output box at FFT plan time)
        self.subformat: int = -1
        if isinstance(distribution, Slab):
            self.distribution_layout[distribution] = self.operand_layout

            if self.options.reshape:
                from_axis, to_axis = ("X", "X") if distribution == Slab.X else ("Y", "Y")
            else:
                from_axis, to_axis = ("X", "Y") if distribution == Slab.X else ("Y", "X")
            self.logger.info(
                f"The operand distribution is Slab, with input partitioned on {from_axis} axis "
                f"and output on {to_axis} (reshape={self.options.reshape})."
            )
        else:
            input_box, output_box = distribution
            self.distribution_layout[input_box] = self.operand_layout

            self.logger.info(f"The operand distribution is based on custom input box {input_box} and output box {output_box}.")

        # Infer result shape and strides.

        self.global_result_extents = list(self.global_extents)
        global_result_extents_padded = None
        if self.fft_abstract_type == "R2C":
            self.global_result_extents[-1] = self.global_result_extents[-1] // 2 + 1
        elif self.fft_abstract_type == "C2R":
            self.global_result_extents[-1] = (self.global_result_extents[-1] - 1) * 2
            if options.last_axis_parity == "odd":
                self.global_result_extents[-1] += 1
            global_result_extents_padded = list(self.global_result_extents)
            global_result_extents_padded[-1] = self.global_extents[-1] * 2

        if not isinstance(distribution, Slab):
            global_boxes = cast(Sequence[Box], problem_spec.distribution)
            lower, upper = global_boxes[1]
            actual_global_result_extents = tuple(int(upper[i] - lower[i]) for i in range(self.operand_dim))
            if actual_global_result_extents != tuple(self.global_result_extents):
                raise ValueError(
                    "The global box derived from the output boxes doesn't have the expected shape: "
                    f"global_input_box={problem_spec.distribution[0]}, global_output_box={problem_spec.distribution[1]}"  # type: ignore
                )

        if self.options.reshape:
            partition_dim = distribution.partition_dim  # type: ignore
            if self.fft_abstract_type == "C2R":
                self.result_shape_padded, _ = _calculate_slab_shape_strides(
                    global_result_extents_padded, partition_dim, rank, nranks
                )
            self.result_shape, self.result_strides = _calculate_slab_shape_strides(
                self.global_result_extents, partition_dim, rank, nranks, global_result_extents_padded
            )

            # The input of the reshape is the output of the FFT and will have these strides.
            # Note the special strides of the C2R output based on the output's padded last
            # axis.
            _, self.intermediate_strides = _calculate_slab_shape_strides(
                self.global_result_extents, 1 - partition_dim, rank, nranks, global_result_extents_padded
            )
        elif not isinstance(self.distribution, Slab):
            output_lower, output_upper = output_box
            self.result_shape = tuple(output_upper[i] - output_lower[i] for i in range(self.operand_dim))
            self.result_strides = calculate_strides(self.result_shape, reversed(range(self.operand_dim)))
            self.distribution_layout[output_box] = TensorLayout(shape=self.result_shape, strides=self.result_strides)
            output_box._bind(self.global_result_extents, shape=self.result_shape)
        else:
            result_partition_dim = 1 - distribution.partition_dim  # type: ignore
            if self.fft_abstract_type == "C2R":
                self.result_shape_padded, _ = _calculate_slab_shape_strides(
                    global_result_extents_padded, result_partition_dim, rank, nranks
                )
            self.result_shape, self.result_strides = _calculate_slab_shape_strides(
                self.global_result_extents, result_partition_dim, rank, nranks, global_result_extents_padded
            )
            self.distribution_layout[Slab.X if distribution == Slab.Y else Slab.Y] = TensorLayout(
                shape=self.result_shape, strides=self.result_strides
            )

        # Obtain the result operand (the one that will be returned to the user with the
        # expected shape and dtype on this rank according to the FFT type and operand
        # distributions). Note that since the FFT is inplace, the result operand shares
        # the same buffer with the input operand.
        self._get_result_operand(collective_error_checking=True)

        # Create handle.
        with utils.device_ctx(self.device_id):
            self.handle = cufft.create()
            # Dummy handle to create a cufft descriptor with initial tiny data buffer.
            # We'll reuse this descriptor to call cufft.xt_exec_descriptor, by
            # setting the data pointer and subformat in the descriptor.
            self.memory_desc_handle = cufft.create()
            if self.options.reshape:
                self.reshape_handle = cufft.create_reshape()

        # Set stream for the FFT.
        with utils.device_ctx(self.device_id):
            cufft.set_stream(self.handle, stream_holder.ptr)  # type: ignore[union-attr]

        # Plan attributes.
        cufft.set_auto_allocation(self.handle, 0)

        self.fft_planned = False
        # Descriptor to call cufft.xt_exec_descriptor (by setting the
        # data pointer and subformat in the descriptor before execute).
        self.memory_desc = None
        # Pointer to tiny data buffer of descriptor when first created.
        self.dummy_desc_data_ptr = None

        # Workspace attributes.
        self.workspace_ptr: None | memory.MemoryPointer = None
        self.workspace_size = 0
        self._workspace_allocated_here = False
        self.reshaped_operand = None

        # Attributes to establish stream ordering.
        self.workspace_stream = None
        self.last_compute_event = None

        self.valid_state = True
        self.logger.info("The distributed FFT operation has been created.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free()

    def _check_valid_fft(self, *args, **kwargs):
        """
        Check if FFT object is alive and well.
        """
        if not self.valid_state:
            raise InvalidFFTState("The FFT object cannot be used after resources are free'd")

    def _free_plan_resources(self, exception: Exception | None = None) -> bool:
        """
        Free resources allocated in planning.
        """
        if self.memory_desc is not None:
            with utils.device_ctx(self.device_id):
                cufft.xt_free(self.memory_desc)
            self.memory_desc = None

        self.fft_planned = False
        return True

    def _internal_operand_package(self, package_name):
        if self.execution_space == "cuda":
            return package_name if package_name != "numpy" else "cuda"
        else:
            return package_name if package_name != "cupy" else "cupy_host"

    def _allocate_reshape_operand(self, exec_stream_holder: StreamHolder | None, log_debug):
        if log_debug:
            self.logger.debug("Beginning empty tensor creation to hold reshape value...")
            self.logger.debug(
                f"The reshape tensor shape = {self.result_shape} with strides = "
                f"{self.result_strides} and data type '{self.result_data_type}'."
            )

        capacity_out_dtype = (
            self.capacity * 2
            if self.fft_abstract_type == "C2R"
            else self.capacity // 2
            if self.fft_abstract_type == "R2C"
            else self.capacity
        )
        # For C2R we preserve the last axis strides of the real output
        # when we reshape.
        result = _allocate_for_fft(
            self.global_result_extents,
            self.result_shape,
            self.distribution,
            self.result_operand.name_to_dtype[self.result_data_type],
            "cuda",
            self.result_operand.module,
            self.fft_abstract_type[::-1],  # type: ignore
            capacity_out_dtype,
            self.rank,
            self.nranks,
        )
        if log_debug:
            self.logger.debug("The reshape output (empty) tensor has been created.")
        return result

    def _get_result_operand(self, collective_error_checking):
        if isinstance(self.distribution, Slab) and self.fft_abstract_type == "C2R":

            def strided_view(x):
                v = _get_view(
                    x, self.result_shape_padded, self.result_data_type, self.communicator, collective_error_checking
                ).tensor
                if not isinstance(v, ndbuffer.NDBuffer):
                    return tensor_wrapper.wrap_operand(v[..., : self.result_shape[-1]])
                else:
                    v = ndbuffer.wrap_external(
                        v, v.data_ptr, self.result_data_type, self.result_shape, v.strides, v.device_id, v.itemsize
                    )
                    return CudaDistributedTensor(v)

            if self.operand_backup is not None:
                self.cpu_result_operand = strided_view(self.operand_backup)
            self.result_operand = strided_view(self.operand)
        else:
            if self.operand_backup is not None:
                self.cpu_result_operand = _get_view(
                    self.operand_backup, self.result_shape, self.result_data_type, self.communicator, collective_error_checking
                )
            self.result_operand = _get_view(
                self.operand, self.result_shape, self.result_data_type, self.communicator, collective_error_checking
            )

    @utils.precondition(_check_valid_fft)
    @utils.atomic(_free_plan_resources, method=True)
    def plan(self, *, stream: AnyStream | None = None):
        """Plan the FFT.

        Args:
            stream: {stream}
        """
        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        if self.fft_planned:
            self.logger.debug("The FFT has already been planned, and redoing the plan is not supported.")
            return

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_op_package)
        self.workspace_stream = stream_holder.obj

        # Set stream for the FFT.
        with utils.device_ctx(self.device_id):
            cufft.set_stream(self.handle, stream_holder.ptr)

        check_inplace_overlapping_layout(self.operand)
        if self.operand_backup is not None:
            check_inplace_overlapping_layout(self.operand_backup)

        if log_debug:
            self.logger.debug(
                f"The operand CUDA type is {NAME_TO_DATA_TYPE[self.operand_data_type].name}, and the result CUDA type is "
                f"{NAME_TO_DATA_TYPE[self.result_data_type].name}."
            )
            self.logger.debug(f"The CUDA type used for compute is {NAME_TO_DATA_TYPE[self.compute_data_type].name}.")
        if log_info:
            self.logger.info("Starting distributed FFT planning...")

        planner = None
        if self.operand_dim == 2:
            planner = cufft.make_plan2d
        elif self.operand_dim == 3:
            planner = cufft.make_plan3d
        else:
            raise AssertionError("Internal error: unsupported dimensionality for distributed FFT in plan().")

        if self.options.reshape:
            # Plan a reshape of the FFT output back to the original slab distribution of the
            # FFT input.
            from_partition_dim, to_partition_dim = (1, 0) if self.distribution == Slab.X else (0, 1)
            # cuFFTMP reshape API only supports 3D, so we broadcast 2D operands.
            X, Y = self.global_result_extents[:2]
            Z = self.global_result_extents[2] if self.operand_dim == 3 else 1
            global_shape = (X, Y, Z)
            reshape_input_box = _calculate_local_box(global_shape, from_partition_dim, self.rank, self.nranks)
            reshape_output_box = _calculate_local_box(global_shape, to_partition_dim, self.rank, self.nranks)
            lower, upper = reshape_input_box
            reshape_input_strides = (
                self.intermediate_strides if self.operand_dim == 3 else tuple(self.intermediate_strides) + (1,)
            )
            reshape_output_strides = self.result_strides if self.operand_dim == 3 else tuple(self.result_strides) + (1,)

        with utils.cuda_call_ctx(stream_holder, blocking=True, timing=log_info) as (
            self.last_compute_event,
            elapsed,
        ):
            if isinstance(self.distribution, Slab):
                self.subformat = self.distribution._cufftmp_value
            else:
                if self.fft_abstract_type == "C2R":
                    # C2R plans only support CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT,
                    # i.e., (lower_input, upper_input) should describe the real data
                    # distribution and (lower_output, upper_output) the complex data
                    # distribution.
                    lower_input, upper_input = self.distribution[1]
                    lower_output, upper_output = self.distribution[0]
                    strides_input = self.result_strides
                    strides_output = self.operand_layout.strides
                else:
                    lower_input, upper_input = self.distribution[0]
                    lower_output, upper_output = self.distribution[1]
                    strides_input = self.operand_layout.strides
                    strides_output = self.result_strides

                cufft.xt_set_distribution(
                    self.handle,
                    self.operand_dim,
                    lower_input,
                    upper_input,
                    lower_output,
                    upper_output,
                    strides_input,
                    strides_output,
                )
                self.box_to_subformat = {}
                self.box_to_subformat[Box(lower_input, upper_input)] = cufft.XtSubFormat.FORMAT_DISTRIBUTED_INPUT
                self.box_to_subformat[Box(lower_output, upper_output)] = cufft.XtSubFormat.FORMAT_DISTRIBUTED_OUTPUT
                self.subformat = (
                    cufft.XtSubFormat.FORMAT_DISTRIBUTED_INPUT
                    if self.fft_abstract_type != "C2R"
                    else cufft.XtSubFormat.FORMAT_DISTRIBUTED_OUTPUT
                )

            fft_concrete_type = _get_fft_concrete_type(self.operand_data_type, self.fft_abstract_type)
            self.logger.debug(f"The FFT concrete type is {fft_concrete_type.name}.")
            # NVSHMEM is already initialized (no need to pass MPI comm to the library).
            cufft.attach_comm(self.handle, cufft.MpCommType.COMM_NONE, 0)
            if self.fft_abstract_type == "C2R":
                self.workspace_size = planner(self.handle, *self.global_result_extents, fft_concrete_type)
            else:
                self.workspace_size = planner(self.handle, *self.global_extents, fft_concrete_type)

            # Create memory descriptor using dummy handle.
            _ = planner(self.memory_desc_handle, *[1] * self.operand_dim, fft_concrete_type)
            self.memory_desc = cufft.xt_malloc(self.memory_desc_handle, cufft.XtSubFormat.FORMAT_INPLACE)

            if self.options.reshape:
                nullptr = 0
                cufft.make_reshape(
                    self.reshape_handle,
                    self.result_operand.itemsize,
                    3,
                    reshape_input_box[0],
                    reshape_input_box[1],
                    reshape_input_strides,
                    reshape_output_box[0],
                    reshape_output_box[1],
                    reshape_output_strides,
                    nullptr,
                    cufft.MpCommType.COMM_NONE,
                )
                reshape_workspace_size = cufft.get_reshape_size(self.reshape_handle)
                self.workspace_size = max(self.workspace_size, reshape_workspace_size)

        self.logger.debug(
            f"The workspace required on process {self.rank} for the distributed"
            f" FFT operation is {formatters.MemoryStr(self.workspace_size)}."
        )

        # Store memory descriptor's buffer pointer, to be able to free it later.
        self.dummy_desc_data_ptr = cufft.set_descriptor_data(self.memory_desc, 0, self.subformat)

        self.fft_planned = True

        if log_info and elapsed.data is not None:
            self.logger.info(f"The FFT planning phase took {elapsed.data:.3f} ms to complete.")

    @utils.precondition(_check_valid_fft)
    def reset_operand(
        self, operand=None, *, distribution: Distribution | Sequence[Box] | None = None, stream: AnyStream | None = None
    ):
        """
        Reset the operand held by this :class:`FFT` instance. This method has two use cases:

        (1) it can be used to provide a new operand for execution
        (2) it can be used to release the internal reference to the previous operand and
            potentially make its memory available for other use by passing
            ``operand=None``.

        Args:
            operand: A tensor (ndarray-like object) compatible with the previous one or
                `None` (default). A value of `None` will release the internal reference to
                the previous operand and user is expected to set a new operand before again
                calling :meth:`execute`. The new operand is considered compatible if all the
                following properties match with the previous one:

                - The operand distribution: (a) if the FFT was planned using a Slab
                  distribution, the reset operand must also use a Slab distribution
                  (both X and Y axes are valid regardless of the slab axis at
                  plan time), (b) if the FFT was planned using a box distribution, the
                  distribution of the reset operand must be (input_box, output_box)
                  or (output_box, input_box) where input_box and output_box are the
                  boxes specified at plan time.
                - The operand data type.
                - The package that the new operand belongs to.
                - The memory space of the new operand (CPU or GPU).
                - The device that new operand belongs to if it is on GPU.

            distribution: {distribution}

            stream: {stream}.

        Examples:

            >>> import cupy as cp
            >>> import nvmath.distributed

            Get MPI communicator used to initialize nvmath.distributed (for information on
            initializing nvmath.distributed, you can refer to the documentation or to the
            FFT examples in `nvmath/examples/distributed/fft
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft>`_):

            >>> comm = nvmath.distributed.get_context().communicator
            >>> nranks = comm.Get_size()

            Create a 3-D complex128 ndarray on GPU symmetric memory, distributed according
            to the Slab distribution on the X axis (the global shape is (128, 128, 128)):

            >>> from nvmath.distributed.distribution import Slab
            >>> shape = 128 // nranks, 128, 128
            >>> dtype = cp.complex128
            >>> a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=dtype)
            >>> a[:] = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

            Create an FFT object as a context manager

            >>> with nvmath.distributed.fft.FFT(a, distribution=Slab.X) as f:
            ...     # Plan the FFT
            ...     f.plan()
            ...
            ...     # Execute the FFT to get the first result.
            ...     r1 = f.execute()
            ...
            ...     # Reset the operand to a new CuPy ndarray.
            ...     b = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=dtype)
            ...     b[:] = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
            ...     f.reset_operand(b)
            ...
            ...     # Execute to get the new result corresponding to the updated operand.
            ...     r2 = f.execute()

            With :meth:`reset_operand`, minimal overhead is achieved as problem
            specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operand` is
            equivalent to updating the operand in-place, i.e, replacing
            ``f.reset_operand(b)`` with ``a[:]=b``. Note that updating the operand in-place
            should be adopted with caution as it can only yield the expected result and
            incur no additional copies under the additional constraints below:

                - The operand's distribution is the same.

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft/example06_stateful_reset_inplace.py>`_.
        """

        log_info = self.logger.isEnabledFor(logging.INFO)

        if operand is None:
            if self.memory_space == "cpu" and self.operand is not None:
                with utils.device_ctx(self.device_id):
                    # Since the execution when user passes CPU operands is blocking, it's
                    # safe to call nvshmem_free here without additional synchronization.
                    self.operand.free_symmetric()
            self.operand = None  # type: ignore
            self.operand_backup = None
            self.logger.info("The operand has been reset to None.")
            return

        self.logger.info("Resetting operand...")
        # First wrap operand.
        operand = tensor_wrapper.wrap_operand(operand)

        # Check package match.
        if self.package != operand.name:
            raise TypeError(f"Library package mismatch: '{self.package}' => '{operand.name}'")

        utils.check_attribute_match(self.operand_data_type, operand.dtype, "data type")

        if len(operand.shape) != self.operand_dim:
            raise ValueError(
                f"The reset operand number of dimensions ({len(operand.shape)}) does not "
                f"match the FFT number of dimensions ({self.operand_dim})"
            )

        stream_holder: StreamHolder = utils.get_or_create_stream(self.device_id, stream, self.internal_op_package)
        self.logger.info(f"The specified stream for reset_operand() is {stream_holder and stream_holder.obj}.")

        # In principle, we could support memory_space change,
        # but to handle it properly we need to update self.memory_space and
        # some dependent properties, like self.blocking, which may be error-prone
        # from the user perspective. It would prevent inplace optimizations as well.
        operand_device_id = operand.device_id
        if operand_device_id != self.operand_device_id:

            def device_str(device_id: int | Literal["cpu"]) -> str:
                return f"cuda:{device_id}" if isinstance(device_id, int) else f"{device_id}"

            raise ValueError(
                f"The new operand must be on the same device as the original one. "
                f"The new operand's device is {device_str(operand_device_id)}, "
                f"the original device is {device_str(self.operand_device_id)}"
            )

        if self.memory_space == "cuda" and not operand.is_symmetric_memory:
            raise TypeError("Distributed FFT requires GPU operand to be on symmetric memory")

        # Check for C memory layout.
        if sorted(operand.strides, reverse=True) != list(operand.strides):
            raise ValueError("The reset operand memory layout is not C")

        # Check that the distribution of the reset operand is compatible.
        if distribution is None:
            raise ValueError("Please specify the distribution of the operand for reset_operand")

        if isinstance(distribution, Distribution):
            distribution = distribution.to(Slab, ndim=self.operand_dim, copy=True)
        else:
            # Must be a Box pair.
            distribution = tuple(cast(Box, box.copy()) for box in distribution)

        distribution_type_old = "slab" if isinstance(self.distribution, Slab) else "box"
        distribution_type_new = "slab" if isinstance(distribution, Slab) else "box"
        if distribution_type_old != distribution_type_new:
            raise ValueError(
                f"This FFT uses {distribution_type_old} distribution, but got "
                f"{distribution_type_new} distribution in reset_operand."
            )

        if self.fft_abstract_type in ("R2C", "C2R") and self.distribution != distribution:
            raise ValueError(f"Can't change distribution with FFT type {self.fft_abstract_type}")

        if distribution_type_old == "slab":
            if self.options.reshape and self.distribution != distribution:
                raise ValueError("Can't change distribution when using reshape=True")

            distribution = cast(Slab, distribution)  # for type checker
            distribution._bind(self.global_extents, shape=operand.shape)
            self.distribution = distribution
            self.subformat = distribution._cufftmp_value

            # Log distribution.
            if log_info:
                if self.options.reshape:
                    from_axis, to_axis = ("X", "X") if distribution == Slab.X else ("Y", "Y")
                else:
                    from_axis, to_axis = ("X", "Y") if distribution == Slab.X else ("Y", "X")
                self.logger.info(
                    f"The operand distribution is Slab, with input partitioned on {from_axis} axis "
                    f"and output on {to_axis} (reshape={self.options.reshape})."
                )
        else:
            distribution = cast(Sequence[Box], distribution)  # for type checker
            input_box, output_box = distribution
            if input_box not in self.box_to_subformat or output_box not in self.box_to_subformat:
                raise ValueError("The reset operand distribution must use the original boxes (in any order)")
            distribution[0]._bind(self.global_extents, shape=operand.shape)
            self.subformat = self.box_to_subformat[input_box]
            self.distribution = distribution

            # Log distribution.
            self.logger.info("The operand distribution is based on custom input and output boxes given on each process.")

        # Set stream for the FFT.
        with utils.device_ctx(self.device_id):
            cufft.set_stream(self.handle, stream_holder.ptr)

        # C2C allows changing distribution in reset_operand, so we may need to adjust
        # the shape of the internal operand.
        internal_operand = (
            None
            if self.operand is None
            else _get_view(self.operand, operand.shape, operand.dtype, self.communicator, collective_error_checking=False)
        )
        self.operand, self.operand_backup = _copy_operand_perhaps(
            internal_operand,
            operand,
            stream_holder,
            self.execution_space,
            self.memory_space,
            self.device_id,
            self.fft_abstract_type,
            self.global_extents,
            distribution,
            self.capacity,
            self.rank,
            self.nranks,
            self.logger,
        )
        operand = self.operand

        # Update operand layout and plan traits.
        self.operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)
        self.logger.info(f"The reset operand shape = {self.operand_layout.shape}, and strides = {self.operand_layout.strides}.")

        if distribution_type_old == "box":
            result_layout = self.distribution_layout[output_box]
            output_box._bind(self.global_result_extents, shape=result_layout.shape)
        elif not self.options.reshape:
            result_layout = self.distribution_layout[Slab.X if distribution == Slab.Y else Slab.Y]
        else:
            if self.fft_abstract_type in ("R2C", "C2R"):
                # Result layout doesn't change.
                result_layout = TensorLayout(shape=self.result_shape, strides=self.result_strides)
            else:
                result_layout = self.operand_layout
        self.result_shape = result_layout.shape
        self.result_strides = result_layout.strides

        self.logger.info(f"The result shape = {self.result_shape}, and strides = {self.result_strides}.")

        # Obtain the result operand (the one that will be returned to the user with the
        # expected shape and dtype on this rank according to the FFT type and operand
        # distributions). Note that since the FFT is inplace, the result operand shares
        # the same buffer with the input operand.
        self._get_result_operand(collective_error_checking=False)

        self.logger.info("The operand has been reset to the specified operand.")

    def _check_planned(self, *args, **kwargs):
        """ """
        what = kwargs["what"]
        if not self.fft_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _check_valid_operand(self, *args, **kwargs):
        """ """
        what = kwargs["what"]
        if self.operand is None:
            raise RuntimeError(
                f"{what} cannot be performed if the input operand has been set to None. Use reset_operand() to set the "
                f"desired input before using performing the {what.lower()}."
            )

    def _free_workspace_memory(self, exception: Exception | None = None) -> bool:
        """
        Free workspace by releasing the MemoryPointer object and reshape operand.
        """
        if self.workspace_ptr is None:
            return True

        with utils.device_ctx(self.device_id):
            # Calling nvshmem_free on memory that's still in use is not safe
            # (nvshmem_free is not stream-ordered), so we need to wait for the
            # computation to finish.
            if self.workspace_stream is not None:
                self.workspace_stream.sync()
            self.workspace_ptr.free()
            if self.reshaped_operand is not None:
                self.reshaped_operand.free_symmetric()
        self.workspace_ptr = None
        self.reshaped_operand = None
        self.logger.debug("[_free_workspace_memory] The workspace has been released.")

        return True

    @utils.precondition(_check_valid_fft)
    @utils.precondition(_check_planned, "Workspace memory allocation")
    @utils.atomic(_free_workspace_memory, method=True)
    def _allocate_workspace_memory(self, stream_holder: StreamHolder):
        """
        Allocate workspace memory using the specified allocator.
        """

        assert self._workspace_allocated_here is False, "Internal Error."

        self.logger.debug("Allocating workspace for performing the FFT...")
        with utils.device_ctx(self.device_id), stream_holder.ctx:
            try:
                self.workspace_ptr = self.allocator.memalloc(self.workspace_size)  # type: ignore[union-attr]
                if self.options.reshape:
                    self.reshaped_operand = self._allocate_reshape_operand(
                        stream_holder, self.logger.isEnabledFor(logging.DEBUG)
                    )
                self._workspace_allocated_here = True
            except TypeError as e:
                message = (
                    "The method 'memalloc' in the allocator object must conform to the interface in the "
                    "'BaseCUDAMemoryManager' protocol."
                )
                raise TypeError(message) from e
            raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
            cufft.set_work_area(self.handle, raw_workspace_ptr)

        self.workspace_stream = stream_holder.obj
        self.logger.debug(
            f"Finished allocating device workspace of size {formatters.MemoryStr(self.workspace_size)} in the context "
            f"of stream {self.workspace_stream}."
        )

    def _allocate_workspace_memory_perhaps(self, stream_holder: StreamHolder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't already been
        done.
        """
        if self.execution_space != "cuda" or self.workspace_ptr is not None:
            return

        return self._allocate_workspace_memory(stream_holder)

    @utils.precondition(_check_valid_fft)
    def _free_workspace_memory_perhaps(self, release_workspace):
        """
        Free workspace memory if if 'release_workspace' is True.
        """
        if not release_workspace:
            return

        # Establish ordering wrt the computation and free workspace if it's more than the
        # specified cache limit.
        if self.last_compute_event is not None:
            with utils.device_ctx(self.device_id):
                self.workspace_stream.wait(self.last_compute_event)
            self.logger.debug("Established ordering with respect to the computation before releasing the workspace.")
            self.last_compute_event = None

        self.logger.debug("[_free_workspace_memory_perhaps] The workspace memory will be released.")
        self._free_workspace_memory()

        return True

    def _release_workspace_memory_perhaps(self, exception: Exception | None = None) -> bool:
        """
        Free workspace memory if it was allocated in this call
        (self._workspace_allocated_here == True) when an exception occurs.
        """
        release_workspace = self._workspace_allocated_here
        self.logger.debug(
            f"[_release_workspace_memory_perhaps] The release_workspace flag is set to {release_workspace} based upon "
            "the value of 'workspace_allocated_here'."
        )
        self._free_workspace_memory_perhaps(release_workspace)
        self._workspace_allocated_here = False
        return True

    @utils.precondition(_check_valid_fft)
    @utils.precondition(_check_planned, "Execution")
    @utils.precondition(_check_valid_operand, "Execution")
    @utils.atomic(_release_workspace_memory_perhaps, method=True)
    def execute(
        self,
        *,
        direction: FFTDirection | None = None,
        stream: AnyStream | None = None,
        release_workspace: bool = False,
        sync_symmetric_memory: bool = True,
    ):
        """
        Execute the FFT operation.

        Args:
            direction: {direction}

            stream: {stream}

            release_workspace: A value of `True` specifies that the FFT object
                should release workspace memory back to the symmetric memory pool on
                function return, while a value of `False` specifies that the object
                should retain the memory. This option may be set to `True` if the
                application performs other operations that consume a lot of memory between
                successive calls to the (same or different) :meth:`execute` API, but incurs
                an overhead due to obtaining and releasing workspace memory from and
                to the symmetric memory pool on every call. The default is `False`.
                **NOTE: All processes must use the same value or the application can
                deadlock.**

            sync_symmetric_memory: {sync_symmetric_memory}

        Returns:
            The transformed operand, which remains on the same device and utilizes the same
            package as the input operand. The data type and shape of the transformed operand
            depend on the type of input operand, and choice of distribution and reshape
            option:

            - For C2C FFT, the data type remains identical to the input.
            - For R2C and C2R FFT, the data type differs from the input. The global output
              shape differs from the global input shape, which affects the shape of the
              result on every process.
            - For slab distribution with reshape=True, the shape on this process is the slab
              shape according to the same distribution as the input operand.
            - For slab distribution with reshape=False, the shape on this process is the
              complementary slab shape.
            - For custom box distribution, the shape will depend on the output box of
              each process.

            For GPU operands, the result will be in symmetric memory and the user is
            responsible for explicitly deallocating it (for example, using
            ``nvmath.distributed.free_symmetric_memory(tensor)``).
        """

        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        if direction is None:
            direction = _get_fft_default_direction(self.fft_abstract_type)
        else:
            direction = _get_validate_direction(direction, self.fft_abstract_type)

        stream_holder: StreamHolder = utils.get_or_create_stream(self.device_id, stream, self.internal_op_package)

        # Set stream for the FFT.
        with utils.device_ctx(self.device_id):
            cufft.set_stream(self.handle, stream_holder.ptr)

        # Allocate workspace if needed.
        self._allocate_workspace_memory_perhaps(stream_holder)
        # cuFFTMp only supports inplace transform.
        result_ptr = self.operand.data_ptr

        if log_info:
            self.logger.info(
                f"Starting distributed FFT {self.fft_abstract_type} calculation in the {direction.name} direction..."  # type: ignore[union-attr]
            )
            self.logger.info(f"{self.call_prologue}")

        with utils.cuda_call_ctx(stream_holder, self.blocking, timing=log_info) as (
            self.last_compute_event,
            elapsed,
        ):
            if log_debug:
                self.logger.debug("The cuFFTMp execution function is 'xt_exec_descriptor'.")
            if sync_symmetric_memory:
                nvshmem.sync_all_on_stream(stream_holder.ptr)
                if log_info:
                    self.logger.info(
                        "sync_symmetric_memory is enabled (this may incur redundant multi-GPU "
                        "synchronization, please refer to the documentation for more information)"
                    )
            elif log_info:
                self.logger.info("sync_symmetric_memory is disabled")
            cufft.set_descriptor_data(self.memory_desc, result_ptr, self.subformat)
            cufft.xt_exec_descriptor(self.handle, self.memory_desc, self.memory_desc, direction)
            if self.options.reshape:
                raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
                assert self.reshaped_operand is not None
                cufft.exec_reshape_async(
                    self.reshape_handle, self.reshaped_operand.data_ptr, result_ptr, raw_workspace_ptr, stream_holder.ptr
                )
                # Copy back to original GPU operand.
                self.result_operand.copy_(self.reshaped_operand, stream_holder=stream_holder)

        if log_info and elapsed.data is not None:
            reshape_addendum = "along with output reshaping" if self.options.reshape else ""
            self.logger.info(f"The distributed FFT calculation {reshape_addendum} took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if it's more than the
        # specified cache limit.
        self._free_workspace_memory_perhaps(release_workspace)

        # reset workspace allocation tracking to False at the end of the methods where
        # workspace memory is potentially allocated. This is necessary to prevent any
        # exceptions raised before method entry from using stale tracking values.
        self._workspace_allocated_here = False

        # Return the result.
        if self.memory_space == self.execution_space:
            out = self.result_operand
        else:
            self.cpu_result_operand.copy_(self.result_operand, stream_holder=stream_holder)
            out = self.cpu_result_operand
        return out.tensor

    def free(self):
        """Free FFT resources.

        It is recommended that the :class:`FFT` object be used within a context, but if it
        is not possible then this method must be called explicitly to ensure that the FFT
        resources (especially internal library objects) are properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the
            # computation.
            if self.last_compute_event is not None:
                with utils.device_ctx(self.device_id):
                    self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None

            self._free_workspace_memory()

            with utils.device_ctx(self.device_id):
                if self.memory_desc is not None:
                    if self.dummy_desc_data_ptr is not None:
                        cufft.set_descriptor_data(self.memory_desc, self.dummy_desc_data_ptr, self.subformat)
                    cufft.xt_free(self.memory_desc)
                    self.memory_desc = None

                if self.handle is not None:
                    cufft.destroy(self.handle)
                    if self.options.reshape:
                        cufft.destroy_reshape(self.reshape_handle)
                    self.handle = None
                    self.reshape_handle = None

                if self.memory_desc_handle is not None:
                    cufft.destroy(self.memory_desc_handle)
                    self.memory_desc_handle = None

                if self.memory_space == "cpu" and self.operand is not None:
                    # In this case, self.operand is an internal GPU operand owned by FFT.
                    # Since the execution when user passes CPU operands is blocking, it's
                    # safe to call nvshmem_free here without additional synchronization.
                    self.operand.free_symmetric()
            self.operand = None
            self.operand_backup = None

        except Exception as e:
            self.logger.critical("Internal error: only part of the FFT object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The FFT object's resources have been released.")


def _fft(
    x,
    /,
    *,
    distribution: Distribution | Sequence[Box],
    direction: FFTDirection | None = None,
    sync_symmetric_memory: bool = True,
    options: FFTOptions | None = None,
    stream: AnyStream | None = None,
    check_dtype: str | None = None,
):
    r"""
    fft({function_signature})

    Perform an N-D *complex-to-complex* (C2C) distributed FFT on the provided complex
    operand.

    Args:
        operand: {operand}
            {operand_admonitions}

        distribution: {distribution}

        sync_symmetric_memory: {sync_symmetric_memory}

        options: {options}

        stream: {stream}

    Returns:
        A transformed operand that retains the same data type as the input. The resulting
        shape will depend on the choice of distribution and reshape option. The operand
        remains on the same device and uses the same package as the input operand.

    .. seealso::
        :func:`ifft`, :func:`irfft`, :func:`rfft`, :class:`FFT`

    Examples:

        >>> import cupy as cp
        >>> import nvmath.distributed

        Get MPI communicator used to initialize nvmath.distributed (for information on
        initializing nvmath.distributed, you can refer to the documentation or to the
        FFT examples in `nvmath/examples/distributed/fft
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft>`_):

        >>> comm = nvmath.distributed.get_context().communicator
        >>> nranks = comm.Get_size()

        Create a 3-D complex128 ndarray on GPU symmetric memory, distributed according to
        the Slab distribution on the Y axis (the global shape is (256, 256, 256)):

        >>> from nvmath.distributed.distribution import Slab
        >>> shape = 256, 256 // nranks, 256
        >>> dtype = cp.complex128
        >>> a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=dtype)
        >>> a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(
        ...     *shape, dtype=cp.float64
        ... )

        Perform a 3-D C2C FFT using :func:`fft`. The result `r` is also a CuPy complex128
        ndarray:

        >>> r = nvmath.distributed.fft.fft(a, distribution=Slab.Y)

        See :class:`FFTOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly
        provided to the FFT operation. This can be done if the FFT operand is computed on a
        different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...     a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=dtype)
        ...     a[:] = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
        >>> r = nvmath.distributed.fft.fft(a, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input
        computation.

        Create a NumPy ndarray on the CPU.

        >>> import numpy as np
        >>> b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        Provide the NumPy ndarray to :func:`fft`, with the result also being a NumPy
        ndarray:

        >>> r = nvmath.distributed.fft.fft(b, distribution=Slab.Y)

    Notes:
        - This function only takes complex operand for C2C transformation. If the user
          wishes to perform full FFT transformation on real input, please cast the input to
          the corresponding complex data type.
        - This function is a convenience wrapper around :class:`FFT` and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API using the default `direction` argument in :meth:`FFT.execute`.

    Further examples can be found in the `nvmath/examples/distributed/fft
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft>`_
    directory.
    """
    if check_dtype is not None:
        assert check_dtype in {"real", "complex"}, "internal error"
        operand = tensor_wrapper.wrap_operand(x)
        if ("complex" in operand.dtype) != (check_dtype == "complex"):
            raise ValueError(f"This function expects {check_dtype} operand, found {operand.dtype}")

    with FFT(x, distribution=distribution, options=options, stream=stream) as fftobj:
        # Plan the FFT.
        fftobj.plan(stream=stream)

        # Execute the FFT.
        result = fftobj.execute(direction=direction, stream=stream, sync_symmetric_memory=sync_symmetric_memory)

    return result


# Forward C2C FFT Function.
fft = functools.wraps(_fft)(functools.partial(_fft, direction=FFTDirection.FORWARD, check_dtype="complex"))
fft.__doc__ = fft.__doc__.format(**SHARED_FFT_DOCUMENTATION)  # type: ignore
fft.__name__ = "fft"


# Forward R2C FFT Function
@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
def rfft(
    operand,
    /,
    *,
    distribution: Distribution | Sequence[Box],
    sync_symmetric_memory: bool = True,
    options: FFTOptions | None = None,
    stream: AnyStream | None = None,
):
    r"""
    rfft({function_signature})

    Perform an N-D *real-to-complex* (R2C) distributed FFT on the provided real operand.

    Args:
        operand: {operand}
            {operand_admonitions}

        distribution: {distribution}

        sync_symmetric_memory: {sync_symmetric_memory}

        options: {options}

        stream: {stream}

    Returns:
        A complex tensor whose shape will depend on the choice of distribution and reshape
        option. The operand remains on the same device and belongs to the same package as
        the input operand. The global extent of the last transformed axis in the result will
        be ``global_extent[-1] // 2 + 1``.

    .. seealso::
        :func:`fft`, :func:`irfft`, :class:`FFT`.
    """
    wrapped_operand = tensor_wrapper.wrap_operand(operand)
    # check if input operand if real type
    if "complex" in wrapped_operand.dtype:
        raise RuntimeError(f"rfft expects a real input, but got {wrapped_operand.dtype}. Please use fft for complex input.")

    return _fft(
        operand,
        distribution=distribution,
        sync_symmetric_memory=sync_symmetric_memory,
        options=options,
        stream=stream,
        check_dtype="real",
    )


# Inverse C2C FFT Function.
ifft = functools.wraps(_fft)(functools.partial(_fft, direction=FFTDirection.INVERSE, check_dtype="complex"))
ifft.__doc__ = """
    ifft({function_signature})

    Perform an N-D *complex-to-complex* (C2C) inverse FFT on the provided complex operand.
    The direction is implicitly inverse.

    Args:
        operand: {operand}
            {operand_admonitions}

        distribution: {distribution}

        sync_symmetric_memory: {sync_symmetric_memory}

        options: {options}

        stream: {stream}

    Returns:
        A transformed operand that retains the same data type as the input. The resulting
        shape will depend on the choice of distribution and reshape option. The operand
        remains on the same device and uses the same package as the input operand.

    .. seealso::
        :func:`fft`, :func:`irfft`, :class:`FFT`.

    Notes:
        - This function only takes complex operand for C2C transformation. If the user wishes
          to perform full FFT transformation on real input, please cast the input to the
          corresponding complex data type.
        - This function is a convenience wrapper around :class:`FFT` and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API by passing the argument ``direction='inverse'`` when calling
          :meth:`FFT.execute`.
""".format(**SHARED_FFT_DOCUMENTATION)
ifft.__name__ = "ifft"


# Inverse C2R FFT Function.
@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
def irfft(
    operand,
    /,
    *,
    distribution: Distribution | Sequence[Box],
    sync_symmetric_memory: bool = True,
    options: FFTOptions | None = None,
    stream: AnyStream | None = None,
):
    """
    irfft({function_signature})

    Perform an N-D *complex-to-real* (C2R) distributed FFT on the provided complex operand.
    The direction is implicitly inverse.

    Args:
        operand: {operand}
            {operand_admonitions}

        distribution: {distribution}

        sync_symmetric_memory: {sync_symmetric_memory}

        options: {options}

        stream: {stream}

    Returns:
        A real tensor whose shape will depend on the choice of distribution and reshape
        option. The operand remains on the same device and belongs to the same package as
        the input operand. The global extent of the last transformed axis in the result
        will be ``(global_extent[-1] - 1) * 2`` if :attr:`FFTOptions.last_axis_parity` is
        ``even``, or ``global_extent[-1] * 2 - 1`` if :attr:`FFTOptions.last_axis_parity`
        is ``odd``.

    .. seealso::
        :func:`fft`, :func:`ifft`, :class:`FFT`.

    Example:

        >>> import cupy as cp
        >>> import nvmath.distributed

        Get MPI communicator used to initialize nvmath.distributed (for information on
        initializing nvmath.distributed, you can refer to the documentation or to the
        FFT examples in `nvmath/examples/distributed/fft
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft>`_):

        >>> comm = nvmath.distributed.get_context().communicator
        >>> nranks = comm.Get_size()
        >>> from nvmath.distributed.fft import Slab

        Create a 3-D symmetric complex128 ndarray on GPU symmetric memory:

        >>> shape = 512 // nranks, 768, 256
        >>> a = nvmath.distributed.allocate_operand(
        ...     shape, cp, input_dtype=cp.float64, distribution=Slab.X, fft_type="R2C"
        ... )
        >>> a[:] = cp.random.rand(*shape, dtype=cp.float64)
        >>> b = nvmath.distributed.fft.rfft(a, distribution=Slab.X)

        Perform a 3-D C2R FFT using the :func:`irfft` wrapper. The result `r` is a CuPy
        float64 ndarray:

        >>> r = nvmath.distributed.fft.irfft(b, distribution=Slab.X)
        >>> r.dtype
        dtype('float64')

    Notes:

        - This function performs an inverse C2R N-D FFT, which is similar to `irfftn` but
          different from `irfft` in various numerical packages.
        - This function is a convenience wrapper around :class:`FFT` and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API by setting :attr:`FFTOptions.fft_type` to ``'C2R'`` and passing the argument
          ``direction='inverse'`` when calling :meth:`FFT.execute`.
        - **The input to this function must be Hermitian-symmetric, otherwise the result is
          undefined.** While the symmetry requirement is partially captured by the different
          global extents in the last transformed dimension between the input and result,
          there are additional `constraints
          <https://docs.nvidia.com/cuda/cufft/#fourier-transform-types>`_. In addition,
          if the input to `irfft` was generated using an R2C FFT with an odd global last
          axis size, :attr:`FFTOptions.last_axis_parity` must be set to ``odd`` to recover
          the original signal.
        - For more details, please refer to `R2C/C2R example
          <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft/example01_cupy_r2c_c2r.py>`_
          and `odd C2R example
          <https://github.com/NVIDIA/nvmath-python/tree/main/examples/distributed/fft/example01_torch_r2c_c2r.py>`_.
    """
    options = cast(FFTOptions, utils.check_or_create_options(FFTOptions, options, "Distributed FFT options"))
    options.fft_type = "C2R"
    return _fft(
        operand,
        distribution=distribution,
        direction=FFTDirection.INVERSE,
        sync_symmetric_memory=sync_symmetric_memory,
        options=options,
        stream=stream,
        check_dtype="complex",
    )
