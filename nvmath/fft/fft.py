# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FFT", "fft", "ifft", "rfft", "irfft", "UnsupportedLayoutError"]

from typing import Literal
from collections.abc import Sequence
from dataclasses import dataclass, astuple as data_cls_astuple
import enum
import functools
import logging
import math
import operator

from ._configuration import ExecutionCPU, ExecutionCUDA, FFTOptions, FFTDirection, DeviceCallable

from nvmath.bindings import cufft  # type: ignore

try:
    from nvmath.bindings.nvpl import fft as fftw  # type: ignore
except ImportError:
    fftw = None  # type: ignore
from nvmath.bindings._internal import utils as _bindings_utils  # type: ignore
from nvmath.fft._exec_utils import _cross_setup_execution_and_options
from nvmath import memory

from nvmath.internal import formatters
from nvmath.internal import tensor_wrapper
from nvmath.internal.typemaps import (
    NAME_TO_DATA_TYPE,
    DATA_TYPE_TO_NAME,
    FFTW_SUPPORTED_SINGLE,
    FFTW_SUPPORTED_DOUBLE,
    FFTW_SUPPORTED_TYPES,
    FFTW_SUPPORTED_FLOAT,
    FFTW_SUPPORTED_COMPLEX,
)
from nvmath.internal import utils
from nvmath._internal.layout import is_contiguous_layout, is_contiguous_in_memory, is_overlapping_layout
from nvmath.internal.package_wrapper import AnyStream, StreamHolder


class UnsupportedLayoutError(Exception):
    """
    Error type for layouts not supported by the library.

    Args:
        message: The error message.

        permutation: The permutation needed to convert the input layout to a supported
            layout to the FFT operation. The same permutation needs to be applied to the
            result to obtain the axis sequence corresponding to the non-permuted input.

        axes: The dimensions along which the FFT is performed corresponding to the permuted
            operand layout.
    """

    def __init__(self, message, permutation, axes):
        self.message = message
        self.permutation = permutation
        self.axes = axes

    def __str__(self):
        return self.message


@dataclass
class TensorLayout:
    """An internal data class for capturing the tensor layout."""

    shape: Sequence[int]
    strides: Sequence[int]


@dataclass
class PlanTraits:
    """An internal data class for capturing FFT plan traits."""

    result_shape: Sequence[int]
    result_strides: Sequence[int]
    ordered_axes: Sequence[int]
    ordered_fft_in_shape: Sequence[int]
    ordered_fft_in_embedding_shape: Sequence[int]
    ordered_fft_out_shape: Sequence[int]
    fft_batch_size: int
    istride: int
    idistance: int
    ostride: int
    odistance: int


class CBLoadType(enum.IntEnum):
    COMPLEX64 = (0x0,)
    COMPLEX128 = (0x1,)
    FLOAT32 = (0x2,)
    FLOAT64 = (0x3,)
    UNDEFINED = 0x8


class CBStoreType(enum.IntEnum):
    COMPLEX64 = (0x4,)
    COMPLEX128 = (0x5,)
    FLOAT32 = (0x6,)
    FLOAT64 = (0x7,)
    UNDEFINED = 0x8


SHARED_FFT_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_FFT_DOCUMENTATION.update(
    {
        "axes": """\
The dimensions along which the FFT is performed. ``axes[-1]`` is the 'last transformed' axis for rffts. Currently, it is
required that the axes are contiguous and include the first or the last dimension. Only up to 3D FFTs are
supported.""".replace("\n", " "),
        #
        "options": """\
Specify options for the FFT as a :class:`FFTOptions` object. Alternatively, a `dict` containing the parameters for the
``FFTOptions`` constructor can also be provided. If not specified, the value will be set to the default-constructed
``FFTOptions`` object.""".replace("\n", " "),
        #
        "execution": """\
Specify execution space options for the FFT as a :class:`ExecutionCUDA` or :class:`ExecutionCPU` object. Alternatively,
a string ('cuda' or 'cpu'), or a `dict` with the 'name' key set to 'cpu' or 'cuda' and optional parameters relevant to
the given execution space. If not specified, the execution space will be selected to match operand's storage (in GPU or
host memory), and the corresponding :class:`ExecutionCUDA` or :class:`ExecutionCPU` object will be
default-constructed.""".replace("\n", " "),
        #
        "prolog": """\
Provide device-callable function in LTO-IR format to use as load-callback as an object of type :class:`DeviceCallable`.
Alternatively, a `dict` containing the parameters for the ``DeviceCallable`` constructor can also be provided. The
default is no prolog. Currently, callbacks are supported only with CUDA execution.""".replace("\n", " "),
        #
        "epilog": """\
Provide device-callable function in LTO-IR format to use as store-callback as an object of type :class:`DeviceCallable`.
Alternatively, a `dict` containing the parameters for the ``DeviceCallable`` constructor can also be provided. The
default is no epilog. Currently, callbacks are supported only with CUDA execution.""".replace("\n", " "),
        #
        "direction": """\
Specify whether forward or inverse FFT is performed (:class:`FFTDirection` object, or as a string from ['forward',
'inverse'], "or as an int from [-1, 1] denoting forward and inverse directions respectively).""".replace("\n", " "),
        #
        "fft_key": """\
A tuple as the key to represent the input FFT problem.""".replace("\n", " "),
        #
        "function_signature": """\
operand,
axes: Sequence[int] | None = None,
options: FFTOptions | None = None,
execution: ExecutionCPU | ExecutionCUDA | None = None,
prolog: DeviceCallable | None = None,
epilog: DeviceCallable | None = None,
stream: AnyStream | None = None
""".replace("\n", " "),
    }
)


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


def _get_default_fft_abstract_type(dtype, fft_type):
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


def _get_fft_default_direction(fft_abstract_type):
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


def _get_size(shape):
    return functools.reduce(operator.mul, shape, 1)


def _get_last_axis_id_and_size(
    axes: Sequence[int],
    operand_shape: Sequence[int],
    fft_abstract_type: Literal["C2C", "C2R", "R2C"],
    last_axis_parity: Literal["even", "odd"],
) -> tuple[int, int]:
    """
    Args:
        axes: The user-specified or default FFT axes.

        operand_shape: The input operand shape.

        fft_abstract_type: The "abstract" type of the FFT ('C2C', 'C2R', 'R2C').

        last_axis_parity: For 'C2R' FFTs, specify whether the last axis size is even or odd.

    Returns the last axis ID and the corresponding axis size required for the result.
    """
    last_axis_id = axes[-1]

    if fft_abstract_type == "C2C":
        return last_axis_id, operand_shape[last_axis_id]

    if fft_abstract_type == "C2R":
        if last_axis_parity == "even":
            return last_axis_id, 2 * (operand_shape[last_axis_id] - 1)
        elif last_axis_parity == "odd":
            return last_axis_id, 2 * operand_shape[last_axis_id] - 1
        else:
            raise AssertionError("Unreachable.")

    if fft_abstract_type == "R2C":
        return last_axis_id, operand_shape[last_axis_id] // 2 + 1


def check_inplace_overlapping_layout(operand: utils.TensorHolder):
    if is_overlapping_layout(operand.shape, operand.strides):
        raise ValueError(
            f"In-place transform is not supported because the tensor with shape "
            f"{operand.shape} and strides {operand.strides} overlaps in memory."
        )


def check_embedding_possible(strides, presorted=False):
    """
    Check if the strides allow for calculating an embedding dimension.
    """
    if not presorted:
        strides = sorted(strides)
    # with a broadcasted view, stride can be 0
    if any(strides[i - 1] == 0 for i in range(1, len(strides))):
        return False
    return all(strides[i] % strides[i - 1] == 0 for i in range(1, len(strides)))


def check_batch_tileable(sorted_batch_shape, sorted_batch_strides):
    """
    Check if FFT layout is tileable across the specified batch layout.
    """
    return is_contiguous_layout(sorted_batch_shape, sorted_batch_strides)


def check_contiguous_layout(axes, strides, shape):
    if not axes:
        return True
    sorted_batch_strides, sorted_batch_shape = zip(*sorted((strides[a], shape[a]) for a in axes), strict=True)
    return is_contiguous_layout(sorted_batch_shape, sorted_batch_strides)


def calculate_embedding_shape(shape: Sequence[int], strides: Sequence[int]):
    """
    Calculate the embedding shape for the given shape and strides.
    """
    n = len(strides)
    # The shape is used to resolve cases like (1, 2, 1) : (2, 1, 1) in CuTe notation.
    ordered_strides, _, order = zip(*sorted(zip(strides, shape, range(n), strict=True)), strict=True)

    ordered_shape = [ordered_strides[i] // ordered_strides[i - 1] for i in range(1, len(ordered_strides))] + [shape[order[-1]]]

    embedding_shape = [0] * n
    for o in range(n):
        embedding_shape[order[o]] = ordered_shape[o]

    return embedding_shape, order


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


def unsupported_layout_exception(operand_dim, axes, message, logger):
    logger.error(message)

    permutation = tuple(a for a in range(operand_dim) if a not in axes) + tuple(axes)
    fft_dim = len(axes)
    axes = tuple(range(operand_dim - fft_dim, operand_dim))

    message = (
        f"To convert to a supported layout, create a transposed view using transpose{permutation} and copy the "
        f"view into a new tensor, using view.copy() for instance, and use axes={axes}."
    )
    logger.error(message)

    raise UnsupportedLayoutError(message, permutation, axes)


def get_null_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    return logger


def get_fft_plan_traits(
    operand_shape: Sequence[int],
    operand_strides: Sequence[int],
    operand_dtype,
    axes: Sequence[int],
    execution: ExecutionCUDA | ExecutionCPU,
    *,
    fft_abstract_type: Literal["C2C", "C2R", "R2C"] = "C2C",
    last_axis_parity: Literal["even", "odd"] = "even",
    result_layout: Literal["optimized", "natural"] = "optimized",
    logger: logging.Logger | None = None,
) -> PlanTraits:
    """
    Extract the FFT shape from the operand shape, compute the ordered axes so that the data
    is C-contiguous in memory, and compute the result shape and strides.

    Args:
        operand_shape: The operand shape

        operand_strides: The operand strides

        axes: The axes over which the FFT is performed. For R2C and C2R transforms, the size
            of the last axis in `axes` will change.

        execution: The execution options, an instance of either ExecutionCUDA or
            ExecutionCPU class.

        fft_abstract_type: The "abstract" type of the FFT ('C2C', 'C2R', 'R2C').

        last_axis_parity: For 'C2R' FFTs, specify whether the last axis size is even or odd.

    The data needed for creating a cuFFT plan is returned in the following order:
    (result_shape, result_strides), ordered_axes, ordered_fft_in_shape,
    ordered_fft_out_shape, (istride, idistance), (ostride, odistance)
    """
    logger = logger if logger is not None else get_null_logger("get_fft_plan_traits_null")

    if len(axes) > 3:
        raise ValueError(
            "Only up to 3D FFTs are currently supported. You can use the 'axes' option to specify up to three axes "
            f"along which to perform the FFT. The current number of dimensions is {len(axes)} corresponding to the "
            f"axes {axes}."
        )

    # Check for duplicate axis IDs.
    if len(axes) != len(set(axes)):
        raise ValueError(f"The specified FFT axes = {axes} contains duplicate axis IDs, which is not supported.")

    operand_dim = len(operand_shape)
    batch_axes = [axis for axis in range(operand_dim) if axis not in axes]

    # Check if an embedding is possible for the provided operand layout.
    if not check_embedding_possible(operand_strides):
        message = (
            f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} is "
            "not currently supported because it does not have a suitable embedding dimension."
        )
        unsupported_layout_exception(operand_dim, axes, message, logger)

    # Compute the embedding shape for the operand.
    operand_embedding_shape, axis_order = calculate_embedding_shape(operand_shape, operand_strides)
    logger.debug(f"The operand embedding shape = {operand_embedding_shape}.")

    # The first or the last *ordered* axis must be present in the specified axes to be able
    # to use the "advanced" layout.
    first, last = axis_order[-1], axis_order[0]
    if first not in axes and last not in axes:
        raise ValueError(
            f"The first ({first}) or the last ({last}) tensor axis in stride order {axis_order} must be present in the "
            f"specified FFT axes {axes}."
        )

    # Compute the embedding input shape for the FFT.
    fft_in_embedding_shape = [operand_embedding_shape[a] for a in axes]

    # Compute the input shape for the FFT.
    fft_in_shape, fft_in_strides = zip(*[(operand_shape[a], operand_strides[a]) for a in axes], strict=True)
    if not is_contiguous_in_memory(fft_in_embedding_shape, fft_in_strides):
        message = (
            f"The FFT axes {axes} cannot be reordered so that the data is contiguous in memory for "
            f"operand shape = {operand_shape} and operand strides = {operand_strides}."
        )
        unsupported_layout_exception(operand_dim, axes, message, logger)

    # Reorder the FFT axes and input shape so that they are contiguous or separated by
    # constant stride in memory.
    quadruple = sorted(
        zip(fft_in_strides, fft_in_shape, fft_in_embedding_shape, axes, strict=True), key=lambda v: v[:2], reverse=True
    )

    ordered_in_strides, ordered_fft_in_shape, ordered_fft_in_embedding_shape, ordered_axes = zip(*quadruple, strict=True)

    # Check if R2C and C2R can be supported without copying.
    if fft_abstract_type in ["R2C", "C2R"] and ordered_axes[-1] != axes[-1]:
        message = (
            f"The last FFT axis specified ({axes[-1]}) must have the smallest stride of all the FFT axes' "
            f"strides {fft_in_strides} for FFT type '{fft_abstract_type}'."
        )
        unsupported_layout_exception(operand_dim, axes, message, logger)

    # Input FFT size and batch size.
    fft_in_size = _get_size(fft_in_shape)
    if fft_in_size == 0:
        raise ValueError("Invalid number of FFT data points (0) specified.")
    fft_batch_size = _get_size(operand_shape) // fft_in_size

    # Output FFT (ordered) shape and size.
    last_axis_id, last_axis_size = _get_last_axis_id_and_size(axes, operand_shape, fft_abstract_type, last_axis_parity)
    if last_axis_size == 0:
        raise ValueError(
            f"The size of the last FFT axis in the result for FFT type '{fft_abstract_type}' is 0 for operand shape = "
            f"{operand_shape} and axes = {axes}. To fix this, provide 'last_axis_parity' = 'odd' to the FFT options."
        )
    ordered_fft_out_shape = list(ordered_fft_in_shape)
    index = ordered_axes.index(last_axis_id)
    ordered_fft_out_shape[index] = last_axis_size
    fft_out_size = _get_size(ordered_fft_out_shape)

    # Check that batch dimensions are tileable, as required by the "advanced" layout.
    sorted_batch_shape: Sequence[int] = []
    sorted_batch_strides: Sequence[int] = []
    if batch_axes:
        sorted_batch_strides, sorted_batch_shape = zip(
            *sorted((operand_strides[a], operand_shape[a]) for a in batch_axes), strict=True
        )
        if not check_embedding_possible(sorted_batch_strides, presorted=True):
            raise ValueError(
                f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} "
                f"together with the specified axes = {axes} is currently not supported because it is not tileable."
            )
        logger.debug(f"The sorted batch shape is {sorted_batch_shape}.")
        logger.debug(f"The sorted batch strides are {sorted_batch_strides}.")
    if not check_batch_tileable(sorted_batch_shape, sorted_batch_strides):
        message = (
            f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} "
            f"together with the specified axes = {axes} is currently not supported because it is not tileable."
        )
        unsupported_layout_exception(operand_dim, axes, message, logger)
    logger.debug(
        f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} together with "
        f"the specified axes = {axes} IS tileable."
    )

    # The result tensor has updated shape for R2C and C2R transforms.
    result_shape = list(operand_shape)
    result_shape[last_axis_id] = last_axis_size

    # The result tensor layout is either natural or chosen for optimal cuFFT performance,
    # based on the operand layout and user-provided option.

    # We can keep the input's layout (i.e. operand's extents order of increasing strides)
    # without performance hit, if the samples do not interleave.
    # Otherwise, we try to keep it only when explicitly asked (result_layout=natural)
    is_sample_interleaved = sorted_batch_strides and sorted_batch_strides[0] <= ordered_in_strides[0]
    logger.debug(f"Are the samples interleaved? {bool(is_sample_interleaved)}.")

    if not is_sample_interleaved or result_layout == "natural":  # Natural (== operand) layout.
        axis_order = axis_order_in_memory(operand_shape, operand_strides)
        result_strides = calculate_strides(result_shape, axis_order)
        # If the resulting output operand is not tilable, keeping the original layout is not
        # possible. If `not is_sample_interleaved` the batch must be tilable, because the
        # min batch stride is bigger than max fft stride
        if is_sample_interleaved:
            if not check_contiguous_layout(batch_axes, result_strides, result_shape):
                message = (
                    f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} "
                    f"together with the specified axes = {axes} is currently not supported with "
                    "result_layout='natural', because the output batch would not be tileable."
                )
                unsupported_layout_exception(operand_dim, axes, message, logger)
            if not check_contiguous_layout(axes, result_strides, result_shape):
                message = (
                    f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} "
                    f"together with the specified axes = {axes} is currently not supported with "
                    "result_layout='natural', because the output sample would be non-contiguous."
                )
                unsupported_layout_exception(operand_dim, axes, message, logger)
    else:  # Optimized layout.
        axis_order = tuple(
            list(reversed(ordered_axes)) + sorted((a for a in batch_axes), key=lambda v: (operand_strides[v], operand_shape[v]))
        )
        result_strides = calculate_strides(result_shape, axis_order)
    logger.debug(f"The result layout is '{result_layout}' with the result_strides {result_strides}.")

    # Compute the operand linear stride and distance needed for the cuFFT plan.
    last_ordered_in_stride = ordered_in_strides[-1]
    min_in_stride = min(operand_strides)

    if last_ordered_in_stride == min_in_stride:
        istride, idistance = (
            min_in_stride,
            _get_size(ordered_fft_in_embedding_shape) if not sorted_batch_strides else sorted_batch_strides[0],
        )
    else:
        istride, idistance = (
            last_ordered_in_stride,
            min_in_stride if not sorted_batch_strides else sorted_batch_strides[0],
        )

    # Compute the result linear stride and distance needed for the cuFFT plan.
    ostride = result_strides[ordered_axes[-1]]  #  minimal output fft stride
    odistance = fft_out_size if not batch_axes else min(result_strides[axis] for axis in batch_axes)

    if execution.name == "cpu":
        if fft_out_size == 1:
            istride = ostride = 1
    else:
        assert execution.name == "cuda"
        if operand_dtype in ("float16", "complex32"):
            if fft_abstract_type == "R2C" and istride != 1:
                raise ValueError(
                    f"The {fft_abstract_type} FFT of half-precision tensor ({operand_dtype}) "
                    f"is currently not supported for strided inputs "
                    f"(got input stride {istride})."
                )
            if fft_abstract_type == "C2R" and ostride != 1:
                raise ValueError(
                    f"The {fft_abstract_type} FFT of half-precision tensor ({operand_dtype}) "
                    f"is currently not supported for strided outputs "
                    f"(got output stride {ostride})."
                )
            if fft_out_size == 1:
                if cufft.get_version() < 10702:  # 10702 is shipped with CTK 11.7
                    raise ValueError(
                        f"The FFT of sample size 1 and half-precision type ({operand_dtype}) "
                        f"of size 1 is not supported by the installed cuFFT version. "
                    )
                # There is a bug that leads to invalid memory access (CTK 12.1) for
                # one-element, strided C2C complex32 tensors (either in the input or output)
                # or results in CUFFT_INVALID_SIZE (CTK 12.3). This workaround relies on the
                # fact that the [i|o]stride effectively does not matter in a one-element
                # sample.
                elif fft_abstract_type == "C2C":
                    istride = ostride = 1

        # There's a bug in cuFFT in CTKs prior to 11.4U2
        if len(axes) == 3 and fft_batch_size > 1 and cufft.get_version() < 10502:
            raise ValueError(
                "The 3D batched FFT is not supported by the installed cuFFT version. "
                "Please update your CUDA Toolkit (to 11.4.2 or newer)"
            )

    plan_traits = PlanTraits(
        result_shape=tuple(result_shape),
        result_strides=tuple(result_strides),
        ordered_axes=tuple(ordered_axes),
        ordered_fft_in_shape=tuple(ordered_fft_in_shape),
        ordered_fft_in_embedding_shape=tuple(ordered_fft_in_embedding_shape),
        ordered_fft_out_shape=tuple(ordered_fft_out_shape),
        fft_batch_size=fft_batch_size,
        istride=istride,
        idistance=idistance,
        ostride=ostride,
        odistance=odistance,
    )
    return plan_traits


def _copy_operand_perhaps(
    internal_operand,
    operand: utils.TensorHolder,
    stream_holder,
    execution_space,
    memory_space,
    device_id: int | Literal["cpu"],
    fft_abstract_type,
    logger,
):
    if execution_space == memory_space:
        if fft_abstract_type != "C2R":
            return operand, None
        else:
            # For C2R, we need to take a copy to avoid input being overwritten
            logger.info("For C2R FFT with input operand on GPU, the input is copied to avoid being overwritten by cuFFT.")
            operand_copy = utils.create_empty_tensor(
                operand.__class__,
                operand.shape,
                operand.dtype,
                device_id,
                stream_holder,
                verify_strides=True,
                strides=operand.strides,
            )
            operand_copy.copy_(operand, stream_holder=stream_holder)
            # We don't need to keep the operand backup, because C2R precludes `inplace=True`
            return operand_copy, None
    else:
        # Copy the `operand` to memory that matches the exec space
        # and keep the original `operand` to handle `options.inplace=True`
        if internal_operand is None:
            if execution_space == "cuda":
                assert isinstance(device_id, int)
                to_device: int | Literal["cpu"] = device_id
            else:
                assert execution_space == "cpu"
                to_device = "cpu"
            exec_space_copy = operand.to(to_device, stream_holder)
            return exec_space_copy, operand
        else:
            # In-place copy to existing pointer
            tensor_wrapper.copy_([operand], [internal_operand], stream_holder)
            return internal_operand, operand


def create_xt_plan_args(*, plan_traits=None, fft_abstract_type=None, operand_data_type=None, inplace=None):
    """
    Create the arguments to xt_make_plan_many() except for the handle. This is also used for
    computing the FFT key.
    """
    assert plan_traits is not None, "Internal error."
    assert fft_abstract_type is not None, "Internal error."
    assert operand_data_type is not None, "Internal error."
    assert inplace is not None, "Internal error."

    result_data_type, compute_data_type = _get_fft_result_and_compute_types(operand_data_type, fft_abstract_type)

    # The input shape to the plan should be the logical FFT shape.
    ordered_plan_shape = plan_traits.ordered_fft_out_shape if fft_abstract_type == "C2R" else plan_traits.ordered_fft_in_shape

    # Handle in-place transforms.
    if inplace:
        ordered_fft_out_shape, ostride, odistance = (
            plan_traits.ordered_fft_in_embedding_shape,
            plan_traits.istride,
            plan_traits.idistance,
        )
    else:
        ordered_fft_out_shape, ostride, odistance = (
            plan_traits.ordered_fft_out_shape,
            plan_traits.ostride,
            plan_traits.odistance,
        )

    return (
        len(ordered_plan_shape),
        ordered_plan_shape,
        plan_traits.ordered_fft_in_embedding_shape,
        plan_traits.istride,
        plan_traits.idistance,
        NAME_TO_DATA_TYPE[operand_data_type],
        ordered_fft_out_shape,
        ostride,
        odistance,
        NAME_TO_DATA_TYPE[result_data_type],
        plan_traits.fft_batch_size,
        NAME_TO_DATA_TYPE[compute_data_type],
    )


def fftw_plan_args(xt_plan_args, operand_ptr, result_ptr, fft_abstract_type, direction):
    """
    Create the arguments for fftw API based on the args created by create_xt_plan_args and
    pointers to the input and the output tensors. Note, that while the pointers to the data
    are required in planning, different pointers may be passed to the same plan in
    subsequent execute call (assuming dtype, memory layout, alignment, and inplace
    properties do not change).
    """
    (
        rank,
        n,
        inembed,
        istride,
        idist,
        input_t,
        onembed,
        ostride,
        odist,
        output_t,
        batch,
        execution_t,
    ) = xt_plan_args

    if input_t in FFTW_SUPPORTED_SINGLE:
        assert output_t in FFTW_SUPPORTED_SINGLE, "Expected single precision output for single precision input"
        precision = fftw.Precision.FLOAT
    elif input_t in FFTW_SUPPORTED_DOUBLE:
        assert output_t in FFTW_SUPPORTED_DOUBLE, "Expected double precision output for double precision input"
        precision = fftw.Precision.DOUBLE
    else:
        supported_types_str = ", ".join(DATA_TYPE_TO_NAME[dtype] for dtype in FFTW_SUPPORTED_TYPES)
        raise ValueError(
            f"Currently, the CPU FFT supports following input types: {supported_types_str}. "
            f"Got input of type {DATA_TYPE_TO_NAME[input_t]}."
        )
    kind = getattr(fftw.Kind, fft_abstract_type)
    if kind == fftw.Kind.C2C:
        supported_in_t, supported_out_t = FFTW_SUPPORTED_COMPLEX, FFTW_SUPPORTED_COMPLEX
    elif kind == fftw.Kind.C2R:
        supported_in_t, supported_out_t = FFTW_SUPPORTED_COMPLEX, FFTW_SUPPORTED_FLOAT
    else:
        assert kind == fftw.Kind.R2C
        supported_in_t, supported_out_t = FFTW_SUPPORTED_FLOAT, FFTW_SUPPORTED_COMPLEX
    if input_t not in supported_in_t:
        supported_types_str = ", ".join(DATA_TYPE_TO_NAME[dtype] for dtype in supported_in_t)
        raise ValueError(
            f"Got unsupported input data type {DATA_TYPE_TO_NAME[input_t]} "
            f"for the {fft_abstract_type} transform. "
            f"Supported types are {supported_types_str}."
        )
    assert output_t in supported_out_t, "Mismatched data type and FFT transform type"
    if direction is None:
        sign = fftw.Sign.UNSPECIFIED
    else:
        sign = fftw.Sign(direction)
    return (
        precision,
        kind,
        sign,
        rank,
        n,
        batch,
        operand_ptr,
        inembed,
        istride,
        idist,
        result_ptr,
        onembed,
        ostride,
        odist,
        fftw.PlannerFlags.ESTIMATE,
    )


def setup_options(operand: utils.TensorHolder, options, execution) -> tuple[FFTOptions, ExecutionCUDA | ExecutionCPU]:
    default_exec_space = operand.device
    execution = utils.check_or_create_one_of_options(
        (ExecutionCUDA, ExecutionCPU),
        execution,
        "'execution' options",
        default_name=default_exec_space,
    )
    # Process options.
    options = utils.check_or_create_options(FFTOptions, options, "FFT options")
    return _cross_setup_execution_and_options(options, execution)


def create_fft_key(
    operand,
    *,
    axes: Sequence[int] | None = None,
    options: FFTOptions | None = None,
    execution: ExecutionCPU | ExecutionCUDA | None = None,
    inplace=None,
    prolog: DeviceCallable | None = None,
    epilog: DeviceCallable | None = None,
    plan_args=None,
):
    """
    This key is not designed to be serialized and used on a different machine. It is meant
    for runtime use only. We use a specific inplace argument instead of taking it from
    options, because self.inplace != self.options.inplace for CPU tensors for efficiency.

    It is the user's responsibility to augment this key with the stream in case they use
    stream-ordered memory pools.
    """
    if plan_args is None:
        operand = tensor_wrapper.wrap_operand(operand)
        options, execution = setup_options(operand, options, execution)
        fft_abstract_type = _get_default_fft_abstract_type(operand.dtype, options.fft_type)
        if axes is None:
            axes = range(len(operand.shape))

        # Determine plan traits.
        plan_traits = get_fft_plan_traits(
            operand.shape,
            operand.strides,
            operand.dtype,
            axes,
            execution,
            fft_abstract_type=fft_abstract_type,
            last_axis_parity=options.last_axis_parity,
            result_layout=options.result_layout,
            logger=None,
        )

        # Inplace is always True when execution space is different than the operand's memory
        # space (as the operand needs to be copied once anyway)
        if inplace is None:
            memory_space = operand.device
            execution_space = execution.name
            assert execution.name in ("cpu", "cuda")
            inplace = memory_space != execution_space or options.inplace

        if inplace:
            check_inplace_overlapping_layout(operand)

        # Get the arguments to xt_make_plan_many.
        plan_args = create_xt_plan_args(
            plan_traits=plan_traits,
            fft_abstract_type=fft_abstract_type,
            operand_data_type=operand.dtype,
            inplace=inplace,
        )

    # Prolog and epilog, if used.
    if prolog is not None or epilog is not None:
        prolog = utils.check_or_create_options(DeviceCallable, prolog, "prolog", keep_none=True)
        epilog = utils.check_or_create_options(DeviceCallable, epilog, "epilog", keep_none=True)

        def get_data(device_callable):
            return None if device_callable is None else (device_callable.ltoir, device_callable.data)

        callable_data = get_data(prolog), get_data(epilog)
    else:
        callable_data = None

    # The key is based on plan arguments, callback data (a callable object of type
    # DeviceCallback or None) and the execution options (in "normalized" form of
    # ("cpu"/"cuda", *execution_options)).
    return plan_args, callable_data, data_cls_astuple(execution)  # type: ignore[arg-type]


def _has_only_small_factors_extent(extent):
    # fast track for powers of 2 (and zero)
    if extent & (extent - 1) == 0:
        return True
    # Divide the `extent` by the product of all the prime factors up to
    # 127 present in the `extent` until there are none left.
    # Considering all the prime factors at once is faster, even though the
    # first call (and only the first call) to gcd operates on ints with precision
    # exceeding 64 bits. For common 2, 3, 5, 7 factors, the higher powers are included,
    # to reduce number of iterations.
    # math.prod(p for p in range(2, 128) if is_prime(p)) * 2**10 * 3**7 * 5**5 * 7**4
    magic_prod = 67455891904760197438286248026720562156610454525830430432000000
    d = math.gcd(extent, magic_prod)
    while d > 1:
        if extent == d:
            return True
        extent //= d
        d = math.gcd(extent, d)
    return False


def _has_only_small_factors_shape(shape):
    return all(extent <= 2048 or _has_only_small_factors_extent(extent) for extent in shape)


def check_is_shape_supported_lto_ea(operand, plan_traits, fft_abstract_type):
    if fft_abstract_type != "C2R":
        shape = plan_traits.ordered_fft_in_shape
    else:
        shape = plan_traits.ordered_fft_out_shape
    if not _has_only_small_factors_shape(shape):
        raise ValueError(
            f"cuFFT LTO EA does not support callbacks with inputs of certain shapes. "
            f"Tensor with extents comprasing prime factors larger than 127 are not supported. "
            f"Got a tensor of shape {operand.shape}."
        )
    if len(shape) == 3 and sum(e == 1 for e in shape) == 1 and shape[-1] == 1:
        raise ValueError(
            "cuFFT LTO EA does not support callbacks with inputs of certain shapes. "
            "3D FFT with the last extent equal 1 are not supported"
        )


def _check_prolog_epilog_traits(prolog, epilog, plan_traits, operand, fft_abstract_type):
    # Since the version 11300, cufft does the validation itself.
    # In earlier versions, it could ignore the callback silently
    # for unsupported shapes
    if (prolog or epilog) and cufft.get_version() < 11300:
        check_is_shape_supported_lto_ea(operand, plan_traits, fft_abstract_type)


def set_prolog_and_epilog(handle, prolog, epilog, operand_dtype, result_dtype, logger):
    def set_callback(cbkind, cbobj, dtype):
        if cbobj is None:
            return

        assert cbkind in ["prolog", "epilog"], "Internal error."
        CBType = CBLoadType if cbkind == "prolog" else CBStoreType

        try:
            cufft.xt_set_jit_callback(handle, cbobj.ltoir, cbobj.size, CBType[dtype.upper()], [cbobj.data])
        except _bindings_utils.FunctionNotFoundError as e:
            version = cufft.get_version()
            raise RuntimeError(
                f"The currently running cuFFT version {version} does not support LTO callbacks. \n"
                f"The following cuFFT releases support LTO callbacks: \n"
                f"1. cuFFT shipped with CUDA 12.6U2 (11.3.0) or newer \n"
                f"2. the older, experimental cuFFT LTO EA (early access) preview build "
                f"(https://developer.nvidia.com/cufftea).\n"
                f"To use version different from the one shipped with CUDA Toolkit, please make "
                f"sure the right 'libcufft.so' takes precedence for nvmath. "
                f"For example, by adjusting the 'LD_LIBRARY_PATH' or 'LD_PRELOAD'."
            ) from e

        logger.info(f"The specified LTO-IR {cbkind} has been set.")
        if isinstance(cbobj.ltoir, int):
            logger.debug(f"The {cbkind} LTO-IR pointer is {cbobj.ltoir}.")
        logger.debug(f"The {cbkind} LTO-IR size is {cbobj.size}, and data is {cbobj.data}.")

    if prolog is not None:
        set_callback("prolog", prolog, operand_dtype)
    if epilog is not None:
        set_callback("epilog", epilog, result_dtype)


class InvalidFFTState(Exception):
    pass


@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
class FFT:
    """
    Create a stateful object that encapsulates the specified FFT computations and required
    resources. This object ensures the validity of resources during use and releases them
    when they are no longer needed to prevent misuse.

    This object encompasses all functionalities of function-form APIs :func:`fft`,
    :func:`ifft`, :func:`rfft`, and :func:`irfft`, which are convenience wrappers around it.
    The stateful object also allows for the amortization of preparatory costs when the same
    FFT operation is to be performed on multiple operands with the same problem
    specification (see :meth:`reset_operand` and :meth:`create_key` for more details).

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

        axes: {axes}

        options: {options}

        execution: {execution}

        stream: {stream}

    See Also:
        :meth:`plan`, :meth:`reset_operand`, :meth:`execute`, :meth:`create_key`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create a 3-D complex128 ndarray on the GPU:

        >>> shape = 128, 128, 128
        >>> a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

        We will define a 2-D C2C FFT operation along the first two dimensions, batched along
        the last dimension:

        >>> axes = 0, 1

        Create an FFT object encapsulating the problem specification above:

        >>> f = nvmath.fft.FFT(a, axes=axes)

        Options can be provided above to control the behavior of the operation using the
        `options` argument (see :class:`FFTOptions`). Similarly, the execution space (CUDA
        or CPU) and execution options can be passed using the `execution` argument (see
        :class:`ExecutionCUDA`, :class:`ExecutionCPU`).

        Next, plan the FFT. Load and/or store callback functions can be provided to
        :meth:`plan` using the `prolog` and `epilog` option:

        >>> f.plan()

        Now execute the FFT, and obtain the result `r1` as a CuPy ndarray. The transform
        will be performed on GPU, because ``execution`` was not explicitly specified and
        ``a`` resides in GPU memory.

        >>> r1 = f.execute()

        Finally, free the FFT object's resources. To avoid this explicit call, it's
        recommended to use the FFT object as a context manager as shown below, if possible.

        >>> f.free()

        Note that all :class:`FFT` methods execute on the current stream by default.
        Alternatively, the `stream` argument can be used to run a method on a specified
        stream.

        Let's now look at the same problem with NumPy ndarrays on the CPU.

        Create a 3-D complex128 NumPy ndarray on the CPU:

        >>> import numpy as np
        >>> shape = 128, 128, 128
        >>> a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        Create an FFT object encapsulating the problem specification described earlier and
        use it as a context manager.

        >>> with nvmath.fft.FFT(a, axes=axes) as f:
        ...     f.plan()
        ...
        ...     # Execute the FFT to get the first result.
        ...     r1 = f.execute()

        All the resources used by the object are released at the end of the block.

        The operation was performed on the CPU because ``a`` resides in host memory. With
        ``execution`` specified to 'cuda', the NumPy array would be temporarily copied to
        device memory and transformed on the GPU:

        >>> with nvmath.fft.FFT(a, axes=axes, execution="cuda") as f:
        ...     f.plan()
        ...
        ...     # Execute the FFT to get the first result.
        ...     r1 = f.execute()

        Further examples can be found in the `nvmath/examples/fft
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft>`_ directory.

    Notes:

        - The input must be Hermitian-symmetric when :attr:`FFTOptions.fft_type` is
          ``'C2R'``, otherwise the result is undefined. As a specific example, if the input
          for a C2R FFT was generated using an R2C FFT with an odd last axis size, then
          :attr:`FFTOptions.last_axis_parity` must be set to `odd` to recover the original
          signal.
    """

    def __init__(
        self,
        operand,
        *,
        axes: Sequence[int] | None = None,
        options: FFTOptions | None = None,
        execution: ExecutionCPU | ExecutionCUDA | None = None,
        stream: AnyStream | None = None,
    ):
        self.operand = operand = tensor_wrapper.wrap_operand(operand)
        options, execution = setup_options(operand, options, execution)
        self.options = options
        self.execution_options = execution

        self.operand_dim = len(operand.shape)

        if not axes and self.operand_dim > 3:
            raise ValueError(
                f"The tensor is {self.operand_dim}-D and FFTs in number of dimensions > 3 is not supported. The FFT "
                "axes need to be specified using the 'axes' option."
            )

        if self.operand_dim == 0:
            raise ValueError(f"The tensor is {self.operand_dim}-D (i.e. a scalar). FFT does not support scalars.")

        self.operand_data_type = operand.dtype
        self.fft_abstract_type = _get_default_fft_abstract_type(self.operand_data_type, options.fft_type)

        self.result_data_type, self.compute_data_type = _get_fft_result_and_compute_types(operand.dtype, self.fft_abstract_type)

        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info(f"The FFT type is {self.fft_abstract_type}.")
        self.logger.info(
            f"The input data type is {self.operand_data_type}, and the result data type is {self.result_data_type}."
        )

        if axes is None:
            axes = range(self.operand_dim)

        if any(axis >= self.operand_dim or axis < -self.operand_dim for axis in axes):
            raise ValueError(f"The specified FFT axes {axes} are out of bounds for a {self.operand_dim}-D tensor.")

        # Handle negative axis indices.
        self.axes = tuple(axis % self.operand_dim for axis in axes)
        self.logger.info(f"The specified FFT axes are {self.axes}.")

        self.package = utils.infer_object_package(operand.tensor)

        # NumPy and CuPy don't support complex32 yet.
        if self.package in ["numpy", "cupy"] and self.result_data_type == "complex32":
            raise TypeError(
                f"The result data type {self.result_data_type} is not supported by the operand package '{self.package}'."
            )

        # Infer operand package, execution space, and memory space.
        if execution.name == "cuda":
            if operand.device == "cuda":  # exec space matches the mem space
                self.memory_space = "cuda"
                self.device_id = operand.device_id
            else:  # we need to move inputs cpu -> gpu and outputs gpu -> cpu
                self.memory_space = "cpu"
                self.device_id = execution.device_id
        else:
            assert execution.name == "cpu"
            self.device_id = "cpu"
            if operand.device_id == "cpu":  # exec space matches the mem space
                self.memory_space = "cpu"
            else:  # we need to move inputs gpu -> cpu and outputs cpu -> gpu
                self.memory_space = "cuda"
        self.execution_space = execution.name
        self.operand_device_id = operand.device_id
        self.internal_op_package = self._internal_operand_package(self.package)
        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)

        self.logger.info(
            f"The input tensor's memory space is {self.memory_space}, and the execution space "
            f"is {self.execution_space}, with device {self.device_id}."
        )

        self.logger.info(
            f"The specified stream for the FFT ctor is "
            f"{(exec_stream_holder or operand_stream_holder) and getattr(exec_stream_holder or operand_stream_holder, 'obj', None)}."  # noqa: E501
        )

        # In-place FFT option, available only for C2C transforms.
        self.inplace = self.options.inplace
        if self.inplace and self.fft_abstract_type != "C2C":
            raise ValueError(
                f"The in-place option (FFTOptions.inplace=True) is only supported for complex-to-complex FFT. "
                f"The FFT type is '{self.fft_abstract_type}'."
            )

        # Copy the operand to execution_space's device if needed.
        self.operand, self.operand_backup = _copy_operand_perhaps(
            None,
            operand,
            operand_stream_holder,
            self.execution_space,
            self.memory_space,
            self.device_id,
            self.fft_abstract_type,
            self.logger,
        )

        operand = self.operand
        # Capture operand layout for consistency checks when resetting operands.
        self.operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)

        self._preallocated_result: utils.TensorHolder | None = None

        if self.options.inplace:  # Don't use self.inplace here, because we always set it to True for CPU tensors.
            self.logger.info("The FFT will be performed in-place, with the result overwriting the input.")
        else:
            self.logger.info("The FFT will be performed out-of-place.")

        # Check if FFT is supported and calculate plan traits.
        self.plan_traits = get_fft_plan_traits(
            operand.shape,
            operand.strides,
            operand.dtype,
            self.axes,
            self.execution_options,
            fft_abstract_type=self.fft_abstract_type,
            last_axis_parity=self.options.last_axis_parity,
            result_layout=self.options.result_layout,
            logger=self.logger,
        )

        self.logger.info(
            f"The operand data type = {self.operand_data_type}, shape = {self.operand_layout.shape}, and "
            f"strides = {self.operand_layout.strides}."
        )
        result_data_type, result_shape, result_strides = (
            (self.operand_data_type, self.operand_layout.shape, self.operand_layout.strides)
            if self.inplace
            else (self.result_data_type, self.plan_traits.result_shape, self.plan_traits.result_strides)
        )
        self.logger.info(f"The result data type = {result_data_type}, shape = {result_shape}, and strides = {result_strides}.")
        self.logger.info(f"The FFT batch size is {self.plan_traits.fft_batch_size}.")

        ordered_fft_out_shape, ostride, odistance = (
            (self.plan_traits.ordered_fft_in_shape, self.plan_traits.istride, self.plan_traits.idistance)
            if self.inplace
            else (self.plan_traits.ordered_fft_out_shape, self.plan_traits.ostride, self.plan_traits.odistance)
        )
        self.logger.debug(
            f"The plan ordered axes = {self.plan_traits.ordered_axes}, ordered input shape = "
            f"{self.plan_traits.ordered_fft_in_shape}, ordered input embedding shape = "
            f"{self.plan_traits.ordered_fft_in_embedding_shape}, ordered output shape = {ordered_fft_out_shape}."
        )
        self.logger.debug(f"The plan input stride is {self.plan_traits.istride} with distance {self.plan_traits.idistance}.")
        self.logger.debug(f"The plan output stride is {ostride} with distance {odistance}.")

        # The result's package and device.
        self.result_class = operand.__class__

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == "cpu" or self.execution_space == "cpu"
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = (
                "This call is non-blocking and will return immediately after the operation is launched on the device."
            )

        # Set memory allocator.
        if self.execution_space == "cpu":
            self.allocator = None  # currently, the nvpl/fftw does not support custom workspace allocation
        else:
            self.allocator = (
                options.allocator
                if options.allocator is not None
                else memory._MEMORY_MANAGER[self.internal_op_package](self.device_id, self.logger)
            )

        if self.execution_space == "cpu":
            # the handle is created alongside planning
            self.handle = None
        else:
            # Create handle.
            with utils.device_ctx(self.device_id):
                self.handle = cufft.create()

            # Set stream for the FFT.
            cufft.set_stream(self.handle, exec_stream_holder.ptr)  # type: ignore[union-attr]

            # Plan attributes.
            cufft.set_auto_allocation(self.handle, 0)

        self.fft_planned = False

        # Workspace attributes.
        self.workspace_ptr: None | memory.MemoryPointer = None
        self.workspace_size = 0
        self._workspace_allocated_here = False

        # Attributes to establish stream ordering.
        self.workspace_stream = None
        self.last_compute_event = None

        # Keep track of key (sans callback) for resetting operands, once plan in available.
        self.orig_key = None

        self.valid_state = True
        self.logger.info("The FFT operation has been created.")

    def get_key(self, *, prolog: DeviceCallable | None = None, epilog: DeviceCallable | None = None):
        """
        Get the key for this object's data supplemented with the callbacks.

        Args:
            prolog: {prolog}
            epilog: {epilog}

        Returns:
            {fft_key}

        See Also:
            :meth:`create_key`
        """
        return create_fft_key(
            self.operand.tensor,
            axes=self.axes,
            options=self.options,
            execution=self.execution_options,
            inplace=self.inplace,
            prolog=prolog,
            epilog=epilog,
        )

    @staticmethod
    def create_key(
        operand,
        *,
        axes: Sequence[int] | None = None,
        options: FFTOptions | None = None,
        execution: ExecutionCPU | ExecutionCUDA | None = None,
        prolog: DeviceCallable | None = None,
        epilog: DeviceCallable | None = None,
    ):
        """
        Create a key as a compact representation of the FFT problem specification based on
        the given operand, axes and the FFT options. Note that different combinations of
        operand layout, axes and options can potentially correspond to the same underlying
        problem specification (key). Users may reuse the FFT objects when different input
        problems map to an identical key.

        Args:
            operand: {operand}

            axes: {axes}

            options: {options}

            execution: {execution}

            prolog: {prolog}

            epilog: {epilog}

        Returns:
            {fft_key}

        Notes:
            - Users may take advantage of this method to create cached version of
              :func:`fft` based on the stateful object APIs (see `caching.py
              <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/caching.py>`_
              for an example implementation).
            - This key is meant for runtime use only and not designed to be serialized or
              used on a different machine.
            - It is the user's responsibility to augment this key with the stream in case
              they use stream-ordered memory pools.
        """
        return create_fft_key(operand, axes=axes, options=options, execution=execution, prolog=prolog, epilog=epilog)

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

        self.workspace_ptr = None
        self.fft_planned = False
        return True

    def _internal_operand_package(self, package_name):
        if self.execution_space == "cuda":
            if package_name == "numpy":
                # TODO: remove this call after cupy is dropped
                tensor_wrapper.maybe_register_package("cupy")
            return package_name if package_name != "numpy" else "cupy"
        else:
            return package_name if package_name != "cupy" else "numpy"

    def _get_or_create_stream_maybe(self, stream: AnyStream) -> tuple[StreamHolder | None, StreamHolder | None]:
        if self.execution_space == "cuda":
            assert isinstance(self.device_id, int), self.device_id
            stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_op_package)
            return stream_holder, stream_holder
        elif self.memory_space == "cuda":
            assert isinstance(self.operand_device_id, int), self.operand_device_id
            operand_device_steam = utils.get_or_create_stream(self.operand_device_id, stream, self.package)
            return None, operand_device_steam
        else:
            return None, None

    def _allocate_result_operand(self, exec_stream_holder: StreamHolder | None, log_debug):
        if log_debug:
            self.logger.debug("Beginning output (empty) tensor creation...")
            self.logger.debug(
                f"The output tensor shape = {self.plan_traits.result_shape} with strides = "
                f"{self.plan_traits.result_strides} and data type '{self.result_data_type}'."
            )
        result = utils.create_empty_tensor(
            self.result_class,
            self.plan_traits.result_shape,
            self.result_data_type,
            self.device_id,
            exec_stream_holder,
            verify_strides=False,  # the strides are computed so that they are contiguous
            strides=self.plan_traits.result_strides,
        )
        if log_debug:
            self.logger.debug("The output (empty) tensor has been created.")
        return result

    def _get_validate_direction(self, direction):
        if isinstance(direction, str) and (d := direction.upper()) in ["FORWARD", "INVERSE"]:
            direction = FFTDirection[d]
        else:
            direction = FFTDirection(direction)

        if self.fft_abstract_type == "C2R":
            if direction != FFTDirection.INVERSE:
                raise ValueError(
                    f"The specified direction {direction.name} is not compatible with the FFT type '{self.fft_abstract_type}'."
                )
        elif self.fft_abstract_type == "R2C":  # noqa: SIM102
            if direction != FFTDirection.FORWARD:
                raise ValueError(
                    f"The specified direction {direction.name} is not compatible with the FFT type '{self.fft_abstract_type}'."
                )
        return direction

    @utils.precondition(_check_valid_fft)
    @utils.atomic(_free_plan_resources, method=True)
    def plan(
        self,
        *,
        prolog: DeviceCallable | None = None,
        epilog: DeviceCallable | None = None,
        stream: AnyStream | None = None,
        direction: FFTDirection | None = None,
    ):
        """Plan the FFT.

        Args:
            prolog: {prolog}

            epilog: {epilog}

            stream: {stream}

            direction: If specified, the same direction must be passed to subsequent
                :meth:`execute` calls. It may be used as a hint to optimize C2C planning for
                CPU FFT calls.
        """
        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        if self.fft_planned:
            self.logger.debug("The FFT has already been planned, and redoing the plan is not supported.")
            return

        if self.execution_space == "cpu":
            stream_holder = None
            if prolog is not None or epilog is not None:
                raise ValueError("The 'prolog' and 'epilog' are not supported with CPU 'execution'.")
        else:
            assert isinstance(self.device_id, int)
            stream_holder = utils.get_or_create_stream(self.device_id, stream, self.internal_op_package)
            self.workspace_stream = stream_holder.obj

            # Set stream for the FFT.
            cufft.set_stream(self.handle, stream_holder.ptr)

            # Set LTO-IR callbacks, if present.
            prolog = utils.check_or_create_options(DeviceCallable, prolog, "prolog", keep_none=True)
            epilog = utils.check_or_create_options(DeviceCallable, epilog, "epilog", keep_none=True)
            _check_prolog_epilog_traits(prolog, epilog, self.plan_traits, self.operand, self.fft_abstract_type)
            set_prolog_and_epilog(self.handle, prolog, epilog, self.operand_data_type, self.result_data_type, self.logger)

        # Get all the arguments to xt_make_plan_many except for the first (the handle).
        if self.inplace:
            check_inplace_overlapping_layout(self.operand)
            if self.operand_backup is not None:
                check_inplace_overlapping_layout(self.operand_backup)

        plan_args = create_xt_plan_args(
            plan_traits=self.plan_traits,
            fft_abstract_type=self.fft_abstract_type,
            operand_data_type=self.operand_data_type,
            inplace=self.inplace,
        )

        # Keep track of original key (sans callback) for resetting operands. Pass in plan
        # args to avoid recomputation.
        self.orig_key = create_fft_key(
            self.operand.tensor,
            axes=self.axes,
            options=self.options,
            execution=self.execution_options,
            plan_args=plan_args,
        )
        if log_debug:
            self.logger.debug(f"The FFT key (sans callback) is {self.orig_key}.")

            self.logger.debug(
                f"The operand CUDA type is {NAME_TO_DATA_TYPE[self.operand_data_type].name}, and the result CUDA type is "
                f"{NAME_TO_DATA_TYPE[self.result_data_type].name}."
            )
            self.logger.debug(f"The CUDA type used for compute is {NAME_TO_DATA_TYPE[self.compute_data_type].name}.")
        if log_info:
            self.logger.info("Starting FFT planning...")

        if self.execution_space == "cpu":
            if direction is not None:
                direction = self._get_validate_direction(direction)
            if self.inplace:
                result_ptr = self.operand.data_ptr
            else:
                # FFTW3 API requires passing pointers to the input and output during
                # planning. Passing different pointers to (properly strided and aligned)
                # data in subsequent execute calls is supported, but it is not clear what
                # planning is allowed to do with the provided pointers. For one, planning
                # can compare the two pointers for equality to decide if it is inplace or
                # out-of-place operation. To avoid subtle issues, just preallocate the
                # result tensor earlier.
                self._preallocated_result = self._allocate_result_operand(None, True)
                result_ptr = self._preallocated_result.data_ptr  # type: ignore[attr-defined, union-attr]
            precision, *plan_args = fftw_plan_args(
                plan_args,
                self.operand.data_ptr,
                result_ptr,
                fft_abstract_type=self.fft_abstract_type,
                direction=direction,
            )
            with utils.host_call_ctx(timing=log_info) as elapsed:
                fftw.plan_with_nthreads(precision, self.execution_options.num_threads)  # type: ignore[union-attr]
                try:
                    assert self.handle is None
                    self.handle = fftw.plan_many(precision, *plan_args)
                except OverflowError as e:
                    raise ValueError(
                        "Currently, the CPU FFT only supports the problem sizes that "
                        "can be expressed as 32-bit signed integer. "
                        "Tensors with shape extents or stride larger than "
                        "`2147483647` are not currently supported."
                    ) from e
        else:
            # FIXME: Move creation of stream_holder into this block, so assertion is not
            # needed
            assert isinstance(self.device_id, int), self.device_id
            assert stream_holder is not None
            with utils.cuda_call_ctx(stream_holder, blocking=True, timing=log_info) as (
                self.last_compute_event,
                elapsed,
            ):
                self.workspace_size = cufft.xt_make_plan_many(self.handle, *plan_args)

        self.fft_planned = True

        if log_info and elapsed.data is not None:
            self.logger.info(f"The FFT planning phase took {elapsed.data:.3f} ms to complete.")

    @utils.precondition(_check_valid_fft)
    def reset_operand(self, operand=None, *, stream: AnyStream | None = None):
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

                - The problem specification key for the new operand. Generally the keys will
                  match if the operand shares the same layout (shape, strides and data
                  type). The keys may still match for certain operands with different
                  layout, see :meth:`create_key` for details.
                - The package that the new operand belongs to.
                - The memory space of the new operand (CPU or GPU).
                - The device that new operand belongs to if it is on GPU.

            stream: {stream}.

        Examples:

            >>> import cupy as cp
            >>> import nvmath

            Create a 3-D complex128 ndarray on the GPU:

            >>> shape = 128, 128, 128
            >>> a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

            Create an FFT object as a context manager

            >>> axes = 0, 1
            >>> with nvmath.fft.FFT(a, axes=axes) as f:
            ...     # Plan the FFT
            ...     f.plan()
            ...
            ...     # Execute the FFT to get the first result.
            ...     r1 = f.execute()
            ...
            ...     # Reset the operand to a new CuPy ndarray.
            ...     b = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
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

                - The operation is not a complex-to-real (C2R) FFT.
                - The operand's memory matches the FFT execution space. More precisely, the
                  operand memory space should be accessible from the execution space (CPU or
                  CUDA).

            For more details, please refer to `inplace update example
            <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example05_stateful_inplace.py>`_.
        """

        if operand is None:
            self.operand = None  # type: ignore
            self.operand_backup = None  # type: ignore
            self.logger.info("The operand has been reset to None.")
            return

        self.logger.info("Resetting operand...")
        # First wrap operand.
        operand = tensor_wrapper.wrap_operand(operand)

        # Check package match.
        package = utils.infer_object_package(operand.tensor)
        if self.package != package:
            message = f"Library package mismatch: '{self.package}' => '{package}'"
            raise TypeError(message)

        utils.check_attribute_match(self.operand_data_type, operand.dtype, "data type")

        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)
        self.logger.info(
            "The specified stream for reset_operand() is "
            f"{(exec_stream_holder or operand_stream_holder) and (exec_stream_holder or operand_stream_holder).obj}."  # type: ignore[union-attr]
        )

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

        if self.orig_key is not None:
            # Compute the key corresponding to the new operand (sans callback).
            try:
                new_key = create_fft_key(
                    operand.tensor,
                    axes=self.axes,
                    options=self.options,
                    execution=self.execution_options,
                    inplace=self.inplace,
                )
            except UnsupportedLayoutError:
                new_key = None
            if self.orig_key != new_key:
                self.logger.debug(f"The FFT key corresponding to the original operand is: {self.orig_key}.")
                if new_key is None:
                    self.logger.debug(
                        "The FFT key for the new operand cannot be computed since the layout "
                        f"(shape = {operand.shape}, strides = {operand.strides}) and axes = {self.axes} combination "
                        "is unsupported."
                    )
                else:
                    self.logger.debug(f"The FFT key corresponding to the new operand is:      {new_key}.")
                raise ValueError(
                    "The new operand's traits (data type, shape, or strides) are incompatible with that of the "
                    "original operand."
                )

        if self.execution_space == "cuda":
            # Set stream for the FFT.
            cufft.set_stream(self.handle, exec_stream_holder.ptr)  # type: ignore[union-attr]

        self.operand, self.operand_backup = _copy_operand_perhaps(
            self.operand,
            operand,
            operand_stream_holder,
            self.execution_space,
            self.memory_space,
            self.device_id,
            self.fft_abstract_type,
            self.logger,
        )
        operand = self.operand

        # Update operand layout and plan traits.
        self.operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)
        self.logger.info(f"The reset operand shape = {self.operand_layout.shape}, and strides = {self.operand_layout.strides}.")

        self.plan_traits = get_fft_plan_traits(
            operand.shape,
            operand.strides,
            operand.dtype,
            self.axes,
            self.execution_options,
            fft_abstract_type=self.fft_abstract_type,
            last_axis_parity=self.options.last_axis_parity,
            result_layout=self.options.result_layout,
            logger=self.logger,
        )
        result_shape, result_strides = (
            (self.operand_layout.shape, self.operand_layout.strides)
            if self.inplace
            else (self.plan_traits.result_shape, self.plan_traits.result_strides)
        )
        self.logger.info(f"The result shape = {result_shape}, and strides = {result_strides}.")

        self.logger.info("The operand has been reset to the specified operand.")

    def get_input_layout(self):
        """
        Returns a pair of tuples: shape and strides of the FFT input.

        .. note::
            In some cases, the FFT operation requires taking a copy of the input tensor
            (e.g. C2R cuFFT, or provided tensor resides on CPU but FFT is executed on GPU).
            The copied tensor strides may differ from the input tensor passed by the user,
            if the original tensor's strides do not conform to dense C-like layout.
        """
        return self.operand_layout.shape, self.operand_layout.strides

    def get_output_layout(self):
        """
        Returns a pair of tuples: shape and strides of the FFT output.
        """
        return (
            (self.operand_layout.shape, self.operand_layout.strides)
            if self.inplace
            else (self.plan_traits.result_shape, self.plan_traits.result_strides)
        )

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
        Free workspace by releasing the MemoryPointer object.
        """
        if self.workspace_ptr is None:
            return True

        self.workspace_ptr = None
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
                if isinstance(self.allocator, memory.BaseCUDAMemoryManagerAsync):
                    self.workspace_ptr = self.allocator.memalloc_async(self.workspace_size, stream_holder.obj)
                else:
                    self.workspace_ptr = self.allocator.memalloc(self.workspace_size)  # type: ignore[union-attr]
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
    def execute(self, direction: FFTDirection | None = None, stream: AnyStream | None = None, release_workspace: bool = False):
        """
        Execute the FFT operation.

        Args:
            direction: {direction}

            stream: {stream}

            release_workspace: {release_workspace}

        Returns:
            The transformed operand, which remains on the same device and utilizes the same
            package as the input operand. The data type and shape of the transformed operand
            depend on the type of input operand:

            - For C2C FFT, the data type and shape remain identical to the input.
            - For R2C and C2R FFT, both data type and shape differ from the input.
        """

        log_info = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        if direction is None:
            direction = _get_fft_default_direction(self.fft_abstract_type)
        else:
            direction = self._get_validate_direction(direction)

        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)

        if self.execution_space == "cuda":
            # Set stream for the FFT.
            cufft.set_stream(self.handle, exec_stream_holder.ptr)  # type: ignore[union-attr]

        # Allocate workspace if needed.
        self._allocate_workspace_memory_perhaps(exec_stream_holder)  # type: ignore[arg-type]
        # Allocate output operand if needed
        if self.inplace:
            result_ptr = self.operand.data_ptr
        else:
            if self._preallocated_result is not None:
                assert self.execution_space == "cpu"
                self.result = self._preallocated_result
                self._preallocated_result = None
            else:
                self.result = self._allocate_result_operand(exec_stream_holder, log_debug)
            result_ptr = self.result.data_ptr

        if log_info:
            self.logger.info(f"Starting FFT {self.fft_abstract_type} calculation in the {direction.name} direction...")  # type: ignore[union-attr]
            self.logger.info(f"{self.call_prologue}")

        if self.execution_space == "cpu":
            with utils.host_call_ctx(timing=log_info) as elapsed:
                fftw.execute(self.handle, self.operand.data_ptr, result_ptr, direction)
        else:
            assert isinstance(self.device_id, int), self.device_id
            assert exec_stream_holder is not None
            with utils.cuda_call_ctx(exec_stream_holder, self.blocking, timing=log_info) as (
                self.last_compute_event,
                elapsed,
            ):
                if log_debug:
                    self.logger.debug("The cuFFT execution function is 'xt_exec'.")
                cufft.xt_exec(self.handle, self.operand.data_ptr, result_ptr, direction)

        if log_info and elapsed.data is not None:
            self.logger.info(f"The FFT calculation took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if it's more than the
        # specified cache limit.
        self._free_workspace_memory_perhaps(release_workspace)

        # reset workspace allocation tracking to False at the end of the methods where
        # workspace memory is potentially allocated. This is necessary to prevent any
        # exceptions raised before method entry from using stale tracking values.
        self._workspace_allocated_here = False

        # Return the result.
        result = self.operand if self.inplace else self.result
        if self.memory_space == self.execution_space:
            out = result.tensor
        else:
            if self.options.inplace:  # Don't use self.inplace here, because we always set it to True for CPU tensors.
                self.operand_backup.copy_(result, stream_holder=operand_stream_holder)
                out = self.operand_backup.tensor
            else:
                target_dev = "cpu" if self.memory_space == "cpu" else self.operand_device_id
                out = result.to(target_dev, stream_holder=operand_stream_holder).tensor  # type: ignore[arg-type]

        # Release internal reference to the result to permit recycling of memory.
        self.result = None  # type: ignore

        return out

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
                self.workspace_stream.wait(self.last_compute_event)
                self.last_compute_event = None

            self._free_workspace_memory()

            if self.handle is not None:
                if self.execution_space == "cuda":
                    cufft.destroy(self.handle)
                else:
                    fftw.destroy(self.handle)
                self.handle = None

        except Exception as e:
            self.logger.critical("Internal error: only part of the FFT object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The FFT object's resources have been released.")


def _fft(
    x,
    *,
    axes: Sequence[int] | None = None,
    direction: FFTDirection | None = None,
    options: FFTOptions | None = None,
    execution: ExecutionCPU | ExecutionCUDA | None = None,
    prolog: DeviceCallable | None = None,
    epilog: DeviceCallable | None = None,
    stream: AnyStream | None = None,
    check_dtype: str | None = None,
):
    r"""
    fft({function_signature})

    Perform an N-D *complex-to-complex* (C2C) FFT on the provided complex operand.

    Args:
        operand: {operand}

        axes: {axes}

        options: {options}

        execution: {execution}

        prolog: {prolog}

        epilog: {epilog}

        stream: {stream}

    Returns:
        A transformed operand that retains the same data type and shape as the input. It
        remains on the same device and uses the same package as the input operand.

    See Also:
        :func:`ifft`, :func:`irfft`, :func:`rfft`, :class:`FFT`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create a 3-D complex128 ndarray on the GPU:

        >>> shape = 256, 256, 256
        >>> a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(
        ...     *shape, dtype=cp.float64
        ... )

        Perform a 3-D C2C FFT using :func:`fft`. The result `r` is also a CuPy complex128
        ndarray:

        >>> r = nvmath.fft.fft(a)

        User may also perform FFT along a subset of dimensions, e.g, 2-D C2C FFT along the
        first two dimensions, batched along the last dimension:

        >>> axes = 0, 1
        >>> r = nvmath.fft.fft(a, axes=axes)

        For C2C type FFT operation, the output can be directly computed inplace thus
        overwriting the input operand. This can be specified using options to the FFT:

        >>> o = nvmath.fft.FFTOptions(inplace=True)
        >>> r = nvmath.fft.fft(a, options=o)
        >>> r is a
        True

        See :class:`FFTOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly
        provided to the FFT operation. This can be done if the FFT operand is computed on a
        different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...     a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
        >>> r = nvmath.fft.fft(a, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input
        computation.

        Create a NumPy ndarray on the CPU.

        >>> import numpy as np
        >>> b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        Provide the NumPy ndarray to :func:`fft`, with the result also being a NumPy
        ndarray:

        >>> r = nvmath.fft.fft(b)

    Notes:
        - This function only takes complex operand for C2C transformation. If the user
          wishes to perform full FFT transformation on real input, please cast the input to
          the corresponding complex data type.
        - This function is a convenience wrapper around :class:`FFT` and and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API using the default `direction` argument in :meth:`FFT.execute`.

    Further examples can be found in the `nvmath/examples/fft
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft>`_ directory.
    """
    if check_dtype is not None:
        assert check_dtype in {"real", "complex"}, "internal error"
        operand = tensor_wrapper.wrap_operand(x)
        if ("complex" in operand.dtype) != (check_dtype == "complex"):
            raise ValueError(f"This function expects {check_dtype} operand, found {operand.dtype}")

    with FFT(x, axes=axes, options=options, execution=execution, stream=stream) as fftobj:
        # Plan the FFT.
        fftobj.plan(stream=stream, prolog=prolog, epilog=epilog, direction=direction)

        # Execute the FFT.
        result = fftobj.execute(direction=direction, stream=stream)

    return result


# Forward C2C FFT Function.
fft = functools.wraps(_fft)(functools.partial(_fft, direction=FFTDirection.FORWARD, check_dtype="complex"))
fft.__doc__ = fft.__doc__.format(**SHARED_FFT_DOCUMENTATION)  # type: ignore
fft.__name__ = "fft"


# Forward R2C FFT Function
@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
def rfft(
    operand,
    *,
    axes: Sequence[int] | None = None,
    options: FFTOptions | None = None,
    execution: ExecutionCPU | ExecutionCUDA | None = None,
    prolog: DeviceCallable | None = None,
    epilog: DeviceCallable | None = None,
    stream: AnyStream | None = None,
):
    r"""
    rfft({function_signature})

    Perform an N-D *real-to-complex* (R2C) FFT on the provided real operand.

    Args:
        operand: {operand}

        axes: {axes}

        options: {options}

        execution: {execution}

        prolog: {prolog}

        epilog: {epilog}

        stream: {stream}

    Returns:
        A complex tensor that remains on the same device and belongs to the same package as
        the input operand. The extent of the last transformed axis in the result will be
        ``operand.shape[axes[-1]] // 2 + 1``.


    See Also:
        :func:`fft`, :func:`irfft`, :class:`FFT`.
    """
    wrapped_operand = tensor_wrapper.wrap_operand(operand)
    # check if input operand if real type
    if "complex" in wrapped_operand.dtype:
        raise RuntimeError(f"rfft expects a real input, but got {wrapped_operand.dtype}. Please use fft for complex input.")

    return _fft(
        operand,
        axes=axes,
        options=options,
        execution=execution,
        prolog=prolog,
        epilog=epilog,
        stream=stream,
        check_dtype="real",
    )


# Inverse C2C/R2C FFT Function.
ifft = functools.wraps(_fft)(functools.partial(_fft, direction=FFTDirection.INVERSE, check_dtype="complex"))
ifft.__doc__ = """
    ifft({function_signature})

    Perform an N-D *complex-to-complex* (C2C) inverse FFT on the provided complex operand.
    The direction is implicitly inverse.

    Args:
        operand: {operand}

        axes: {axes}

        options: {options}

        execution: {execution}

        prolog: {prolog}

        epilog: {epilog}

        stream: {stream}

    Returns:
        A transformed operand that retains the same data type and shape as the input. It
        remains on the same device and uses the same package as the input operand.

    See Also:
        :func:`fft`, :func:`irfft`, :class:`FFT`.

    Notes:
        - This function only takes complex operand for C2C transformation. If users wishes
          to perform full FFT transformation on real input, please cast the input to the
          corresponding complex data type.
        - This function is a convenience wrapper around :class:`FFT` and and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API by passing the argument ``direction='inverse'`` when calling
          :meth:`FFT.execute`.
""".format(**SHARED_FFT_DOCUMENTATION)
ifft.__name__ = "ifft"


# Inverse C2R FFT Function.
@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
def irfft(
    x,
    *,
    axes: Sequence[int] | None = None,
    options: FFTOptions | None = None,
    execution: ExecutionCPU | ExecutionCUDA | None = None,
    prolog: DeviceCallable | None = None,
    epilog: DeviceCallable | None = None,
    stream: AnyStream | None = None,
):
    """
    irfft({function_signature})

    Perform an N-D *complex-to-real* (C2R) FFT on the provided complex operand. The
    direction is implicitly inverse.

    Args:
        operand: {operand}

        axes: {axes}

        options: {options}

        execution: {execution}

        prolog: {prolog}

        epilog: {epilog}

        stream: {stream}

    Returns:
        A real tensor that remains on the same device and belongs to the same package as the
        input operand. The extent of the last transformed axis in the result will be
        ``(operand.shape[axes[-1]] - 1) * 2`` if :attr:`FFTOptions.last_axis_parity` is
        ``even``, or ``operand.shape[axes[-1]] * 2 - 1`` if
        :attr:`FFTOptions.last_axis_parity` is ``odd``.

    See Also:
        :func:`fft`, :func:`ifft`, :class:`FFT`.

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create a 3-D symmetric complex128 ndarray on the GPU:

        >>> shape = 512, 768, 256
        >>> a = nvmath.fft.rfft(cp.random.rand(*shape, dtype=cp.float64))

        Perform a 3-D C2R FFT using the :func:`irfft` wrapper. The result `r` is a CuPy
        float64 ndarray:

        >>> r = nvmath.fft.irfft(a)
        >>> r.dtype
        dtype('float64')

    Notes:

        - This function performs an inverse C2R N-D FFT, which is similar to `irfftn` but
          different from `irfft` in various numerical packages.
        - This function is a convenience wrapper around :class:`FFT` and and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API by setting :attr:`FFTOptions.fft_type` to ``'C2R'`` and passing the argument
          ``direction='inverse'`` when calling :meth:`FFT.execute`.
        - **The input to this function must be Hermitian-symmetric, otherwise the result is
          undefined.** While the symmetry requirement is partially captured by the different
          extents in the last transformed dimension between the input and result, there are
          additional `constraints
          <https://docs.nvidia.com/cuda/cufft/#fourier-transform-types>`_. As a specific
          example, 1-D transforms require the first element (and the last element, if the
          extent is even) of the input to be purely real-valued. In addition, if the input
          to `irfft` was generated using an R2C FFT with an odd last axis size,
          :attr:`FFTOptions.last_axis_parity` must be set to ``odd`` to recover the original
          signal.
        - For more details, please refer to `C2R example
          <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example07_c2r.py>`_
          and `odd C2R example
          <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example07_c2r_odd.py>`_.
    """
    options = utils.check_or_create_options(FFTOptions, options, "FFT options")
    assert options is not None
    options.fft_type = "C2R"
    return _fft(
        x,
        axes=axes,
        direction=FFTDirection.INVERSE,
        options=options,
        execution=execution,
        prolog=prolog,
        epilog=epilog,
        stream=stream,
        check_dtype="complex",
    )
