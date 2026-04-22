# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FFT", "fft", "ifft", "rfft", "irfft", "UnsupportedLayoutError", "estimate_workspace_size"]

import enum
import functools
import logging
import operator
from collections.abc import Sequence
from dataclasses import astuple as data_cls_astuple
from dataclasses import dataclass
from typing import Literal

from nvmath.bindings import cufft  # type: ignore

from ._configuration import DeviceCallable, ExecutionCPU, ExecutionCUDA, FFTDirection, FFTOptions

try:
    from nvmath.bindings.nvpl import fft as fftw  # type: ignore
except ImportError:
    fftw = None  # type: ignore
from nvmath import memory
from nvmath._internal.layout import is_contiguous_in_memory, is_contiguous_layout, is_overlapping_layout
from nvmath.bindings._internal import utils as _bindings_utils  # type: ignore
from nvmath.fft._exec_utils import _cross_setup_execution_and_options
from nvmath.internal import formatters, tensor_wrapper, utils
from nvmath.internal.package_wrapper import AnyStream, StreamHolder
from nvmath.internal.typemaps import (
    DATA_TYPE_TO_NAME,
    FFTW_SUPPORTED_COMPLEX,
    FFTW_SUPPORTED_DOUBLE,
    FFTW_SUPPORTED_FLOAT,
    FFTW_SUPPORTED_SINGLE,
    FFTW_SUPPORTED_TYPES,
    NAME_TO_DATA_TYPE,
    cudaDataType,
)


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
    optimized_result_layout: bool
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
/,
*,
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
    is_sample_interleaved = bool(sorted_batch_strides and sorted_batch_strides[0] <= ordered_in_strides[0])
    logger.debug(f"Are the samples interleaved? {is_sample_interleaved}.")

    use_optimized_result_layout = is_sample_interleaved and result_layout != "natural"
    if not use_optimized_result_layout:  # Natural (== operand) layout.
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
        optimized_result_layout=use_optimized_result_layout,
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


def _allocate_operand_or_identity(
    user_operand: utils.TensorHolder,
    stream_holder,
    execution_space,
    memory_space,
    device_id: int | Literal["cpu"],
    fft_abstract_type,
    logger,
):
    """
    Allocate the internal operand for the given execution/memory space, or
    return the user operand as-is (identity) when
    no internal buffer/mirror is needed.

    Used during construction and after release_operand() when the internal
    buffer needs to be created from scratch.
    """
    if execution_space == memory_space:
        if fft_abstract_type != "C2R":
            return user_operand, None
        else:
            # For C2R, we need to take a copy to avoid input being overwritten
            logger.info("For C2R FFT with input operand on GPU, the input is copied to avoid being overwritten by cuFFT.")
            internal_operand = utils.create_empty_tensor(
                user_operand.__class__,
                user_operand.shape,
                user_operand.dtype,
                device_id,
                stream_holder,
                verify_strides=True,
                strides=user_operand.strides,
            )
            internal_operand.copy_(user_operand, stream_holder=stream_holder)
            # We don't need to keep the operand backup, because C2R precludes `inplace=True`
            return internal_operand, None
    else:
        # Copy the `operand` to memory that matches the exec space
        # and keep the original `operand` to handle `options.inplace=True`
        if execution_space == "cuda":
            assert isinstance(device_id, int)
            to_device: int | Literal["cpu"] = device_id
        else:
            assert execution_space == "cpu"
            to_device = "cpu"
        exec_space_copy = user_operand.to(to_device, stream_holder)
        return exec_space_copy, user_operand


def _copy_operand_or_identity(
    internal_operand: utils.TensorHolder,
    new_operand: utils.TensorHolder,
    stream_holder,
    execution_space,
    memory_space,
    fft_abstract_type,
    logger,
):
    """
    Copy new operand data into an existing internal buffer, or return the
    new operand as-is (identity) when no internal buffer is needed.

    Used by reset_operand() when the operand has NOT been released, so the
    internal buffer is still valid and can be reused via in-place copy.
    """
    if execution_space == memory_space:
        if fft_abstract_type != "C2R":
            return new_operand, None
        else:
            logger.info("For C2R FFT with input operand on GPU, the input is copied to avoid being overwritten by cuFFT.")
            internal_operand.copy_(new_operand, stream_holder=stream_holder)
            return internal_operand, None
    else:
        internal_operand.copy_(new_operand, stream_holder=stream_holder)
        return internal_operand, new_operand


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


def _compute_fft_plan_args_and_traits(
    operand: utils.TensorHolder,
    *,
    axes: Sequence[int] | None = None,
    options: FFTOptions,
    execution: ExecutionCPU | ExecutionCUDA,
    inplace: bool | None = None,
) -> tuple[tuple, PlanTraits]:
    """
    (private method) Compute plan_args and plan_traits for FFT operations.

    Args:
        operand: The input operand wrapped as a TensorHolder.
        axes: The axes along which to perform the FFT.
        options: Normalized FFTOptions object (use setup_options to normalize).
        execution: ExecutionCPU or ExecutionCUDA object.
        inplace: Whether the operation is in-place. If None, computed from
            options.inplace with cross-device override.

    Returns:
        tuple: (plan_args, plan_traits) containing the plan arguments and traits.
    """
    fft_abstract_type = _get_default_fft_abstract_type(operand.dtype, options.fft_type)
    if axes is None:
        axes = range(len(operand.shape))
    else:
        operand_dim = len(operand.shape)
        # Mirror FFT.__init__: enforce bounds and support negative indices.
        if any(axis >= operand_dim or axis < -operand_dim for axis in axes):
            raise ValueError(f"The specified FFT axes {tuple(axes)} are out of bounds for a {operand_dim}-D tensor.")
        axes = tuple(axis % operand_dim for axis in axes)

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

    # If inplace is not explicitly provided, use options.inplace.
    # For cross-device, inplace is always True (the operand needs to be copied once anyway).
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

    return plan_args, plan_traits


def create_fft_key(
    plan_args: tuple,
    execution: ExecutionCPU | ExecutionCUDA,
    *,
    prolog: DeviceCallable | None = None,
    epilog: DeviceCallable | None = None,
) -> tuple[tuple, tuple | None, tuple]:
    """
    Create an FFT key from plan_args and execution options.

    This key is not designed to be serialized and used on a different machine. It is meant
    for runtime use only.

    It is the user's responsibility to augment this key with the stream in case they use
    stream-ordered memory pools.

    Args:
        plan_args: The plan arguments from create_xt_plan_args.
        execution: The execution options (ExecutionCPU or ExecutionCUDA).
        prolog: Optional prolog callback.
        epilog: Optional epilog callback.

    Returns:
        tuple: (plan_args, callable_data, execution_tuple) representing the FFT key.
    """
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


_CUDA_TYPES_TO_CUFFT_TYPE = {
    (cudaDataType.CUDA_C_32F, cudaDataType.CUDA_C_32F): cufft.Type.C2C,
    (cudaDataType.CUDA_R_32F, cudaDataType.CUDA_C_32F): cufft.Type.R2C,
    (cudaDataType.CUDA_C_32F, cudaDataType.CUDA_R_32F): cufft.Type.C2R,
    (cudaDataType.CUDA_C_64F, cudaDataType.CUDA_C_64F): cufft.Type.Z2Z,
    (cudaDataType.CUDA_R_64F, cudaDataType.CUDA_C_64F): cufft.Type.D2Z,
    (cudaDataType.CUDA_C_64F, cudaDataType.CUDA_R_64F): cufft.Type.Z2D,
}


def estimate_workspace_size(
    key: tuple[tuple, tuple | None, tuple],
    *,
    technique: str = "default",
    handle: int | None = None,
) -> int:
    """
    Estimate the workspace size in bytes for the given FFT key.

    Args:
        key: The FFT key returned by :meth:`FFT.create_key`.

        technique: The estimation technique to use.

          - ``"default"``: Uses ``cufftEstimateMany``, which returns
            the workspace size for default plan settings without
            requiring a handle. Does not support half or bfloat16
            precision keys.
          - ``"refined"``: Uses ``cufftXtGetSizeMany``, which requires
            a ``handle`` and returns the workspace size accounting for
            any settings in the handle (e.g., plan properties).
            With a freshly-created handle, this returns the
            same value as ``"default"``.

        handle: A cuFFT handle as returned by
          ``nvmath.bindings.cufft.create()``. Required when
          ``technique="refined"``; ignored otherwise. The caller is
          responsible to ensure the handle was created on the same
          CUDA device that the key targets.

    Returns:
        int: The estimated workspace size in bytes.

    Semantics:
        - This function does not create a full plan. The callback data
          (prolog/epilog) in the key is ignored, as the underlying
          cuFFT size estimation APIs do not account for callbacks.

        - The returned value is 0 in two cases: (1) the key specifies
          CPU execution, since workspace management is not applicable to
          the CPU backend, or (2) the key specifies CUDA execution but
          no temporary workspace is needed for the given FFT configuration.

    .. seealso::
        :meth:`FFT.create_key`
    """
    if technique not in ("default", "refined"):
        raise ValueError(f"technique must be 'default' or 'refined', got {technique!r}.")

    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError(f"The key must be a 3-tuple as returned by FFT.create_key(). Got {key}.")

    plan_args, _, execution_tuple = key
    if not isinstance(plan_args, tuple) or len(plan_args) != 12:
        raise ValueError("The first element of the key (plan_args) must be a 12-element tuple as returned by FFT.create_key().")
    if not isinstance(execution_tuple, tuple) or len(execution_tuple) < 1:
        raise ValueError("The third element of the key (execution) must be a non-empty tuple as returned by FFT.create_key().")

    assert execution_tuple[0] in ("cpu", "cuda"), (
        f"Internal error: expected execution_tuple[0] to be 'cuda' or 'cpu', got {execution_tuple[0]!r}."
    )
    if execution_tuple[0] == "cpu":
        return 0

    if technique == "refined":
        if handle is None:
            raise ValueError("A cuFFT handle is required when technique='refined'.")
        return cufft.xt_get_size_many(handle, *plan_args)

    # technique == "default"
    inputtype = plan_args[5]
    outputtype = plan_args[9]
    cufft_type = _CUDA_TYPES_TO_CUFFT_TYPE.get((inputtype, outputtype))
    if cufft_type is None:
        raise ValueError(
            f"technique='default' (cufftEstimateMany) does not support "
            f"the data types in this key (inputtype={inputtype}, "
            f"outputtype={outputtype}). Use technique='refined' instead."
        )
    return cufft.estimate_many(
        plan_args[0],  # rank
        plan_args[1],  # n
        plan_args[2],  # inembed
        plan_args[3],  # istride
        plan_args[4],  # idist
        plan_args[6],  # onembed
        plan_args[7],  # ostride
        plan_args[8],  # odist
        cufft_type,
        plan_args[10],  # batch
    )


def set_prolog_and_epilog(handle, prolog, epilog, operand_dtype, result_dtype, logger):
    def set_callback(cbkind, cbobj, dtype):
        if cbobj is None:
            return

        assert cbkind in ["prolog", "epilog"], "Internal error."
        CBType = CBLoadType if cbkind == "prolog" else CBStoreType

        try:
            cufft.xt_set_jit_callback(handle, 0, cbobj.ltoir, cbobj.size, CBType[dtype.upper()], [cbobj.data])
        except _bindings_utils.FunctionNotFoundError as e:
            version = cufft.get_version()
            raise RuntimeError(
                f"The currently running cuFFT version {version} does not support LTO callbacks. \n"
                f"cuFFT LTO callbacks are supported starting with cuFFT 11.3, "
                f"shipped with CUDA Toolkit 12.6U2 (11.3.0) or newer. \n"
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
    specification (see :meth:`reset_operand`, :meth:`reset_operand_unchecked`,
    and :meth:`create_key` for more details).

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

    .. versionchanged:: 0.9.0
        The `operand` parameter is now positional-only.

    Args:
        operand: {operand}

        axes: {axes}

        options: {options}

        execution: {execution}

        stream: {stream}

    .. seealso::
        :meth:`plan`, :meth:`reset_operand`, :meth:`reset_operand_unchecked`,
        :meth:`release_operand`, :meth:`execute`, :meth:`create_key`

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
        /,
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

        # Key and plan_args computed from the user's original operand
        # before any copy across memory spaces occurs.
        #
        # When a copy occurs (cross-device or C2R), the copied operand may have different
        # strides than the original. We always compute the key from the user's original
        # operand to match what create_key() does. This ensures:
        # 1. get_key() returns the same value as create_key() for the same operand
        # 2. reset_operand() validates against the user-facing key, not the internal copy
        # We only need one key because reset_operand() validation
        # ensures the new operand produces the same key.
        self.key_from_user_operand: tuple[tuple, tuple | None, tuple] | None = None
        self.plan_args_from_user_operand: tuple | None = None

        # Compute from original operand whenever a copy will occur:
        # - Cross-device: .to() may change strides
        # - C2R: copy preserves strides, but we compute for consistency
        if self.memory_space != self.execution_space or self.fft_abstract_type == "C2R":
            try:
                # Pass inplace explicitly to match what create_key() does.
                self.plan_args_from_user_operand, _ = _compute_fft_plan_args_and_traits(
                    operand,
                    axes=self.axes,
                    options=self.options,
                    execution=self.execution_options,
                    inplace=self.options.inplace,
                )
                self.key_from_user_operand = create_fft_key(self.plan_args_from_user_operand, self.execution_options)
            except UnsupportedLayoutError:
                # If the layout is unsupported, we won't be able to compute a key.
                # This is fine - validation will fall back to other checks.
                self.key_from_user_operand = None
                self.plan_args_from_user_operand = None

        # Copy the operand to execution_space's device if needed.
        self.operand, self.operand_backup = _allocate_operand_or_identity(
            operand,
            operand_stream_holder,
            self.execution_space,
            self.memory_space,
            self.device_id,
            self.fft_abstract_type,
            self.logger,
        )

        # Track whether the user has called release_operand(). This flag is
        # checked in _check_valid_operand to prevent execution after the user
        # has released their operand. It is cleared by reset_operand() and
        # reset_operand_unchecked().
        self._operand_released = False

        # For C2R transforms, cuFFT and FFTW overwrite the input buffer
        # during execute(). This flag marks the operand as stale after each
        # execute() call so that users must call reset_operand() or
        # reset_operand_unchecked() before executing again.
        self._c2r_operand_stale = False

        # For C2R with same-space CUDA, we allocated an auxiliary buffer above
        # on operand_stream_holder. Track its allocation stream so we can ensure
        # proper ordering, for example when the buffer is freed in free().
        self.c2r_buffer_stream = None
        if self.execution_space == "cuda" and self.execution_space == self.memory_space and self.fft_abstract_type == "C2R":
            assert operand_stream_holder is not None
            self.c2r_buffer_stream = operand_stream_holder.obj

        operand = self.operand
        # Class invariant: self.internal_operand_layout always reflects the layout of
        # self.operand. For same-space this is the user's operand; for
        # cross-space it is the execution-space copy.
        self.internal_operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)

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

        # Derive plan_args_from_user_operand from plan_traits if not already set.
        if self.plan_args_from_user_operand is None:
            self.plan_args_from_user_operand = create_xt_plan_args(
                plan_traits=self.plan_traits,
                fft_abstract_type=self.fft_abstract_type,
                operand_data_type=self.operand_data_type,
                inplace=self.options.inplace,
            )

        # Ensure key_from_user_operand is set (for same-device case
        # and cross-device fallback).
        if self.key_from_user_operand is None:
            self.key_from_user_operand = create_fft_key(self.plan_args_from_user_operand, self.execution_options)

        self.logger.info(
            f"The operand data type = {self.operand_data_type}, shape = {self.internal_operand_layout.shape}, and "
            f"strides = {self.internal_operand_layout.strides}."
        )
        result_data_type, result_shape, result_strides = (
            (self.operand_data_type, self.internal_operand_layout.shape, self.internal_operand_layout.strides)
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

        self.valid_state = True
        self.logger.info("The FFT operation has been created.")

    def _check_planned(self, *args, **kwargs):
        """ """
        what = kwargs["what"]
        if not self.fft_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    @utils.precondition(_check_planned, "Returning the key")
    def get_key(self, *, prolog: DeviceCallable | None = None, epilog: DeviceCallable | None = None):
        """
        Get the key based on the current state of this FFT object,
        supplemented with the callbacks. The key is computed from the object's
        current plan, execution options, and operand properties.

        Args:
            prolog: {prolog}
            epilog: {epilog}

        Returns:
            {fft_key}

        .. seealso::
            :meth:`create_key`, :func:`estimate_workspace_size`
        """
        # plan_args_from_user_operand is guaranteed to be set after __init__
        return create_fft_key(
            self.plan_args_from_user_operand,  # type: ignore[arg-type]
            self.execution_options,
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

        .. seealso::
            :meth:`get_key`, :func:`estimate_workspace_size`
        """
        operand = tensor_wrapper.wrap_operand(operand)
        options, execution = setup_options(operand, options, execution)
        plan_args, _ = _compute_fft_plan_args_and_traits(
            operand, axes=axes, options=options, execution=execution, inplace=options.inplace
        )
        return create_fft_key(plan_args, execution, prolog=prolog, epilog=epilog)

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
            return package_name if package_name != "numpy" else "cuda"
        else:
            return package_name if package_name != "cupy" else "cupy_host"

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

            # Set stream for the FFT.
            cufft.set_stream(self.handle, stream_holder.ptr)

            # Set LTO-IR callbacks, if present.
            prolog = utils.check_or_create_options(DeviceCallable, prolog, "prolog", keep_none=True)
            epilog = utils.check_or_create_options(DeviceCallable, epilog, "epilog", keep_none=True)
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

        if log_debug:
            self.logger.debug(f"The FFT key (sans callback) is {self.key_from_user_operand}.")

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

    def _maybe_wait_c2r_aux_buffer_on_last_compute(self):
        """
        (private) Make the C2R same-space auxiliary buffer's alloc stream
        wait on ``self.last_compute_event``, so any in-flight compute using
        the buffer is ordered before its (potentially stream-ordered-async)
        free. No-op in every other configuration, or when there is no
        outstanding compute event.
        """
        if (
            self.execution_space == self.memory_space
            and self.fft_abstract_type == "C2R"
            and self.c2r_buffer_stream is not None
            and self.last_compute_event is not None
        ):
            self.c2r_buffer_stream.wait(self.last_compute_event)

    def _reset_operand_validate(self, operand):
        """(private method) Validate operand compatibility.
        This method is used by :meth:`reset_operand` to validate that the new
        user-provided operand satisfies all the requirements listed in
        the :meth:`reset_operand` docstring.

        Args:
            operand: The new operand tensor to validate.

        Raises:
            TypeError: If the library package or data type doesn't match.
            ValueError: If the device doesn't match or if the operand's traits
                        are incompatible with the original operand.
        """

        # Validate package match
        package = utils.infer_object_package(operand.tensor)
        if self.package != package:
            message = f"Library package mismatch: '{self.package}' => '{package}'"
            raise TypeError(message)

        # Validate data type match
        utils.check_attribute_match(self.operand_data_type, operand.dtype, "data type")

        # Validate device match
        # In principle, we could support memory_space change, but it would require
        # updating self.memory_space and dependent properties like self.blocking,
        # which could be error-prone and would prevent inplace optimizations.
        operand_device_id = operand.device_id
        if operand_device_id != self.operand_device_id:

            def device_str(device_id: int | Literal["cpu"]) -> str:
                return f"cuda:{device_id}" if isinstance(device_id, int) else f"{device_id}"

            raise ValueError(
                f"The new operand must be on the same device as the original one. "
                f"The new operand's device is {device_str(operand_device_id)}, "
                f"the original device is {device_str(self.operand_device_id)}"
            )

        # Validate FFT plan compatibility using key_from_user_operand,
        # matches create_key() behavior.
        # key_from_user_operand is always set after __init__.
        assert self.key_from_user_operand is not None, "Internal error: key_from_user_operand not set."

        try:
            # Pass inplace explicitly to match what create_key() does.
            new_plan_args, _ = _compute_fft_plan_args_and_traits(
                operand,
                axes=self.axes,
                options=self.options,
                execution=self.execution_options,
                inplace=self.options.inplace,
            )
            new_key = create_fft_key(new_plan_args, self.execution_options)
        except UnsupportedLayoutError:
            new_key = None

        if self.key_from_user_operand != new_key:
            self.logger.debug(f"The FFT key corresponding to the original operand is: {self.key_from_user_operand}.")
            if new_key is None:
                self.logger.debug(
                    "The FFT key for the new operand cannot be computed since the layout "
                    f"(shape = {operand.shape}, strides = {operand.strides}) and axes = {self.axes} combination "
                    "is unsupported."
                )
            else:
                self.logger.debug(f"The FFT key corresponding to the new operand is: {new_key}.")
            raise ValueError(
                "The new operand's traits (data type, shape, or strides) are incompatible with that of the original operand."
            )

        # Validation passed. No updates needed because:
        # - key_from_user_operand == new_key
        # - plan_args_from_user_operand produces same key, get_key() returns correct value
        # - plan_traits stays unchanged (copy produces the same layout)
        self.logger.debug("Validation passed; no updates needed since keys match.")

    def _update_plan_traits_for_new_operand(self, new_operand_shape, new_operand_strides):
        """
        (private) Recompute plan_traits.result_shape/strides after
        a reset whose operand has different properties but a matching key.

        Args:
            new_operand_shape: Shape of the internal buffer. Batch
                dimensions may differ from the original (e.g.
                (2,3,64) -> (3,2,64) with axes=(2,)).
            new_operand_strides: Strides of the internal buffer. Used to
                recompute the result axis order.

        The FFT axis sizes in result_shape may differ from the operand
        (R2C/C2R), so we preserve them from the old result_shape and only
        replace the batch dimensions from the new operand.

        The axis order used to compute result_strides depends on the
        layout strategy chosen at plan time (stored in
        plan_traits.optimized_result_layout):
        - Natural: axis_order_in_memory (mirrors the operand layout).
        - Optimized: reversed FFT axes + batch axes sorted by stride.
        """
        new_shape = list(new_operand_shape)
        old_result_shape = self.plan_traits.result_shape
        # Preserve FFT axis sizes (they differ from operand for R2C/C2R).
        for ax in self.plan_traits.ordered_axes:
            new_shape[ax] = old_result_shape[ax]
        new_result_shape = tuple(new_shape)

        if self.plan_traits.optimized_result_layout:
            fft_axes_set = set(self.plan_traits.ordered_axes)
            batch_axes = [a for a in range(len(new_operand_shape)) if a not in fft_axes_set]
            axis_order = tuple(
                list(reversed(self.plan_traits.ordered_axes))
                + sorted(batch_axes, key=lambda v: (new_operand_strides[v], new_operand_shape[v]))
            )
        else:
            axis_order = axis_order_in_memory(new_operand_shape, new_operand_strides)

        self.plan_traits.result_shape = new_result_shape
        self.plan_traits.result_strides = calculate_strides(new_result_shape, axis_order)

    def _update_layout_and_plan_traits_if_needed(self):
        """
        (private) Update internal_operand_layout from the current internal operand,
        and recompute plan_traits if shape or strides changed.

        The comparison is against self.operand (the internal buffer), not the
        user's original operand. For cross-space scenarios, the internal
        buffer is always a contiguous copy, so stride-only user changes are
        invisible here — which is correct because cuFFT sees contiguous data.

        When does _update_plan_traits_for_new_operand get called?
        The columns below refer to the *internal buffer* (self.operand),
        not the user's original operand.

                                                  internal   internal
                                                  shape      strides   plan_traits
            Scenario (user's operand)             changed    changed   updated
            --------------------------------      --------   --------  -----------
            Same operand (no change)              No         No        No
            Different shape, same-space           Yes        Yes       Yes
            Different shape, cross-space          Yes        Yes       Yes
            Different strides only, same-space    No         Yes       Yes
            Different strides only, cross-space   No         No        No
              (cross-space copy is always contiguous, so internal strides
               are unchanged regardless of the user's strides)
        """
        new_layout = TensorLayout(shape=self.operand.shape, strides=self.operand.strides)
        if tuple(self.internal_operand_layout.shape) != tuple(new_layout.shape) or tuple(
            self.internal_operand_layout.strides
        ) != tuple(new_layout.strides):
            self._update_plan_traits_for_new_operand(new_layout.shape, new_layout.strides)

        # Always update: the new operand may have a different layout but a
        # compatible key. Class invariant: self.internal_operand_layout
        # always reflects the layout of the currently stored operand.
        self.internal_operand_layout = new_layout

    def _reset_operand_set_stream_and_update_operand(self, operand, stream: AnyStream | None):
        """(private method) Set the stream, copy the operand, and update layout information.
        This method is used by ``reset_operand()`` after validation has passed.
        It performs the following operations:

        1. Gets or creates the execution and operand streams
        2. Sets the stream for the cuFFT handle (for CUDA execution)
        3. Copies the operand tensor if necessary (reallocates when shape changed)
        4. Updates internal_operand_layout and plan_traits if needed

        Args:
            operand: The validated operand tensor to use.
            stream: Optional stream to use for execution.
                    If None, a stream is created or retrieved.
        """

        exec_stream_holder, operand_stream_holder = self._get_or_create_stream_maybe(stream)
        self.logger.info(
            "The specified stream for reset_operand() is "
            f"{(exec_stream_holder or operand_stream_holder) and (exec_stream_holder or operand_stream_holder).obj}."  # type: ignore[union-attr]
        )

        # Allocate a new internal buffer when the previous one was released
        # or when the new operand has a different shape (e.g., same FFT key but
        # rearranged batch dimensions). copy_() rejects incompatible shapes
        # due to broadcasting, and the underlying tensor type varies across
        # backends, so there is no uniform reshape; we reallocate instead.
        needs_reallocation = self._operand_released or self.operand.shape != operand.shape
        if needs_reallocation:
            # When the shape changes, the old self.operand buffer is implicitly
            # released here (replaced by the new allocation). For the C2R
            # same-space case this is an internal aux buffer the user cannot
            # stream-order against; in the other cases the old buffer is
            # either the user's tensor (same-space non-C2R, user orders) or a
            # cross-space mirror (cross-space operations are blocking).
            # When the operand was already released, release_operand() already
            # performed this ordering, so we skip it here.
            if not self._operand_released:
                self._maybe_wait_c2r_aux_buffer_on_last_compute()
            self.operand, self.operand_backup = _allocate_operand_or_identity(
                operand,
                operand_stream_holder,
                self.execution_space,
                self.memory_space,
                self.device_id,
                self.fft_abstract_type,
                self.logger,
            )
        else:
            self.operand, self.operand_backup = _copy_operand_or_identity(
                self.operand,
                operand,
                operand_stream_holder,
                self.execution_space,
                self.memory_space,
                self.fft_abstract_type,
                self.logger,
            )

        # No-op for non-C2R; for C2R same-space, the operand copy above
        # provides fresh data so the stale flag can be cleared.
        self._c2r_operand_stale = False

        if (
            needs_reallocation
            and self.execution_space == "cuda"
            and self.execution_space == self.memory_space
            and self.fft_abstract_type == "C2R"
        ):
            assert operand_stream_holder is not None
            self.c2r_buffer_stream = operand_stream_holder.obj

        self._update_layout_and_plan_traits_if_needed()

        # Log final result layout
        self.logger.info(
            f"The reset operand shape = {self.internal_operand_layout.shape}, "
            f"and strides = {self.internal_operand_layout.strides}."
        )
        result_shape, result_strides = (
            (self.internal_operand_layout.shape, self.internal_operand_layout.strides)
            if self.inplace
            else (self.plan_traits.result_shape, self.plan_traits.result_strides)
        )
        self.logger.info(f"The result shape = {result_shape}, and strides = {result_strides}.")
        self.logger.info("The operand has been reset to the specified operand.")

    @utils.precondition(_check_valid_fft)
    def reset_operand(self, operand, *, stream: AnyStream | None = None):
        """
        Reset the operand held by this :class:`FFT` instance to a new compatible
        operand for subsequent execution.

        Args:
            operand: A tensor (ndarray-like object) compatible with the previous one.
                The new operand is considered compatible if all the
                following properties match with the previous one:

                - The problem specification key for the new operand. Generally the keys will
                  match if the operand shares the same layout (shape, strides and data
                  type). The keys may still match for certain operands with different
                  layout, see :meth:`create_key` for details.
                - The package that the new operand belongs to.
                - The memory space of the new operand (CPU or GPU).
                - The device that new operand belongs to if it is on GPU.

            stream: {stream}

        Semantics:
            - If execution space == memory space and the FFT is not a C2R transform:
              operand reference update with no data copying.

            - If execution space == memory space, the FFT is a C2R transform:
              one data copy to an auxiliary tensor, required to prevent cuFFT from
              overwriting the user's input.

            - If execution space != memory space:
              data must be copied between different memory spaces.

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
            However it still performs validation to ensure that the operand is compatible
            with the original, and, if enabled, logging. See :meth:`reset_operand_unchecked`
            for an alternative when the caller has already validated the operand or chooses
            to skip validation and logging.

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

        .. seealso::
            :meth:`reset_operand_unchecked`, :meth:`release_operand`
        """

        self.logger.info("Resetting operand...")
        if operand is None:
            raise ValueError("Resetting operand requires a valid operand. Use release_operand() to release the operand.")

        operand = tensor_wrapper.wrap_operand(operand)
        self._reset_operand_validate(operand)
        self._reset_operand_set_stream_and_update_operand(operand, stream)
        self._operand_released = False

    def reset_operand_unchecked(self, operand, *, stream: AnyStream | None = None):
        """
        .. experimental:: method

        This method is a performance-optimized alternative to :meth:`reset_operand` that
        eliminates validation and logging overhead, making it ideal for
        performance-critical loops where operand compatibility is guaranteed by the caller.

        Args:
            operand: A tensor (ndarray-like object) that is **guaranteed** by the user
                to be compatible with the original operand used during planning.
                See the ``operand`` parameter in :meth:`reset_operand` for the definition
                of compatibility.

            stream: {stream}

        Returns:
            None

        Semantics:
            The semantics are the same as in :meth:`reset_operand`,
            except that this method does not perform any validation (e.g. package
            match, data type match, key match, etc.) or logging.

        When to Use:
            - Performance-critical loops with repeated FFT executions on different operands

            - After verifying correctness with :meth:`reset_operand` during development

            - When operand compatibility is guaranteed by construction or invariant

        Examples:

            **Example 1: Optimizing a processing loop**

            .. code-block:: python

                import cupy as cp
                import nvmath

                shape = (1024, 1024)
                operand = cp.random.rand(*shape, dtype=cp.complex64)

                fft = nvmath.fft.FFT(operand, execution="cuda")
                with fft:
                    fft.plan()
                    for i in range(10000):
                        # Process and create new operand with the same shape, dtype,
                        # and device as the original operand
                        new_operand = process_data(...)
                        fft.reset_operand_unchecked(new_operand)
                        result = fft.execute()
                        # block until the result is ready
                        ...

            **Example 2: Streaming data processing**

            Processing a stream of incoming data operands with identical layout:

            .. code-block:: python

                import cupy as cp
                import nvmath

                # Create a stateful FFT object and prepare it once.
                shape = (512, 512)
                initial_operand = cp.empty(shape, dtype=cp.complex64)

                fft = nvmath.fft.FFT(initial_operand, execution="cuda")
                with fft:
                    fft.plan()

                    # Process stream of incoming operands
                    for operand in incoming_data_stream():
                        # The user guarantees that the operand is compatible
                        # with the original (same shape, dtype, device, ...).
                        fft.reset_operand_unchecked(operand)
                        result = fft.execute()
                        # block until the result is ready
                        process_spectrum(result)
                        ...

        .. seealso::
            :meth:`reset_operand`: Safe, validated method for changing operands.
            :meth:`release_operand`: Release the internal references to the operand.
            :meth:`create_key`: For understanding FFT key compatibility.
            :func:`estimate_workspace_size`: For estimating workspace size.
        """
        # Case 1: Same execution and memory space, non-C2R transform
        # The user guarantees that the operand is in the same memory space as the original
        # operand so we can directly update the operand reference without copying data.
        # The TensorHolder wrapper is always alive (release_operand only clears
        # .tensor), so we can unconditionally swap the inner tensor.
        if self.execution_space == self.memory_space and self.fft_abstract_type != "C2R":
            self.operand.tensor = operand
            self.operand_backup = None
            self._operand_released = False
            self._update_layout_and_plan_traits_if_needed()
            return

        # Cases 2 and 3 require data copying, so we need the stream
        _, operand_stream_holder = self._get_or_create_stream_maybe(stream)
        # and also require the wrapped operand
        operand_wrapped = tensor_wrapper.wrap_operand(operand)

        # Case 2: C2R transform with same execution and memory space
        # Data must be copied to prevent cuFFT from overwriting the user's input buffer.
        # This is a corner case that stems from cuFFT behavior.
        if self.execution_space == self.memory_space:
            needs_reallocation = self._operand_released or self.operand.shape != operand_wrapped.shape
            if needs_reallocation:
                self.operand = utils.create_empty_tensor(
                    operand_wrapped.__class__,
                    operand_wrapped.shape,
                    operand_wrapped.dtype,
                    self.device_id,
                    operand_stream_holder,
                    verify_strides=True,
                    strides=operand_wrapped.strides,
                )
                if self.execution_space == "cuda":
                    self.c2r_buffer_stream = operand_stream_holder.obj  # type: ignore[union-attr]

            self.operand.copy_(operand_wrapped, stream_holder=operand_stream_holder)
            self.operand_backup = None
            self._operand_released = False
            self._c2r_operand_stale = False
            self._update_layout_and_plan_traits_if_needed()
            return

        # Case 3: Cross-space scenario (execution_space != memory_space)
        # Example: CPU operand with CUDA execution, or CUDA operand with CPU execution.
        # Data must be copied between memory spaces.
        needs_reallocation = self._operand_released or self.operand.shape != operand_wrapped.shape
        if needs_reallocation:
            if self.execution_space == "cuda":
                to_device = self.device_id
            else:
                to_device = "cpu"
            self.operand = operand_wrapped.to(to_device, operand_stream_holder)
        else:
            self.operand.copy_(operand_wrapped, stream_holder=operand_stream_holder)
        self.operand_backup.tensor = operand_wrapped.tensor
        self._operand_released = False
        self._c2r_operand_stale = False
        self._update_layout_and_plan_traits_if_needed()

    @utils.precondition(_check_valid_fft)
    def release_operand(self):
        """
        {release_operand}
        """
        # We release the references to the user-provided
        # operand and/or GPU mirrors of the user-provided operand
        # and/or the internal auxiliary buffer for C2R transforms
        # since it can be non-negligible memory.
        # Note that we do not release the whole wrapper objects,
        # but only their internal tensor references, which allow
        # us to reuse them without re-wrapping saving some overhead.

        # Case 1 (same-space, non-C2R):
        #   self.operand is the user's tensor,
        #   self.operand_backup is None.
        # Case 2 (same-space, C2R):
        #   self.operand references the internal auxiliary buffer
        #   The user's tensor is not referenced. We must ensure the compute
        #   that used this buffer has completed before releasing it.
        # Case 3 (cross-space):
        #   self.operand is an internal mirror, self.operand_backup
        #   is the user's tensor. Release both.

        self._maybe_wait_c2r_aux_buffer_on_last_compute()
        self.c2r_buffer_stream = None
        self.operand.tensor = None
        if self.operand_backup is not None:
            self.operand_backup.tensor = None

        self._operand_released = True
        self.logger.info("User-provided operand has been released.")

    def get_input_layout(self):
        """
        Returns a pair of tuples: shape and strides of the FFT input.

        .. note::
            In some cases, the FFT operation requires taking a copy of the input tensor
            (e.g. C2R cuFFT, or provided tensor resides on CPU but FFT is executed on GPU).
            The copied tensor strides may differ from the input tensor passed by the user,
            if the original tensor's strides do not conform to dense C-like layout.
        """
        return self.internal_operand_layout.shape, self.internal_operand_layout.strides

    def get_output_layout(self):
        """
        Returns a pair of tuples: shape and strides of the FFT output.
        """
        return (
            (self.internal_operand_layout.shape, self.internal_operand_layout.strides)
            if self.inplace
            else (self.plan_traits.result_shape, self.plan_traits.result_strides)
        )

    def _check_valid_operand(self, *args, **kwargs):
        """ """
        what = kwargs["what"]
        if self._operand_released:
            raise RuntimeError(
                f"{what} cannot be performed after the operand has been released. Use reset_operand() to provide a new "
                f"operand before performing the {what.lower()}."
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

        if self.fft_abstract_type == "C2R" and self._c2r_operand_stale:
            raise RuntimeError(
                "For C2R FFTs, execute() cannot be called multiple times without "
                "resetting the operand in between. Please call reset_operand() or "
                "reset_operand_unchecked() before executing again."
            )

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

        if self.fft_abstract_type == "C2R":
            self._c2r_operand_stale = True

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
            # Ensure ordering with respect to the last computation
            # to avoid race conditions when releasing internal resources.
            self._maybe_wait_c2r_aux_buffer_on_last_compute()
            if self.last_compute_event is not None and self.workspace_stream is not None:
                self.workspace_stream.wait(self.last_compute_event)
            self.last_compute_event = None

            self._free_workspace_memory()

            if self.handle is not None:
                if self.execution_space == "cuda":
                    cufft.destroy(self.handle)
                else:
                    fftw.destroy(self.handle)
                self.handle = None

            # Set all attributes to None except for logger and valid_state
            _keep = {"logger", "valid_state"}
            for attr in list(vars(self)):
                if attr not in _keep:
                    setattr(self, attr, None)

        except Exception as e:
            self.logger.critical("Internal error: only part of the FFT object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The FFT object's resources have been released.")


def _fft(
    operand,
    /,
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

    .. seealso::
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
        - This function is a convenience wrapper around :class:`FFT` and is specifically
          meant for *single* use. The same computation can be performed with the stateful
          API using the default `direction` argument in :meth:`FFT.execute`.

    Further examples can be found in the `nvmath/examples/fft
    <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft>`_ directory.
    """
    if check_dtype is not None:
        assert check_dtype in {"real", "complex"}, "internal error"
        wrapped = tensor_wrapper.wrap_operand(operand)
        if ("complex" in wrapped.dtype) != (check_dtype == "complex"):
            raise ValueError(f"This function expects {check_dtype} operand, found {wrapped.dtype}")

    with FFT(operand, axes=axes, options=options, execution=execution, stream=stream) as fftobj:
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
    /,
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

    .. versionchanged:: 0.9.0
        The ``operand`` parameter is now positional-only.

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


    .. seealso::
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

    .. seealso::
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
        - This function is a convenience wrapper around :class:`FFT` and is specifically
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
        operand,
        axes=axes,
        direction=FFTDirection.INVERSE,
        options=options,
        execution=execution,
        prolog=prolog,
        epilog=epilog,
        stream=stream,
        check_dtype="complex",
    )
