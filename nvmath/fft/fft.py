# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ['FFT', 'fft', 'ifft', 'rfft', 'irfft', 'UnsupportedLayoutError']

from dataclasses import dataclass
import enum
import functools
import logging
import operator
from typing import Sequence

import cupy as cp

from nvmath.bindings import cufft
from nvmath.fft import configuration
from nvmath import memory

from nvmath._internal import formatters
from nvmath._internal import tensor_wrapper
from nvmath._internal.typemaps import NAME_TO_DATA_TYPE
from nvmath._internal import utils

class UnsupportedLayoutError(Exception):
   """
   Error type for layouts not supported by the library.

   Args:
       message: The error message.
       permutation: The permutation needed to convert the input layout to a supported layout to the FFT operation. The same
           permutation needs to be applied to the result to obtain the axis sequence corresponding to the non-permuted input.
       axes: The dimensions along which the FFT is performed corresponding to the permuted operand layout.
   """
   def __init__(self, message, permutation, axes):
       self.message = message
       self.permutation = permutation
       self.axes = axes

   def __str__(self):
       return self.message

@dataclass
class TensorLayout:
    """An internal data class for capturing the tensor layout.
    """
    shape : Sequence[int] = None
    strides : Sequence[int] = None

@dataclass
class PlanTraits:
    """An internal data class for capturing FFT plan traits.
    """
    result_shape : Sequence[int] = None
    result_strides : Sequence[int] = None
    ordered_axes : Sequence[int] = None
    ordered_fft_in_shape : Sequence[int] = None
    ordered_fft_in_embedding_shape : Sequence[int] = None
    ordered_fft_out_shape : Sequence[int] = None
    fft_batch_size : int = None
    istride : int = None
    idistance : int = None
    ostride : int = None
    odistance : int = None

class CBLoadType(enum.IntEnum):
    COMPLEX64  = 0x0,
    COMPLEX128 = 0x1,
    FLOAT32    = 0x2,
    FLOAT64    = 0x3,
    UNDEFINED  = 0x8

class CBStoreType(enum.IntEnum):
    COMPLEX64  = 0x4,
    COMPLEX128 = 0x5,
    FLOAT32    = 0x6,
    FLOAT64    = 0x7,
    UNDEFINED  = 0x8

SHARED_FFT_DOCUMENTATION = utils.COMMON_SHARED_DOC_MAP.copy()
SHARED_FFT_DOCUMENTATION.update({
    'axes':
        "The dimensions along which the FFT is performed. Currently, it is required that the axes are contiguous "
        "and include the first or the last dimension. Only up to 3D FFTs are supported.",
    'options':
        "Specify options for the FFT as a :class:`FFTOptions` object. "
        "Alternatively, a `dict` containing the parameters for the ``FFTOptions`` constructor can also be provided. "
        "If not specified, the value will be set to the default-constructed ``FFTOptions`` object.",
    'prolog':
        "Provide device-callable function in LTO-IR format to use as load-callback as an object of type :class:`DeviceCallable`. "
        "Alternatively, a `dict` containing the parameters for the ``DeviceCallable`` constructor can also be provided. The default is no prolog.",
    'epilog':
        "Provide device-callable function in LTO-IR format to use as store-callback as an object of type :class:`DeviceCallable`. "
        "Alternatively, a `dict` containing the parameters for the ``DeviceCallable`` constructor can also be provided. The default is no epilog.",
    'direction':
        "Specify whether forward or inverse FFT is performed (`FFTDirection` object, or as a string from ['forward', 'inverse'], "
        "or as an int from [-1, 1] denoting forward and inverse directions respectively).",
    'fft_key': "A tuple as the key to represent the input FFT problem."
})

def complex_to_real_equivalent(name):
    assert 'complex' in name, f"Internal Error ({name=})"
    m = name.split('complex')
    assert len(m) in (1, 2)
    size = int(m[-1]) // 2
    if len(m) == 1:
        return f'float{size}'
    else:
        return f'{m[0]}float{size}'

def real_to_complex_equivalent(name):
    assert 'float' in name, f"Internal Error ({name=})"
    m = name.split('float')
    assert len(m) in (1, 2)
    size = int(m[-1])
    if len(m) == 1:
        return f'complex{size*2}'
    else:
        return f'{m[0]}complex{size*2}'

def _get_default_fft_abstract_type(dtype, fft_type):
    if fft_type is not None:
        return fft_type

    f, c = "float", "complex"
    if dtype[:len(f)] == f:
        fft_type = 'R2C'
    elif  dtype[:len(c)] == c:
        fft_type = 'C2C'
    else:
        raise ValueError(f"Unsupported dtype '{dtype}' for FFT.")
    return fft_type

def _get_fft_result_and_compute_types(dtype, fft_abstract_type):
    """
    Return result and compute data type given the input data type and the FFT type.
    """
    if fft_abstract_type == 'C2C':
        return dtype, dtype
    elif fft_abstract_type == 'C2R':
        return complex_to_real_equivalent(dtype), dtype
    elif fft_abstract_type == 'R2C':
        return real_to_complex_equivalent(dtype), dtype
    else:
        raise ValueError(f"Unsupported FFT Type: '{fft_abstract_type}'")

def _get_fft_default_direction(fft_abstract_type):
    """
    Return the default FFT direction (as object of type configuration.FFTDirection) based on the FFT type.
    """
    if fft_abstract_type in ['C2C', 'R2C']:
        return configuration.FFTDirection.FORWARD
    elif fft_abstract_type == 'C2R':
        return configuration.FFTDirection.INVERSE
    else:
        raise ValueError(f"Unsupported FFT Type: '{fft_abstract_type}'")

def _get_size(shape):
    return functools.reduce(operator.mul, shape, 1)

def _get_last_axis_id_and_size(axes, operand_shape, fft_abstract_type, last_axis_size):
    """
    axes                  = The user-specified or default FFT axes.
    operand_shape         = The input operand shape.
    fft_abstract_type     = The "abstract" type of the FFT ('C2C', 'C2R', 'R2C').
    last_axis_size        = For 'C2R' FFTs, specify whether the last axis size is even or odd.

    Returns the last axis ID and the corresponding axis size required for the result.
    """
    last_axis_id = axes[-1]

    if fft_abstract_type == 'C2C':
        return last_axis_id, operand_shape[last_axis_id]

    if fft_abstract_type == 'C2R':
        if last_axis_size == 'even':
            return last_axis_id, 2 * (operand_shape[last_axis_id] - 1)
        elif last_axis_size == 'odd':
            return last_axis_id, 2 * operand_shape[last_axis_id]  - 1
        else:
            assert False, "Unreachable."

    if fft_abstract_type == 'R2C':
        return last_axis_id, operand_shape[last_axis_id] // 2 + 1

def _contiguous_layout(sorted_shape, sorted_strides):
    for s in range(1, len(sorted_strides)):
        if sorted_shape[s-1] * sorted_strides[s-1] != sorted_strides[s]:
            return False
    return True

def contiguous_in_memory(shape, strides):
    """
    Check if the provided (shape, strides) result in a contiguous memory layout.
    """
    sorted_strides, sorted_shape =  zip(*sorted(zip(strides, shape)))
    return _contiguous_layout(sorted_shape, sorted_strides)

def overlapping_layout(shape, strides):
    sorted_strides, sorted_shape =  zip(*sorted(zip(strides, shape)))
    for s in range(1, len(sorted_strides)):
        if sorted_shape[s-1] * sorted_strides[s-1] > sorted_strides[s]:
            return True
    return False

def check_embedding_possible(strides, presorted=False):
    """
    Check if the strides allow for calculating an embedding dimension.
    """
    if not presorted:
        strides = sorted(strides)
    # with a broadcasted view, stride can be 0
    if any(strides[i-1] == 0 for i in range(1, len(strides))):
        return False
    return all(strides[i] % strides[i-1] == 0 for i in range(1, len(strides)))

def check_batch_tileable(sorted_batch_shape, sorted_batch_strides):
    """
    Check if FFT layout is tileable across the specified batch layout.
    """
    return _contiguous_layout(sorted_batch_shape, sorted_batch_strides)

def check_contiguous_layout(axes, strides, shape):
    if not axes:
        return True
    sorted_batch_strides, sorted_batch_shape = zip(*sorted(((strides[a], shape[a]) for a in axes)))
    return _contiguous_layout(sorted_batch_shape, sorted_batch_strides)

def calculate_embedding_shape(shape, strides):
    """
    Calculate the embedding shape for the given shape and strides.
    """
    n = len(strides)
    # The shape is used to resolve cases like (1, 2, 1) : (2, 1, 1) in CuTe notation.
    ordered_strides, _, order = zip(*sorted(zip(strides, shape, range(n))))

    ordered_shape = [ordered_strides[i] // ordered_strides[i-1] for i in range(1, len(ordered_strides))] + [shape[order[-1]]]

    embedding_shape = [0] * n
    for o in range(n):
        embedding_shape[order[o]] = ordered_shape[o]

    return embedding_shape, order

def axis_order_in_memory(shape, strides):
    """
    Compute the order in which the axes appear in memory.
    """
    # The shape is used to resolve cases like (1, 2, 1) : (2, 1, 1) in CuTe notation.
    _, _, axis_order = zip(*sorted(zip(strides, shape, range(len(strides)))))

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

    message = f"To convert to a supported layout, create a transposed view using transpose{permutation} and copy the view into a new tensor, using view.copy() for instance, and use axes={axes}."
    logger.error(message)

    raise UnsupportedLayoutError(message, permutation, axes)

def get_null_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    return logger

def get_fft_plan_traits(operand_shape, operand_strides, operand_dtype, axes, *, fft_abstract_type='C2C', last_axis_size='even', result_layout='optimized', logger=None):
    """
    Extract the FFT shape from the operand shape, compute the ordered axes so that the data is C-contiguous in memory, and compute the result shape and strides.

    operand_shape         = The operand shape
    operand_strides       = The operand strides
    axes                  = The axes over which the FFT is performed. For R2C and C2R transforms, the size of the last axis in `axes` will change.
    fft_abstract_type     = The "abstract" type of the FFT ('C2C', 'C2R', 'R2C').
    last_axis_size        = For 'C2R' FFTs, specify whether the last axis size is even or odd.

    The data needed for creating a cuFFT plan is returned in the following order:
    (result_shape, result_strides), ordered_axes, ordered_fft_in_shape, ordered_fft_out_shape, (istride, idistance), (ostride, odistance)
    """
    logger = logger if logger is not None else get_null_logger('get_fft_plan_traits_null')

    if len(axes) > 3:
        raise ValueError(f"Only upto 3D FFTs are currently supported. You can use the 'axes' option to specify upto three axes along which to perform the FFT. The current number of dimensions is {len(axes)} corresponding to the axes {axes}.")

    # Check for duplicate axis IDs.
    if len(axes) != len(set(axes)):
        raise ValueError(f"The specified FFT axes = {axes} contains duplicate axis IDs, which is not supported.")

    operand_dim = len(operand_shape)
    batch_axes = [axis for axis in range(operand_dim) if axis not in axes]

    # Check if an embedding is possible for the provided operand layout.
    if not check_embedding_possible(operand_strides):
        message = f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} is not currently supported because it does not have a suitable embedding dimension."
        unsupported_layout_exception(operand_dim, axes, message, logger)

    # Compute the embedding shape for the operand.
    operand_embedding_shape, axis_order = calculate_embedding_shape(operand_shape, operand_strides)
    logger.debug(f"The operand embedding shape = {operand_embedding_shape}.")

    # The first or the last *ordered* axis must be present in the specified axes to be able to use the "advanced" layout.
    first, last = axis_order[-1], axis_order[0]
    if first not in axes and last not in axes:
        raise ValueError(f"The first ({first}) or the last ({last}) tensor axis in stride order {axis_order} must be present in the specified FFT axes {axes}.")

    # Compute the embedding input shape for the FFT.
    fft_in_embedding_shape = [operand_embedding_shape[a] for a in axes]

    # Compute the input shape for the FFT.
    fft_in_shape, fft_in_strides = zip(*[(operand_shape[a], operand_strides[a]) for a in axes])
    if not contiguous_in_memory(fft_in_embedding_shape, fft_in_strides):
        message = f"The FFT axes {axes} cannot be reordered so that the data is contiguous in memory for operand shape = {operand_shape} and operand strides = {operand_strides}."
        unsupported_layout_exception(operand_dim, axes, message, logger)

    # Reorder the FFT axes and input shape so that they are contiguous or separated by constant stride in memory.
    quadruple = sorted(zip(fft_in_strides, fft_in_shape, fft_in_embedding_shape, axes), key=lambda v: v[:2], reverse=True)

    ordered_in_strides, ordered_fft_in_shape, ordered_fft_in_embedding_shape, ordered_axes = zip(*quadruple)

    # Check if R2C and C2R can be supported without copying.
    if fft_abstract_type in ['R2C', 'C2R'] and ordered_axes[-1] != axes[-1]:
        message = f"The last FFT axis specified ({axes[-1]}) must have the smallest stride of all the FFT axes' strides {fft_in_strides} for FFT type '{fft_abstract_type}'."
        unsupported_layout_exception(operand_dim, axes, message, logger)

    # Input FFT size and batch size.
    fft_in_size    = _get_size(fft_in_shape)
    if fft_in_size == 0:
        raise ValueError("Invalid number of FFT data points (0) specified.")
    fft_batch_size = _get_size(operand_shape) // fft_in_size

    # Output FFT (ordered) shape and size.
    last_axis_id, last_axis_size = _get_last_axis_id_and_size(axes, operand_shape, fft_abstract_type, last_axis_size)
    if last_axis_size == 0:
        raise ValueError(f"The size of the last FFT axis in the result for FFT type '{fft_abstract_type}' is 0 for operand shape = {operand_shape} and axes = {axes}. To fix this, provide 'last_axis_size' = 'odd' to the FFT options.")
    ordered_fft_out_shape        = list(ordered_fft_in_shape)
    index = ordered_axes.index(last_axis_id)
    ordered_fft_out_shape[index] = last_axis_size
    fft_out_size                 = _get_size(ordered_fft_out_shape)

    # Check that batch dimensions are tileable, as required by the "advanced" layout.
    sorted_batch_shape, sorted_batch_strides = list(), list()
    if batch_axes:
        sorted_batch_strides, sorted_batch_shape = zip(*sorted(((operand_strides[a], operand_shape[a]) for a in batch_axes)))
        if not check_embedding_possible(sorted_batch_strides, presorted=True):
            raise ValueError(f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} together with the specified axes = {axes} is currently not supported because it is not tileable.")
        logger.debug(f"The sorted batch shape is {sorted_batch_shape}.")
        logger.debug(f"The sorted batch strides are {sorted_batch_strides}.")
    if not check_batch_tileable(sorted_batch_shape, sorted_batch_strides):
        message = f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} together with the specified axes = {axes} is currently not supported because it is not tileable."
        unsupported_layout_exception(operand_dim, axes, message, logger)
    logger.debug(f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} together with the specified axes = {axes} IS tileable.")

    # The result tensor has updated shape for R2C and C2R transforms.
    result_shape = list(operand_shape)
    result_shape[last_axis_id] = last_axis_size

    # The result tensor layout is either natural or chosen for optimal cuFFT performance, based on the operand layout and user-provided option.

    # We can keep the input's layout (i.e. operand's extents order of increasing strides)
    # without performance hit, if the samples do not interleave.
    # Otherwise, we try to keep it only when explicitly asked (result_layout=natural)
    is_sample_interleaved = sorted_batch_strides and sorted_batch_strides[0] <= ordered_in_strides[0]
    logger.debug(f"Are the samples interleaved? {is_sample_interleaved }.")

    if not is_sample_interleaved or result_layout == 'natural':    # Natural (== operand) layout.
        axis_order = axis_order_in_memory(operand_shape, operand_strides)
        result_strides = calculate_strides(result_shape, axis_order)
        # If the resulting output operand is not tilable, keeping the original layout is not possible.
        # If `not is_sample_interleaved` the batch must be tilable,
        # because the min batch stride is bigger than max fft stride
        if is_sample_interleaved:
            if not check_contiguous_layout(batch_axes, result_strides, result_shape):
                message = f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} together with the specified axes = {axes} is currently not supported with result_layout='natural', because the output batch would not be tileable."
                unsupported_layout_exception(operand_dim, axes, message, logger)
            if not check_contiguous_layout(axes, result_strides, result_shape):
                message = f"The operand layout corresponding to shape = {operand_shape} and strides = {operand_strides} together with the specified axes = {axes} is currently not supported with result_layout='natural', because the output sample would be non-contiguous."
                unsupported_layout_exception(operand_dim, axes, message, logger)
    else:    # Optimized layout.
        axis_order = tuple(list(reversed(ordered_axes)) + sorted((a for a in batch_axes), key=lambda v: (operand_strides[v], operand_shape[v])))
        result_strides = calculate_strides(result_shape, axis_order)
    logger.debug(f"The result layout is '{result_layout}' with the result_strides {result_strides}.")

    # Compute the operand linear stride and distance needed for the cuFFT plan.
    last_ordered_in_stride  = ordered_in_strides[-1]
    min_in_stride           = min(operand_strides)

    if last_ordered_in_stride == min_in_stride:
        istride, idistance = min_in_stride, _get_size(ordered_fft_in_embedding_shape) if not sorted_batch_strides else sorted_batch_strides[0]
    else:
        istride, idistance = last_ordered_in_stride, min_in_stride if not sorted_batch_strides else sorted_batch_strides[0]

    # Compute the result linear stride and distance needed for the cuFFT plan.
    ostride = result_strides[ordered_axes[-1]] #  minimal output fft stride
    odistance = fft_out_size if not batch_axes else min(result_strides[axis] for axis in batch_axes)

    if operand_dtype in ("float16", "complex32"):
        if fft_abstract_type == "R2C" and istride != 1:
            raise ValueError(
                f"The {fft_abstract_type} FFT of half-precision tensor ({operand_dtype}) "
                f"is currently not supported for strided inputs "
                f"(got input stride {istride}).")
        if fft_abstract_type == "C2R" and ostride != 1:
            raise ValueError(
                f"The {fft_abstract_type} FFT of half-precision tensor ({operand_dtype}) "
                f"is currently not supported for strided outputs "
                f"(got output stride {ostride}).")
        if fft_out_size == 1:
            if cufft.get_version() < 10702:  # 10702 is shipped with CTK 11.7
                raise ValueError(
                    f"The FFT of sample size 1 and half-precision type ({operand_dtype}) "
                    f"of size 1 is not supported by the installed cuFFT version. ")
            # There is a bug that leads to invalid memory access (CTK 12.1) for one-element,
            # strided C2C complex32 tensors (either in the input or output) or results in
            # CUFFT_INVALID_SIZE (CTK 12.3). This workaround relies on the fact that the
            # [i|o]stride effectively does not matter in a one-element sample.
            elif fft_abstract_type == "C2C":
                istride = ostride = 1

    # There's a bug in cuFFT in CTKs prior to 11.4U2
    if len(axes) == 3 and fft_batch_size > 1 and cufft.get_version() < 10502:
        raise ValueError(
            "The 3D batched FFT is not supported by the installed cuFFT version. "
            "Please update your CUDA Toolkit (to 11.4.2 or newer)"
        )

    plan_traits = PlanTraits(result_shape=tuple(result_shape), result_strides=tuple(result_strides), ordered_axes=tuple(ordered_axes),
                             ordered_fft_in_shape=tuple(ordered_fft_in_shape), ordered_fft_in_embedding_shape=tuple(ordered_fft_in_embedding_shape),
                             ordered_fft_out_shape=tuple(ordered_fft_out_shape),
                             fft_batch_size=fft_batch_size, istride=istride, idistance=idistance, ostride=ostride, odistance=odistance)
    return plan_traits

def create_xt_plan_args(*, plan_traits=None, fft_abstract_type=None, operand_data_type=None, operand_layout=None, inplace=None):
    """
    Create the arguments to xt_make_plan_many() except for the handle. This is also used for computing the FFT key.
    """
    assert plan_traits is not None, "Internal error."
    assert fft_abstract_type is not None, "Internal error."
    assert operand_data_type is not None, "Internal error."
    assert inplace is not None, "Internal error."
    assert operand_layout is not None, "Internal error."

    result_data_type, compute_data_type = _get_fft_result_and_compute_types(operand_data_type, fft_abstract_type)

    # The input shape to the plan should be the logical FFT shape.
    ordered_plan_shape = plan_traits.ordered_fft_out_shape if fft_abstract_type == 'C2R' else plan_traits.ordered_fft_in_shape

    # Handle in-place transforms.
    if inplace:
        if overlapping_layout(operand_layout.shape, operand_layout.strides):
            raise ValueError(f"In-place transform is not supported because the tensor with shape {operand_layout.shape} and strides {operand_layout.strides} overlaps in memory.")
        ordered_fft_out_shape, ostride, odistance = plan_traits.ordered_fft_in_embedding_shape, plan_traits.istride, plan_traits.idistance
    else:
        ordered_fft_out_shape, ostride, odistance = plan_traits.ordered_fft_out_shape, plan_traits.ostride, plan_traits.odistance

    return len(ordered_plan_shape), ordered_plan_shape, plan_traits.ordered_fft_in_embedding_shape, plan_traits.istride, plan_traits.idistance, NAME_TO_DATA_TYPE[operand_data_type], ordered_fft_out_shape, ostride, odistance, NAME_TO_DATA_TYPE[result_data_type], plan_traits.fft_batch_size, NAME_TO_DATA_TYPE[compute_data_type]

def create_fft_key(operand, *, axes=None, options=None, inplace=None, prolog=None, epilog=None, plan_args=None):
    """
    This key is not designed to be serialized and used on a different machine. It is meant for runtime use only.
    We use a specific inplace argument instead of taking it from options, because self.inplace != self.options.inplace
    for CPU tensors for efficiency.

    It is the user's responsiblity to augment this key with the stream in case they use stream-ordered memory pools.
    """
    if plan_args is None:
        # Process options.
        options = utils.check_or_create_options(configuration.FFTOptions, options, "FFT options")

        operand           = tensor_wrapper.wrap_operand(operand)
        fft_abstract_type = _get_default_fft_abstract_type(operand.dtype, options.fft_type)

        # Determine plan traits.
        plan_traits = get_fft_plan_traits(operand.shape, operand.strides, operand.dtype, axes, fft_abstract_type=fft_abstract_type, last_axis_size=options.last_axis_size, result_layout=options.result_layout, logger=None)

        # Inplace is always True for CPU tensors for efficiency.
        if inplace is None:
            inplace = True if operand.device_id is None else options.inplace

        # Get the arguments to xt_make_plan_many.
        plan_args = create_xt_plan_args(plan_traits=plan_traits, fft_abstract_type=fft_abstract_type, operand_data_type=operand.dtype, operand_layout=TensorLayout(shape=operand.shape, strides=operand.strides), inplace=inplace)

    # Prolog and epilog, if used.
    if prolog is not None or epilog is not None:
        get_data = lambda device_callable: None if device_callable is None else (device_callable.ltoir, device_callable.data)
        callable_data = get_data(prolog), get_data(epilog)
    else:
        callable_data = None

    # The key is based on plan arguments and callback data (a callable object of type DeviceCallback or None).
    return plan_args, callable_data

def set_prolog_and_epilog(handle, prolog, epilog, operand_dtype, result_dtype, logger):

    def set_callback(cbkind, cbobj, dtype):
        if cbobj is None:
            return

        assert cbkind in ['prolog', 'epilog'], "Internal error."
        CBType = CBLoadType if cbkind=="prolog" else CBStoreType

        cufft.xt_set_jit_callback(handle, cbobj.ltoir, cbobj.size, CBType[dtype.upper()], [cbobj.data])

        logger.info(f"The specified LTO-IR {cbkind} has been set.")
        if isinstance(cbobj.ltoir, int):
            logger.debug(f"The {cbkind} LTO-IR pointer is {cbobj.ltoir}.")
        logger.debug(f"The {cbkind} LTO-IR size is {cbobj.size}, and data is {cbobj.data}.")

    if prolog is not None:
        set_callback('prolog', prolog, operand_dtype)
    if epilog is not None:
        set_callback('epilog', epilog, result_dtype)


class InvalidFFTState(Exception):
    pass

@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
class FFT:
    """
    FFT(operand, *, axes=None, options=None, stream=None)

    Create a stateful object that encapsulates the specified FFT computations and required resources.
    This object ensures the validity of resources during use and releases them when they are no longer needed to prevent misuse.

    This object encompasses all functionalities of function-form APIs :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`, which are convenience wrappers around it.
    The stateful object also allows for the amortization of preparatory costs when the same FFT operation is to be performed on multiple operands with the same problem specification (see :meth:`reset_operand` and :meth:`create_key` for more details).

    Using the stateful object typically involves the following steps:

    1. **Problem Specification**: Initialize the object with a defined operation and options.
    2. **Preparation**: Use :meth:`plan` to determine the best algorithmic implementation for this specific FFT operation.
    3. **Execution**: Perform the FFT computation with :meth:`execute`, which can be either forward or inverse FFT transformation.
    4. **Resource Management**: Ensure all resources are released either by explicitly calling :meth:`free` or by managing the stateful object within a context manager.

    Detailed information on each step described above can be obtained by passing in a :class:`logging.Logger` object
    to :class:`FFTOptions` or by setting the appropriate options in the root logger object, which is used by default:

        >>> import logging
        >>> logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    Args:
        operand: {operand}
        axes: {axes}
        options: {options}
        stream: {stream}

    See Also:
        :meth:`plan`, :meth:`reset_operand`, :meth:`execute`, :meth:`create_key`

    Examples:

        >>> import numpy as np
        >>> import nvmath

        Create a 3-D complex128 ndarray on the CPU:

        >>> shape = 128, 128, 128
        >>> a = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        We will define a 2-D C2C FFT operation along the first two dimensions, batched along the last dimension:

        >>> axes = 0, 1

        Create an FFT object encapsulating the problem specification above:

        >>> f = nvmath.fft.FFT(a, axes=axes)

        Options can be provided above to control the behavior of the operation using the `options` argument (see :class:`FFTOptions`).

        Next, plan the FFT. Load and/or store callback functions can be provided to :meth:`plan` using the `prolog` and `epilog` option:

        >>> f.plan()

        Now execute the FFT, and obtain the result `r1` as a NumPy ndarray.

        >>> r1 = f.execute()

        Finally, free the FFT object's resources. _To avoid having to explictly making this call, it's recommended to use the FFT object as
        a context manager as shown below, if possible.

        >>> f.free()

        Note that all :class:`FFT` methods execute on the current stream by default. Alternatively, the `stream` argument can be used to run a method on a specified stream.

        Let's now look at the same problem with CuPy ndarrays on the GPU.

        Create a 3-D complex128 CuPy ndarray on the GPU:

        >>> import cupy as cp
        >>> shape = 128, 128, 128
        >>> a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)

        Create an FFT object encapsulating the problem specification described earlier and use it as a context manager.

        >>> with nvmath.fft.FFT(a, axes=axes) as f:
        ...    f.plan()
        ...
        ...    # Execute the FFT to get the first result.
        ...    r1 = f.execute()

        All the resources used by the object are released at the end of the block.

        Further examples can be found in the `nvmath/examples/fft <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft>`_ directory.

    Notes:

        - The input must be Hermitian-symmetric when :attr:`FFTOptions.fft_type` is ``'C2R'``, otherwise the result is undefined. As a specific example, if the input for a C2R FFT was generated using an R2C FFT with an odd last axis size,
          then :attr:`FFTOptions.last_axis_size` must be set to `odd` to recover the original signal.
    """


    def __init__(self, operand, *, axes=None, options=None, stream=None):

        options = utils.check_or_create_options(configuration.FFTOptions, options, "FFT options")
        self.options = options

        self.operand = operand = tensor_wrapper.wrap_operand(operand)

        # Capture operand layout for consistency checks when resetting operands.
        self.operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)
        self.operand_dim    = len(operand.shape)

        if not axes and self.operand_dim > 3:
            raise ValueError(f"The tensor is {self.operand_dim}-D and FFTs in number of dimensions > 3 is not supported. The FFT axes need to be specified using the 'axes' option.")

        if self.operand_dim == 0:
            raise ValueError(f"The tensor is {self.operand_dim}-D (i.e. a scalar). FFT does not support scalars.")

        self.operand_data_type = operand.dtype
        self.fft_abstract_type = _get_default_fft_abstract_type(self.operand_data_type, options.fft_type)

        self.result_data_type, self.compute_data_type = _get_fft_result_and_compute_types(operand.dtype, self.fft_abstract_type)

        self.logger = options.logger if options.logger is not None else logging.getLogger()
        self.logger.info(f"The FFT type is {self.fft_abstract_type}.")
        self.logger.info(f"The input data type is {self.operand_data_type}, and the result data type is {self.result_data_type}.")

        if axes is None: axes = range(self.operand_dim)

        if any(axis >= self.operand_dim or axis < -self.operand_dim for axis in axes):
            raise ValueError(f"The specified FFT axes {axes} are out of bounds for a {self.operand_dim}-D tensor.")

        # Handle negative axis indices.
        self.axes = tuple(axis % self.operand_dim for axis in axes)
        self.logger.info(f"The specified FFT axes are {self.axes}.")

        self.package = utils.infer_object_package(operand.tensor)

        # NumPy and CuPy don't support complex32 yet.
        if self.package in ['numpy', 'cupy'] and self.result_data_type == 'complex32':
            raise TypeError(f"The result data type {self.result_data_type} is not supported by the operand package '{self.package}'.")

        # Infer operand package, execution space, and memory space.
        self.memory_space = 'cuda'
        self.device_id = operand.device_id
        if self.device_id is None:
            self.package = self.operand.name
            if self.package == 'numpy':
                self.package = 'cupy'
            self.memory_space = 'cpu'
            self.device_id = options.device_id
        self.logger.info(f"The input tensor's memory space is {self.memory_space}, and the execution space is on device {self.device_id}.")

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.logger.info(f"The specified stream for the FFT ctor is {stream_holder.obj}.")

        # In-place FFT option, available only for C2C transforms.
        self.inplace = self.options.inplace
        if self.inplace and self.fft_abstract_type != 'C2C':
            raise ValueError(f"The in-place option (FFTOptions.inplace=True) is only supported for complex-to-complex FFT. The FFT type is '{self.fft_abstract_type}'.")

        # Copy operand to device if needed.
        if self.memory_space == 'cpu':
            self.cpu_operand = self.operand
            self.operand = operand = tensor_wrapper.to(operand, self.device_id, stream_holder)
        else:
            # take a copy of input for C2R since cufft overwrites input
            if self.fft_abstract_type == 'C2R':
                self.logger.info("For C2R FFT with input operand on GPU, the input is copied to avoid being overwritten by cuFFT.")
                operand_copy = utils.create_empty_tensor(operand.__class__,
                    operand.shape, operand.dtype, self.device_id, stream_holder, operand.strides)
                operand_copy.copy_(operand.tensor, stream_holder=stream_holder)
                assert operand_copy.strides == operand.strides and operand_copy.data_ptr != operand.data_ptr
                self.operand = operand = operand_copy

        if self.options.inplace:    # Don't use self.inplace here, because we always set it to True for CPU tensors.
            self.logger.info(f"The FFT will be performed in-place, with the result overwriting the input.")
        else:
            self.logger.info(f"The FFT will be performed out-of-place.")

        # Check if FFT is supported and calculate plan traits.
        self.plan_traits = get_fft_plan_traits(operand.shape, operand.strides, operand.dtype, self.axes, fft_abstract_type=self.fft_abstract_type, last_axis_size=self.options.last_axis_size, result_layout=self.options.result_layout, logger=self.logger)

        self.logger.info(f"The operand data type = {self.operand_data_type}, shape = {self.operand_layout.shape}, and strides = {self.operand_layout.strides}.")
        result_data_type, result_shape, result_strides = (self.operand_data_type, self.operand_layout.shape, self.operand_layout.strides) if self.inplace else (self.result_data_type, self.plan_traits.result_shape, self.plan_traits.result_strides)
        self.logger.info(f"The result data type = {result_data_type}, shape = {result_shape}, and strides = {result_strides}.")
        self.logger.info(f"The FFT batch size is {self.plan_traits.fft_batch_size}.")

        ordered_fft_out_shape, ostride, odistance = (self.plan_traits.ordered_fft_in_shape, self.plan_traits.istride, self.plan_traits.idistance) if self.inplace else (self.plan_traits.ordered_fft_out_shape, self.plan_traits.ostride, self.plan_traits.odistance)
        self.logger.debug(f"The plan ordered axes = {self.plan_traits.ordered_axes}, ordered input shape = {self.plan_traits.ordered_fft_in_shape}, ordered input embedding shape = {self.plan_traits.ordered_fft_in_embedding_shape}, ordered output shape = {ordered_fft_out_shape}.")
        self.logger.debug(f"The plan input stride is {self.plan_traits.istride} with distance {self.plan_traits.idistance}.")
        self.logger.debug(f"The plan output stride is {ostride} with distance {odistance}.")

        # The result's package and device.
        self.result_class = operand.__class__
        self.device = cp.cuda.Device(self.device_id)

        # Set blocking or non-blocking behavior.
        self.blocking = self.options.blocking is True or self.memory_space == 'cpu'
        if self.blocking:
            self.call_prologue = "This call is blocking and will return only after the operation is complete."
        else:
            self.call_prologue = "This call is non-blocking and will return immediately after the operation is launched on the device."

        # Set memory allocator.
        self.allocator = options.allocator if options.allocator is not None else memory._MEMORY_MANAGER[self.package](self.device_id, self.logger)

        # Create handle.
        with utils.device_ctx(self.device_id):
            self.handle = cufft.create()

        # Set stream for the FFT.
        cufft.set_stream(self.handle, stream_holder.ptr)

        # Plan attributes.
        cufft.set_auto_allocation(self.handle, 0)
        self.fft_planned = False

        # Workspace attributes.
        self.workspace_ptr, self.workspace_size = None, None
        self._workspace_allocated_here = False

        # Attributes to establish stream ordering.
        self.workspace_stream = None
        self.last_compute_event = None

        # Keep track of key (sans callback) for resetting operands, once plan in available.
        self.orig_key = None

        self.valid_state = True
        self.logger.info("The FFT operation has been created.")

    def get_key(self, *, prolog=None, epilog=None):
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
        return create_fft_key(self.operand.tensor, axes=self.axes, options=self.options, inplace=self.inplace, prolog=prolog, epilog=epilog)

    @staticmethod
    def create_key(operand, *, axes=None, options=None, prolog=None, epilog=None):
        """
        Create a key as a compact representation of the FFT problem specification based on the given operand, axes and the FFT options.
        Note that different combinations of operand layout, axes and options can potentially correspond to the same underlying problem specification (key).
        Users may reuse the FFT objects when different input problems map to an identical key.

        Args:
            operand: {operand}
            axes: {axes}
            options: {options}
            prolog: {prolog}
            epilog: {epilog}

        Returns:
            {fft_key}

        Notes:
            - Users may take advantage of this method to create cached version of :func:`fft` based on the stateful object APIs
              (see `caching.py <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/caching.py>`_ for an example implementation).
            - This key is meant for runtime use only and not designed to be serialized or used on a different machine.
            - It is the user's responsiblity to augment this key with the stream in case they use stream-ordered memory pools.
        """
        return create_fft_key(operand, axes=axes, options=options, prolog=prolog, epilog=epilog)

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

    def _free_plan_resources(self, exception=None):
        """
        Free resources allocated in planning.
        """

        self.workspace_ptr = None
        self.fft_planned   = False
        return True

    @utils.precondition(_check_valid_fft)
    @utils.atomic(_free_plan_resources, method=True)
    def plan(self, *, prolog=None, epilog=None, stream=None):
        """Plan the FFT.

        Args:
            prolog: {prolog}
            epilog: {epilog}
            stream: {stream}
        """

        if self.fft_planned:
            self.logger.debug("The FFT has already been planned, and redoing the plan is not supported.")
            return

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.workspace_stream = stream_holder.obj

        # Set stream for the FFT.
        cufft.set_stream(self.handle, stream_holder.ptr)

        # Set LTO-IR callbacks, if present.
        prolog = utils.check_or_create_options(configuration.DeviceCallable, prolog, "prolog", keep_none=True)
        epilog = utils.check_or_create_options(configuration.DeviceCallable, epilog, "epilog", keep_none=True)
        set_prolog_and_epilog(self.handle, prolog, epilog, self.operand_data_type, self.result_data_type, self.logger)

        # Get all the arguments to xt_make_plan_many except for the first (the handle).
        plan_args = create_xt_plan_args(plan_traits=self.plan_traits, fft_abstract_type=self.fft_abstract_type, operand_data_type=self.operand_data_type, operand_layout=self.operand_layout, inplace=self.inplace)

        # Keep track of original key (sans callback) for resetting operands. Pass in plan args to avoid recomputation.
        self.orig_key = create_fft_key(self.operand.tensor, axes=self.axes, options=self.options, plan_args=plan_args)
        self.logger.debug(f"The FFT key (sans callback) is {self.orig_key}.")

        self.logger.debug(f"The operand CUDA type is {NAME_TO_DATA_TYPE[self.operand_data_type].name}, and the result CUDA type is {NAME_TO_DATA_TYPE[self.result_data_type].name}.")
        self.logger.debug(f"The CUDA type used for compute is {NAME_TO_DATA_TYPE[self.compute_data_type].name}.")
        timing =  bool(self.logger and self.logger.handlers)
        self.logger.info("Starting FFT planning...")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, blocking=True, timing=timing) as (self.last_compute_event, elapsed):
            self.workspace_size = cufft.xt_make_plan_many(self.handle, *plan_args)

        self.fft_planned = True

        if elapsed.data is not None:
            self.logger.info(f"The FFT planning phase took {elapsed.data:.3f} ms to complete.")

    @utils.precondition(_check_valid_fft)
    def reset_operand(self, operand=None, *, stream=None):
        """
        Reset the operand held by this :class:`FFT` instance. This method has two use cases: (1) it can be used to provide a new operand for execution when the
        operand is on the CPU, and (2) it can be used to release the internal reference to the previous operand and potentially make its memory available for
        other use by passing ``operand=None``.

        Args:
            operand: A tensor (ndarray-like object) compatible with the previous one or `None` (default).
                A value of `None` will release the internal reference to the previous operand and user is expected to set a new operand before again calling :meth:`execute`.
                The new operand is considered compatible if all following properties match with the previous one:

                    - The problem specification key for the new operand. Generally the keys will match if the operand shares the same layout (shape, strides and data type).
                      The keys may still match for certain operands with different layout, see :meth:`create_key` for details.
                    - The package that the new operand belongs to .
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
            ...    # Plan the FFT
            ...    f.plan()
            ...
            ...    # Execute the FFT to get the first result.
            ...    r1 = f.execute()
            ...
            ...    # Reset the operand to a new CuPy ndarray.
            ...    b = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
            ...    f.reset_operand(b)
            ...
            ...    # Execute to get the new result corresponding to the updated operand.
            ...    r2 = f.execute()

            With :meth:`reset_operand`, minimal overhead is achieved as problem specification and planning are only performed once.

            For the particular example above, explicitly calling :meth:`reset_operand` is equivalent to updating the operand in-place, i.e, replacing ``f.reset_operand(b)`` with ``a[:]=b``.
            Note that updating the operand in-place should be adopted with caution as it can only yield the expected result under the additional constraints below:

                - The operation is not a complex-to-real (C2R) FFT.
                - The operand is on the GPU (more precisely, the operand memory space should be accessible from the execution space).

            For more details, please refer to `inplace update example <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example05_stateful_inplace.py>`_.
        """

        if operand is None:
            self.operand = None
            self.logger.info(f"The operand has been reset to None.")
            return

        self.logger.info("Resetting operand...")
        # First wrap operand.
        operand = tensor_wrapper.wrap_operand(operand)

        if self.orig_key is not None:
            # Compute the key corresponding to the new operand (sans callback).
            try:
                new_key  = create_fft_key(operand.tensor, axes=self.axes, options=self.options, inplace=self.inplace)
            except UnsupportedLayoutError:
                new_key = None
            if self.orig_key != new_key:
                self.logger.debug(f"The FFT key corresponding to the original operand is: {self.orig_key}.")
                if new_key is None:
                    self.logger.debug(f"The FFT key for the new operand cannot be computed since the layout (shape = {operand.shape}, strides = {operand.strides}) and axes = {self.axes} combination is unsupported.")
                else:
                    self.logger.debug(f"The FFT key corresponding to the new operand is:      {new_key}.")
                raise ValueError("The new operand's traits (data type, shape, or strides) are incompatible with that of the original operand.")

        # Check package match.
        package = utils.infer_object_package(operand.tensor)
        utils.check_attribute_match(self.operand_data_type, operand.dtype, "data type")

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        self.logger.info(f"The specified stream for reset_operand() is {stream_holder.obj}.")

        # Set stream for the FFT.
        cufft.set_stream(self.handle, stream_holder.ptr)

        device_id = operand.device_id
        if device_id is None:
            package = 'cupy' if package == 'numpy' else package   # Handle the NumPy <=> CuPy asymmetry.
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            if self.operand is None:
                # Copy operand across memory spaces (CPU to GPU).
                self.operand = tensor_wrapper.to(operand, self.device_id, stream_holder)
            else:
                # In-place copy to existing device pointer because the new operand is on the CPU.
                tensor_wrapper.copy_([operand], [self.operand], stream_holder)
            # Finally, replace the original CPU operand by the new one to handle cases like inplace transforms.
            self.cpu_operand = operand
        else:
            if self.package != package:
                message = f"Library package mismatch: '{self.package}' => '{package}'"
                raise TypeError(message)

            if self.device_id != device_id:
                raise ValueError(f"The new operand must be on the same device ({device_id}) as the original operand "
                                 f"({self.device_id}).")

            # Finally, replace the original operand by the new one.
            if self.fft_abstract_type == 'C2R':
                # For C2R, we need to take a copy to avoid input being overwritten
                self.logger.info("For C2R FFT with input operand on GPU, the input is copied to avoid being overwritten by cuFFT.")
                operand_copy = utils.create_empty_tensor(operand.__class__,
                    operand.shape, operand.dtype, self.device_id, stream_holder, operand.strides)
                operand_copy.copy_(operand.tensor, stream_holder=stream_holder)
                self.operand = operand = operand_copy
            else:
                self.operand = operand

        # Update operand layout and plan traits.
        self.operand_layout = TensorLayout(shape=operand.shape, strides=operand.strides)
        self.logger.info(f"The reset operand shape = {self.operand_layout.shape}, and strides = {self.operand_layout.strides}.")

        self.plan_traits    = get_fft_plan_traits(operand.shape, operand.strides, operand.dtype, self.axes, fft_abstract_type=self.fft_abstract_type,
                                                  last_axis_size=self.options.last_axis_size, result_layout=self.options.result_layout, logger=self.logger)
        result_shape, result_strides = (self.operand_layout.shape, self.operand_layout.strides) if self.inplace else (self.plan_traits.result_shape, self.plan_traits.result_strides)
        self.logger.info(f"The result shape = {result_shape}, and strides = {result_strides}.")

        self.logger.info("The operand has been reset to the specified operand.")

    def _check_planned(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        if not self.fft_planned:
            raise RuntimeError(f"{what} cannot be performed before plan() has been called.")

    def _check_valid_operand(self, *args, **kwargs):
        """
        """
        what = kwargs['what']
        if self.operand is None:
            raise RuntimeError(f"{what} cannot be performed if the input operand has been set to None. Use reset_operand() to set the desired input before using performing the {what.lower()}.")

    def _free_workspace_memory(self, exception=None):
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
    def _allocate_workspace_memory_perhaps(self, stream_holder):
        """
        Allocate workspace memory using the specified allocator, if it hasn't already been done.
        """

        assert self._workspace_allocated_here is False, "Internal Error."

        if self.workspace_ptr is not None:
            return

        self.logger.debug(f"Allocating workspace for performing the FFT...")
        with utils.device_ctx(self.device_id), stream_holder.ctx:
            try:
                self.workspace_ptr = self.allocator.memalloc(self.workspace_size)
                self._workspace_allocated_here = True
            except TypeError as e:
                message = "The method 'memalloc' in the allocator object must conform to the interface in the "\
                          "'BaseCUDAMemoryManager' protocol."
                raise TypeError(message) from e
            raw_workspace_ptr = utils.get_ptr_from_memory_pointer(self.workspace_ptr)
            cufft.set_work_area(self.handle, raw_workspace_ptr)

        self.workspace_stream = stream_holder.obj
        self.logger.debug(f"Finished allocating device workspace of size {formatters.MemoryStr(self.workspace_size)} in the context of stream {self.workspace_stream}.")

    @utils.precondition(_check_valid_fft)
    def _free_workspace_memory_perhaps(self, release_workspace):
        """
        Free workspace memory if if 'release_workspace' is True.
        """
        if not release_workspace:
            return

        # Establish ordering wrt the computation and free workspace if it's more than the specified cache limit.
        if self.last_compute_event is not None:
            self.workspace_stream.wait_event(self.last_compute_event)
            self.logger.debug("Established ordering with respect to the computation before releasing the workspace.")

        self.logger.debug("[_free_workspace_memory_perhaps] The workspace memory will be released.")
        self._free_workspace_memory()

        return True

    def _release_workspace_memory_perhaps(self, exception=None):
        """
        Free workspace memory if it was allocated in this call (self._workspace_allocated_here == True) when an exception occurs.
        """
        release_workspace = self._workspace_allocated_here
        self.logger.debug(f"[_release_workspace_memory_perhaps] The release_workspace flag is set to {release_workspace} based upon the value of 'workspace_allocated_here'.")
        self._free_workspace_memory_perhaps(release_workspace)
        return True

    @utils.precondition(_check_valid_fft)
    @utils.precondition(_check_planned, "Execution")
    @utils.precondition(_check_valid_operand, "Execution")
    @utils.atomic(_release_workspace_memory_perhaps, method=True)
    def execute(self, direction=None, stream=None, release_workspace=False):
        """
        Execute the FFT operation.

        Args:
            direction: {direction}
            stream: {stream}
            release_workspace: {release_workspace}

        Returns:
            The transformed operand, which remains on the same device and utilizes the same package as the input operand.
            The data type and shape of the transformed operand depend on the type of input operand:

                - For C2C FFT, the data type and shape remain identical to the input.
                - For R2C and C2R FFT, both data type and shape differ from the input.
        """

        log_info  = self.logger.isEnabledFor(logging.INFO)
        log_debug = self.logger.isEnabledFor(logging.DEBUG)

        if direction is None:
            direction = _get_fft_default_direction(self.fft_abstract_type)
        elif isinstance(direction, str) and (d := direction.upper()) in ['FORWARD', 'INVERSE']:
            direction = configuration.FFTDirection[d]
        else:
            direction = configuration.FFTDirection(direction)

            if self.fft_abstract_type == 'C2R':
                if direction != configuration.FFTDirection.INVERSE:
                    raise ValueError(f"The specified direction {direction.name} is not compatible with the FFT type '{self.fft_abstract_type}'.")
            elif self.fft_abstract_type == 'R2C':
                if direction != configuration.FFTDirection.FORWARD:
                    raise ValueError(f"The specified direction {direction.name} is not compatible with the FFT type '{self.fft_abstract_type}'.")

        stream_holder = utils.get_or_create_stream(self.device_id, stream, self.package)
        if log_info: self.logger.info(f"The specified stream for execute() is {stream_holder.obj}.")

        # Set stream for the FFT.
        cufft.set_stream(self.handle, stream_holder.ptr)

        # Allocate workspace if needed.
        self._allocate_workspace_memory_perhaps(stream_holder)

        # Create empty tensor for the result, if it's not an in-place operation.
        if not self.inplace:
            if log_debug: self.logger.debug("Beginning output (empty) tensor creation...")
            if log_debug: self.logger.debug(f"The output tensor shape = {self.plan_traits.result_shape} with strides = {self.plan_traits.result_strides} and data type '{self.result_data_type}'.")
            self.result = utils.create_empty_tensor(self.result_class, self.plan_traits.result_shape, self.result_data_type, self.device_id, stream_holder, self.plan_traits.result_strides)
            if log_debug: self.logger.debug("The output (empty) tensor has been created.")

        result_ptr = self.operand.data_ptr if self.inplace else self.result.data_ptr

        timing =  bool(self.logger and self.logger.handlers)
        if log_info: self.logger.info(f"Starting FFT {self.fft_abstract_type} calculation in the {direction.name} direction...")
        if log_info: self.logger.info(f"{self.call_prologue}")
        with utils.device_ctx(self.device_id), utils.cuda_call_ctx(stream_holder, self.blocking, timing) as (self.last_compute_event, elapsed):
            if log_debug: self.logger.debug(f"The cuFFT execution function is 'xt_exec'.")
            cufft.xt_exec(self.handle, self.operand.data_ptr, result_ptr, direction)

        if elapsed.data is not None:
            if log_info: self.logger.info(f"The FFT calculation took {elapsed.data:.3f} ms to complete.")

        # Establish ordering wrt the computation and free workspace if it's more than the specified cache limit.
        self._free_workspace_memory_perhaps(release_workspace)

        # reset workspace allocation tracking to False at the end of the methods where workspace memory is potentially allocated.
        # This is necessary to prevent any exceptions raised before method entry from using stale tracking values.
        self._workspace_allocated_here = False

        # Return the result.
        result = self.operand if self.inplace else self.result
        if self.memory_space == 'cpu':
            if self.options.inplace:    # Don't use self.inplace here, because we always set it to True for CPU tensors.
                out = self.cpu_operand.copy_(result.tensor, stream_holder=stream_holder)
            else:
                out = result.to('cpu', stream_holder=stream_holder)
        else:
            out = result.tensor

        # Release internal reference to the result to permit recycling of memory.
        self.result = None

        return out

    def free(self):
        """Free FFT resources.

        It is recommended that the :class:`FFT` object be used within a context, but if it is not possible then this
        method must be called explicitly to ensure that the FFT resources (especially internal library objects) are
        properly cleaned up.
        """

        if not self.valid_state:
            return

        try:
            # Future operations on the workspace stream should be ordered after the computation.
            if self.last_compute_event is not None:
                self.workspace_stream.wait_event(self.last_compute_event)

            self._free_workspace_memory()

            if self.handle is not None:
                cufft.destroy(self.handle)
                self.handle = None

        except Exception as e:
            self.logger.critical("Internal error: only part of the FFT object's resources have been released.")
            self.logger.critical(str(e))
            raise e
        finally:
            self.valid_state = False

        self.logger.info("The FFT object's resources have been released.")

def _fft(x, *, axes=None, direction=None, options=None,  prolog=None, epilog=None, stream=None, check_dtype=None):
    r"""
    fft(operand, axes=None, direction=None, options=None, prolog=None, epilog=None, stream=None)

    Perform an N-D *complex-to-complex* (C2C) FFT on the provided complex operand.

    Args:
        operand: {operand}
        axes: {axes}
        options: {options}
        prolog: {prolog}
        epilog: {epilog}
        stream: {stream}

    Returns:
        A transformed operand that retains the same data type and shape as the input. It remains on the same device and uses the same package as the input operand.

    See Also:
        :func:`ifft`, :func:`irfft`, :func:`rfft`, :class:`FFT`

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create a 3-D complex128 ndarray on the GPU:

        >>> a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)

        Perform a 3-D C2C FFT using :func:`fft`. The result `r` is also a CuPy complex128 ndarray:

        >>> r = nvmath.fft.fft(a)

        User may also perform FFT along a subset of dimensions, e.g, 2-D C2C FFT along the first two dimensions, batched along the last dimension:

        >>> axes = 0, 1
        >>> r = nvmath.fft.fft(a, axes=axes)

        For C2C type FFT operation, the output can be directly computed inplace thus overwriting the input operand. This can be specified using options to the FFT:

        >>> o = nvmath.fft.FFTOptions(inplace=True)
        >>> r = nvmath.fft.fft(a, options=o)
        >>> r is a

        See `FFTOptions` for the complete list of available options.

        The package current stream is used by default, but a stream can be explicitly provided to the FFT operation. This can be done if the
        FFT operand is computed on a different stream, for example:

        >>> s = cp.cuda.Stream()
        >>> with s:
        ...    a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
        >>> nvmath.fft.fft(a, stream=s)

        The operation above runs on stream `s` and is ordered with respect to the input computation.

        Create a NumPy ndarray on the CPU.

        >>> import numpy as np
        >>> b = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        Provide the NumPy ndarray to :func:`fft`, with the result also being a NumPy ndarray:

        >>> r = nvmath.fft.fft(b)

    Notes:
        - This function only takes complex operand for C2C transformation. If the user wishes to perform full FFT transformation on real input,
          please cast the input to the corresponding complex data type.
        - This function is a convenience wrapper around :class:`FFT` and and is specifically meant for *single* use.
          The same computation can be performed with the stateful API using the default `direction` argument in :meth:`FFT.execute`.

    Further examples can be found in the `nvmath/examples/fft <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft>`_ directory.
    """
    if check_dtype is not None:
        assert check_dtype in {'real', 'complex'}, "internal error"
        operand = tensor_wrapper.wrap_operand(x)
        if ('complex' in operand.dtype) != (check_dtype == 'complex'):
            raise ValueError(f"This function expects {check_dtype} operand, found {operand.dtype}")

    with FFT(x, axes=axes, options=options, stream=stream) as fftobj:

        # Plan the FFT.
        fftobj.plan(stream=stream, prolog=prolog, epilog=epilog)

        # Execute the FFT.
        result = fftobj.execute(direction=direction, stream=stream)

    return result


# Forward C2C FFT Function.
fft  = functools.wraps(_fft)(functools.partial(_fft, direction=configuration.FFTDirection.FORWARD, check_dtype='complex'))
fft.__doc__ = fft.__doc__.format(**SHARED_FFT_DOCUMENTATION)
fft.__name__ = 'fft'

# Forward R2C FFT Function
@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
def rfft(operand, *, axes=None, options=None, prolog=None, epilog=None, stream=None):
    r"""
    rfft(operand, axes=None, options=None, prolog=None, epilog=None, stream=None)

    Perform an N-D *real-to-complex* (R2C) FFT on the provided real operand.

    Args:
        operand: {operand}
        axes: {axes}
        options: {options}
        prolog: {prolog}
        epilog: {epilog}
        stream: {stream}

    Returns:
        A complex tensor that remains on the same device and belongs to the same package as the input operand. The extent of the last
        transformed axis in the result will be ``operand.shape[axes[-1]] // 2 + 1``.


    See Also:
        :func:`fft`, :func:`irfft`, :class:`FFT`.
    """
    wrapped_operand = tensor_wrapper.wrap_operand(operand)
    # check if input operand if real type
    if 'complex' in wrapped_operand.dtype:
        raise RuntimeError(f"rfft expects a real input, but got {wrapped_operand.dtype}. Please use fft for complex input.")

    return _fft(operand, axes=axes, options=options, prolog=prolog, epilog=epilog, stream=stream, check_dtype='real')


# Inverse C2C/R2C FFT Function.
ifft = functools.wraps(_fft)(functools.partial(_fft, direction=configuration.FFTDirection.INVERSE, check_dtype='complex'))
ifft.__doc__ = """
    ifft(operand, axes=None, options=None, prolog=None, epilog=None, stream=None)

    Perform an N-D *complex-to-complex* (C2C) inverse FFT on the provided complex operand. The direction is implicitly inverse.

    Args:
        operand: {operand}
        axes: {axes}
        options: {options}
        prolog: {prolog}
        epilog: {epilog}
        stream: {stream}

    Returns:
        A transformed operand that retains the same data type and shape as the input. It remains on the same device and uses the same package as the input operand.

    See Also:
        :func:`fft`, :func:`irfft`, :class:`FFT`.

    Notes:
        - This function only takes complex operand for C2C transformation. If users wishes to perform full FFT transformation on real input,
          please cast the input to the corresponding complex data type.
        - This function is a convenience wrapper around :class:`FFT` and and is specifically meant for *single* use.
          The same computation can be performed with the stateful API by passing the argument ``direction='inverse'`` when calling :meth:`FFT.execute`.
""".format(**SHARED_FFT_DOCUMENTATION)
ifft.__name__  = 'ifft'


# Inverse C2R FFT Function.
@utils.docstring_decorator(SHARED_FFT_DOCUMENTATION, skip_missing=False)
def irfft(x, *, axes=None, options=None,  prolog=None, epilog=None, stream=None):
    """
    irfft(operand, axes=None, options=None, prolog=None, epilog=None, stream=None)

    Perform an N-D *complex-to-real* (C2R) FFT on the provided complex operand. The direction is implicitly inverse.

    Args:
        operand: {operand}
        axes: {axes}
        options: {options}
        prolog: {prolog}
        epilog: {epilog}
        stream: {stream}

    Returns:
        A real tensor that remains on the same device and belongs to the same package as the input operand. The extent of the last
        transformed axis in the result will be ``(operand.shape[axes[-1]] - 1) * 2`` if :attr:`FFTOptions.last_axis_size` is ``even``, or
        ``operand.shape[axes[-1]] * 2 - 1`` if :attr:`FFTOptions.last_axis_size` is ``odd``.

    See Also:
        :func:`fft`, :func:`ifft`, :class:`FFT`.

    Examples:

        >>> import cupy as cp
        >>> import nvmath

        Create a 3-D symmetric complex128 ndarray on the GPU:

        >>> shape = 512, 768, 256
        >>> a = nvmath.fft.rfft(cp.random.rand(*shape, dtype=cp.float64))

        Perform a 3-D C2R FFT using the :func:`irfft` wrapper. The result `r` is a CuPy float64 ndarray:

        >>> r = nvmath.fft.irfft(a)
        >>> r.dtype

    Notes:

        - This function performs an inverse C2R N-D FFT, which is similar to `irfftn` but different from `irfft` in various numerical packages.
        - This function is a convenience wrapper around :class:`FFT` and and is specifically meant for *single* use.
          The same computation can be performed with the stateful API by setting :attr:`FFTOptions.fft_type` to ``'C2R'`` and passing the argument ``direction='inverse'`` when calling :meth:`FFT.execute`.
        - **The input to this function must be Hermitian-symmetric, otherwise the result is undefined.** While the symmetry requirement is partially captured by the different extents in the last transformed
          dimension between the input and result, there are additional `constraints <https://docs.nvidia.com/cuda/cufft/#fourier-transform-types>`_. As
          a specific example, 1-D transforms require the first element (and the last element, if the extent is even) of the input to be purely real-valued.
          In addition, if the input to `irfft` was generated using an R2C FFT with an odd last axis size, :attr:`FFTOptions.last_axis_size` must be set to ``odd`` to recover the original signal.
        - For more details, please refer to `C2R example <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example07_c2r.py>`_
          and `odd C2R example <https://github.com/NVIDIA/nvmath-python/tree/main/examples/fft/example07_c2r_odd.py>`_.
    """
    options = utils.check_or_create_options(configuration.FFTOptions, options, "FFT options")
    options.fft_type = 'C2R'
    return _fft(x, axes=axes, direction=configuration.FFTDirection.INVERSE, options=options, prolog=prolog, epilog=epilog, stream=stream, check_dtype='complex')
