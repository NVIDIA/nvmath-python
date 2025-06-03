# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FFTDirection", "FFTOptions", "DeviceCallable", "ExecutionCUDA", "ExecutionCPU"]

from dataclasses import dataclass, field
from enum import IntEnum
from logging import Logger
from typing import Literal
import warnings

from nvmath.memory import BaseCUDAMemoryManager


@dataclass
class ExecutionCUDA:
    """
    A data class for providing GPU execution options to the :class:`FFT` object and the
    family of wrapper functions :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`.

    Attributes:
        device_id: CUDA device ordinal (only used if the operand resides on the CPU). The
            default value is 0.

    See Also:
       :class:`ExecutionCPU`, :class:`FFT`, :func:`fft`, :func:`ifft`, :func:`rfft`, and
       :func:`irfft`.
    """

    name: Literal["cuda"] = field(default="cuda", init=False)
    # If not specified, it defaults to the deprecated FFTOptions.device_id or 0
    # Keep None as an option to differentiate between the user-provided 0 and the default 0.
    device_id: int | None = None


@dataclass
class ExecutionCPU:
    """
    A data class for providing CPU execution options to the :class:`FFT` object and the
    family of wrapper functions :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`.

    Attributes:
        num_threads: The number of CPU threads used to execute the plan.
                     If not specified, defaults to the number of CPU cores available to the
                     process.

    See Also:
       :class:`ExecutionCUDA`, :class:`FFT`, :func:`fft`, :func:`ifft`, :func:`rfft`, and
       :func:`irfft`.
    """

    name: Literal["cpu"] = field(default="cpu", init=False)
    num_threads: int | None = None


@dataclass
class FFTOptions:
    """
    A data class for providing options to the :class:`FFT` object and the family of wrapper
    functions :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`.

    Attributes:
        fft_type: The type of FFT to perform, available options include ``'C2C'``,
            ``'C2R'``, and ``'R2C'``. The default is ``'C2C'`` for complex input and
            ``'R2C'`` for real input.

        inplace: Specify if the operation is in-place (`True` or `False`). The operand is
            overwritten by the result if ``inplace`` is `True`. The default is `False`.

        last_axis_parity: For complex-to-real FFT (corresponding to ``fft_type='C2R'``),
            specify whether the size of the last axis in the result should be even or odd.
            The even size is calculated as :math:`2 * (m - 1)`, where :math:`m` is the the
            size of the last axis of the operand, and the odd size is calculated as :math:`2
            * (m - 1) + 1`. The specified value should be either ``'even'`` or ``'odd'``,
            with the default being ``'even'``.

        result_layout: The layout to use for the result, either ``'natural'`` or
            ``'optimized'``. For the ``'natural'`` option, the result layout is the same as
            that of the operand. The default is ``'optimized'``, which generally provides
            much better performance and should be used if the user doesn't care about the
            result layout matching the operand layout. However in rare cases, depending on
            the device type, shape and strides of the operand, and the FFT dimensions, the
            ``'natural'`` layout may perform better. This option is ignored if ``inplace``
            is specified to be True.

        device_id: CUDA device ordinal (used if the operand resides on the CPU).
            Device 0 will be used if not specified.

            .. deprecated:: 0.2.0
                The ``device_id`` should be specified as ``execution`` option,
                see :attr:`ExecutionCUDA.device_id`.


        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`fft` and :meth:`FFT.execute`. When ``blocking`` is `True`, the
            execution methods do not return until the operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the input tensor
            is on the GPU and ``execution`` is set to 'CUDA'. The execution methods always
            block when the input tensor is on the CPU or ``execution`` is specified to 'CPU'
            to ensure that the user doesn't inadvertently use the result before it becomes
            available. The default is ``"auto"``.

        allocator: An object that supports the :class:`BaseCUDAMemoryManager` protocol, used
            to draw device memory. If an allocator is not provided, a memory allocator from
            the library package will be used (:func:`torch.cuda.caching_allocator_alloc` for
            PyTorch operands, :func:`cupy.cuda.alloc` otherwise). Currently, CPU execution
            does not allow specifying a memory pool.

    See Also:
        :class:`ExecutionCUDA`, :class:`ExecutionCPU`, :class:`FFT`, :func:`fft`,
        :func:`ifft`, :func:`rfft`, and :func:`irfft`.
    """

    fft_type: Literal["C2C", "C2R", "R2C"] | None = None
    inplace: bool = False
    last_axis_parity: Literal["even", "odd"] = "even"
    result_layout: Literal["natural", "optimized"] = "optimized"
    device_id: int | None = None
    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"
    allocator: BaseCUDAMemoryManager | None = None

    def __post_init__(self):
        valid_fft_types = [None, "C2C", "C2R", "R2C"]
        if self.fft_type not in valid_fft_types:
            raise ValueError(f"The value specified for 'fft_type' must be one of {valid_fft_types}.")

        if not isinstance(self.inplace, bool):
            raise ValueError("The value specified for 'inplace' must be of type bool (True or False).")

        valid_last_axis_parity = ["even", "odd"]
        if self.last_axis_parity not in valid_last_axis_parity:
            raise ValueError(f"The value specified for 'last_axis_parity' must be one of {valid_last_axis_parity}.")

        valid_result_layout_options = ["natural", "optimized"]
        if self.result_layout not in valid_result_layout_options:
            raise ValueError(f"The value specified for 'result_layout' must be one of {valid_result_layout_options}.")

        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for 'blocking' must be either True or 'auto'.")

        if self.device_id is not None:
            msg = "`FFTOptions.device_id` is deprecated. Use `ExecutionCUDA.device_id` instead."
            warnings.warn(msg, DeprecationWarning)


@dataclass
class DeviceCallable:
    """A data class capturing LTO-IR callables.

    Attributes:
        ltoir: A device-callable function in LTO-IR format, which can be provided as as
            either as a :class:`bytes` object or as a pointer to the LTO-IR as Python
            :class:`int`.

        size: The size of the LTO-IR callable. If not specified and a :class:`bytes` object
            is passed for ``ltoir``, the size is calculated from it. If a pointer is
            provided for the LTO-IR, `size` must be specified.

        data:  A device pointer to user data used in the callback. The default is None,
            which means a null pointer will be used in the callback.

    See Also:
        :meth:`FFT.plan`, :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`.
    """

    ltoir: int | bytes | None = None
    size: int | None = None
    data: int | None = None

    def __post_init__(self):
        if self.ltoir is None:
            return
        if not isinstance(self.ltoir, int | bytes):
            raise ValueError(
                "The LTO-IR code must be provided as a bytes object or as a Python int "
                "representing the pointer to the LTO-IR code."
            )
        if isinstance(self.ltoir, int) and self.size is None:
            raise ValueError(
                "The size of the LTO-IR code specified as a pointer must be explicitly provided using the 'size' option."
            )
        if isinstance(self.ltoir, bytes):
            self.size = len(self.ltoir)
        if not isinstance(self.size, int):
            raise ValueError(f"Invalid size value: {self.size}.")
        if self.data is None:
            self.data = 0
        if not isinstance(self.data, int):
            raise ValueError("The 'data' attribute must be a Python int representing a device pointer to user data.")


KeywordArgType = dict


class FFTDirection(IntEnum):
    """An IntEnum class specifying the direction of the transform.

    See Also:
        :meth:`FFT.execute`, :func:`fft`
    """

    FORWARD = -1
    INVERSE = 1
