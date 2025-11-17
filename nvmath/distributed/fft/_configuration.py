# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FFTDirection", "FFTOptions"]

from dataclasses import dataclass
from enum import IntEnum
from logging import Logger
from typing import Literal


@dataclass
class FFTOptions:
    """
    A data class for providing options to the :class:`FFT` object and the family of wrapper
    functions :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`.

    Attributes:
        fft_type: The type of FFT to perform, available options include ``'C2C'``,
            ``'C2R'``, and ``'R2C'``. The default is ``'C2C'`` for complex input and
            ``'R2C'`` for real input.

        reshape: Reshape the output distribution to the same slab distribution used by the
            input. This only applies when using a Slab distribution. The default is `True`.

        last_axis_parity: For complex-to-real FFT (corresponding to ``fft_type='C2R'``),
            specify whether the global size of the last axis in the result should be even
            or odd. The even size is calculated as :math:`2 * (m - 1)`, where :math:`m` is
            the size of the last axis of the operand, and the odd size is calculated as
            :math:`2 * (m - 1) + 1`. The specified value should be either ``'even'`` or
            ``'odd'``, with the default being ``'even'``.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`fft` and :meth:`FFT.execute`. When ``blocking`` is `True`, the
            execution methods do not return until the operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the input tensor
            is on the GPU. The execution methods always block when the input tensor is on
            the CPU, to ensure that the user doesn't inadvertently use the result before it
            becomes available. The default is ``"auto"``.

    .. seealso::
        :class:`FFT`, :func:`fft`, :func:`ifft`, :func:`rfft`, and :func:`irfft`.
    """

    fft_type: Literal["C2C", "C2R", "R2C"] | None = None
    reshape: bool = True
    last_axis_parity: Literal["even", "odd"] = "even"
    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"

    def __post_init__(self):
        valid_fft_types = [None, "C2C", "C2R", "R2C"]
        if self.fft_type not in valid_fft_types:
            raise ValueError(f"The value specified for 'fft_type' must be one of {valid_fft_types}.")

        if not isinstance(self.reshape, bool):
            raise ValueError("The value specified for 'reshape' must be of type bool (True or False).")

        valid_last_axis_parity = ["even", "odd"]
        if self.last_axis_parity not in valid_last_axis_parity:
            raise ValueError(f"The value specified for 'last_axis_parity' must be one of {valid_last_axis_parity}.")

        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for 'blocking' must be either True or 'auto'.")


class FFTDirection(IntEnum):
    """An IntEnum class specifying the direction of the transform.

    .. seealso::
        :meth:`FFT.execute`, :func:`fft`
    """

    FORWARD = -1
    INVERSE = 1
