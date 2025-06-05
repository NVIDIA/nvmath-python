# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["FFTDirection", "FFTOptions", "Slab"]

from dataclasses import dataclass
from enum import IntEnum
from logging import Logger
from typing import Literal
from nvmath.bindings import cufftMp  # type: ignore


@dataclass
class FFTOptions:
    """
    A data class for providing options to the :class:`FFT` object and the family of wrapper
    functions :func:`fft` and :func:`ifft`.

    Attributes:
        fft_type: The type of FFT to perform, available options include ``'C2C'``.

        reshape: Reshape the output distribution to the same slab distribution used by the
            input. This only applies when using a Slab distribution. The default is `True`.

        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`fft` and :meth:`FFT.execute`. When ``blocking`` is `True`, the
            execution methods do not return until the operation is complete. When
            ``blocking`` is ``"auto"``, the methods return immediately when the input tensor
            is on the GPU. The execution methods always block when the input tensor is on
            the CPU, to ensure that the user doesn't inadvertently use the result before it
            becomes available. The default is ``"auto"``.

    See Also:
        :class:`FFT`, :func:`fft` and :func:`ifft`.
    """

    fft_type: Literal["C2C"] | None = None
    reshape: bool = True
    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"

    def __post_init__(self):
        valid_fft_types = [None, "C2C"]
        if self.fft_type not in valid_fft_types:
            raise ValueError(f"The value specified for 'fft_type' must be one of {valid_fft_types}.")

        if not isinstance(self.reshape, bool):
            raise ValueError("The value specified for 'reshape' must be of type bool (True or False).")

        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for 'blocking' must be either True or 'auto'.")


class FFTDirection(IntEnum):
    """An IntEnum class specifying the direction of the transform.

    See Also:
        :meth:`FFT.execute`, :func:`fft`
    """

    FORWARD = -1
    INVERSE = 1


class Slab(IntEnum):
    """An IntEnum class to specify a cuFFTMp Slab distribution.

    Given an array of size X * Y * Z distributed over n GPUs, there are two possible slab
    distributions depending on whether the data is partitioned on the X or Y axis:

    * X axis partitioning: the first X % n GPUs each own (X/n+1) * Y * Z elements and
      the remaining GPUs each own (X/n) * Y * Z elements.

    * Y axis partitioning: the first Y % n GPUs each own X * (Y/n+1) * Z elements and
      the remaining GPUs each own X * (Y/n) * Z elements.

    See Also:
        :class:`FFT`, :func:`fft`
    """

    X = cufftMp.XtSubFormat.FORMAT_INPLACE
    Y = cufftMp.XtSubFormat.FORMAT_INPLACE_SHUFFLED
