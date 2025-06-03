# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["ReshapeOptions"]

from dataclasses import dataclass
from logging import Logger
from typing import Literal


@dataclass
class ReshapeOptions:
    """
    A data class for providing options to the :class:`Reshape` object and the wrapper
    function :func:`reshape`.

    Attributes:
        logger (logging.Logger): Python Logger object. The root logger will be used if a
            logger object is not provided.

        blocking: A flag specifying the behavior of the execution functions and methods,
            such as :func:`reshape` and :meth:`Reshape.execute`. When ``blocking`` is
            `True`, the execution methods do not return until the operation is complete.
            When ``blocking`` is ``"auto"``, the methods return immediately when the input
            tensor is on the GPU. The execution methods always block when the input tensor
            is on the CPU to ensure that the user doesn't inadvertently use the result
            before it becomes available. The default is ``"auto"``.

    See Also:
        :class:`Reshape` and :func:`reshape`.
    """

    logger: Logger | None = None
    blocking: Literal[True, "auto"] = "auto"

    def __post_init__(self):
        if self.blocking not in (True, "auto"):
            raise ValueError("The value specified for 'blocking' must be either True or 'auto'.")
