# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["current_device_lto", "ComputeCapability", "CodeType", "ISAVersion", "Code", "Dim3", "MAX_SUPPORTED_CC"]

from typing import NamedTuple
from cuda.bindings import runtime as cudart, driver as cudadrv
import logging
from collections import namedtuple


# Code = CodeType + ISAVersion + buffer
class Code(namedtuple("Code", ("code_type", "isa_version", "data"))):
    """
    A namedtuple class that encapsulates code type, version, and buffer.

    Attributes:
        code_type (CodeType): The underlying code type.
        isa_version (ISAVersion): The instruction set architecture version for the code.
        data: The buffer pointer.
    """

    pass


# CodeType = type + CC
class CodeType(namedtuple("CodeType", ("kind", "cc"))):
    """
    A namedtuple class that encapsulates code kind and compute capability.

    Attributes:
        kind (str): A string denoting the nature of the code, e.g, ``'lto'``.
        cc (ComputeCapability): The current GPU compute capability.
    """

    pass


# CC = e.g. SM 9.0 (Hopper)
class ComputeCapability(NamedTuple):
    """
    A namedtuple class that encapsulates the major and minor compute capability.

    Attributes:
        major (int): The major compute capability.
        minor (int): The minor compute capability.
    """

    major: int
    minor: int

    @property
    def integer(self) -> int:
        """Integer representation of the ISAVersion"""
        return self.major * 100 + self.minor * 10

    def __str__(self):
        """String representation of the ComputeCapability"""
        return f"{self.major}.{self.minor}"

    pass


MAX_SUPPORTED_CC = ComputeCapability(12, 1)


class Dim3(namedtuple("Dim3", ("x", "y", "z"), defaults=(1, 1, 1))):
    """
    A namedtuple class that encapsulates the dimensions for grids and blocks.

    Attributes:
        x (int): The dimension in the x direction (default 1).
        y (int): The dimension in the y direction (default 1).
        z (int): The dimension in the z direction (default 1).
    """

    pass


# ISAVersion = e.g. 12.3 (from CUDA 12.3)
class ISAVersion(NamedTuple):
    """
    A namedtuple class that encapsulates the code version.

    Attributes:
        major (int): The major code version.
        minor (int): The minor code version.
    """

    major: int
    minor: int

    @classmethod
    def from_integer(cls, isa: int):
        major = isa // 1000
        minor = (isa % 1000) // 10

        return cls(major, minor)


def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDArt Error: {str} ({err})")


def CHECK_CUDA(err):
    if err != cudadrv.CUresult.CUDA_SUCCESS:
        err2, str = cudadrv.cuGetErrorName(err)
        raise RuntimeError(f"CUDA Error: {str} ({err})")


def get_current_device_cc():
    (err,) = cudadrv.cuInit(0)
    CHECK_CUDA(err)
    # Check if a context exist
    err, pctx = cudadrv.cuCtxGetCurrent()
    CHECK_CUDA(err)
    if int(pctx) == 0:
        # If not, return the CC of device 0
        device = 0
    else:
        err, device = cudart.cudaGetDevice()
        CHECK_CUDART(err)
    err, prop = cudart.cudaGetDeviceProperties(device)
    CHECK_CUDART(err)
    major, minor = prop.major, prop.minor
    if (major, minor) > MAX_SUPPORTED_CC:
        logging.info(
            "The current device supports compute capability "
            f"{prop.major}.{prop.minor}, but the generated LTO version is "
            f"capped at {MAX_SUPPORTED_CC}."
        )
        major, minor = MAX_SUPPORTED_CC
    logging.info(f"Using device {device} for default compute capability, found cc = {prop.major}.{prop.minor}")
    return ComputeCapability(major, minor)


def get_default_code_type() -> CodeType:
    try:
        return CodeType("lto", get_current_device_cc())
    except RuntimeError as e:
        raise ValueError(f"Failed to get the current GPU compute capability (got {e}). Set code_type explicitly.")


def current_device_lto():
    """
    A helper function to get the default code type for link time optimization (LTO) on the
    current device.

    Returns:
        A :class:`CodeType` object representing the default LTO code type for the current
        device.
    """
    return get_default_code_type()
