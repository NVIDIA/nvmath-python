# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ['current_device_lto', 'ComputeCapability', 'CodeType', 'ISAVersion', 'Code', 'Symbol', 'Dim3']

from cuda import cudart, cuda
import logging
from collections import namedtuple

# Code = CodeType + ISAVersion + buffer
class Code(namedtuple('Code', ('code_type', 'isa_version', 'data'))):
    """
    A namedtuple class that encapsulates code type, version, and buffer.

    Attributes:
        code_type (CodeType): The underlying code type.
        isa_version (ISAVersion): The instruction set architecture version for the code.
        data: The buffer pointer.
    """
    pass


# CodeType = type + CC
class CodeType(namedtuple('CodeType', ('kind', 'cc'))):
    """
    A namedtuple class that encapsulates code kind and compute capability.

    Attributes:
        kind (str): A string denoting the nature of the code, e.g, ``'lto'``.
        cc (ComputeCapability): The current GPU compute capability.
    """
    pass


# CC = e.g. SM 9.0 (Hopper)
class ComputeCapability(namedtuple('ComputeCapability', ('major', 'minor'))):
    """
    A namedtuple class that encapsulates the major and minor compute capability.

    Attributes:
        major (int): The major compute capability.
        minor (int): The minor compute capability.
    """
    pass


class Dim3(namedtuple('Dim3', ('x', 'y', 'z'), defaults=(1, 1, 1))):
    """
    A namedtuple class that encapsulates the dimensions for grids and blocks.

    Attributes:
        x (int): The dimension in the x direction (default 1).
        y (int): The dimension in the y direction (default 1).
        z (int): The dimension in the z direction (default 1).
    """
    pass


class Symbol(namedtuple('symbol', ('variant', 'name'))):
    """
    A namedtuple class that encapsulates a device function symbol and which API it maps to.

    Attributes:
        variant (str): A short description of what API this symbol corresponds to.
        name (str): The (mangled) name of the device function.
    """
    pass


# ISAVersion = e.g. 12.3 (from CUDA 12.3)
class ISAVersion(namedtuple('ISAVersion', ('major', 'minor'))):
    """
    A namedtuple class that encapsulates the code version.

    Attributes:
        major (int): The major code version.
        minor (int): The minor code version.
    """
    pass 

def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDArt Error: {str} ({err})")

def CHECK_CUDA(err):
    if err != cuda.CUresult.CUDA_SUCCESS:
        err2, str = cuda.cuGetErrorName(err)
        raise RuntimeError(f"CUDA Error: {str} ({err})")

def get_current_device_cc():
    err, = cuda.cuInit(0)
    CHECK_CUDA(err)
    # Check if a context exist
    err, pctx = cuda.cuCtxGetCurrent()
    CHECK_CUDA(err)
    if int(pctx) == 0:
        # If not, return the CC of device 0
        device = 0
    else:
        err, device = cudart.cudaGetDevice()
        CHECK_CUDART(err)
    err, prop = cudart.cudaGetDeviceProperties(device)
    CHECK_CUDART(err)
    logging.info(f"Using device {device} for default compute capability, found cc = {prop.major}.{prop.minor}")
    return ComputeCapability(prop.major, prop.minor)


def get_default_code_type():
    try:
        return CodeType('lto', get_current_device_cc())
    except RuntimeError as e:
        raise ValueError(f"Failed to get the current GPU compute capability (got {e}). Set code_type explicitly.")

def current_device_lto():
    """
    A helper function to get the default code type for link time optimization (LTO) on the current device.

    Returns:
        A :class:`CodeType` object representing the default LTO code type for the current device.
    """
    return get_default_code_type()
