"""
BLAS function wrapper utilities for dynamically loading and wrapping Level-3 matrix
multiplication functions from cuBLAS and NVPL BLAS backends. Handles data type to BLAS
abbreviation mapping, function name generation, and provides unified access with proper
handle and stream management.
"""

import logging
import typing

from nvmath._internal.templates import ExecutionCPU, ExecutionCUDA
from nvmath.bindings._internal.utils import FunctionNotFoundError
from nvmath.internal import typemaps, utils
from nvmath.linalg._internal.utils import get_handle
import nvmath.bindings.cublas as cublas
import nvmath.bindings.nvpl.blas as blas


def get_value_zeroth_element(array):
    """Returns the value of zeroth element."""
    return array[0]


def get_address_zeroth_element(array):
    """Returns the memory address of the zeroth element."""
    return array.ctypes.data


def _blas_dtype_abbreviation(dtype: typemaps.cudaDataType) -> str:
    """Return the BLAS Level-3 API abbreviation of a dtype."""
    match dtype:
        case typemaps.cudaDataType.CUDA_R_32F:
            return "s"
        case typemaps.cudaDataType.CUDA_R_64F:
            return "d"
        case typemaps.cudaDataType.CUDA_C_32F:
            return "c"
        case typemaps.cudaDataType.CUDA_C_64F:
            return "z"
        case typemaps.cudaDataType.CUDA_R_16F:
            return "h"
        case _:
            msg = f"'{dtype.name}' has no known BLAS abbreviation."
            raise ValueError(msg)


def _netlib_mm_function_name(dtype: typemaps.cudaDataType, matrix_descr_abbreviation: str) -> str:
    """Return a netlib API Level-3 function name based on the parameters."""
    return f"{_blas_dtype_abbreviation(dtype)}{matrix_descr_abbreviation}mm"


def _cublas_mm_function_name(
    dtype: typemaps.cudaDataType,
    matrix_descr_abbreviation: str,
    batch_type: typing.Literal["", "stride", "group"] = "",
) -> str:
    """Return a cuBLAS API Level-3 function name based on the parameters."""
    match batch_type:
        case "stride":
            suffix = "_strided_batched"
        case "group":
            suffix = "_grouped_batched"
        case "":
            suffix = ""
        case _:
            raise ValueError("batch_type is invalid.")
    return _netlib_mm_function_name(dtype, matrix_descr_abbreviation) + suffix + "_64"


def cublas_mm_function(
    execution: ExecutionCUDA | ExecutionCPU,
    dtype: typemaps.cudaDataType,
    matrix_descr_abbreviation: str,
    logger: logging.Logger,
    batch_type: typing.Literal["", "stride", "group"] = "",
) -> typing.Callable:
    """Return a cublas API Level-3 function from nvmath.bindings.cublas."""
    # We get the cublas handle and set the stream here because other BLAS implementations do
    # not have these constructs
    assert isinstance(execution, ExecutionCUDA)
    handle = get_handle(device_id=execution.device_id, binding="cublas")
    function_name = _cublas_mm_function_name(dtype, matrix_descr_abbreviation, batch_type)
    try:
        function = getattr(cublas, function_name)
        try:
            function()
        except TypeError:
            pass

        def wrapped_with_handle_and_stream(*args, stream_holder: utils.StreamHolder):
            cublas.set_stream(handle, stream_holder.ptr)
            function(handle, *args)

        wrapped_with_handle_and_stream.__name__ = function.__name__

        logger.info("Loaded a cuBLAS API function named %s", function.__name__)
        return wrapped_with_handle_and_stream
    except (AttributeError, FunctionNotFoundError) as e:
        # The user may try to call a newer function with older cuBLAS.
        cublas_version = cublas.get_version(handle)
        msg = (
            f"{function_name}() is an unknown cuBLAS function "
            f"for cuBLAS version {cublas_version}. Please check the cuBLAS Level-3 Function Reference "
            f"to see whether this function should exists for cuBLAS version {cublas_version}."
        )
        raise NotImplementedError(msg) from e


def cublas_enum_mapper(enum):
    """Maps cuBLAS enums to cuBLAS enum."""
    return enum


def _nvpl_mm_function_name(
    dtype: typemaps.cudaDataType,
    matrix_descr_abbreviation: str,
    batch_type: typing.Literal["", "stride", "group"] = "",
) -> str:
    """Return an NVPL API Level-3 function name based on the parameters."""
    match batch_type:
        case "stride":
            suffix = "_batch_strided"
        case "group":
            suffix = "_batch_grouped"
        case "":
            suffix = ""
        case _:
            raise ValueError("batch_type is invalid.")
    return _netlib_mm_function_name(dtype, matrix_descr_abbreviation) + suffix


def nvpl_mm_function(
    execution: ExecutionCUDA | ExecutionCPU,
    dtype: typemaps.cudaDataType,
    matrix_descr_abbreviation: str,
    logger: logging.Logger,
    batch_type: typing.Literal["", "stride", "group"] = "",
) -> typing.Callable:
    """Return an NVPL API Level-3 function from nvmath.bindings.nvpl.blas."""
    if matrix_descr_abbreviation == "tr":
        # FIXME: Reconcile API differences
        raise NotImplementedError("trmm on CPU is unsupported at this time because the cuBLAS API differs.")
    assert isinstance(execution, ExecutionCPU)
    function_name = _nvpl_mm_function_name(dtype, matrix_descr_abbreviation, batch_type)
    try:
        function = getattr(blas, function_name)
        try:
            function()
        except TypeError:
            pass

        new_num_threads = 0 if execution.num_threads is None else execution.num_threads

        for set_num_threads_local_name in [
            "set_num_threads_local",
            "mkl_set_num_threads_local",
            "openblas_set_num_threads_local",
        ]:
            try:
                set_num_threads_local = getattr(blas, set_num_threads_local_name)
                old_num_threads = set_num_threads_local(new_num_threads)
                set_num_threads_local(old_num_threads)
            except FunctionNotFoundError as e:
                logger.debug(e)
                pass
            else:
                logger.debug(f"function {set_num_threads_local_name} is valid.")
                break
        else:
            # If none of the local setting functions are valid, implement a dummy function
            def set_num_threads_local(x):
                pass

        def wrapped_with_threads_and_stream(*args, stream_holder: None):
            old_num_threads = set_num_threads_local(new_num_threads)
            function(blas.ORDER.ColMajor, *args)
            set_num_threads_local(old_num_threads)

        wrapped_with_threads_and_stream.__name__ = function.__name__

        logger.info("Loaded a NVPL BLAS API function named %s", function.__name__)
        return wrapped_with_threads_and_stream
    except (AttributeError, FunctionNotFoundError) as e:
        # The user may try to call a newer function with older cuBLAS.
        try:
            blas_version = blas.get_version()
        except FunctionNotFoundError:
            blas_version = "unknown"
        msg = (
            f"{function_name}() is an unknown NVPL BLAS function "
            f"for NVPL BLAS version {blas_version}. Please check the BLAS Level-3 Function Reference "
            f"to see whether this function should exists for NVPL BLAS version {blas_version}."
        )
        raise NotImplementedError(msg) from e


# NOTE: We have to map from name (str) to enum because of value collisions in the enums
_CUBLAS_ENUM_TO_NVPL_ENUM: dict[str, int] = {
    cublas.DiagType.NON_UNIT.name: blas.DIAG.NonUnit,
    cublas.DiagType.UNIT.name: blas.DIAG.Unit,
    cublas.FillMode.LOWER.name: blas.UPLO.Lower,
    cublas.FillMode.UPPER.name: blas.UPLO.Upper,
    cublas.Operation.C.name: blas.TRANSPOSE.ConjTrans,
    cublas.Operation.N.name: blas.TRANSPOSE.NoTrans,
    cublas.Operation.T.name: blas.TRANSPOSE.Trans,
    cublas.SideMode.LEFT.name: blas.SIDE.Left,
    cublas.SideMode.RIGHT.name: blas.SIDE.Right,
}


def nvpl_enum_mapper(enum):
    """Maps cuBLAS enums to BLAS enums."""
    return _CUBLAS_ENUM_TO_NVPL_ENUM[enum.name]
