# This code was automatically generated with version 0.2.3. Do not modify it directly.

cimport cython  # NOQA

from enum import IntEnum as _IntEnum

from ._internal.utils cimport get_buffer_pointer, get_resource_ptr, nullable_unique_ptr, get_resource_ptrs
from libcpp.vector cimport vector

###############################################################################
# POD
###############################################################################




###############################################################################
# Enum
###############################################################################

class CommondxValueType(_IntEnum):
    """See `commondxValueType`."""
    R_8F_E5M2 = COMMONDX_R_8F_E5M2
    C_8F_E5M2 = COMMONDX_C_8F_E5M2
    R_8F_E4M3 = COMMONDX_R_8F_E4M3
    C_8F_E4M3 = COMMONDX_C_8F_E4M3
    R_16BF = COMMONDX_R_16BF
    C_16BF = COMMONDX_C_16BF
    R_16F2 = COMMONDX_R_16F2
    R_16F = COMMONDX_R_16F
    C_16F = COMMONDX_C_16F
    C_16F2 = COMMONDX_C_16F2
    R_32TF = COMMONDX_R_32TF
    C_32TF = COMMONDX_C_32TF
    R_32F = COMMONDX_R_32F
    C_32F = COMMONDX_C_32F
    R_64F = COMMONDX_R_64F
    C_64F = COMMONDX_C_64F
    R_8I = COMMONDX_R_8I
    C_8I = COMMONDX_C_8I
    R_16I = COMMONDX_R_16I
    C_16I = COMMONDX_C_16I
    R_32I = COMMONDX_R_32I
    C_32I = COMMONDX_C_32I
    R_64I = COMMONDX_R_64I
    C_64I = COMMONDX_C_64I
    R_8UI = COMMONDX_R_8UI
    C_8UI = COMMONDX_C_8UI
    R_16UI = COMMONDX_R_16UI
    C_16UI = COMMONDX_C_16UI
    R_32UI = COMMONDX_R_32UI
    C_32UI = COMMONDX_C_32UI
    R_64UI = COMMONDX_R_64UI
    C_64UI = COMMONDX_C_64UI

class CommondxStatusType(_IntEnum):
    """See `commondxStatusType`."""
    SUCCESS = COMMONDX_SUCCESS
    INVALID_VALUE = COMMONDX_INVALID_VALUE
    INTERNAL_ERROR = COMMONDX_INTERNAL_ERROR
    COMPILATION_ERROR = COMMONDX_COMPILATION_ERROR
    CUFFT_ERROR = COMMONDX_CUFFT_ERROR

class CommondxPrecision(_IntEnum):
    """See `commondxPrecision`."""
    F8_E5M2 = COMMONDX_PRECISION_F8_E5M2
    F8_E4M3 = COMMONDX_PRECISION_F8_E4M3
    BF16 = COMMONDX_PRECISION_BF16
    F16 = COMMONDX_PRECISION_F16
    TF32 = COMMONDX_PRECISION_TF32
    F32 = COMMONDX_PRECISION_F32
    F64 = COMMONDX_PRECISION_F64
    I8 = COMMONDX_PRECISION_I8
    I16 = COMMONDX_PRECISION_I16
    I32 = COMMONDX_PRECISION_I32
    I64 = COMMONDX_PRECISION_I64
    UI8 = COMMONDX_PRECISION_UI8
    UI16 = COMMONDX_PRECISION_UI16
    UI32 = COMMONDX_PRECISION_UI32
    UI64 = COMMONDX_PRECISION_UI64

class CommondxOption(_IntEnum):
    """See `commondxOption`."""
    SYMBOL_NAME = COMMONDX_OPTION_SYMBOL_NAME
    TARGET_SM = COMMONDX_OPTION_TARGET_SM
    CODE_CONTAINER = COMMONDX_OPTION_CODE_CONTAINER
    CODE_ISA = COMMONDX_OPTION_CODE_ISA
    EXTRA_NVTRC_ARGS = COMMONDX_OPTION_EXTRA_NVTRC_ARGS

class CommondxExecution(_IntEnum):
    """See `commondxExecution`."""
    THREAD = COMMONDX_EXECUTION_THREAD
    BLOCK = COMMONDX_EXECUTION_BLOCK

class CommondxCodeContainer(_IntEnum):
    """See `commondxCodeContainer`."""
    LTOIR = COMMONDX_CODE_CONTAINER_LTOIR
    FATBIN = COMMONDX_CODE_CONTAINER_FATBIN

class CublasdxApi(_IntEnum):
    """See `cublasdxApi`."""
    SMEM = CUBLASDX_API_SMEM
    SMEM_DYNAMIC_LD = CUBLASDX_API_SMEM_DYNAMIC_LD
    TENSORS = CUBLASDX_API_TENSORS

class CublasdxType(_IntEnum):
    """See `cublasdxType`."""
    REAL = CUBLASDX_TYPE_REAL
    COMPLEX = CUBLASDX_TYPE_COMPLEX

class CublasdxTransposeMode(_IntEnum):
    """See `cublasdxTransposeMode`."""
    NON_TRANSPOSED = CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED
    TRANSPOSED = CUBLASDX_TRANSPOSE_MODE_TRANSPOSED
    CONJ_TRANSPOSED = CUBLASDX_TRANSPOSE_MODE_CONJ_TRANSPOSED

class CublasdxArrangement(_IntEnum):
    """See `cublasdxArrangement`."""
    COL_MAJOR = CUBLASDX_ARRANGEMENT_COL_MAJOR
    ROW_MAJOR = CUBLASDX_ARRANGEMENT_ROW_MAJOR

class CublasdxFunction(_IntEnum):
    """See `cublasdxFunction`."""
    MM = CUBLASDX_FUNCTION_MM

class CublasdxOperatorType(_IntEnum):
    """See `cublasdxOperatorType`."""
    FUNCTION = CUBLASDX_OPERATOR_FUNCTION
    SIZE = CUBLASDX_OPERATOR_SIZE
    TYPE = CUBLASDX_OPERATOR_TYPE
    PRECISION = CUBLASDX_OPERATOR_PRECISION
    SM = CUBLASDX_OPERATOR_SM
    EXECUTION = CUBLASDX_OPERATOR_EXECUTION
    BLOCK_DIM = CUBLASDX_OPERATOR_BLOCK_DIM
    LEADING_DIMENSION = CUBLASDX_OPERATOR_LEADING_DIMENSION
    TRANSPOSE_MODE = CUBLASDX_OPERATOR_TRANSPOSE_MODE
    API = CUBLASDX_OPERATOR_API
    ARRANGEMENT = CUBLASDX_OPERATOR_ARRANGEMENT
    ALIGNMENT = CUBLASDX_OPERATOR_ALIGNMENT
    STATIC_BLOCK_DIM = CUBLASDX_OPERATOR_STATIC_BLOCK_DIM

class CublasdxTraitType(_IntEnum):
    """See `cublasdxTraitType`."""
    VALUE_TYPE = CUBLASDX_TRAIT_VALUE_TYPE
    SIZE = CUBLASDX_TRAIT_SIZE
    BLOCK_SIZE = CUBLASDX_TRAIT_BLOCK_SIZE
    BLOCK_DIM = CUBLASDX_TRAIT_BLOCK_DIM
    LEADING_DIMENSION = CUBLASDX_TRAIT_LEADING_DIMENSION
    SYMBOL_NAME = CUBLASDX_TRAIT_SYMBOL_NAME
    ARRANGEMENT = CUBLASDX_TRAIT_ARRANGEMENT
    ALIGNMENT = CUBLASDX_TRAIT_ALIGNMENT
    SUGGESTED_LEADING_DIMENSION = CUBLASDX_TRAIT_SUGGESTED_LEADING_DIMENSION
    SUGGESTED_BLOCK_DIM = CUBLASDX_TRAIT_SUGGESTED_BLOCK_DIM
    MAX_THREADS_PER_BLOCK = CUBLASDX_TRAIT_MAX_THREADS_PER_BLOCK

class CublasdxTensorType(_IntEnum):
    """See `cublasdxTensorType`."""
    SMEM_A = CUBLASDX_TENSOR_SMEM_A
    SMEM_B = CUBLASDX_TENSOR_SMEM_B
    SMEM_C = CUBLASDX_TENSOR_SMEM_C
    SUGGESTED_SMEM_A = CUBLASDX_TENSOR_SUGGESTED_SMEM_A
    SUGGESTED_SMEM_B = CUBLASDX_TENSOR_SUGGESTED_SMEM_B
    SUGGESTED_SMEM_C = CUBLASDX_TENSOR_SUGGESTED_SMEM_C
    SUGGESTED_RMEM_C = CUBLASDX_TENSOR_SUGGESTED_RMEM_C
    GMEM_A = CUBLASDX_TENSOR_GMEM_A
    GMEM_B = CUBLASDX_TENSOR_GMEM_B
    GMEM_C = CUBLASDX_TENSOR_GMEM_C

class CublasdxTensorOption(_IntEnum):
    """See `cublasdxTensorOption`."""
    ALIGNMENT_BYTES = CUBLASDX_TENSOR_OPTION_ALIGNMENT_BYTES

class CublasdxTensorTrait(_IntEnum):
    """See `cublasdxTensorTrait`."""
    STORAGE_BYTES = CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES
    ALIGNMENT_BYTES = CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES
    UID = CUBLASDX_TENSOR_TRAIT_UID
    OPAQUE_NAME = CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME

class CublasdxDeviceFunctionTrait(_IntEnum):
    """See `cublasdxDeviceFunctionTrait`."""
    SYMBOL = CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL

class CublasdxDeviceFunctionOption(_IntEnum):
    """See `cublasdxDeviceFunctionOption`."""
    COPY_ALIGNMENT = CUBLASDX_DEVICE_FUNCTION_OPTION_COPY_ALIGNMENT

class CublasdxDeviceFunctionType(_IntEnum):
    """See `cublasdxDeviceFunctionType`."""
    EXECUTE = CUBLASDX_DEVICE_FUNCTION_EXECUTE
    COPY = CUBLASDX_DEVICE_FUNCTION_COPY
    COPY_WAIT = CUBLASDX_DEVICE_FUNCTION_COPY_WAIT
    CLEAR = CUBLASDX_DEVICE_FUNCTION_CLEAR
    AXPBY = CUBLASDX_DEVICE_FUNCTION_AXPBY

class CufftdxApi(_IntEnum):
    """See `cufftdxApi`."""
    LMEM = CUFFTDX_API_LMEM
    SMEM = CUFFTDX_API_SMEM

class CufftdxType(_IntEnum):
    """See `cufftdxType`."""
    C2C = CUFFTDX_TYPE_C2C
    R2C = CUFFTDX_TYPE_R2C
    C2R = CUFFTDX_TYPE_C2R

class CufftdxDirection(_IntEnum):
    """See `cufftdxDirection`."""
    FORWARD = CUFFTDX_DIRECTION_FORWARD
    INVERSE = CUFFTDX_DIRECTION_INVERSE

class CufftdxComplexLayout(_IntEnum):
    """See `cufftdxComplexLayout`."""
    NATURAL = CUFFTDX_COMPLEX_LAYOUT_NATURAL
    PACKED = CUFFTDX_COMPLEX_LAYOUT_PACKED
    FULL = CUFFTDX_COMPLEX_LAYOUT_FULL

class CufftdxRealMode(_IntEnum):
    """See `cufftdxRealMode`."""
    NORMAL = CUFFTDX_REAL_MODE_NORMAL
    FOLDED = CUFFTDX_REAL_MODE_FOLDED

class CufftdxCodeType(_IntEnum):
    """See `cufftdxCodeType`."""
    PTX = CUFFTDX_CODE_TYPE_PTX
    LTOIR = CUFFTDX_CODE_TYPE_LTOIR

class CufftdxOperatorType(_IntEnum):
    """See `cufftdxOperatorType`."""
    SIZE = CUFFTDX_OPERATOR_SIZE
    DIRECTION = CUFFTDX_OPERATOR_DIRECTION
    TYPE = CUFFTDX_OPERATOR_TYPE
    PRECISION = CUFFTDX_OPERATOR_PRECISION
    SM = CUFFTDX_OPERATOR_SM
    EXECUTION = CUFFTDX_OPERATOR_EXECUTION
    FFTS_PER_BLOCK = CUFFTDX_OPERATOR_FFTS_PER_BLOCK
    ELEMENTS_PER_THREAD = CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD
    BLOCK_DIM = CUFFTDX_OPERATOR_BLOCK_DIM
    REAL_FFT_OPTIONS = CUFFTDX_OPERATOR_REAL_FFT_OPTIONS
    API = CUFFTDX_OPERATOR_API
    CODE_TYPE = CUFFTDX_OPERATOR_CODE_TYPE

class CufftdxKnobType(_IntEnum):
    """See `cufftdxKnobType`."""
    ELEMENTS_PER_THREAD = CUFFTDX_KNOB_ELEMENTS_PER_THREAD
    FFTS_PER_BLOCK = CUFFTDX_KNOB_FFTS_PER_BLOCK

class CufftdxTraitType(_IntEnum):
    """See `cufftdxTraitType`."""
    VALUE_TYPE = CUFFTDX_TRAIT_VALUE_TYPE
    INPUT_TYPE = CUFFTDX_TRAIT_INPUT_TYPE
    OUTPUT_TYPE = CUFFTDX_TRAIT_OUTPUT_TYPE
    IMPLICIT_TYPE_BATCHING = CUFFTDX_TRAIT_IMPLICIT_TYPE_BATCHING
    ELEMENTS_PER_THREAD = CUFFTDX_TRAIT_ELEMENTS_PER_THREAD
    STORAGE_SIZE = CUFFTDX_TRAIT_STORAGE_SIZE
    STRIDE = CUFFTDX_TRAIT_STRIDE
    BLOCK_DIM = CUFFTDX_TRAIT_BLOCK_DIM
    SHARED_MEMORY_SIZE = CUFFTDX_TRAIT_SHARED_MEMORY_SIZE
    FFTS_PER_BLOCK = CUFFTDX_TRAIT_FFTS_PER_BLOCK
    SYMBOL_NAME = CUFFTDX_TRAIT_SYMBOL_NAME
    INPUT_LENGTH = CUFFTDX_TRAIT_INPUT_LENGTH
    OUTPUT_LENGTH = CUFFTDX_TRAIT_OUTPUT_LENGTH
    INPUT_ELEMENTS_PER_THREAD = CUFFTDX_TRAIT_INPUT_ELEMENTS_PER_THREAD
    OUTPUT_ELEMENTS_PER_THREAD = CUFFTDX_TRAIT_OUTPUT_ELEMENTS_PER_THREAD
    SUGGESTED_FFTS_PER_BLOCK = CUFFTDX_TRAIT_SUGGESTED_FFTS_PER_BLOCK

class CusolverdxApi(_IntEnum):
    """See `cusolverdxApi`."""
    SMEM = CUSOLVERDX_API_SMEM
    SMEM_DYNAMIC_LD = CUSOLVERDX_API_SMEM_DYNAMIC_LD

class CusolverdxType(_IntEnum):
    """See `cusolverdxType`."""
    REAL = CUSOLVERDX_TYPE_REAL
    COMPLEX = CUSOLVERDX_TYPE_COMPLEX

class CusolverdxFunction(_IntEnum):
    """See `cusolverdxFunction`."""
    GETRF_NO_PIVOT = CUSOLVERDX_FUNCTION_GETRF_NO_PIVOT
    GETRS_NO_PIVOT = CUSOLVERDX_FUNCTION_GETRS_NO_PIVOT
    POTRF = CUSOLVERDX_FUNCTION_POTRF
    POTRS = CUSOLVERDX_FUNCTION_POTRS
    TRSM = CUSOLVERDX_FUNCTION_TRSM

class CusolverdxArrangement(_IntEnum):
    """See `cusolverdxArrangement`."""
    COL_MAJOR = CUSOLVERDX_ARRANGEMENT_COL_MAJOR
    ROW_MAJOR = CUSOLVERDX_ARRANGEMENT_ROW_MAJOR

class CusolverdxFillMode(_IntEnum):
    """See `cusolverdxFillMode`."""
    UPPER = CUSOLVERDX_FILL_MODE_UPPER
    LOWER = CUSOLVERDX_FILL_MODE_LOWER

class CusolverdxSide(_IntEnum):
    """See `cusolverdxSide`."""
    LEFT = CUSOLVERDX_SIDE_LEFT
    RIGHT = CUSOLVERDX_SIDE_RIGHT

class CusolverdxDiag(_IntEnum):
    """See `cusolverdxDiag`."""
    UNIT = CUSOLVERDX_DIAG_UNIT
    NON_UNIT = CUSOLVERDX_DIAG_NON_UNIT

class CusolverdxOperatorType(_IntEnum):
    """See `cusolverdxOperatorType`."""
    SIZE = CUSOLVERDX_OPERATOR_SIZE
    TYPE = CUSOLVERDX_OPERATOR_TYPE
    PRECISION = CUSOLVERDX_OPERATOR_PRECISION
    SM = CUSOLVERDX_OPERATOR_SM
    EXECUTION = CUSOLVERDX_OPERATOR_EXECUTION
    BLOCK_DIM = CUSOLVERDX_OPERATOR_BLOCK_DIM
    API = CUSOLVERDX_OPERATOR_API
    FUNCTION = CUSOLVERDX_OPERATOR_FUNCTION
    ARRANGEMENT = CUSOLVERDX_OPERATOR_ARRANGEMENT
    FILL_MODE = CUSOLVERDX_OPERATOR_FILL_MODE
    SIDE = CUSOLVERDX_OPERATOR_SIDE
    DIAG = CUSOLVERDX_OPERATOR_DIAG

class CusolverdxTraitType(_IntEnum):
    """See `cusolverdxTraitType`."""
    SHARED_MEMORY_SIZE = CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE
    SYMBOL_NAME = CUSOLVERDX_TRAIT_SYMBOL_NAME


###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    COMMONDX_SUCCESS          : 'LIBMATHDX_SUCCESS',
    COMMONDX_INVALID_VALUE    : 'LIBMATHDX_INVALID_VALUE',
    COMMONDX_INTERNAL_ERROR   : 'LIBMATHDX_INTERNAL_ERROR',
    COMMONDX_COMPILATION_ERROR: 'LIBMATHDX_COMPILATION_ERROR',
}

class LibMathDxError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = STATUS[status]
        super(LibMathDxError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise LibMathDxError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef long long int commondx_create_code() except? 0:
    """Creates a code handle.

    Returns:
        long long int: A pointer to the output code handle.

    .. seealso:: `commondxCreateCode`
    """
    cdef commondxCode code
    with nogil:
        status = commondxCreateCode(&code)
    check_status(status)
    return <long long int>code


cpdef commondx_set_code_option_int64(long long int code, int option, long long int value):
    """Set an option on a code handle.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        option (CommondxOption): The option to set the code to.
        value (long long int): A corresponding value for the selected option.

    .. seealso:: `commondxSetCodeOptionInt64`
    """
    with nogil:
        status = commondxSetCodeOptionInt64(<commondxCode>code, <_CommondxOption>option, value)
    check_status(status)


cpdef commondx_set_code_option_str(long long int code, int option, value):
    """Set a C-string option on a code handle.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        option (CommondxOption): The option to set the code to.
        value (str): A C-string. Cannot be ``NULL``.

    .. seealso:: `commondxSetCodeOptionStr`
    """
    if not isinstance(value, str):
        raise TypeError("value must be a Python str")
    cdef bytes _temp_value_ = (<str>value).encode()
    cdef char* _value_ = _temp_value_
    with nogil:
        status = commondxSetCodeOptionStr(<commondxCode>code, <_CommondxOption>option, <const char*>_value_)
    check_status(status)


cpdef long long int commondx_get_code_option_int64(long long int code, int option) except? 0:
    """Get option from a code handle.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        option (CommondxOption): The option to get.

    Returns:
        long long int: The option value.

    .. seealso:: `commondxGetCodeOptionInt64`
    """
    cdef long long int value
    with nogil:
        status = commondxGetCodeOptionInt64(<commondxCode>code, <_CommondxOption>option, &value)
    check_status(status)
    return value


cpdef commondx_get_code_options_int64s(long long int code, int option, size_t size, array):
    """Get options (as an array) from a code handle, with one option per output code.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        option (CommondxOption): The option to get.
        size (size_t): The array size, as result from commondxGetCodeNumLTOIRs.
        array (object): A pointer to the beginning of the output array. Must be a pointer to a buffer of at least ``size`` elements. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `commondxGetCodeOptionsInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = commondxGetCodeOptionsInt64s(<commondxCode>code, <_CommondxOption>option, size, <long long int*>(_array_.data()))
    check_status(status)


cpdef size_t commondx_get_code_ltoir_size(long long int code) except? 0:
    """Extract the LTOIR size, in bytes.

    Args:
        code (long long int): A code handle from commondxCreateCode.

    Returns:
        size_t: The LTOIR size, in bytes.

    .. seealso:: `commondxGetCodeLTOIRSize`
    """
    cdef size_t size
    with nogil:
        status = commondxGetCodeLTOIRSize(<commondxCode>code, &size)
    check_status(status)
    return size


cpdef commondx_get_code_ltoir(long long int code, size_t size, out):
    """Extract the LTOIR.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        size (size_t): The LTOIR size, as returned by commondxGetCodeLTOIRSize.
        out (bytes): The LTOIR. Must be a pointer to a buffer of at least size byte.

    .. seealso:: `commondxGetCodeLTOIR`
    """
    cdef void* _out_ = get_buffer_pointer(out, size, readonly=False)
    with nogil:
        status = commondxGetCodeLTOIR(<commondxCode>code, size, <void*>_out_)
    check_status(status)


cpdef size_t commondx_get_code_num_ltoirs(long long int code) except? 0:
    """Returns the number the LTOIR chunks.

    Args:
        code (long long int): A code handle from commondxCreateCode.

    Returns:
        size_t: The number of LTOIR chunks.

    .. seealso:: `commondxGetCodeNumLTOIRs`
    """
    cdef size_t size
    with nogil:
        status = commondxGetCodeNumLTOIRs(<commondxCode>code, &size)
    check_status(status)
    return size


cpdef commondx_get_code_ltoir_sizes(long long int code, size_t size, out):
    """Returns the size of all LTOIR chunks.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        size (size_t): The number of LTOIR chunks, as returned by commondxGetCodeNumLTOIRs.
        out (object): On output, ``out[i]`` is the size, in byte, of the ith LTOIR chunk. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``size_t``.


    .. seealso:: `commondxGetCodeLTOIRSizes`
    """
    cdef nullable_unique_ptr[ vector[size_t] ] _out_
    get_resource_ptr[size_t](_out_, out, <size_t*>NULL)
    with nogil:
        status = commondxGetCodeLTOIRSizes(<commondxCode>code, size, <size_t*>(_out_.data()))
    check_status(status)


cpdef commondx_get_code_ltoirs(long long int code, size_t size, out):
    """Returns all LTOIR chunks.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        size (size_t): The number of LTOIR chunks, as returned by commondxGetCodeNumLTOIRs.
        out (object): On output, ``out[i]`` is filled with the ith LTOIR chunk. ``out[i]`` must point to a buffer of at least ``size[i]`` bytes, with ``size`` the output of commondxGetCodeLTOIRSizes. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of :class:`int`\s (as pointer addresses).


    .. seealso:: `commondxGetCodeLTOIRs`
    """
    cdef nullable_unique_ptr[ vector[void*] ] _out_
    get_resource_ptrs[void](_out_, out, <void*>NULL)
    with nogil:
        status = commondxGetCodeLTOIRs(<commondxCode>code, size, <void**>(_out_.data()))
    check_status(status)


cpdef commondx_destroy_code(long long int code):
    """Destroys a code handle.

    Args:
        code (long long int): A code handle from commondxCreateCode.

    .. seealso:: `commondxDestroyCode`
    """
    with nogil:
        status = commondxDestroyCode(<commondxCode>code)
    check_status(status)


cpdef str commondx_status_to_str(int status):
    """Convert a status enum to a human readable C-string.

    Args:
        status (CommondxStatusType): The status enum to convert.

    .. seealso:: `commondxStatusToStr`
    """
    cdef bytes _output_
    _output_ = commondxStatusToStr(<_CommondxStatusType>status)
    return _output_.decode()


cpdef int get_version() except? 0:
    """Returns the libmathdx version as a single integer.

    Returns:
        int: The version, encoded as 1000 * major + 100 * minor + patch.

    .. seealso:: `mathdxGetVersion`
    """
    cdef int version
    with nogil:
        status = mathdxGetVersion(&version)
    check_status(status)
    return version


cpdef tuple get_version_ex():
    """Returns the libmathdx version as a triplet of integers.

    Returns:
        A 3-tuple containing:

        - int: The major version.
        - int: The minor version.
        - int: The patch version.

    .. seealso:: `mathdxGetVersionEx`
    """
    cdef int major
    cdef int minor
    cdef int patch
    with nogil:
        status = mathdxGetVersionEx(&major, &minor, &patch)
    check_status(status)
    return (major, minor, patch)


cpdef long long int cublasdx_create_descriptor() except? 0:
    """Creates a cuBLASDx descriptor.

    Returns:
        long long int: A pointer to a handle.

    .. seealso:: `cublasdxCreateDescriptor`
    """
    cdef cublasdxDescriptor handle
    with nogil:
        status = cublasdxCreateDescriptor(&handle)
    check_status(status)
    return <long long int>handle


cpdef cublasdx_set_option_str(long long int handle, int option, value):
    """Sets a C-string option on a cuBLASDx descriptor.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        option (CommondxOption): An option to set the descriptor to.
        value (str): A pointer to a C-string. Cannot be ``NULL``.

    .. seealso:: `cublasdxSetOptionStr`
    """
    if not isinstance(value, str):
        raise TypeError("value must be a Python str")
    cdef bytes _temp_value_ = (<str>value).encode()
    cdef char* _value_ = _temp_value_
    with nogil:
        status = cublasdxSetOptionStr(<cublasdxDescriptor>handle, <_CommondxOption>option, <const char*>_value_)
    check_status(status)


cpdef cublasdx_set_operator_int64(long long int handle, int op, long long int value):
    """Set an operator on a cuBLASDx descriptor to an integer value.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        op (CublasdxOperatorType): An operator to set the descriptor to.
        value (long long int): The operator's value.

    .. seealso:: `cublasdxSetOperatorInt64`
    """
    with nogil:
        status = cublasdxSetOperatorInt64(<cublasdxDescriptor>handle, <_CublasdxOperatorType>op, value)
    check_status(status)


cpdef cublasdx_set_operator_int64s(long long int handle, int op, size_t count, array):
    """Set an operator on a cuBLASDx descriptor to an integer array.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        op (CublasdxOperatorType): An option to set the descriptor to.
        count (size_t): The size of the operator array, as indicated by the cublasdxOperatorType_t documentation.
        array (object): A pointer to an array containing the operator's value. Must point to at least ``count`` elements. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cublasdxSetOperatorInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cublasdxSetOperatorInt64s(<cublasdxDescriptor>handle, <_CublasdxOperatorType>op, count, <const long long int*>(_array_.data()))
    check_status(status)


cpdef long long int cublasdx_bind_tensor(long long int handle, int tensor_type) except? 0:
    """Binds (aka create) a tensor handle to a tensor The tensor is bound to the cuBLASDx descriptor and will be freed when the cuBLASDx descriptor is destroyed.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        tensor_type (CublasdxTensorType): The tensor type to bind to the handle.

    Returns:
        long long int: A valid tensor handle.

    .. seealso:: `cublasdxBindTensor`
    """
    cdef cublasdxTensor tensor
    with nogil:
        status = cublasdxBindTensor(<cublasdxDescriptor>handle, <_CublasdxTensorType>tensor_type, &tensor)
    check_status(status)
    return <long long int>tensor


cpdef cublasdx_set_tensor_option_int64(long long int tensor, int option, long long int value):
    """Set an option on a tensor. This must be called before the tensor is finalized.

    Args:
        tensor (long long int): A cuBLASDx tensor, output of cublasdxBindTensor.
        option (CublasdxTensorOption): The option to set on the tensor.
        value (long long int): A value for the option.

    .. seealso:: `cublasdxSetTensorOptionInt64`
    """
    with nogil:
        status = cublasdxSetTensorOptionInt64(<cublasdxTensor>tensor, <_CublasdxTensorOption>option, value)
    check_status(status)


cpdef cublasdx_finalize_tensors(long long int handle, size_t count, array):
    """Finalize the tensors. This is required before traits can be queried.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        count (size_t): The number of tensors to finalized.
        array (object): The array of tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxTensor``.


    .. seealso:: `cublasdxFinalizeTensors`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cublasdxFinalizeTensors(<cublasdxDescriptor>handle, count, <const cublasdxTensor*>(_array_.data()))
    check_status(status)


cpdef long long int cublasdx_get_tensor_trait_int64(long long int tensor, int trait) except? 0:
    """Query an integer trait value from a finalized tensor.

    Args:
        tensor (long long int): A finalized tensor handle, output of cublasdxBindTensor.
        trait (CublasdxTensorTrait): The trait to query.

    Returns:
        long long int: The trait value.

    .. seealso:: `cublasdxGetTensorTraitInt64`
    """
    cdef long long int value
    with nogil:
        status = cublasdxGetTensorTraitInt64(<cublasdxTensor>tensor, <_CublasdxTensorTrait>trait, &value)
    check_status(status)
    return value


cpdef size_t cublasdx_get_tensor_trait_str_size(long long int tensor, int trait) except? 0:
    """Query an C-string trait's size from a finalized tensor.

    Args:
        tensor (long long int): A finalized tensor handle, output of cublasdxBindTensor.
        trait (CublasdxTensorTrait): The trait to query.

    Returns:
        size_t: The C-string size (including the ``\0``).

    .. seealso:: `cublasdxGetTensorTraitStrSize`
    """
    cdef size_t size
    with nogil:
        status = cublasdxGetTensorTraitStrSize(<cublasdxTensor>tensor, <_CublasdxTensorTrait>trait, &size)
    check_status(status)
    return size


cpdef cublasdx_get_tensor_trait_str(long long int tensor, int trait, size_t size, value):
    """Query a C-string trait value from a finalized tensor.

    Args:
        tensor (long long int): A finalized tensor handle, output of cublasdxBindTensor.
        trait (CublasdxTensorTrait): The trait to query.
        size (size_t): The C-string size, as returned by cublasdxGetTensorTraitStrSize.
        value (bytes): The C-string trait value.

    .. seealso:: `cublasdxGetTensorTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        status = cublasdxGetTensorTraitStr(<cublasdxTensor>tensor, <_CublasdxTensorTrait>trait, size, <char*>_value_)
    check_status(status)


cpdef long long int cublasdx_bind_device_function(long long int handle, int device_function_type, size_t count, array) except? 0:
    """Binds (aka create) a device function from a set of tensor.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        device_function_type (CublasdxDeviceFunctionType): The device function to create.
        count (size_t): The number of input & output tensors to the device function.
        array (object): The array of input & output tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxTensor``.


    Returns:
        long long int: The device function.

    .. seealso:: `cublasdxBindDeviceFunction`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    cdef cublasdxDeviceFunction device_function
    with nogil:
        status = cublasdxBindDeviceFunction(<cublasdxDescriptor>handle, <_CublasdxDeviceFunctionType>device_function_type, count, <const cublasdxTensor*>(_array_.data()), &device_function)
    check_status(status)
    return <long long int>device_function


cpdef cublasdx_finalize_device_functions(long long int code, size_t count, array):
    """Finalize (aka codegen) a set of device function into a code handle.

    Args:
        code (long long int): A code handle, output from commondxCreateCode.
        count (size_t): The number of device functions to codegen.
        array (object): The array of device functions. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxDeviceFunction``.


    .. seealso:: `cublasdxFinalizeDeviceFunctions`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cublasdxFinalizeDeviceFunctions(<commondxCode>code, count, <const cublasdxDeviceFunction*>(_array_.data()))
    check_status(status)


cpdef size_t cublasdx_get_device_function_trait_str_size(long long int device_function, int trait) except? 0:
    """Query a device function C-string trait value size.

    Args:
        device_function (long long int): A device function handle, output from cublasdxFinalizeDeviceFunctions.
        trait (CublasdxDeviceFunctionTrait): The trait to query the device function.

    Returns:
        size_t: The size of the trait value C-string, including the ``\0``.

    .. seealso:: `cublasdxGetDeviceFunctionTraitStrSize`
    """
    cdef size_t size
    with nogil:
        status = cublasdxGetDeviceFunctionTraitStrSize(<cublasdxDeviceFunction>device_function, <_CublasdxDeviceFunctionTrait>trait, &size)
    check_status(status)
    return size


cpdef cublasdx_get_device_function_trait_str(long long int device_function, int trait, size_t size, value):
    """Query a device function C-string trait value.

    Args:
        device_function (long long int): A device function handle, output from cublasdxFinalizeDeviceFunctions.
        trait (CublasdxDeviceFunctionTrait): The trait to query the device function.
        size (size_t): The size of the trait value C-string as returned by cublasdxGetDeviceFunctionTraitStrSize.
        value (bytes): The trait value as a C-string. Must point to at least ``size`` bytes.

    .. seealso:: `cublasdxGetDeviceFunctionTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        status = cublasdxGetDeviceFunctionTraitStr(<cublasdxDeviceFunction>device_function, <_CublasdxDeviceFunctionTrait>trait, size, <char*>_value_)
    check_status(status)


cpdef size_t cublasdx_get_ltoir_size(long long int handle) except? 0:
    """Returns the LTOIR size, in bytes.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.

    Returns:
        size_t: The size of the LTOIR.

    .. seealso:: `cublasdxGetLTOIRSize`
    """
    cdef size_t lto_size
    with nogil:
        status = cublasdxGetLTOIRSize(<cublasdxDescriptor>handle, &lto_size)
    check_status(status)
    return lto_size


cpdef cublasdx_get_ltoir(long long int handle, size_t size, lto):
    """Returns the LTOIR.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        size (size_t): The size, in bytes, of the LTOIR, as returned by cublasdxGetLTOIRSize.
        lto (bytes): A pointer to at least ``size`` bytes containing the LTOIR.

    .. seealso:: `cublasdxGetLTOIR`
    """
    cdef void* _lto_ = get_buffer_pointer(lto, size, readonly=False)
    with nogil:
        status = cublasdxGetLTOIR(<cublasdxDescriptor>handle, size, <void*>_lto_)
    check_status(status)


cpdef size_t cublasdx_get_trait_str_size(long long int handle, int trait) except? 0:
    """Returns the size of a C-string trait.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        trait (CublasdxTraitType): The trait to query the size of.

    Returns:
        size_t: The size of the C-string value, including the ``\0``.

    .. seealso:: `cublasdxGetTraitStrSize`
    """
    cdef size_t size
    with nogil:
        status = cublasdxGetTraitStrSize(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, &size)
    check_status(status)
    return size


cpdef cublasdx_get_trait_str(long long int handle, int trait, size_t size, value):
    """Returns a C-string trait's value.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        trait (CublasdxTraitType): The trait to query on the descriptor.
        size (size_t): The size of the C-string (including the ``\0``).
        value (bytes): The C-string trait value. Must point to at least ``size`` bytes.

    .. seealso:: `cublasdxGetTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        status = cublasdxGetTraitStr(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, size, <char*>_value_)
    check_status(status)


cpdef long long int cublasdx_get_trait_int64(long long int handle, int trait) except? 0:
    """Returns an integer trait's value.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        trait (CublasdxTraitType): A trait to query the handle for.

    Returns:
        long long int: The trait value.

    .. seealso:: `cublasdxGetTraitInt64`
    """
    cdef long long int value
    with nogil:
        status = cublasdxGetTraitInt64(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, &value)
    check_status(status)
    return value


cpdef cublasdx_get_trait_int64s(long long int handle, int trait, size_t count, array):
    """Returns an array trait's value.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        trait (CublasdxTraitType): A trait to query handle for.
        count (size_t): The number of elements in the trait array, as indicated in the cublasdxTraitType_t documentation.
        array (object): A pointer to at least count integers. As output, an array filled with the trait value. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cublasdxGetTraitInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cublasdxGetTraitInt64s(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, count, <long long int*>(_array_.data()))
    check_status(status)


cpdef str cublasdx_operator_type_to_str(int op):
    """Convert an operator enum to a human readable C-string.

    Args:
        op (CublasdxOperatorType): The operator enum to convert.

    .. seealso:: `cublasdxOperatorTypeToStr`
    """
    cdef bytes _output_
    _output_ = cublasdxOperatorTypeToStr(<_CublasdxOperatorType>op)
    return _output_.decode()


cpdef str cublasdx_trait_type_to_str(int trait):
    """Convert a trait enum to a human readable C-string.

    Args:
        trait (CublasdxTraitType): The trait enum to convert.

    .. seealso:: `cublasdxTraitTypeToStr`
    """
    cdef bytes _output_
    _output_ = cublasdxTraitTypeToStr(<_CublasdxTraitType>trait)
    return _output_.decode()


cpdef cublasdx_finalize_code(long long int code, long long int handle):
    """Fill an instance of commondxCode with the code from the cuBLASDx descriptor.

    Args:
        code (long long int): A commondxCode code.
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.

    .. seealso:: `cublasdxFinalizeCode`
    """
    with nogil:
        status = cublasdxFinalizeCode(<commondxCode>code, <cublasdxDescriptor>handle)
    check_status(status)


cpdef cublasdx_destroy_descriptor(long long int handle):
    """Destroy a cuBLASDx descriptor.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.

    .. seealso:: `cublasdxDestroyDescriptor`
    """
    with nogil:
        status = cublasdxDestroyDescriptor(<cublasdxDescriptor>handle)
    check_status(status)


cpdef long long int cufftdx_create_descriptor() except? 0:
    """Creates a cuFFTDx descriptor.

    Returns:
        long long int: A pointer to a cuFFTDx descriptor handle, and as output, a valid initialized descriptor.

    .. seealso:: `cufftdxCreateDescriptor`
    """
    cdef cufftdxDescriptor handle
    with nogil:
        status = cufftdxCreateDescriptor(&handle)
    check_status(status)
    return <long long int>handle


cpdef cufftdx_set_option_str(long long int handle, int opt, value):
    """Set a C-string option on a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        opt (CommondxOption): The option to set.
        value (str): The C-string to set the option to.

    .. seealso:: `cufftdxSetOptionStr`
    """
    if not isinstance(value, str):
        raise TypeError("value must be a Python str")
    cdef bytes _temp_value_ = (<str>value).encode()
    cdef char* _value_ = _temp_value_
    with nogil:
        status = cufftdxSetOptionStr(<cufftdxDescriptor>handle, <_CommondxOption>opt, <const char*>_value_)
    check_status(status)


cpdef size_t cufftdx_get_knob_int64size(long long int handle, size_t num_knobs, knobs_ptr) except? 0:
    """Returns the number of knobs for a set of knobs.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        num_knobs (size_t): The number of knobs.
        knobs_ptr (object): An array of num_knobs knobs. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_CufftdxKnobType``.


    Returns:
        size_t: The number of distinct sets of knobs.

    .. seealso:: `cufftdxGetKnobInt64Size`
    """
    cdef nullable_unique_ptr[ vector[int] ] _knobs_ptr_
    get_resource_ptr[int](_knobs_ptr_, knobs_ptr, <int*>NULL)
    cdef size_t size
    with nogil:
        status = cufftdxGetKnobInt64Size(<cufftdxDescriptor>handle, num_knobs, <_CufftdxKnobType*>(_knobs_ptr_.data()), &size)
    check_status(status)
    return size


cpdef cufftdx_get_knob_int64s(long long int handle, size_t num_knobs, knobs_ptr, size_t size, intptr_t values):
    """Returns the knobs values for a set of knobs.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        num_knobs (size_t): The number of knobs.
        knobs_ptr (object): A pointer to an array of num_knobs knobs. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_CufftdxKnobType``.

        size (size_t): The number of knobs.
        values (intptr_t): The knob values. Must be a pointer to an array of at least size knobs values (integer).

    .. seealso:: `cufftdxGetKnobInt64s`
    """
    cdef nullable_unique_ptr[ vector[int] ] _knobs_ptr_
    get_resource_ptr[int](_knobs_ptr_, knobs_ptr, <int*>NULL)
    with nogil:
        status = cufftdxGetKnobInt64s(<cufftdxDescriptor>handle, num_knobs, <_CufftdxKnobType*>(_knobs_ptr_.data()), size, <long long int*>values)
    check_status(status)


cpdef cufftdx_set_operator_int64(long long int handle, int op, long long int value):
    """Set an integer operator to a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        op (CufftdxOperatorType): The operator to set the descriptor to.
        value (long long int): The value to set the operator to.

    .. seealso:: `cufftdxSetOperatorInt64`
    """
    with nogil:
        status = cufftdxSetOperatorInt64(<cufftdxDescriptor>handle, <_CufftdxOperatorType>op, value)
    check_status(status)


cpdef cufftdx_set_operator_int64s(long long int handle, int op, size_t count, array):
    """Set an array operator to a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        op (CufftdxOperatorType): The operator to set the descriptor to.
        count (size_t): The array size.
        array (object): A pointer to at least count integers, the arrat to set the descriptor to. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cufftdxSetOperatorInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cufftdxSetOperatorInt64s(<cufftdxDescriptor>handle, <_CufftdxOperatorType>op, count, <const long long int*>(_array_.data()))
    check_status(status)


cpdef size_t cufftdx_get_ltoir_size(long long int handle) except? 0:
    """Get the LTOIR's size from a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.

    Returns:
        size_t: The size (in bytes) of the LTOIR.

    .. seealso:: `cufftdxGetLTOIRSize`
    """
    cdef size_t lto_size
    with nogil:
        status = cufftdxGetLTOIRSize(<cufftdxDescriptor>handle, &lto_size)
    check_status(status)
    return lto_size


cpdef cufftdx_get_ltoir(long long int handle, size_t size, lto):
    """Get the LTOIR from a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        size (size_t): The LTOIR size, in bytes.
        lto (bytes): The LTOIR code.

    .. seealso:: `cufftdxGetLTOIR`
    """
    cdef void* _lto_ = get_buffer_pointer(lto, size, readonly=False)
    with nogil:
        status = cufftdxGetLTOIR(<cufftdxDescriptor>handle, size, <void*>_lto_)
    check_status(status)


cpdef size_t cufftdx_get_trait_str_size(long long int handle, int trait) except? 0:
    """Returns a C-string trait's value size.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        trait (CufftdxTraitType): The trait to query the descriptor for.

    Returns:
        size_t: The C-string length (including ``\0``).

    .. seealso:: `cufftdxGetTraitStrSize`
    """
    cdef size_t size
    with nogil:
        status = cufftdxGetTraitStrSize(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, &size)
    check_status(status)
    return size


cpdef cufftdx_get_trait_str(long long int handle, int trait, size_t size, value):
    """Returns a C-string trait value.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        trait (CufftdxTraitType): The trait to query the descriptor for.
        size (size_t): The C-string size (including the ``\0``).
        value (bytes): As output, the C-string trait value.

    .. seealso:: `cufftdxGetTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        status = cufftdxGetTraitStr(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, size, <char*>_value_)
    check_status(status)


cpdef long long int cufftdx_get_trait_int64(long long int handle, int trait) except? 0:
    """Returns an integer trait.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        trait (CufftdxTraitType): The trait to query the descriptor for.

    Returns:
        long long int: The trait integer value.

    .. seealso:: `cufftdxGetTraitInt64`
    """
    cdef long long int value
    with nogil:
        status = cufftdxGetTraitInt64(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, &value)
    check_status(status)
    return value


cpdef cufftdx_get_trait_int64s(long long int handle, int trait, size_t count, array):
    """Returns an array of integers trait.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        trait (CufftdxTraitType): The trait to query the descriptor for.
        count (size_t): The array size.
        array (object): The trait array. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cufftdxGetTraitInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cufftdxGetTraitInt64s(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, count, <long long int*>(_array_.data()))
    check_status(status)


cpdef int cufftdx_get_trait_commondx_data_type(long long int handle, int trait) except? -1:
    """Return a commondxValueType trait.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        trait (CufftdxTraitType): The trait to query the descriptor for, of value commondxValueType.

    Returns:
        int: As output, the valuetype for the given input trait.

    .. seealso:: `cufftdxGetTraitCommondxDataType`
    """
    cdef _CommondxValueType value
    with nogil:
        status = cufftdxGetTraitCommondxDataType(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, &value)
    check_status(status)
    return <int>value


cpdef cufftdx_finalize_code(long long int code, long long int handle):
    """Generate code from the cuFFTDx descriptor and stores it in code.

    Args:
        code (long long int): A commondxCode instance, output of commondxCreateCode.
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.

    .. seealso:: `cufftdxFinalizeCode`
    """
    with nogil:
        status = cufftdxFinalizeCode(<commondxCode>code, <cufftdxDescriptor>handle)
    check_status(status)


cpdef cufftdx_destroy_descriptor(long long int handle):
    """Destroys a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.

    .. seealso:: `cufftdxDestroyDescriptor`
    """
    with nogil:
        status = cufftdxDestroyDescriptor(<cufftdxDescriptor>handle)
    check_status(status)


cpdef str cufftdx_operator_type_to_str(int op):
    """Convert a cufftdxOperatorType instance to a human readable C-string.

    Args:
        op (CufftdxOperatorType): A cufftdxOperatorType instance.

    .. seealso:: `cufftdxOperatorTypeToStr`
    """
    cdef bytes _output_
    _output_ = cufftdxOperatorTypeToStr(<_CufftdxOperatorType>op)
    return _output_.decode()


cpdef str cufftdx_trait_type_to_str(int op):
    """Convert a cufftdxTraitType instance to a human readable C-string.

    Args:
        op (CufftdxTraitType): A cufftdxTraitType instance.

    .. seealso:: `cufftdxTraitTypeToStr`
    """
    cdef bytes _output_
    _output_ = cufftdxTraitTypeToStr(<_CufftdxTraitType>op)
    return _output_.decode()


cpdef long long int cusolverdx_create_descriptor() except? 0:
    """Creates a cuSOLVERDx descriptor.

    Returns:
        long long int: A pointer to a descriptor handle. As output, an initialized cuSOLVERDx descriptor.

    .. seealso:: `cusolverdxCreateDescriptor`
    """
    cdef cusolverdxDescriptor handle
    with nogil:
        status = cusolverdxCreateDescriptor(&handle)
    check_status(status)
    return <long long int>handle


cpdef cusolverdx_set_option_str(long long int handle, int opt, value):
    """Sets a C-string option on a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        opt (CommondxOption): The option to set.
        value (str): The value for the option.

    .. seealso:: `cusolverdxSetOptionStr`
    """
    if not isinstance(value, str):
        raise TypeError("value must be a Python str")
    cdef bytes _temp_value_ = (<str>value).encode()
    cdef char* _value_ = _temp_value_
    with nogil:
        status = cusolverdxSetOptionStr(<cusolverdxDescriptor>handle, <_CommondxOption>opt, <const char*>_value_)
    check_status(status)


cpdef cusolverdx_set_operator_int64(long long int handle, int op, long long int value):
    """Sets an integer operator on a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        op (CusolverdxOperatorType): The operator to set.
        value (long long int): A value for the operator.

    .. seealso:: `cusolverdxSetOperatorInt64`
    """
    with nogil:
        status = cusolverdxSetOperatorInt64(<cusolverdxDescriptor>handle, <_CusolverdxOperatorType>op, value)
    check_status(status)


cpdef cusolverdx_set_operator_int64s(long long int handle, int op, size_t count, array):
    """Sets a integer array operator on a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        op (CusolverdxOperatorType): The operator to set.
        count (size_t): The number of entries in the array value, as indicated in the cusolverdxOperatorType_t documentation.
        array (object): A pointer to at least count integers, the array operator to set. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cusolverdxSetOperatorInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        status = cusolverdxSetOperatorInt64s(<cusolverdxDescriptor>handle, <_CusolverdxOperatorType>op, count, <const long long int*>(_array_.data()))
    check_status(status)


cpdef size_t cusolverdx_get_ltoir_size(long long int handle) except? 0:
    """Extract the size of the LTOIR for a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.

    Returns:
        size_t: As output, the size of the LTOIR.

    .. seealso:: `cusolverdxGetLTOIRSize`
    """
    cdef size_t lto_size
    with nogil:
        status = cusolverdxGetLTOIRSize(<cusolverdxDescriptor>handle, &lto_size)
    check_status(status)
    return lto_size


cpdef cusolverdx_get_ltoir(long long int handle, size_t size, lto):
    """Extract the LTOIR from a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        size (size_t): The LTOIR size, output of cusolverdxGetLTOIRSize.
        lto (bytes): The buffer contains the LTOIR.

    .. seealso:: `cusolverdxGetLTOIR`
    """
    cdef void* _lto_ = get_buffer_pointer(lto, size, readonly=False)
    with nogil:
        status = cusolverdxGetLTOIR(<cusolverdxDescriptor>handle, size, <void*>_lto_)
    check_status(status)


cpdef size_t cusolverdx_get_universal_fatbin_size(long long int handle) except? 0:
    """Returns the size of the universal fatbin for cuSOLVERDx.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.

    Returns:
        size_t: The size of the fatbin, in bytes.

    .. seealso:: `cusolverdxGetUniversalFATBINSize`
    """
    cdef size_t fatbin_size
    with nogil:
        status = cusolverdxGetUniversalFATBINSize(<cusolverdxDescriptor>handle, &fatbin_size)
    check_status(status)
    return fatbin_size


cpdef cusolverdx_get_universal_fatbin(long long int handle, size_t fatbin_size, fatbin):
    """Returns a universal fatbin for cuSOLVERDx.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        fatbin_size (size_t): The size of the fatbin, output from cusolverdxGetUniversalFATBINSize.
        fatbin (bytes): The universal fatbin. Must pointer to at least ``fatbin_size`` bytes.

    .. seealso:: `cusolverdxGetUniversalFATBIN`
    """
    cdef void* _fatbin_ = get_buffer_pointer(fatbin, fatbin_size, readonly=False)
    with nogil:
        status = cusolverdxGetUniversalFATBIN(<cusolverdxDescriptor>handle, fatbin_size, <void*>_fatbin_)
    check_status(status)


cpdef size_t cusolverdx_get_trait_str_size(long long int handle, int trait) except? 0:
    """Returns the size of a C-string trait value.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        trait (CusolverdxTraitType): A trait to query the descriptor for.

    Returns:
        size_t: The size of the C-string value for the trait (including the ``\0``).

    .. seealso:: `cusolverdxGetTraitStrSize`
    """
    cdef size_t size
    with nogil:
        status = cusolverdxGetTraitStrSize(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, &size)
    check_status(status)
    return size


cpdef cusolverdx_get_trait_str(long long int handle, int trait, size_t size, value):
    """Returns a C-string trait value.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        trait (CusolverdxTraitType): A trait to query the descriptor for.
        size (size_t): The size of the C-string, output from cusolverdxGetTraitStrSize.
        value (bytes): The C-string trait value.

    .. seealso:: `cusolverdxGetTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        status = cusolverdxGetTraitStr(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, size, <char*>_value_)
    check_status(status)


cpdef long long int cusolverdx_get_trait_int64(long long int handle, int trait) except? 0:
    """Returns an integer trait value.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        trait (CusolverdxTraitType): A trait to query the descriptor for.

    Returns:
        long long int: The trait value.

    .. seealso:: `cusolverdxGetTraitInt64`
    """
    cdef long long int value
    with nogil:
        status = cusolverdxGetTraitInt64(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, &value)
    check_status(status)
    return value


cpdef cusolverdx_finalize_code(long long int code, long long int handle):
    """Fills a code handle with the descriptor's device function code.

    Args:
        code (long long int): A code handle.
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.

    .. seealso:: `cusolverdxFinalizeCode`
    """
    with nogil:
        status = cusolverdxFinalizeCode(<commondxCode>code, <cusolverdxDescriptor>handle)
    check_status(status)


cpdef cusolverdx_destroy_descriptor(long long int handle):
    """Destroys a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.

    .. seealso:: `cusolverdxDestroyDescriptor`
    """
    with nogil:
        status = cusolverdxDestroyDescriptor(<cusolverdxDescriptor>handle)
    check_status(status)


cpdef str cusolverdx_operator_type_to_str(int op):
    """Converts an operator enum to a human readable C-string.

    Args:
        op (CusolverdxOperatorType): An operator enum.

    .. seealso:: `cusolverdxOperatorTypeToStr`
    """
    cdef bytes _output_
    _output_ = cusolverdxOperatorTypeToStr(<_CusolverdxOperatorType>op)
    return _output_.decode()


cpdef str cusolverdx_trait_type_to_str(int trait):
    """Converts a trait enum to a human readable C-string.

    Args:
        trait (CusolverdxTraitType): A trait enum.

    .. seealso:: `cusolverdxTraitTypeToStr`
    """
    cdef bytes _output_
    _output_ = cusolverdxTraitTypeToStr(<_CusolverdxTraitType>trait)
    return _output_.decode()
