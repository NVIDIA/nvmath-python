# This code was automatically generated across versions from 0.3.1 to 0.3.2, generator version 0.3.1.dev1418+g63712a33a.d20260318. Do not modify it directly.

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
    """
    The set of value types supported by the library.Note that, for complex
    numbers, this combines real and imaginary part, and that all complex
    numbers are overaligned on their size.

    See `commondxValueType`.
    """
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
    """
    The set of status values that can be returned by APIs defined in the
    library.

    See `commondxStatusType`.
    """
    SUCCESS = COMMONDX_SUCCESS
    INVALID_VALUE = COMMONDX_INVALID_VALUE
    INTERNAL_ERROR = COMMONDX_INTERNAL_ERROR
    COMPILATION_ERROR = COMMONDX_COMPILATION_ERROR
    CUFFT_ERROR = COMMONDX_CUFFT_ERROR

class CommondxPrecision(_IntEnum):
    """
    The set of precisions supported by the library.For the actual input and
    output value types, see commondxValueType_t

    See `commondxPrecision`.
    """
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

class CommondxArchModifier(_IntEnum):
    """
    The set of compute feature to target. See COMMONDX_OPTION_TARGET_SM
    .See also https://docs.nvidia.com/cuda/cuda-c-programming-
    guide/index.html#feature-set-compiler-targets

    See `commondxArchModifier`.
    """
    GENERIC = COMMONDX_ARCH_MODIFIER_GENERIC
    ARCH_SPECIFIC = COMMONDX_ARCH_MODIFIER_ARCH_SPECIFIC
    FAMILY_SPECIFIC = COMMONDX_ARCH_MODIFIER_FAMILY_SPECIFIC

class CommondxOption(_IntEnum):
    """
    Options to tweak code generation.

    See `commondxOption`.
    """
    SYMBOL_NAME = COMMONDX_OPTION_SYMBOL_NAME
    TARGET_SM = COMMONDX_OPTION_TARGET_SM
    CODE_CONTAINER = COMMONDX_OPTION_CODE_CONTAINER
    CODE_ISA = COMMONDX_OPTION_CODE_ISA
    EXTRA_NVRTC_ARGS = COMMONDX_OPTION_EXTRA_NVRTC_ARGS
    TARGET_CODE = COMMONDX_OPTION_TARGET_CODE

class CommondxExecution(_IntEnum):
    """
    The set of execution modes supported by the library.

    See `commondxExecution`.
    """
    THREAD = COMMONDX_EXECUTION_THREAD
    BLOCK = COMMONDX_EXECUTION_BLOCK

class CommondxCodeContainer(_IntEnum):
    """
    The set of device code container types supported by the library. See
    commondxOption_t::COMMONDX_OPTION_CODE_CONTAINER.

    See `commondxCodeContainer`.
    """
    LTOIR = COMMONDX_CODE_CONTAINER_LTOIR
    FATBIN = COMMONDX_CODE_CONTAINER_FATBIN
    PTX = COMMONDX_CODE_CONTAINER_PTX

class CublasdxApi(_IntEnum):
    """
    Type of cublasDx API.Handling problems with default or custom/dynamic
    leading dimensions. Check cublasdx::LeadingDimension operator
    documentation for more details (https://docs.nvidia.com/cuda/cublasdx/a
    pi/operators.html#leadingdimension-operator)

    See `cublasdxApi`.
    """
    SMEM = CUBLASDX_API_SMEM
    SMEM_DYNAMIC_LD = CUBLASDX_API_SMEM_DYNAMIC_LD
    TENSORS = CUBLASDX_API_TENSORS

class CublasdxDevicePipelineType(_IntEnum):
    """
    Type of device pipeline.

    See `cublasdxDevicePipelineType`.
    """
    SUGGESTED = CUBLASDX_DEVICE_PIPELINE_SUGGESTED

class CublasdxTilePipelineType(_IntEnum):
    """
    Type of tile pipeline.

    See `cublasdxTilePipelineType`.
    """
    PIPELINE_DEFAULT = CUBLASDX_TILE_PIPELINE_DEFAULT

class CublasdxBlockSizeStrategy(_IntEnum):
    """
    Type of block size strategy used in device pipeline and tile pipeline
    creation. This enum affects register usage in pipelining GEMM kernels
    and should be considered for different architectures.

    See `cublasdxBlockSizeStrategy`.
    """
    HEURISTIC = CUBLASDX_BLOCK_SIZE_STRATEGY_HEURISTIC
    FIXED = CUBLASDX_BLOCK_SIZE_STRATEGY_FIXED

class CublasdxType(_IntEnum):
    """
    Type of computation data.Check cubladx::Type operator documentation for
    more details
    (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#type-
    operator)

    See `cublasdxType`.
    """
    REAL = CUBLASDX_TYPE_REAL
    COMPLEX = CUBLASDX_TYPE_COMPLEX

class CublasdxMemorySpace(_IntEnum):
    """
    Memory space.

    See `cublasdxMemorySpace`.
    """
    RMEM = CUBLASDX_MEMORY_SPACE_RMEM
    SMEM = CUBLASDX_MEMORY_SPACE_SMEM
    GMEM = CUBLASDX_MEMORY_SPACE_GMEM
    ANY = CUBLASDX_MEMORY_SPACE_ANY

class CublasdxTransposeMode(_IntEnum):
    """
    Tensor transpose mode.The transpose mode depends on
    cubladx::TransposeMode operator which is deprecated since cublasDx
    0.2.0 and might be removed in future versions of mathDx libraries
    Check cubladx::TransposeMode operator documentation for more details (h
    ttps://docs.nvidia.com/cuda/cublasdx/api/operators.html#transposemode-
    operator)

    See `cublasdxTransposeMode`.
    """
    NON_TRANSPOSED = CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED
    TRANSPOSED = CUBLASDX_TRANSPOSE_MODE_TRANSPOSED
    CONJ_TRANSPOSED = CUBLASDX_TRANSPOSE_MODE_CONJ_TRANSPOSED

class CublasdxArrangement(_IntEnum):
    """
    Data arrangement mode.Defines data arrangements in tensors' taking part
    in the calculation.  Check cubladx::TransposeMode operator
    documentation for more details
    (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#arrangement-
    operator)

    See `cublasdxArrangement`.
    """
    COL_MAJOR = CUBLASDX_ARRANGEMENT_COL_MAJOR
    ROW_MAJOR = CUBLASDX_ARRANGEMENT_ROW_MAJOR

class CublasdxFunction(_IntEnum):
    """
    BLAS function.Sets the BLAS function that will be executed.  Check
    cubladx::Function operator documentation for more details
    (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#function-
    operator)

    See `cublasdxFunction`.
    """
    MM = CUBLASDX_FUNCTION_MM

class CublasdxOperatorType(_IntEnum):
    """
    cublasDx operatorsThe set of supported cublasDx operators.  Check
    cublaDx description operator documentation for more details
    (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#function-
    operator)  Check cublasDx execution operator documentation for more
    details
    (https://docs.nvidia.com/cuda/cublasdx/api/operators.html#execution-
    operators)

    See `cublasdxOperatorType`.
    """
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
    ENABLE_INPUT_STREAMING = CUBLASDX_OPERATOR_ENABLE_INPUT_STREAMING
    WITH_PIPELINE = CUBLASDX_OPERATOR_WITH_PIPELINE

class CublasdxTraitType(_IntEnum):
    """
    cublasDx traitsThe set of supported types of traits that can be
    accessed from finalized sources that use cublasDx.  Check cublasDx
    Execution Block Traits documentation for more details
    (https://docs.nvidia.com/cuda/cublasdx/api/traits.html#block-traits)

    See `cublasdxTraitType`.
    """
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
    """
    cuBLASDx desired tensor typeTensor types are opaque (layout is
    unspecified), non-owning, and defined by  - Memory space (global,
    shared or register memory)  - Size & alignment (in bytes)  Tensor's
    representation in-memory and in-device depends on their memory space.
    Shared & register tensors are defined as  **View CUDA Toolkit
    Documentation for a C++ code example**  where `ptr` points to the
    associated data.  Global memory tensors have an associated runtime
    leading dimension (64b signed integer), and their representation is
    **View CUDA Toolkit Documentation for a C++ code example**  where `ptr`
    points to the associated data and `strides[0]` is the leading
    dimension.  In either case, `ptr` must point to some storage (with
    appropriate size and alignment, see below) and is not owning. The user
    is expected to keep memory allocated beyond any use of the tensor.
    `strides[1]` should be a signed, 64bit integer (`long long`) equal to
    the leading dimension of the global memory tensor. The leading
    dimension is the number of `elements` between two successive rows or
    columns (not bytes), depending on the context.  All tensor APIs take
    their argument by value (not by pointer) and expect the struct to be
    passed as-is on the stack.  Each opaque tensor type is uniquely
    identified by a unique ID and name, see cublasdxTensorTrait_t .

    See `cublasdxTensorType`.
    """
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
    SUGGESTED_ACCUMULATOR_C = CUBLASDX_TENSOR_SUGGESTED_ACCUMULATOR_C
    RMEM_C = CUBLASDX_TENSOR_RMEM_C
    ACCUMULATOR_C = CUBLASDX_TENSOR_ACCUMULATOR_C

class CublasdxTensorOption(_IntEnum):
    """
    Tensor options.

    See `cublasdxTensorOption`.
    """
    ALIGNMENT_BYTES = CUBLASDX_TENSOR_OPTION_ALIGNMENT_BYTES

class CublasdxTensorTrait(_IntEnum):
    """
    Tensor traits, used to query informations.

    See `cublasdxTensorTrait`.
    """
    STORAGE_BYTES = CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES
    ALIGNMENT_BYTES = CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES
    UID = CUBLASDX_TENSOR_TRAIT_UID
    OPAQUE_NAME = CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME
    LOGICAL_SIZE = CUBLASDX_TENSOR_TRAIT_LOGICAL_SIZE
    MEMORY_SPACE = CUBLASDX_TENSOR_TRAIT_MEMORY_SPACE

class CublasdxPipelineTrait(_IntEnum):
    """
    Pipeline traits, used to query informations.

    See `cublasdxPipelineTrait`.
    """
    STORAGE_BYTES = CUBLASDX_PIPELINE_TRAIT_STORAGE_BYTES
    STORAGE_ALIGNMENT_BYTES = CUBLASDX_PIPELINE_TRAIT_STORAGE_ALIGNMENT_BYTES
    BUFFER_SIZE = CUBLASDX_PIPELINE_TRAIT_BUFFER_SIZE
    BUFFER_ALIGNMENT_BYTES = CUBLASDX_PIPELINE_TRAIT_BUFFER_ALIGNMENT_BYTES
    OPAQUE_NAME = CUBLASDX_PIPELINE_TRAIT_OPAQUE_NAME
    BLOCK_DIM = CUBLASDX_PIPELINE_TRAIT_BLOCK_DIM

class CublasdxDeviceFunctionTrait(_IntEnum):
    """
    Device function traits, used to query informations.

    See `cublasdxDeviceFunctionTrait`.
    """
    SYMBOL = CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL

class CublasdxDeviceFunctionOption(_IntEnum):
    """
    Device function options.

    See `cublasdxDeviceFunctionOption`.
    """
    SYMBOL_NAME = CUBLASDX_DEVICE_FUNCTION_OPTION_SYMBOL_NAME
    COPY_ALIGNMENT = CUBLASDX_DEVICE_FUNCTION_OPTION_COPY_ALIGNMENT
    CALLBACK = CUBLASDX_DEVICE_FUNCTION_OPTION_CALLBACK
    NUM_THREADS = CUBLASDX_DEVICE_FUNCTION_OPTION_NUM_THREADS

class CublasdxDeviceFunctionType(_IntEnum):
    """
    Device functions supported by the library.

    See `cublasdxDeviceFunctionType`.
    """
    EXECUTE = CUBLASDX_DEVICE_FUNCTION_EXECUTE
    COPY = CUBLASDX_DEVICE_FUNCTION_COPY
    COPY_WAIT = CUBLASDX_DEVICE_FUNCTION_COPY_WAIT
    CLEAR = CUBLASDX_DEVICE_FUNCTION_CLEAR
    AXPBY = CUBLASDX_DEVICE_FUNCTION_AXPBY
    MAP_IDX2CRD = CUBLASDX_DEVICE_FUNCTION_MAP_IDX2CRD
    MAP_IDX2CRD_PARTITIONER = CUBLASDX_DEVICE_FUNCTION_MAP_IDX2CRD_PARTITIONER
    MAP_CRD2IDX = CUBLASDX_DEVICE_FUNCTION_MAP_CRD2IDX
    IS_THREAD_ACTIVE = CUBLASDX_DEVICE_FUNCTION_IS_THREAD_ACTIVE
    IS_PREDICATED = CUBLASDX_DEVICE_FUNCTION_IS_PREDICATED
    IS_INDEX_IN_BOUNDS = CUBLASDX_DEVICE_FUNCTION_IS_INDEX_IN_BOUNDS
    CREATE = CUBLASDX_DEVICE_FUNCTION_CREATE
    DESTROY = CUBLASDX_DEVICE_FUNCTION_DESTROY
    RESET = CUBLASDX_DEVICE_FUNCTION_RESET
    EPILOGUE = CUBLASDX_DEVICE_FUNCTION_EPILOGUE

class CufftdxApi(_IntEnum):
    """
    Type of cufftDx API.Handling problems with input in register or in
    shared memory buffers.  Check cufftdx::execute method documentation for
    more details
    (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#block-execute-
    method)

    See `cufftdxApi`.
    """
    LMEM = CUFFTDX_API_LMEM
    SMEM = CUFFTDX_API_SMEM

class CufftdxType(_IntEnum):
    """
    Type of computation data.Check cufftdx::Type operator documentation for
    more details
    (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#block-execute-
    method)

    See `cufftdxType`.
    """
    C2C = CUFFTDX_TYPE_C2C
    R2C = CUFFTDX_TYPE_R2C
    C2R = CUFFTDX_TYPE_C2R

class CufftdxDirection(_IntEnum):
    """
    FFT direction.Check cufftdx::Direction operator documentation for more
    details
    (https://docs.nvidia.com/cuda/cufftdx/api/operators.html#direction-
    operator)

    See `cufftdxDirection`.
    """
    FORWARD = CUFFTDX_DIRECTION_FORWARD
    INVERSE = CUFFTDX_DIRECTION_INVERSE

class CufftdxComplexLayout(_IntEnum):
    """
    Complex data layout.Check cufftdx layout documentation for more details
    (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#complex-element-
    layouts)

    See `cufftdxComplexLayout`.
    """
    NATURAL = CUFFTDX_COMPLEX_LAYOUT_NATURAL
    PACKED = CUFFTDX_COMPLEX_LAYOUT_PACKED
    FULL = CUFFTDX_COMPLEX_LAYOUT_FULL

class CufftdxRealMode(_IntEnum):
    """
    Real data mode.Check cufftdx real data mode documentation for more
    details (https://docs.nvidia.com/cuda/cufftdx/api/methods.html#real-
    element-layouts)

    See `cufftdxRealMode`.
    """
    NORMAL = CUFFTDX_REAL_MODE_NORMAL
    FOLDED = CUFFTDX_REAL_MODE_FOLDED

class CufftdxCodeType(_IntEnum):
    """
    Code type.Check cufftdx code type documentation for more details
    (https://docs.nvidia.com/cuda/1.4.0-ea/cufftdx/api/methods.html#code-
    type)

    See `cufftdxCodeType`.
    """
    PTX = CUFFTDX_CODE_TYPE_PTX
    LTOIR = CUFFTDX_CODE_TYPE_LTOIR

class CufftdxOperatorType(_IntEnum):
    """
    cufftDx operatorsThe set of supported cufftDx operators.  Check cufftDx
    description operators documentation for more details
    (https://docs.nvidia.com/cuda/cufftdx/api/operators.html#description-
    operators)  Check cufftDx execution operators documentation for more
    details
    (https://docs.nvidia.com/cuda/cufftdx/api/operators.html#execution-
    operators)

    See `cufftdxOperatorType`.
    """
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
    """
    cufftDx configuration knobsThe set of supported knobs used for
    accessing cufftDx operator's performance configuration details.  Check
    cufftDx Execution Traits documentation for more details
    (https://docs.nvidia.com/cuda/cufftdx/api/traits.html#execution-traits)

    See `cufftdxKnobType`.
    """
    ELEMENTS_PER_THREAD = CUFFTDX_KNOB_ELEMENTS_PER_THREAD
    FFTS_PER_BLOCK = CUFFTDX_KNOB_FFTS_PER_BLOCK

class CufftdxTraitType(_IntEnum):
    """
    cufftDx traitsThe set of supported types of traits that can be accessed
    from finalized sources that use cufftDx.  Check cufftDx Execution
    Thread Traits documentation for more details
    (https://docs.nvidia.com/cuda/cufftdx/api/traits.html#thread-traits)
    Check cufftDx Execution Block Traits documentation for more details
    (https://docs.nvidia.com/cuda/cufftdx/api/traits.html#block-traits)

    See `cufftdxTraitType`.
    """
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
    """
    Type of cusolverdx API.

    See `cusolverdxApi`.
    """
    SMEM = CUSOLVERDX_API_SMEM
    SMEM_DYNAMIC_LD = CUSOLVERDX_API_SMEM_DYNAMIC_LD

class CusolverdxType(_IntEnum):
    """
    Type of input values.

    See `cusolverdxType`.
    """
    REAL = CUSOLVERDX_TYPE_REAL
    COMPLEX = CUSOLVERDX_TYPE_COMPLEX

class CusolverdxFunction(_IntEnum):
    """
    Type of device function.The function signatures depend on the selected
    cusolverdxApi_t:  - For CUSOLVERDX_API_SMEM: Leading dimensions are
    fixed at compile time.  - For CUSOLVERDX_API_SMEM_DYNAMIC_LD: Leading
    dimensions are passed as pointers to `unsigned` at runtime.  Type
    definitions used in function signatures:  - `value_type`: The value
    type determined by CUSOLVERDX_OPERATOR_TYPE and
    CUSOLVERDX_OPERATOR_PRECISION. For real types: `float` or `double`. For
    complex types: `cuFloatComplex` or `cuDoubleComplex`.  - `status_type`:
    A signed 32b integer (`int`) used to return status information.  All
    functions are `extern "C"` and the symbol name can be queried using
    CUSOLVERDX_TRAIT_SYMBOL_NAME.

    See `cusolverdxFunction`.
    """
    GETRF_NO_PIVOT = CUSOLVERDX_FUNCTION_GETRF_NO_PIVOT
    GETRS_NO_PIVOT = CUSOLVERDX_FUNCTION_GETRS_NO_PIVOT
    POTRF = CUSOLVERDX_FUNCTION_POTRF
    POTRS = CUSOLVERDX_FUNCTION_POTRS
    TRSM = CUSOLVERDX_FUNCTION_TRSM
    GETRF_PARTIAL_PIVOT = CUSOLVERDX_FUNCTION_GETRF_PARTIAL_PIVOT
    GETRS_PARTIAL_PIVOT = CUSOLVERDX_FUNCTION_GETRS_PARTIAL_PIVOT
    GEQRF = CUSOLVERDX_FUNCTION_GEQRF
    UNMQR = CUSOLVERDX_FUNCTION_UNMQR
    GELQF = CUSOLVERDX_FUNCTION_GELQF
    UNMLQ = CUSOLVERDX_FUNCTION_UNMLQ
    POSV = CUSOLVERDX_FUNCTION_POSV
    GESV_NO_PIVOT = CUSOLVERDX_FUNCTION_GESV_NO_PIVOT
    GESV_PARTIAL_PIVOT = CUSOLVERDX_FUNCTION_GESV_PARTIAL_PIVOT
    GELS = CUSOLVERDX_FUNCTION_GELS
    UNGQR = CUSOLVERDX_FUNCTION_UNGQR
    UNGLQ = CUSOLVERDX_FUNCTION_UNGLQ
    GTSV_NO_PIVOT = CUSOLVERDX_FUNCTION_GTSV_NO_PIVOT
    HTEV = CUSOLVERDX_FUNCTION_HTEV
    HEEV = CUSOLVERDX_FUNCTION_HEEV

class CusolverdxArrangement(_IntEnum):
    """
    Data arrangement mode.Defines data arrangements in tensors' taking part
    in the calculation.

    See `cusolverdxArrangement`.
    """
    COL_MAJOR = CUSOLVERDX_ARRANGEMENT_COL_MAJOR
    ROW_MAJOR = CUSOLVERDX_ARRANGEMENT_ROW_MAJOR

class CusolverdxTransposeMode(_IntEnum):
    """
    Transpose mode.Indicates inputs or outputs must be considered
    transposed.

    See `cusolverdxTransposeMode`.
    """
    NON_TRANSPOSED = CUSOLVERDX_TRANSPOSE_MODE_NON_TRANSPOSED
    TRANSPOSED = CUSOLVERDX_TRANSPOSE_MODE_TRANSPOSED
    CONJ_TRANSPOSED = CUSOLVERDX_TRANSPOSE_MODE_CONJ_TRANSPOSED

class CusolverdxFillMode(_IntEnum):
    """
    Tensor fill mode.For symmetric matrix the fill mode can be upper or
    lower triangular

    See `cusolverdxFillMode`.
    """
    UPPER = CUSOLVERDX_FILL_MODE_UPPER
    LOWER = CUSOLVERDX_FILL_MODE_LOWER

class CusolverdxSide(_IntEnum):
    """
    Side operator.Indicates which side an operator is applied from in
    operations such as matrix multiplication.

    See `cusolverdxSide`.
    """
    LEFT = CUSOLVERDX_SIDE_LEFT
    RIGHT = CUSOLVERDX_SIDE_RIGHT

class CusolverdxDiag(_IntEnum):
    """
    Diag operator.Indicates whether the matrix diagonal is unit (all ones)
    or non-unit.

    See `cusolverdxDiag`.
    """
    UNIT = CUSOLVERDX_DIAG_UNIT
    NON_UNIT = CUSOLVERDX_DIAG_NON_UNIT

class CusolverdxOperatorType(_IntEnum):
    """
    Operators.The set of supported cusolverDx operators.

    See `cusolverdxOperatorType`.
    """
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
    TRANSPOSE_MODE = CUSOLVERDX_OPERATOR_TRANSPOSE_MODE
    LEADING_DIMENSION = CUSOLVERDX_OPERATOR_LEADING_DIMENSION
    BATCHES_PER_BLOCK = CUSOLVERDX_OPERATOR_BATCHES_PER_BLOCK
    JOB = CUSOLVERDX_OPERATOR_JOB

class CusolverdxTraitType(_IntEnum):
    """
    Traits.The set of supported types of traits that can be accessed from
    finalized sources that use cusolverdx.

    See `cusolverdxTraitType`.
    """
    SHARED_MEMORY_SIZE = CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE
    SYMBOL_NAME = CUSOLVERDX_TRAIT_SYMBOL_NAME
    BLOCK_DIM = CUSOLVERDX_TRAIT_BLOCK_DIM
    SUGGESTED_BLOCK_DIM = CUSOLVERDX_TRAIT_SUGGESTED_BLOCK_DIM
    SUGGESTED_BATCHES_PER_BLOCK = CUSOLVERDX_TRAIT_SUGGESTED_BATCHES_PER_BLOCK
    ARRANGEMENT = CUSOLVERDX_TRAIT_ARRANGEMENT
    BATCHES_PER_BLOCK = CUSOLVERDX_TRAIT_BATCHES_PER_BLOCK
    TYPE = CUSOLVERDX_TRAIT_TYPE
    LEADING_DIMENSION = CUSOLVERDX_TRAIT_LEADING_DIMENSION
    VALUE_TYPE = CUSOLVERDX_TRAIT_VALUE_TYPE
    WORKSPACE_SIZE = CUSOLVERDX_TRAIT_WORKSPACE_SIZE
    SUGGESTED_LEADING_DIMENSION = CUSOLVERDX_TRAIT_SUGGESTED_LEADING_DIMENSION

class CuranddxDistribution(_IntEnum):
    """
    Distribution type.Indicate the random number generation to use. Can be
    set using CURANDDX_OPERATOR_DISTRIBUTION and further customized using
    CURANDDX_OPERATOR_NORMAL_METHOD. See cuRANDDx distributions in the
    documentation:
    https://docs.nvidia.com/cuda/curanddx/api/methods.html#random-number-
    generation-with-distributions.

    See `curanddxDistribution`.
    """
    UNIFORM_BITS = CURANDDX_DISTRIBUTION_UNIFORM_BITS
    UNIFORM = CURANDDX_DISTRIBUTION_UNIFORM
    NORMAL = CURANDDX_DISTRIBUTION_NORMAL
    LOG_NORMAL = CURANDDX_DISTRIBUTION_LOG_NORMAL
    POISSON = CURANDDX_DISTRIBUTION_POISSON

class CuranddxNormalMethod(_IntEnum):
    """
    Normal distribution method type.Specifies the method to use for normal
    and log-normal distributions. Set using CURANDDX_OPERATOR_NORMAL_METHOD
    . See cuRANDDx normal distribution documentation:
    https://docs.nvidia.com/cuda/curanddx/api/methods.html#normal-
    distribution

    See `curanddxNormalMethod`.
    """
    ICDF = CURANDDX_NORMAL_METHOD_ICDF
    BOX_MULLER = CURANDDX_NORMAL_METHOD_BOX_MULLER

class CuranddxGenerateMethod(_IntEnum):
    """
    Generation method type.Specifies which generation method to use for
    cuRANDDx distributions. Set using CURANDDX_OPERATOR_GENERATE_METHOD .
    See cuRANDDx execution methods in the documentation:
    https://docs.nvidia.com/cuda/curanddx/api/methods.html

    See `curanddxGenerateMethod`.
    """
    SINGLE = CURANDDX_GENERATE_METHOD_SINGLE
    PAIR = CURANDDX_GENERATE_METHOD_PAIR
    QUAD = CURANDDX_GENERATE_METHOD_QUAD

class CuranddxGenerator(_IntEnum):
    """
    Generator type.Matches cuRANDDx generator kinds. Set using
    CURANDDX_OPERATOR_GENERATOR . See cuRANDDx generators in the
    documentation: https://docs.nvidia.com/cuda/curanddx/api/description_op
    s.html#generator-operator-label

    See `curanddxGenerator`.
    """
    XORWOW = CURANDDX_GENERATOR_XORWOW
    MRG32K3A = CURANDDX_GENERATOR_MRG32K3A
    PHILOX4_32 = CURANDDX_GENERATOR_PHILOX4_32
    PCG = CURANDDX_GENERATOR_PCG
    SOBOL32 = CURANDDX_GENERATOR_SOBOL32
    SCRAMBLED_SOBOL32 = CURANDDX_GENERATOR_SCRAMBLED_SOBOL32
    SOBOL64 = CURANDDX_GENERATOR_SOBOL64
    SCRAMBLED_SOBOL64 = CURANDDX_GENERATOR_SCRAMBLED_SOBOL64

class CuranddxOperatorType(_IntEnum):
    """
    Operators.The set of supported curandDx operators.

    See `curanddxOperatorType`.
    """
    GENERATOR = CURANDDX_OPERATOR_GENERATOR
    PHILOX_ROUNDS = CURANDDX_OPERATOR_PHILOX_ROUNDS
    SM = CURANDDX_OPERATOR_SM
    EXECUTION = CURANDDX_OPERATOR_EXECUTION
    DISTRIBUTION = CURANDDX_OPERATOR_DISTRIBUTION
    OUTPUT_TYPE = CURANDDX_OPERATOR_OUTPUT_TYPE
    GENERATE_METHOD = CURANDDX_OPERATOR_GENERATE_METHOD
    NORMAL_METHOD = CURANDDX_OPERATOR_NORMAL_METHOD
    DEVICE_FUNCTIONS = CURANDDX_OPERATOR_DEVICE_FUNCTIONS
    DISTRIBUTION_PARAMETERS = CURANDDX_OPERATOR_DISTRIBUTION_PARAMETERS

class CuranddxDeviceFunctionType(_IntEnum):
    """
    Device functions.The set of supported device functions. Must be passed
    using CURANDDX_OPERATOR_DEVICE_FUNCTIONS and can be or'ed to indicate
    that libmathdx should generate multiple device functions.

    See `curanddxDeviceFunctionType`.
    """
    GENERATE = CURANDDX_DEVICE_FUNCTION_GENERATE
    INIT_STATE = CURANDDX_DEVICE_FUNCTION_INIT_STATE
    DESTROY_STATE = CURANDDX_DEVICE_FUNCTION_DESTROY_STATE
    SKIP_OFFSET = CURANDDX_DEVICE_FUNCTION_SKIP_OFFSET
    SKIP_SUBSEQUENCE = CURANDDX_DEVICE_FUNCTION_SKIP_SUBSEQUENCE
    SKIP_SEQUENCE = CURANDDX_DEVICE_FUNCTION_SKIP_SEQUENCE

class CuranddxTraitType(_IntEnum):
    """
    Traits.The set of supported types of traits that can be accessed from
    finalized sources that use curanddx.

    See `curanddxTraitType`.
    """
    SYMBOL_GENERATE_NAME = CURANDDX_TRAIT_SYMBOL_GENERATE_NAME
    SYMBOL_INIT_STATE_NAME = CURANDDX_TRAIT_SYMBOL_INIT_STATE_NAME
    SYMBOL_DESTROY_STATE_NAME = CURANDDX_TRAIT_SYMBOL_DESTROY_STATE_NAME
    SYMBOL_SKIP_OFFSET_NAME = CURANDDX_TRAIT_SYMBOL_SKIP_OFFSET_NAME
    SYMBOL_SKIP_SUBSEQUENCE_NAME = CURANDDX_TRAIT_SYMBOL_SKIP_SUBSEQUENCE_NAME
    SYMBOL_SKIP_SEQUENCE_NAME = CURANDDX_TRAIT_SYMBOL_SKIP_SEQUENCE_NAME
    STATE_SIZE = CURANDDX_TRAIT_STATE_SIZE
    STATE_ALIGNMENT = CURANDDX_TRAIT_STATE_ALIGNMENT

class CommondxCodeType(_IntEnum):
    """
    The set of code types supported by the library. See
    commondxOption_t::COMMONDX_OPTION_TARGET_CODE.

    See `commondxCodeType`.
    """
    LTO = COMMONDX_CODE_TYPE_LTO
    PTX = COMMONDX_CODE_TYPE_PTX

class CusolverdxJob(_IntEnum):
    """
    Job operator.Specifies whether and how eigenvectors are computed and
    stored in eigenvalue operations.

    See `cusolverdxJob`.
    """
    NO_VECTORS = CUSOLVERDX_JOB_NO_VECTORS
    ALL_VECTORS = CUSOLVERDX_JOB_ALL_VECTORS
    MULTIPLY_VECTORS = CUSOLVERDX_JOB_MULTIPLY_VECTORS
    OVERWRITE_VECTORS = CUSOLVERDX_JOB_OVERWRITE_VECTORS


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
        cdef str err = STATUS.get(status, f"Unknown error {status}")

        cdef size_t err_size = 0
        cdef commondxStatusType err_code
        cdef vector[char] err_str

        if commondxGetLastErrorStrSize(&err_size) == COMMONDX_SUCCESS and err_size > 0:
            err_str.resize(err_size)
            if commondxGetLastErrorStr(&err_code, err_size, err_str.data()) == COMMONDX_SUCCESS:
                err += ": " + err_str.data().decode('utf-8', 'replace')

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
        __status__ = commondxCreateCode(&code)
    check_status(__status__)
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
        __status__ = commondxSetCodeOptionInt64(<commondxCode>code, <_CommondxOption>option, value)
    check_status(__status__)


cpdef commondx_set_code_option_int64s(long long int code, int option, size_t count, values):
    """Set an option on a code handle.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        option (CommondxOption): The option to set the code to.
        count (size_t): The length of the array.
        values (object): A pointer to ``count`` entries. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `commondxSetCodeOptionInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _values_
    get_resource_ptr[int64_t](_values_, values, <int64_t*>NULL)
    with nogil:
        __status__ = commondxSetCodeOptionInt64s(<commondxCode>code, <_CommondxOption>option, count, <long long int*>(_values_.data()))
    check_status(__status__)


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
        __status__ = commondxSetCodeOptionStr(<commondxCode>code, <_CommondxOption>option, <const char*>_value_)
    check_status(__status__)


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
        __status__ = commondxGetCodeOptionInt64(<commondxCode>code, <_CommondxOption>option, &value)
    check_status(__status__)
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
        __status__ = commondxGetCodeOptionsInt64s(<commondxCode>code, <_CommondxOption>option, size, <long long int*>(_array_.data()))
    check_status(__status__)


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
        __status__ = commondxGetCodeLTOIRSize(<commondxCode>code, &size)
    check_status(__status__)
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
        __status__ = commondxGetCodeLTOIR(<commondxCode>code, size, <void*>_out_)
    check_status(__status__)


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
        __status__ = commondxGetCodeNumLTOIRs(<commondxCode>code, &size)
    check_status(__status__)
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
        __status__ = commondxGetCodeLTOIRSizes(<commondxCode>code, size, <size_t*>(_out_.data()))
    check_status(__status__)


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
        __status__ = commondxGetCodeLTOIRs(<commondxCode>code, size, <void**>(_out_.data()))
    check_status(__status__)


cpdef commondx_destroy_code(long long int code):
    """Destroys a code handle.

    Args:
        code (long long int): A code handle from commondxCreateCode.

    .. seealso:: `commondxDestroyCode`
    """
    with nogil:
        __status__ = commondxDestroyCode(<commondxCode>code)
    check_status(__status__)


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
        __status__ = mathdxGetVersion(&version)
    check_status(__status__)
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
        __status__ = mathdxGetVersionEx(&major, &minor, &patch)
    check_status(__status__)
    return (major, minor, patch)


cpdef long long int cublasdx_create_descriptor() except? 0:
    """Creates a cuBLASDx descriptor.

    Returns:
        long long int: A pointer to a handle.

    .. seealso:: `cublasdxCreateDescriptor`
    """
    cdef cublasdxDescriptor handle
    with nogil:
        __status__ = cublasdxCreateDescriptor(&handle)
    check_status(__status__)
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
        __status__ = cublasdxSetOptionStr(<cublasdxDescriptor>handle, <_CommondxOption>option, <const char*>_value_)
    check_status(__status__)


cpdef cublasdx_set_operator_int64(long long int handle, int op, long long int value):
    """Set an operator on a cuBLASDx descriptor to an integer value.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        op (CublasdxOperatorType): An operator to set the descriptor to.
        value (long long int): The operator's value.

    .. seealso:: `cublasdxSetOperatorInt64`
    """
    with nogil:
        __status__ = cublasdxSetOperatorInt64(<cublasdxDescriptor>handle, <_CublasdxOperatorType>op, value)
    check_status(__status__)


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
        __status__ = cublasdxSetOperatorInt64s(<cublasdxDescriptor>handle, <_CublasdxOperatorType>op, count, <const long long int*>(_array_.data()))
    check_status(__status__)


cpdef long long int cublasdx_create_tensor(long long int handle, int tensor_type) except? 0:
    """Create a tensor handle.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        tensor_type (CublasdxTensorType): The tensor type to bind to the handle.

    Returns:
        long long int: A valid tensor handle.

    .. seealso:: `cublasdxCreateTensor`
    """
    cdef cublasdxTensor tensor
    with nogil:
        __status__ = cublasdxCreateTensor(<cublasdxDescriptor>handle, <_CublasdxTensorType>tensor_type, &tensor)
    check_status(__status__)
    return <long long int>tensor


cpdef long long int cublasdx_create_tensor_strided(int memory_space, int value_type, intptr_t ptr, long long int rank, shape, stride) except? 0:
    """Create a tensor handle for a N-dimensional strided tensor.

    Args:
        memory_space (CublasdxMemorySpace): The memory space for the tensor.
        value_type (CommondxValueType): The datatype of the individual elements.
        ptr (intptr_t): A pointer to the data. Currently, only ``NULL`` is supported.
        rank (long long int): The rank of of tensor.
        shape (object): An array of size ``rank`` indicating the tensor shape. LIBMATHDX_RUNTIME can be used to indicate a runtime shape. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.

        stride (object): An array of size ``rank`` indicating the tensor stride. LIBMATHDX_RUNTIME can be used to indicate a runtime stride. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    Returns:
        long long int: The tensor handle.

    .. seealso:: `cublasdxCreateTensorStrided`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _shape_
    get_resource_ptr[int64_t](_shape_, shape, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _stride_
    get_resource_ptr[int64_t](_stride_, stride, <int64_t*>NULL)
    cdef cublasdxTensor tensor
    with nogil:
        __status__ = cublasdxCreateTensorStrided(<_CublasdxMemorySpace>memory_space, <_CommondxValueType>value_type, <void*>ptr, rank, <long long int*>(_shape_.data()), <long long int*>(_stride_.data()), &tensor)
    check_status(__status__)
    return <long long int>tensor


cpdef long long int cublasdx_make_tensor_like(long long int input, int value_type) except? 0:
    """Create an opaque tensor with a identical layout (smem/gmem) or partitioner (rmem), but with a different datatype.

    Args:
        input (long long int): An opaque tensors.
        value_type (CommondxValueType): The new datatype.

    Returns:
        long long int: The output tensor.

    .. seealso:: `cublasdxMakeTensorLike`
    """
    cdef cublasdxTensor output
    with nogil:
        __status__ = cublasdxMakeTensorLike(<cublasdxTensor>input, <_CommondxValueType>value_type, &output)
    check_status(__status__)
    return <long long int>output


cpdef cublasdx_destroy_tensor(long long int tensor):
    """Destroys a tensor handle created using cublasdxCreateTensor or cublasdxMakeTensorLike.

    Args:
        tensor (long long int): The tensor to destroy.

    .. seealso:: `cublasdxDestroyTensor`
    """
    with nogil:
        __status__ = cublasdxDestroyTensor(<cublasdxTensor>tensor)
    check_status(__status__)


cpdef cublasdx_destroy_pipeline(long long int pipeline):
    """Destroys a pipeline handle created using cublasdxCreateDevicePipeline or cublasdxCreateTilePipeline.

    Args:
        pipeline (long long int): The pipeline to destroy.

    .. seealso:: `cublasdxDestroyPipeline`
    """
    with nogil:
        __status__ = cublasdxDestroyPipeline(<cublasdxPipeline>pipeline)
    check_status(__status__)


cpdef long long int cublasdx_create_device_pipeline(long long int handle, int device_pipeline_type, long long int pipeline_depth, int block_size_strategy, long long int tensor_a, long long int tensor_b) except? 0:
    """Create a device pipeline handle.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        device_pipeline_type (CublasdxDevicePipelineType): The type of the device pipeline.
        pipeline_depth (long long int): The depth of the pipeline. If LIBMATHDX_MAX_PIPELINE_DEPTH is passed, the pipeline depth will be set to the maximal depth based on available shared memory.
        block_size_strategy (CublasdxBlockSizeStrategy): The block size strategy to use, fixed (you use the number of threads specified) or heuristic (cuBLASDx can use more threads if needed).
        tensor_a (long long int): The tensor handle for global matrix A.
        tensor_b (long long int): The tensor handle for global matrix B.

    Returns:
        long long int: A valid device pipeline handle.

    .. seealso:: `cublasdxCreateDevicePipeline`
    """
    cdef cublasdxPipeline device_pipeline
    with nogil:
        __status__ = cublasdxCreateDevicePipeline(<cublasdxDescriptor>handle, <_CublasdxDevicePipelineType>device_pipeline_type, pipeline_depth, <_CublasdxBlockSizeStrategy>block_size_strategy, <cublasdxTensor>tensor_a, <cublasdxTensor>tensor_b, &device_pipeline)
    check_status(__status__)
    return <long long int>device_pipeline


cpdef long long int cublasdx_create_tile_pipeline(long long int handle, int tile_pipeline_type, long long int device_pipeline) except? 0:
    """Create a tile pipeline handle.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        tile_pipeline_type (CublasdxTilePipelineType): The type of the tile pipeline.
        device_pipeline (long long int): The device pipeline handle this tile pipeline is associated with.

    Returns:
        long long int: A valid tile pipeline handle.

    .. seealso:: `cublasdxCreateTilePipeline`
    """
    cdef cublasdxPipeline tile_pipeline
    with nogil:
        __status__ = cublasdxCreateTilePipeline(<cublasdxDescriptor>handle, <_CublasdxTilePipelineType>tile_pipeline_type, <cublasdxPipeline>device_pipeline, &tile_pipeline)
    check_status(__status__)
    return <long long int>tile_pipeline


cpdef cublasdx_set_tensor_option_int64(long long int tensor, int option, long long int value):
    """Set an option on a tensor. This must be called before the tensor is finalized.

    Args:
        tensor (long long int): A cuBLASDx tensor, output of cublasdxCreateTensor.
        option (CublasdxTensorOption): The option to set on the tensor.
        value (long long int): A value for the option.

    .. seealso:: `cublasdxSetTensorOptionInt64`
    """
    with nogil:
        __status__ = cublasdxSetTensorOptionInt64(<cublasdxTensor>tensor, <_CublasdxTensorOption>option, value)
    check_status(__status__)


cpdef cublasdx_finalize_tensors(size_t count, array):
    """Finalize the tensors. This is required before traits can be queried.

    Args:
        count (size_t): The number of tensors to finalized.
        array (object): The array of tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxTensor``.


    .. seealso:: `cublasdxFinalizeTensors`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        __status__ = cublasdxFinalizeTensors(count, <const cublasdxTensor*>(_array_.data()))
    check_status(__status__)


cpdef cublasdx_finalize_pipelines(size_t count, array):
    """Finalize the pipelines. This is required before traits can be queried.

    Args:
        count (size_t): The number of pipelines to finalized.
        array (object): The array of pipelines. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxPipeline``.


    .. seealso:: `cublasdxFinalizePipelines`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        __status__ = cublasdxFinalizePipelines(count, <const cublasdxPipeline*>(_array_.data()))
    check_status(__status__)


cpdef cublasdx_finalize(size_t count_tensors, tensors, size_t count_pipelines, pipelines):
    """Finalize both tensors and pipelines. This is required before traits can be queried. Internally calls cublasdxFinalizeTensors and cublasdxFinalizePipelines.

    Args:
        count_tensors (size_t): The number of tensors to finalized.
        tensors (object): The array of tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxTensor``.

        count_pipelines (size_t): The number of pipelines to finalized.
        pipelines (object): The array of pipelines. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxPipeline``.


    .. seealso:: `cublasdxFinalize`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensors_
    get_resource_ptr[int64_t](_tensors_, tensors, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _pipelines_
    get_resource_ptr[int64_t](_pipelines_, pipelines, <int64_t*>NULL)
    with nogil:
        __status__ = cublasdxFinalize(count_tensors, <const cublasdxTensor*>(_tensors_.data()), count_pipelines, <const cublasdxPipeline*>(_pipelines_.data()))
    check_status(__status__)


cpdef long long int cublasdx_get_tensor_trait_int64(long long int tensor, int trait) except? 0:
    """Query an integer trait value from a finalized tensor.

    Args:
        tensor (long long int): A finalized tensor handle, output of cublasdxCreateTensor.
        trait (CublasdxTensorTrait): The trait to query.

    Returns:
        long long int: The trait value.

    .. seealso:: `cublasdxGetTensorTraitInt64`
    """
    cdef long long int value
    with nogil:
        __status__ = cublasdxGetTensorTraitInt64(<cublasdxTensor>tensor, <_CublasdxTensorTrait>trait, &value)
    check_status(__status__)
    return value


cpdef long long int cublasdx_get_pipeline_trait_int64(long long int pipeline, int trait) except? 0:
    """Query an integer trait value from a finalized pipeline.

    Args:
        pipeline (long long int): A finalized pipeline handle, output of cublasdxCreateDevicePipeline or cublasdxCreateTilePipeline.
        trait (CublasdxPipelineTrait): The trait to query.

    Returns:
        long long int: The trait value.

    .. seealso:: `cublasdxGetPipelineTraitInt64`
    """
    cdef long long int value
    with nogil:
        __status__ = cublasdxGetPipelineTraitInt64(<cublasdxPipeline>pipeline, <_CublasdxPipelineTrait>trait, &value)
    check_status(__status__)
    return value


cpdef cublasdx_get_pipeline_trait_int64s(long long int pipeline, int trait, size_t count, array):
    """Returns an array trait's value from a finalized pipeline.

    Args:
        pipeline (long long int): A finalized pipeline handle, output of cublasdxCreateDevicePipeline or cublasdxCreateTilePipeline.
        trait (CublasdxPipelineTrait): The trait to query.
        count (size_t): The number of values to query.
        array (object): The array of trait values. Must point to exactly ``count`` elements of type ``long long int``. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cublasdxGetPipelineTraitInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    with nogil:
        __status__ = cublasdxGetPipelineTraitInt64s(<cublasdxPipeline>pipeline, <_CublasdxPipelineTrait>trait, count, <long long int*>(_array_.data()))
    check_status(__status__)


cpdef size_t cublasdx_get_tensor_trait_str_size(long long int tensor, int trait) except? 0:
    """Query an C-string trait's size from a finalized tensor.

    Args:
        tensor (long long int): A finalized tensor handle, output of cublasdxCreateTensor.
        trait (CublasdxTensorTrait): The trait to query.

    Returns:
        size_t: The C-string size (including the ``\0``).

    .. seealso:: `cublasdxGetTensorTraitStrSize`
    """
    cdef size_t size
    with nogil:
        __status__ = cublasdxGetTensorTraitStrSize(<cublasdxTensor>tensor, <_CublasdxTensorTrait>trait, &size)
    check_status(__status__)
    return size


cpdef size_t cublasdx_get_pipeline_trait_str_size(long long int pipeline, int trait) except? 0:
    """Query an C-string trait's size from a finalized pipeline.

    Args:
        pipeline (long long int): A finalized pipeline handle, output of cublasdxCreateDevicePipeline or cublasdxCreateTilePipeline.
        trait (CublasdxPipelineTrait): The trait to query.

    Returns:
        size_t: The C-string size (including the ``\0``).

    .. seealso:: `cublasdxGetPipelineTraitStrSize`
    """
    cdef size_t size
    with nogil:
        __status__ = cublasdxGetPipelineTraitStrSize(<cublasdxPipeline>pipeline, <_CublasdxPipelineTrait>trait, &size)
    check_status(__status__)
    return size


cpdef cublasdx_get_tensor_trait_str(long long int tensor, int trait, size_t size, value):
    """Query a C-string trait value from a finalized tensor.

    Args:
        tensor (long long int): A finalized tensor handle, output of cublasdxCreateTensor.
        trait (CublasdxTensorTrait): The trait to query.
        size (size_t): The C-string size, as returned by cublasdxGetTensorTraitStrSize.
        value (bytes): The C-string trait value.

    .. seealso:: `cublasdxGetTensorTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        __status__ = cublasdxGetTensorTraitStr(<cublasdxTensor>tensor, <_CublasdxTensorTrait>trait, size, <char*>_value_)
    check_status(__status__)


cpdef cublasdx_get_pipeline_trait_str(long long int pipeline, int trait, size_t size, value):
    """Query a C-string trait value from a finalized pipeline.

    Args:
        pipeline (long long int): A finalized pipeline handle, output of cublasdxCreateDevicePipeline or cublasdxCreateTilePipeline.
        trait (CublasdxPipelineTrait): The trait to query.
        size (size_t): The C-string size, as returned by cublasdxGetPipelineTraitStrSize.
        value (bytes): The C-string trait value.

    .. seealso:: `cublasdxGetPipelineTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        __status__ = cublasdxGetPipelineTraitStr(<cublasdxPipeline>pipeline, <_CublasdxPipelineTrait>trait, size, <char*>_value_)
    check_status(__status__)


cpdef long long int cublasdx_create_device_function(long long int handle, int device_function_type, size_t count, array) except? 0:
    """Create a device function from a set of tensor.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor or LIBMATHDX_NONE if no descriptor is required.
        device_function_type (CublasdxDeviceFunctionType): The device function to create.
        count (size_t): The number of input & output tensors to the device function.
        array (object): The array of input & output tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxTensor``.


    Returns:
        long long int: The device function.

    .. seealso:: `cublasdxCreateDeviceFunction`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _array_
    get_resource_ptr[int64_t](_array_, array, <int64_t*>NULL)
    cdef cublasdxDeviceFunction device_function
    with nogil:
        __status__ = cublasdxCreateDeviceFunction(<cublasdxDescriptor>handle, <_CublasdxDeviceFunctionType>device_function_type, count, <const cublasdxTensor*>(_array_.data()), &device_function)
    check_status(__status__)
    return <long long int>device_function


cpdef cublasdx_destroy_device_function(long long int device_function):
    """Destroys a device function handle.

    Args:
        device_function (long long int): A cuBLASDx device function, output of cublasdxCreateDeviceFunction.

    .. seealso:: `cublasdxDestroyDeviceFunction`
    """
    with nogil:
        __status__ = cublasdxDestroyDeviceFunction(<cublasdxDeviceFunction>device_function)
    check_status(__status__)


cpdef long long int cublasdx_create_device_function_with_pipelines(long long int handle, int device_function_type, size_t tensor_count, tensors, size_t pipeline_count, pipelines) except? 0:
    """Binds (aka create) a device function from a set of tensor.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.
        device_function_type (CublasdxDeviceFunctionType): The device function to create.
        tensor_count (size_t): The number of input & output tensors to the device function.
        tensors (object): The array of input & output tensors. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxTensor``.

        pipeline_count (size_t): The number of pipelines to the device function.
        pipelines (object): The array of pipelines. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``cublasdxPipeline``.


    Returns:
        long long int: The device function.

    .. seealso:: `cublasdxCreateDeviceFunctionWithPipelines`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _tensors_
    get_resource_ptr[int64_t](_tensors_, tensors, <int64_t*>NULL)
    cdef nullable_unique_ptr[ vector[int64_t] ] _pipelines_
    get_resource_ptr[int64_t](_pipelines_, pipelines, <int64_t*>NULL)
    cdef cublasdxDeviceFunction device_function
    with nogil:
        __status__ = cublasdxCreateDeviceFunctionWithPipelines(<cublasdxDescriptor>handle, <_CublasdxDeviceFunctionType>device_function_type, tensor_count, <const cublasdxTensor*>(_tensors_.data()), pipeline_count, <const cublasdxPipeline*>(_pipelines_.data()), &device_function)
    check_status(__status__)
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
        __status__ = cublasdxFinalizeDeviceFunctions(<commondxCode>code, count, <const cublasdxDeviceFunction*>(_array_.data()))
    check_status(__status__)


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
        __status__ = cublasdxGetDeviceFunctionTraitStrSize(<cublasdxDeviceFunction>device_function, <_CublasdxDeviceFunctionTrait>trait, &size)
    check_status(__status__)
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
        __status__ = cublasdxGetDeviceFunctionTraitStr(<cublasdxDeviceFunction>device_function, <_CublasdxDeviceFunctionTrait>trait, size, <char*>_value_)
    check_status(__status__)


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
        __status__ = cublasdxGetLTOIRSize(<cublasdxDescriptor>handle, &lto_size)
    check_status(__status__)
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
        __status__ = cublasdxGetLTOIR(<cublasdxDescriptor>handle, size, <void*>_lto_)
    check_status(__status__)


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
        __status__ = cublasdxGetTraitStrSize(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, &size)
    check_status(__status__)
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
        __status__ = cublasdxGetTraitStr(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, size, <char*>_value_)
    check_status(__status__)


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
        __status__ = cublasdxGetTraitInt64(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, &value)
    check_status(__status__)
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
        __status__ = cublasdxGetTraitInt64s(<cublasdxDescriptor>handle, <_CublasdxTraitType>trait, count, <long long int*>(_array_.data()))
    check_status(__status__)


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
        __status__ = cublasdxFinalizeCode(<commondxCode>code, <cublasdxDescriptor>handle)
    check_status(__status__)


cpdef cublasdx_destroy_descriptor(long long int handle):
    """Destroy a cuBLASDx descriptor.

    Args:
        handle (long long int): A cuBLASDx descriptor, output of cublasdxCreateDescriptor.

    .. seealso:: `cublasdxDestroyDescriptor`
    """
    with nogil:
        __status__ = cublasdxDestroyDescriptor(<cublasdxDescriptor>handle)
    check_status(__status__)


cpdef long long int cufftdx_create_descriptor() except? 0:
    """Creates a cuFFTDx descriptor.

    Returns:
        long long int: A pointer to a cuFFTDx descriptor handle, and as output, a valid initialized descriptor.

    .. seealso:: `cufftdxCreateDescriptor`
    """
    cdef cufftdxDescriptor handle
    with nogil:
        __status__ = cufftdxCreateDescriptor(&handle)
    check_status(__status__)
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
        __status__ = cufftdxSetOptionStr(<cufftdxDescriptor>handle, <_CommondxOption>opt, <const char*>_value_)
    check_status(__status__)


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
        __status__ = cufftdxGetKnobInt64Size(<cufftdxDescriptor>handle, num_knobs, <_CufftdxKnobType*>(_knobs_ptr_.data()), &size)
    check_status(__status__)
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
        __status__ = cufftdxGetKnobInt64s(<cufftdxDescriptor>handle, num_knobs, <_CufftdxKnobType*>(_knobs_ptr_.data()), size, <long long int*>values)
    check_status(__status__)


cpdef cufftdx_set_operator_int64(long long int handle, int op, long long int value):
    """Set an integer operator to a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.
        op (CufftdxOperatorType): The operator to set the descriptor to.
        value (long long int): The value to set the operator to.

    .. seealso:: `cufftdxSetOperatorInt64`
    """
    with nogil:
        __status__ = cufftdxSetOperatorInt64(<cufftdxDescriptor>handle, <_CufftdxOperatorType>op, value)
    check_status(__status__)


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
        __status__ = cufftdxSetOperatorInt64s(<cufftdxDescriptor>handle, <_CufftdxOperatorType>op, count, <const long long int*>(_array_.data()))
    check_status(__status__)


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
        __status__ = cufftdxGetLTOIRSize(<cufftdxDescriptor>handle, &lto_size)
    check_status(__status__)
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
        __status__ = cufftdxGetLTOIR(<cufftdxDescriptor>handle, size, <void*>_lto_)
    check_status(__status__)


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
        __status__ = cufftdxGetTraitStrSize(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, &size)
    check_status(__status__)
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
        __status__ = cufftdxGetTraitStr(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, size, <char*>_value_)
    check_status(__status__)


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
        __status__ = cufftdxGetTraitInt64(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, &value)
    check_status(__status__)
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
        __status__ = cufftdxGetTraitInt64s(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, count, <long long int*>(_array_.data()))
    check_status(__status__)


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
        __status__ = cufftdxGetTraitCommondxDataType(<cufftdxDescriptor>handle, <_CufftdxTraitType>trait, &value)
    check_status(__status__)
    return <int>value


cpdef cufftdx_finalize_code(long long int code, long long int handle):
    """Generate code from the cuFFTDx descriptor and stores it in code.

    Args:
        code (long long int): A commondxCode instance, output of commondxCreateCode.
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.

    .. seealso:: `cufftdxFinalizeCode`
    """
    with nogil:
        __status__ = cufftdxFinalizeCode(<commondxCode>code, <cufftdxDescriptor>handle)
    check_status(__status__)


cpdef cufftdx_destroy_descriptor(long long int handle):
    """Destroys a cuFFTDx descriptor.

    Args:
        handle (long long int): A cuFFTDx descriptor, output of cufftdxCreateDescriptor.

    .. seealso:: `cufftdxDestroyDescriptor`
    """
    with nogil:
        __status__ = cufftdxDestroyDescriptor(<cufftdxDescriptor>handle)
    check_status(__status__)


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
        __status__ = cusolverdxCreateDescriptor(&handle)
    check_status(__status__)
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
        __status__ = cusolverdxSetOptionStr(<cusolverdxDescriptor>handle, <_CommondxOption>opt, <const char*>_value_)
    check_status(__status__)


cpdef cusolverdx_set_operator_int64(long long int handle, int op, long long int value):
    """Sets an integer operator on a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        op (CusolverdxOperatorType): The operator to set.
        value (long long int): A value for the operator.

    .. seealso:: `cusolverdxSetOperatorInt64`
    """
    with nogil:
        __status__ = cusolverdxSetOperatorInt64(<cusolverdxDescriptor>handle, <_CusolverdxOperatorType>op, value)
    check_status(__status__)


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
        __status__ = cusolverdxSetOperatorInt64s(<cusolverdxDescriptor>handle, <_CusolverdxOperatorType>op, count, <const long long int*>(_array_.data()))
    check_status(__status__)


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
        __status__ = cusolverdxGetLTOIRSize(<cusolverdxDescriptor>handle, &lto_size)
    check_status(__status__)
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
        __status__ = cusolverdxGetLTOIR(<cusolverdxDescriptor>handle, size, <void*>_lto_)
    check_status(__status__)


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
        __status__ = cusolverdxGetUniversalFATBINSize(<cusolverdxDescriptor>handle, &fatbin_size)
    check_status(__status__)
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
        __status__ = cusolverdxGetUniversalFATBIN(<cusolverdxDescriptor>handle, fatbin_size, <void*>_fatbin_)
    check_status(__status__)


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
        __status__ = cusolverdxGetTraitStrSize(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, &size)
    check_status(__status__)
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
        __status__ = cusolverdxGetTraitStr(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, size, <char*>_value_)
    check_status(__status__)


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
        __status__ = cusolverdxGetTraitInt64(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, &value)
    check_status(__status__)
    return value


cpdef cusolverdx_get_trait_int64s(long long int handle, int trait, size_t count, values):
    """Returns an integer array trait value.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor .
        trait (CusolverdxTraitType): A trait to query the descriptor for.
        count (size_t): The size of the array to retrieve.
        values (object): The trait values. Must point to an array of ``count`` values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``int64_t``.


    .. seealso:: `cusolverdxGetTraitInt64s`
    """
    cdef nullable_unique_ptr[ vector[int64_t] ] _values_
    get_resource_ptr[int64_t](_values_, values, <int64_t*>NULL)
    with nogil:
        __status__ = cusolverdxGetTraitInt64s(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, count, <long long int*>(_values_.data()))
    check_status(__status__)


cpdef cusolverdx_finalize_code(long long int code, long long int handle):
    """Fills a code handle with the descriptor's device function code.

    Args:
        code (long long int): A code handle.
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.

    .. seealso:: `cusolverdxFinalizeCode`
    """
    with nogil:
        __status__ = cusolverdxFinalizeCode(<commondxCode>code, <cusolverdxDescriptor>handle)
    check_status(__status__)


cpdef cusolverdx_destroy_descriptor(long long int handle):
    """Destroys a cuSOLVERDx descriptor.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.

    .. seealso:: `cusolverdxDestroyDescriptor`
    """
    with nogil:
        __status__ = cusolverdxDestroyDescriptor(<cusolverdxDescriptor>handle)
    check_status(__status__)


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


cpdef tuple curanddx_get_version():
    """Returns the major.minor.patch version of cuRANDDx.

    Returns:
        A 3-tuple containing:

        - int: The major version.
        - int: The minor version.
        - int: The patch version.

    .. seealso:: `curanddxGetVersion`
    """
    cdef int major
    cdef int minor
    cdef int patch
    with nogil:
        __status__ = curanddxGetVersion(&major, &minor, &patch)
    check_status(__status__)
    return (major, minor, patch)


cpdef long long int curanddx_create_descriptor() except? 0:
    """Creates a cuRANDDx descriptor.

    Returns:
        long long int: A pointer to a descriptor handle. As output, an initialized cuRANDDx descriptor.

    .. seealso:: `curanddxCreateDescriptor`
    """
    cdef curanddxDescriptor handle
    with nogil:
        __status__ = curanddxCreateDescriptor(&handle)
    check_status(__status__)
    return <long long int>handle


cpdef curanddx_set_option_str(long long int handle, int opt, value):
    """Sets a C-string option on a cuRANDDx descriptor.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .
        opt (CommondxOption): The option to set.
        value (str): The value for the option.

    .. seealso:: `curanddxSetOptionStr`
    """
    if not isinstance(value, str):
        raise TypeError("value must be a Python str")
    cdef bytes _temp_value_ = (<str>value).encode()
    cdef char* _value_ = _temp_value_
    with nogil:
        __status__ = curanddxSetOptionStr(<curanddxDescriptor>handle, <_CommondxOption>opt, <const char*>_value_)
    check_status(__status__)


cpdef curanddx_set_operator_int64(long long int handle, int op, long long int value):
    """Sets an integer operator on a cuRANDDx descriptor.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .
        op (CuranddxOperatorType): The operator to set.
        value (long long int): A value for the operator.

    .. seealso:: `curanddxSetOperatorInt64`
    """
    with nogil:
        __status__ = curanddxSetOperatorInt64(<curanddxDescriptor>handle, <_CuranddxOperatorType>op, value)
    check_status(__status__)


cpdef curanddx_set_operator_doubles(long long int handle, int op, size_t count, values):
    """Sets the parameters for the distribution.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .
        op (CuranddxOperatorType): The operator to set. Only supports CURANDDX_OPERATOR_DISTRIBUTION_PARAMETERS .
        count (size_t): The number of parameters. Must be 1 or 2.
        values (object): The parameters. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``float``.


    .. seealso:: `curanddxSetOperatorDoubles`
    """
    cdef nullable_unique_ptr[ vector[double] ] _values_
    get_resource_ptr[double](_values_, values, <double*>NULL)
    with nogil:
        __status__ = curanddxSetOperatorDoubles(<curanddxDescriptor>handle, <_CuranddxOperatorType>op, count, <double*>(_values_.data()))
    check_status(__status__)


cpdef size_t curanddx_get_trait_str_size(long long int handle, int trait) except? 0:
    """Returns the size of a C-string trait value.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .
        trait (CuranddxTraitType): A trait to query the descriptor for.

    Returns:
        size_t: The size of the C-string value for the trait (including the ``\0``).

    .. seealso:: `curanddxGetTraitStrSize`
    """
    cdef size_t size
    with nogil:
        __status__ = curanddxGetTraitStrSize(<curanddxDescriptor>handle, <_CuranddxTraitType>trait, &size)
    check_status(__status__)
    return size


cpdef curanddx_get_trait_str(long long int handle, int trait, size_t size, value):
    """Returns a C-string trait value.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .
        trait (CuranddxTraitType): A trait to query the descriptor for.
        size (size_t): The size of the C-string, output from curanddxGetTraitStrSize .
        value (bytes): The C-string trait value.

    .. seealso:: `curanddxGetTraitStr`
    """
    cdef void* _value_ = get_buffer_pointer(value, size, readonly=False)
    with nogil:
        __status__ = curanddxGetTraitStr(<curanddxDescriptor>handle, <_CuranddxTraitType>trait, size, <char*>_value_)
    check_status(__status__)


cpdef long long int curanddx_get_trait_int64(long long int handle, int trait) except? 0:
    """Returns an integer trait value.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .
        trait (CuranddxTraitType): A trait to query the descriptor for.

    Returns:
        long long int: The trait value.

    .. seealso:: `curanddxGetTraitInt64`
    """
    cdef long long int value
    with nogil:
        __status__ = curanddxGetTraitInt64(<curanddxDescriptor>handle, <_CuranddxTraitType>trait, &value)
    check_status(__status__)
    return value


cpdef curanddx_finalize_code(long long int code, long long int handle):
    """Fills a code handle with the descriptor's device function code.

    Args:
        code (long long int): A code handle output from commondxCreateCode .
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .

    .. seealso:: `curanddxFinalizeCode`
    """
    with nogil:
        __status__ = curanddxFinalizeCode(<commondxCode>code, <curanddxDescriptor>handle)
    check_status(__status__)


cpdef curanddx_destroy_descriptor(long long int handle):
    """Destroys a cuRANDDx descriptor.

    Args:
        handle (long long int): A cuRANDDx descriptor, output of curanddxCreateDescriptor .

    .. seealso:: `curanddxDestroyDescriptor`
    """
    with nogil:
        __status__ = curanddxDestroyDescriptor(<curanddxDescriptor>handle)
    check_status(__status__)


cpdef str curanddx_operator_type_to_str(int op):
    """Converts an operator enum to a human readable C-string.

    Args:
        op (CuranddxOperatorType): An operator enum.

    .. seealso:: `curanddxOperatorTypeToStr`
    """
    cdef bytes _output_
    _output_ = curanddxOperatorTypeToStr(<_CuranddxOperatorType>op)
    return _output_.decode()


cpdef str curanddx_distribution_to_str(int dist):
    """Converts a distribution enum to a human readable C-string.

    Args:
        dist (CuranddxDistribution): A distribution enum.

    .. seealso:: `curanddxDistributionToStr`
    """
    cdef bytes _output_
    _output_ = curanddxDistributionToStr(<_CuranddxDistribution>dist)
    return _output_.decode()


cpdef str curanddx_generator_to_str(int generator):
    """Converts a generator enum to a human readable C-string.

    Args:
        generator (CuranddxGenerator): A generator enum.

    .. seealso:: `curanddxGeneratorToStr`
    """
    cdef bytes _output_
    _output_ = curanddxGeneratorToStr(<_CuranddxGenerator>generator)
    return _output_.decode()


cpdef str curanddx_generate_method_to_str(int generate_method):
    """Converts a generate method enum to a human readable C-string.

    Args:
        generate_method (CuranddxGenerateMethod): A generate method enum.

    .. seealso:: `curanddxGenerateMethodToStr`
    """
    cdef bytes _output_
    _output_ = curanddxGenerateMethodToStr(<_CuranddxGenerateMethod>generate_method)
    return _output_.decode()


cpdef str curanddx_normal_method_to_str(int normal_method):
    """Converts a normal method enum to a human readable C-string.

    Args:
        normal_method (CuranddxNormalMethod): A normal method enum.

    .. seealso:: `curanddxNormalMethodToStr`
    """
    cdef bytes _output_
    _output_ = curanddxNormalMethodToStr(<_CuranddxNormalMethod>normal_method)
    return _output_.decode()


cpdef str curanddx_trait_type_to_str(int trait):
    """Converts a trait enum to a human readable C-string.

    Args:
        trait (CuranddxTraitType): A trait enum.

    .. seealso:: `curanddxTraitTypeToStr`
    """
    cdef bytes _output_
    _output_ = curanddxTraitTypeToStr(<_CuranddxTraitType>trait)
    return _output_.decode()


cpdef size_t commondx_get_code_ptx_size(long long int code) except? 0:
    """Extract the PTX size, in bytes.

    Args:
        code (long long int): A code handle from commondxCreateCode.

    Returns:
        size_t: The PTX size, in bytes (including null terminator).

    .. seealso:: `commondxGetCodePTXSize`
    """
    cdef size_t size
    with nogil:
        __status__ = commondxGetCodePTXSize(<commondxCode>code, &size)
    check_status(__status__)
    return size


cpdef commondx_get_code_ptx(long long int code, size_t size, out):
    """Extract the PTX.

    Args:
        code (long long int): A code handle from commondxCreateCode.
        size (size_t): The PTX size, as returned by commondxGetCodePTXSize.
        out (bytes): The PTX. Must be a pointer to a buffer of at least size bytes.

    .. seealso:: `commondxGetCodePTX`
    """
    cdef void* _out_ = get_buffer_pointer(out, size, readonly=False)
    with nogil:
        __status__ = commondxGetCodePTX(<commondxCode>code, size, <void*>_out_)
    check_status(__status__)


cpdef cusolverdx_get_trait_commondx_data_types(long long int handle, int trait, size_t count, array):
    """Returns an array of CommonDxValueType values representing value type traits.

    Args:
        handle (long long int): A cuSOLVERDx descriptor, output of cusolverdxCreateDescriptor.
        trait (CusolverdxTraitType): A trait to query handle for.
        count (size_t): The number of elements in the trait array, as indicated in the cusolverdxTraitType_t documentation.
        array (object): The CommonDxValueType values. Must point to an array of ``count`` values. It can be:

            - an :class:`int` as the pointer address to the array, or
            - a Python sequence of ``_CommondxValueType``.


    .. seealso:: `cusolverdxGetTraitCommondxDataTypes`
    """
    cdef nullable_unique_ptr[ vector[int] ] _array_
    get_resource_ptr[int](_array_, array, <int*>NULL)
    with nogil:
        __status__ = cusolverdxGetTraitCommondxDataTypes(<cusolverdxDescriptor>handle, <_CusolverdxTraitType>trait, count, <_CommondxValueType*>(_array_.data()))
    check_status(__status__)
