# This code was automatically generated with version 0.2.1. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

from libc.stdint cimport int64_t


###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum commondxValueType "commondxValueType":
    COMMONDX_R_8F_E5M2 "COMMONDX_R_8F_E5M2" = 0
    COMMONDX_C_8F_E5M2 "COMMONDX_C_8F_E5M2" = 1
    COMMONDX_R_8F_E4M3 "COMMONDX_R_8F_E4M3" = 2
    COMMONDX_C_8F_E4M3 "COMMONDX_C_8F_E4M3" = 3
    COMMONDX_R_16BF "COMMONDX_R_16BF" = 4
    COMMONDX_C_16BF "COMMONDX_C_16BF" = 5
    COMMONDX_R_16F2 "COMMONDX_R_16F2" = 6
    COMMONDX_R_16F "COMMONDX_R_16F" = 7
    COMMONDX_C_16F "COMMONDX_C_16F" = 8
    COMMONDX_C_16F2 "COMMONDX_C_16F2" = 9
    COMMONDX_R_32TF "COMMONDX_R_32TF" = 10
    COMMONDX_C_32TF "COMMONDX_C_32TF" = 11
    COMMONDX_R_32F "COMMONDX_R_32F" = 12
    COMMONDX_C_32F "COMMONDX_C_32F" = 13
    COMMONDX_R_64F "COMMONDX_R_64F" = 14
    COMMONDX_C_64F "COMMONDX_C_64F" = 15
    COMMONDX_R_8I "COMMONDX_R_8I" = 16
    COMMONDX_C_8I "COMMONDX_C_8I" = 17
    COMMONDX_R_16I "COMMONDX_R_16I" = 18
    COMMONDX_C_16I "COMMONDX_C_16I" = 19
    COMMONDX_R_32I "COMMONDX_R_32I" = 20
    COMMONDX_C_32I "COMMONDX_C_32I" = 21
    COMMONDX_R_64I "COMMONDX_R_64I" = 22
    COMMONDX_C_64I "COMMONDX_C_64I" = 23
    COMMONDX_R_8UI "COMMONDX_R_8UI" = 24
    COMMONDX_C_8UI "COMMONDX_C_8UI" = 25
    COMMONDX_R_16UI "COMMONDX_R_16UI" = 26
    COMMONDX_C_16UI "COMMONDX_C_16UI" = 27
    COMMONDX_R_32UI "COMMONDX_R_32UI" = 28
    COMMONDX_C_32UI "COMMONDX_C_32UI" = 29
    COMMONDX_R_64UI "COMMONDX_R_64UI" = 30
    COMMONDX_C_64UI "COMMONDX_C_64UI" = 31

ctypedef enum commondxStatusType "commondxStatusType":
    COMMONDX_SUCCESS "COMMONDX_SUCCESS" = 0
    COMMONDX_INVALID_VALUE "COMMONDX_INVALID_VALUE" = 1
    COMMONDX_INTERNAL_ERROR "COMMONDX_INTERNAL_ERROR" = 2
    COMMONDX_COMPILATION_ERROR "COMMONDX_COMPILATION_ERROR" = 3
    COMMONDX_CUFFT_ERROR "COMMONDX_CUFFT_ERROR" = 4
    _COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR "_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR" = -42

ctypedef enum commondxPrecision "commondxPrecision":
    COMMONDX_PRECISION_F8_E5M2 "COMMONDX_PRECISION_F8_E5M2" = 0
    COMMONDX_PRECISION_F8_E4M3 "COMMONDX_PRECISION_F8_E4M3" = 1
    COMMONDX_PRECISION_BF16 "COMMONDX_PRECISION_BF16" = 2
    COMMONDX_PRECISION_F16 "COMMONDX_PRECISION_F16" = 3
    COMMONDX_PRECISION_TF32 "COMMONDX_PRECISION_TF32" = 4
    COMMONDX_PRECISION_F32 "COMMONDX_PRECISION_F32" = 5
    COMMONDX_PRECISION_F64 "COMMONDX_PRECISION_F64" = 6
    COMMONDX_PRECISION_I8 "COMMONDX_PRECISION_I8" = 7
    COMMONDX_PRECISION_I16 "COMMONDX_PRECISION_I16" = 8
    COMMONDX_PRECISION_I32 "COMMONDX_PRECISION_I32" = 9
    COMMONDX_PRECISION_I64 "COMMONDX_PRECISION_I64" = 10
    COMMONDX_PRECISION_UI8 "COMMONDX_PRECISION_UI8" = 11
    COMMONDX_PRECISION_UI16 "COMMONDX_PRECISION_UI16" = 12
    COMMONDX_PRECISION_UI32 "COMMONDX_PRECISION_UI32" = 13
    COMMONDX_PRECISION_UI64 "COMMONDX_PRECISION_UI64" = 14

ctypedef enum commondxOption "commondxOption":
    COMMONDX_OPTION_SYMBOL_NAME "COMMONDX_OPTION_SYMBOL_NAME" = 0
    COMMONDX_OPTION_TARGET_SM "COMMONDX_OPTION_TARGET_SM" = 1
    COMMONDX_OPTION_CODE_CONTAINER "COMMONDX_OPTION_CODE_CONTAINER" = 2
    COMMONDX_OPTION_CODE_ISA "COMMONDX_OPTION_CODE_ISA" = 3
    COMMONDX_OPTION_EXTRA_NVTRC_ARGS "COMMONDX_OPTION_EXTRA_NVTRC_ARGS" = 4

ctypedef enum commondxExecution "commondxExecution":
    COMMONDX_EXECUTION_THREAD "COMMONDX_EXECUTION_THREAD" = 0
    COMMONDX_EXECUTION_BLOCK "COMMONDX_EXECUTION_BLOCK" = 1

ctypedef enum commondxCodeContainer "commondxCodeContainer":
    COMMONDX_CODE_CONTAINER_LTOIR "COMMONDX_CODE_CONTAINER_LTOIR" = 0
    COMMONDX_CODE_CONTAINER_FATBIN "COMMONDX_CODE_CONTAINER_FATBIN" = 1

ctypedef enum cublasdxApi "cublasdxApi":
    CUBLASDX_API_SMEM "CUBLASDX_API_SMEM" = 0
    CUBLASDX_API_SMEM_DYNAMIC_LD "CUBLASDX_API_SMEM_DYNAMIC_LD" = 1
    CUBLASDX_API_TENSORS "CUBLASDX_API_TENSORS" = 2

ctypedef enum cublasdxType "cublasdxType":
    CUBLASDX_TYPE_REAL "CUBLASDX_TYPE_REAL" = 0
    CUBLASDX_TYPE_COMPLEX "CUBLASDX_TYPE_COMPLEX" = 1

ctypedef enum cublasdxTransposeMode "cublasdxTransposeMode":
    CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED "CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED" = 0
    CUBLASDX_TRANSPOSE_MODE_TRANSPOSED "CUBLASDX_TRANSPOSE_MODE_TRANSPOSED" = 1
    CUBLASDX_TRANSPOSE_MODE_CONJ_TRANSPOSED "CUBLASDX_TRANSPOSE_MODE_CONJ_TRANSPOSED" = 2

ctypedef enum cublasdxArrangement "cublasdxArrangement":
    CUBLASDX_ARRANGEMENT_COL_MAJOR "CUBLASDX_ARRANGEMENT_COL_MAJOR" = 0
    CUBLASDX_ARRANGEMENT_ROW_MAJOR "CUBLASDX_ARRANGEMENT_ROW_MAJOR" = 1

ctypedef enum cublasdxFunction "cublasdxFunction":
    CUBLASDX_FUNCTION_MM "CUBLASDX_FUNCTION_MM" = 0

ctypedef enum cublasdxOperatorType "cublasdxOperatorType":
    CUBLASDX_OPERATOR_FUNCTION "CUBLASDX_OPERATOR_FUNCTION" = 0
    CUBLASDX_OPERATOR_SIZE "CUBLASDX_OPERATOR_SIZE" = 1
    CUBLASDX_OPERATOR_TYPE "CUBLASDX_OPERATOR_TYPE" = 2
    CUBLASDX_OPERATOR_PRECISION "CUBLASDX_OPERATOR_PRECISION" = 3
    CUBLASDX_OPERATOR_SM "CUBLASDX_OPERATOR_SM" = 4
    CUBLASDX_OPERATOR_EXECUTION "CUBLASDX_OPERATOR_EXECUTION" = 5
    CUBLASDX_OPERATOR_BLOCK_DIM "CUBLASDX_OPERATOR_BLOCK_DIM" = 6
    CUBLASDX_OPERATOR_LEADING_DIMENSION "CUBLASDX_OPERATOR_LEADING_DIMENSION" = 7
    CUBLASDX_OPERATOR_TRANSPOSE_MODE "CUBLASDX_OPERATOR_TRANSPOSE_MODE" = 8
    CUBLASDX_OPERATOR_API "CUBLASDX_OPERATOR_API" = 9
    CUBLASDX_OPERATOR_ARRANGEMENT "CUBLASDX_OPERATOR_ARRANGEMENT" = 10
    CUBLASDX_OPERATOR_ALIGNMENT "CUBLASDX_OPERATOR_ALIGNMENT" = 11
    CUBLASDX_OPERATOR_STATIC_BLOCK_DIM "CUBLASDX_OPERATOR_STATIC_BLOCK_DIM" = 12

ctypedef enum cublasdxTraitType "cublasdxTraitType":
    CUBLASDX_TRAIT_VALUE_TYPE "CUBLASDX_TRAIT_VALUE_TYPE" = 0
    CUBLASDX_TRAIT_SIZE "CUBLASDX_TRAIT_SIZE" = 1
    CUBLASDX_TRAIT_BLOCK_SIZE "CUBLASDX_TRAIT_BLOCK_SIZE" = 2
    CUBLASDX_TRAIT_BLOCK_DIM "CUBLASDX_TRAIT_BLOCK_DIM" = 3
    CUBLASDX_TRAIT_LEADING_DIMENSION "CUBLASDX_TRAIT_LEADING_DIMENSION" = 4
    CUBLASDX_TRAIT_SYMBOL_NAME "CUBLASDX_TRAIT_SYMBOL_NAME" = 5
    CUBLASDX_TRAIT_ARRANGEMENT "CUBLASDX_TRAIT_ARRANGEMENT" = 6
    CUBLASDX_TRAIT_ALIGNMENT "CUBLASDX_TRAIT_ALIGNMENT" = 7
    CUBLASDX_TRAIT_SUGGESTED_LEADING_DIMENSION "CUBLASDX_TRAIT_SUGGESTED_LEADING_DIMENSION" = 8
    CUBLASDX_TRAIT_SUGGESTED_BLOCK_DIM "CUBLASDX_TRAIT_SUGGESTED_BLOCK_DIM" = 9
    CUBLASDX_TRAIT_MAX_THREADS_PER_BLOCK "CUBLASDX_TRAIT_MAX_THREADS_PER_BLOCK" = 10

ctypedef enum cublasdxTensorType "cublasdxTensorType":
    CUBLASDX_TENSOR_SMEM_A "CUBLASDX_TENSOR_SMEM_A" = 0
    CUBLASDX_TENSOR_SMEM_B "CUBLASDX_TENSOR_SMEM_B" = 1
    CUBLASDX_TENSOR_SMEM_C "CUBLASDX_TENSOR_SMEM_C" = 2
    CUBLASDX_TENSOR_SUGGESTED_SMEM_A "CUBLASDX_TENSOR_SUGGESTED_SMEM_A" = 3
    CUBLASDX_TENSOR_SUGGESTED_SMEM_B "CUBLASDX_TENSOR_SUGGESTED_SMEM_B" = 4
    CUBLASDX_TENSOR_SUGGESTED_SMEM_C "CUBLASDX_TENSOR_SUGGESTED_SMEM_C" = 5
    CUBLASDX_TENSOR_SUGGESTED_RMEM_C "CUBLASDX_TENSOR_SUGGESTED_RMEM_C" = 6
    CUBLASDX_TENSOR_GMEM_A "CUBLASDX_TENSOR_GMEM_A" = 7
    CUBLASDX_TENSOR_GMEM_B "CUBLASDX_TENSOR_GMEM_B" = 8
    CUBLASDX_TENSOR_GMEM_C "CUBLASDX_TENSOR_GMEM_C" = 9

ctypedef enum cublasdxTensorOption "cublasdxTensorOption":
    CUBLASDX_TENSOR_OPTION_ALIGNMENT_BYTES "CUBLASDX_TENSOR_OPTION_ALIGNMENT_BYTES" = 0

ctypedef enum cublasdxTensorTrait "cublasdxTensorTrait":
    CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES "CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES" = 0
    CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES "CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES" = 1
    CUBLASDX_TENSOR_TRAIT_UID "CUBLASDX_TENSOR_TRAIT_UID" = 2
    CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME "CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME" = 4

ctypedef enum cublasdxDeviceFunctionTrait "cublasdxDeviceFunctionTrait":
    CUBLASDX_DEVICE_FUNCTION_TRAIT_NAME "CUBLASDX_DEVICE_FUNCTION_TRAIT_NAME" = 0
    CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL "CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL" = 1

ctypedef enum cublasdxDeviceFunctionOption "cublasdxDeviceFunctionOption":
    CUBLASDX_DEVICE_FUNCTION_OPTION_COPY_ALIGNMENT "CUBLASDX_DEVICE_FUNCTION_OPTION_COPY_ALIGNMENT" = 0

ctypedef enum cublasdxDeviceFunctionType "cublasdxDeviceFunctionType":
    CUBLASDX_DEVICE_FUNCTION_EXECUTE "CUBLASDX_DEVICE_FUNCTION_EXECUTE" = 0
    CUBLASDX_DEVICE_FUNCTION_COPY "CUBLASDX_DEVICE_FUNCTION_COPY" = 1
    CUBLASDX_DEVICE_FUNCTION_COPY_WAIT "CUBLASDX_DEVICE_FUNCTION_COPY_WAIT" = 2
    CUBLASDX_DEVICE_FUNCTION_CLEAR "CUBLASDX_DEVICE_FUNCTION_CLEAR" = 3
    CUBLASDX_DEVICE_FUNCTION_AXPBY "CUBLASDX_DEVICE_FUNCTION_AXPBY" = 4

ctypedef enum cufftdxApi "cufftdxApi":
    CUFFTDX_API_LMEM "CUFFTDX_API_LMEM" = 0
    CUFFTDX_API_SMEM "CUFFTDX_API_SMEM" = 1

ctypedef enum cufftdxType "cufftdxType":
    CUFFTDX_TYPE_C2C "CUFFTDX_TYPE_C2C" = 0
    CUFFTDX_TYPE_R2C "CUFFTDX_TYPE_R2C" = 1
    CUFFTDX_TYPE_C2R "CUFFTDX_TYPE_C2R" = 2

ctypedef enum cufftdxDirection "cufftdxDirection":
    CUFFTDX_DIRECTION_FORWARD "CUFFTDX_DIRECTION_FORWARD" = 0
    CUFFTDX_DIRECTION_INVERSE "CUFFTDX_DIRECTION_INVERSE" = 1

ctypedef enum cufftdxComplexLayout "cufftdxComplexLayout":
    CUFFTDX_COMPLEX_LAYOUT_NATURAL "CUFFTDX_COMPLEX_LAYOUT_NATURAL" = 0
    CUFFTDX_COMPLEX_LAYOUT_PACKED "CUFFTDX_COMPLEX_LAYOUT_PACKED" = 1
    CUFFTDX_COMPLEX_LAYOUT_FULL "CUFFTDX_COMPLEX_LAYOUT_FULL" = 2

ctypedef enum cufftdxRealMode "cufftdxRealMode":
    CUFFTDX_REAL_MODE_NORMAL "CUFFTDX_REAL_MODE_NORMAL" = 0
    CUFFTDX_REAL_MODE_FOLDED "CUFFTDX_REAL_MODE_FOLDED" = 1

ctypedef enum cufftdxCodeType "cufftdxCodeType":
    CUFFTDX_CODE_TYPE_PTX "CUFFTDX_CODE_TYPE_PTX" = 0
    CUFFTDX_CODE_TYPE_LTOIR "CUFFTDX_CODE_TYPE_LTOIR" = 1

ctypedef enum cufftdxOperatorType "cufftdxOperatorType":
    CUFFTDX_OPERATOR_SIZE "CUFFTDX_OPERATOR_SIZE" = 0
    CUFFTDX_OPERATOR_DIRECTION "CUFFTDX_OPERATOR_DIRECTION" = 1
    CUFFTDX_OPERATOR_TYPE "CUFFTDX_OPERATOR_TYPE" = 2
    CUFFTDX_OPERATOR_PRECISION "CUFFTDX_OPERATOR_PRECISION" = 3
    CUFFTDX_OPERATOR_SM "CUFFTDX_OPERATOR_SM" = 4
    CUFFTDX_OPERATOR_EXECUTION "CUFFTDX_OPERATOR_EXECUTION" = 5
    CUFFTDX_OPERATOR_FFTS_PER_BLOCK "CUFFTDX_OPERATOR_FFTS_PER_BLOCK" = 6
    CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD "CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD" = 7
    CUFFTDX_OPERATOR_BLOCK_DIM "CUFFTDX_OPERATOR_BLOCK_DIM" = 8
    CUFFTDX_OPERATOR_REAL_FFT_OPTIONS "CUFFTDX_OPERATOR_REAL_FFT_OPTIONS" = 9
    CUFFTDX_OPERATOR_API "CUFFTDX_OPERATOR_API" = 10
    CUFFTDX_OPERATOR_CODE_TYPE "CUFFTDX_OPERATOR_CODE_TYPE" = 11

ctypedef enum cufftdxKnobType "cufftdxKnobType":
    CUFFTDX_KNOB_ELEMENTS_PER_THREAD "CUFFTDX_KNOB_ELEMENTS_PER_THREAD" = 0
    CUFFTDX_KNOB_FFTS_PER_BLOCK "CUFFTDX_KNOB_FFTS_PER_BLOCK" = 1

ctypedef enum cufftdxTraitType "cufftdxTraitType":
    CUFFTDX_TRAIT_VALUE_TYPE "CUFFTDX_TRAIT_VALUE_TYPE" = 0
    CUFFTDX_TRAIT_INPUT_TYPE "CUFFTDX_TRAIT_INPUT_TYPE" = 1
    CUFFTDX_TRAIT_OUTPUT_TYPE "CUFFTDX_TRAIT_OUTPUT_TYPE" = 2
    CUFFTDX_TRAIT_IMPLICIT_TYPE_BATCHING "CUFFTDX_TRAIT_IMPLICIT_TYPE_BATCHING" = 3
    CUFFTDX_TRAIT_ELEMENTS_PER_THREAD "CUFFTDX_TRAIT_ELEMENTS_PER_THREAD" = 4
    CUFFTDX_TRAIT_STORAGE_SIZE "CUFFTDX_TRAIT_STORAGE_SIZE" = 5
    CUFFTDX_TRAIT_STRIDE "CUFFTDX_TRAIT_STRIDE" = 6
    CUFFTDX_TRAIT_BLOCK_DIM "CUFFTDX_TRAIT_BLOCK_DIM" = 7
    CUFFTDX_TRAIT_SHARED_MEMORY_SIZE "CUFFTDX_TRAIT_SHARED_MEMORY_SIZE" = 8
    CUFFTDX_TRAIT_FFTS_PER_BLOCK "CUFFTDX_TRAIT_FFTS_PER_BLOCK" = 9
    CUFFTDX_TRAIT_SYMBOL_NAME "CUFFTDX_TRAIT_SYMBOL_NAME" = 10
    CUFFTDX_TRAIT_INPUT_LENGTH "CUFFTDX_TRAIT_INPUT_LENGTH" = 11
    CUFFTDX_TRAIT_OUTPUT_LENGTH "CUFFTDX_TRAIT_OUTPUT_LENGTH" = 12
    CUFFTDX_TRAIT_INPUT_ELEMENTS_PER_THREAD "CUFFTDX_TRAIT_INPUT_ELEMENTS_PER_THREAD" = 13
    CUFFTDX_TRAIT_OUTPUT_ELEMENTS_PER_THREAD "CUFFTDX_TRAIT_OUTPUT_ELEMENTS_PER_THREAD" = 14
    CUFFTDX_TRAIT_SUGGESTED_FFTS_PER_BLOCK "CUFFTDX_TRAIT_SUGGESTED_FFTS_PER_BLOCK" = 15

ctypedef enum cusolverdxApi "cusolverdxApi":
    CUSOLVERDX_API_SMEM "CUSOLVERDX_API_SMEM" = 0
    CUSOLVERDX_API_SMEM_DYNAMIC_LD "CUSOLVERDX_API_SMEM_DYNAMIC_LD" = 1

ctypedef enum cusolverdxType "cusolverdxType":
    CUSOLVERDX_TYPE_REAL "CUSOLVERDX_TYPE_REAL" = 0
    CUSOLVERDX_TYPE_COMPLEX "CUSOLVERDX_TYPE_COMPLEX" = 1

ctypedef enum cusolverdxFunction "cusolverdxFunction":
    CUSOLVERDX_FUNCTION_GETRF_NO_PIVOT "CUSOLVERDX_FUNCTION_GETRF_NO_PIVOT" = 0
    CUSOLVERDX_FUNCTION_GETRS_NO_PIVOT "CUSOLVERDX_FUNCTION_GETRS_NO_PIVOT" = 1
    CUSOLVERDX_FUNCTION_POTRF "CUSOLVERDX_FUNCTION_POTRF" = 2
    CUSOLVERDX_FUNCTION_POTRS "CUSOLVERDX_FUNCTION_POTRS" = 3
    CUSOLVERDX_FUNCTION_TRSM "CUSOLVERDX_FUNCTION_TRSM" = 4

ctypedef enum cusolverdxArrangement "cusolverdxArrangement":
    CUSOLVERDX_ARRANGEMENT_COL_MAJOR "CUSOLVERDX_ARRANGEMENT_COL_MAJOR" = 0
    CUSOLVERDX_ARRANGEMENT_ROW_MAJOR "CUSOLVERDX_ARRANGEMENT_ROW_MAJOR" = 1

ctypedef enum cusolverdxFillMode "cusolverdxFillMode":
    CUSOLVERDX_FILL_MODE_UPPER "CUSOLVERDX_FILL_MODE_UPPER" = 0
    CUSOLVERDX_FILL_MODE_LOWER "CUSOLVERDX_FILL_MODE_LOWER" = 1

ctypedef enum cusolverdxSide "cusolverdxSide":
    CUSOLVERDX_SIDE_LEFT "CUSOLVERDX_SIDE_LEFT" = 0
    CUSOLVERDX_SIDE_RIGHT "CUSOLVERDX_SIDE_RIGHT" = 1

ctypedef enum cusolverdxDiag "cusolverdxDiag":
    CUSOLVERDX_DIAG_UNIT "CUSOLVERDX_DIAG_UNIT" = 0
    CUSOLVERDX_DIAG_NON_UNIT "CUSOLVERDX_DIAG_NON_UNIT" = 1

ctypedef enum cusolverdxOperatorType "cusolverdxOperatorType":
    CUSOLVERDX_OPERATOR_SIZE "CUSOLVERDX_OPERATOR_SIZE" = 0
    CUSOLVERDX_OPERATOR_TYPE "CUSOLVERDX_OPERATOR_TYPE" = 1
    CUSOLVERDX_OPERATOR_PRECISION "CUSOLVERDX_OPERATOR_PRECISION" = 2
    CUSOLVERDX_OPERATOR_SM "CUSOLVERDX_OPERATOR_SM" = 3
    CUSOLVERDX_OPERATOR_EXECUTION "CUSOLVERDX_OPERATOR_EXECUTION" = 4
    CUSOLVERDX_OPERATOR_BLOCK_DIM "CUSOLVERDX_OPERATOR_BLOCK_DIM" = 5
    CUSOLVERDX_OPERATOR_API "CUSOLVERDX_OPERATOR_API" = 6
    CUSOLVERDX_OPERATOR_FUNCTION "CUSOLVERDX_OPERATOR_FUNCTION" = 7
    CUSOLVERDX_OPERATOR_ARRANGEMENT "CUSOLVERDX_OPERATOR_ARRANGEMENT" = 8
    CUSOLVERDX_OPERATOR_FILL_MODE "CUSOLVERDX_OPERATOR_FILL_MODE" = 9
    CUSOLVERDX_OPERATOR_SIDE "CUSOLVERDX_OPERATOR_SIDE" = 10
    CUSOLVERDX_OPERATOR_DIAG "CUSOLVERDX_OPERATOR_DIAG" = 11

ctypedef enum cusolverdxTraitType "cusolverdxTraitType":
    CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE "CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE" = 1
    CUSOLVERDX_TRAIT_SYMBOL_NAME "CUSOLVERDX_TRAIT_SYMBOL_NAME" = 2

# types
ctypedef long long int commondxCode 'commondxCode'
ctypedef long long int cublasdxDescriptor 'cublasdxDescriptor'
ctypedef long long int cublasdxTensor 'cublasdxTensor'
ctypedef long long int cublasdxDeviceFunction 'cublasdxDeviceFunction'
ctypedef long long int cufftdxDescriptor 'cufftdxDescriptor'
ctypedef long long int cusolverdxDescriptor 'cusolverdxDescriptor'


###############################################################################
# Functions
###############################################################################

cdef commondxStatusType commondxCreateCode(commondxCode* code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxSetCodeOptionInt64(commondxCode code, commondxOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxSetCodeOptionStr(commondxCode code, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeOptionInt64(commondxCode code, commondxOption option, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeOptionsInt64s(commondxCode code, commondxOption option, size_t size, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeLTOIRSize(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeLTOIR(commondxCode code, size_t size, void* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeNumLTOIRs(commondxCode code, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeLTOIRSizes(commondxCode code, size_t size, size_t* out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxGetCodeLTOIRs(commondxCode code, size_t size, void** out) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType commondxDestroyCode(commondxCode code) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* commondxStatusToStr(commondxStatusType status) except?NULL nogil
cdef commondxStatusType mathdxGetVersion(int* version) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType mathdxGetVersionEx(int* major, int* minor, int* patch) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxCreateDescriptor(cublasdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxSetOptionStr(cublasdxDescriptor handle, commondxOption option, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxSetOperatorInt64(cublasdxDescriptor handle, cublasdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxSetOperatorInt64s(cublasdxDescriptor handle, cublasdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxBindTensor(cublasdxDescriptor handle, cublasdxTensorType tensor_type, cublasdxTensor* tensor) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxSetTensorOptionInt64(cublasdxTensor tensor, cublasdxTensorOption option, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxFinalizeTensors(cublasdxDescriptor handle, size_t count, const cublasdxTensor* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTensorTraitInt64(cublasdxTensor tensor, cublasdxTensorTrait trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTensorTraitStrSize(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTensorTraitStr(cublasdxTensor tensor, cublasdxTensorTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxBindDeviceFunction(cublasdxDescriptor handle, cublasdxDeviceFunctionType device_function_type, size_t count, const cublasdxTensor* array, cublasdxDeviceFunction* device_function) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxFinalizeDeviceFunctions(commondxCode code, size_t count, const cublasdxDeviceFunction* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetDeviceFunctionTraitStrSize(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetDeviceFunctionTraitStr(cublasdxDeviceFunction device_function, cublasdxDeviceFunctionTrait trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetLTOIRSize(cublasdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetLTOIR(cublasdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTraitStrSize(cublasdxDescriptor handle, cublasdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTraitStr(cublasdxDescriptor handle, cublasdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTraitInt64(cublasdxDescriptor handle, cublasdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxGetTraitInt64s(cublasdxDescriptor handle, cublasdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* cublasdxOperatorTypeToStr(cublasdxOperatorType op) except?NULL nogil
cdef const char* cublasdxTraitTypeToStr(cublasdxTraitType trait) except?NULL nogil
cdef commondxStatusType cublasdxFinalizeCode(commondxCode code, cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cublasdxDestroyDescriptor(cublasdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxCreateDescriptor(cufftdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxSetOptionStr(cufftdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetKnobInt64Size(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetKnobInt64s(cufftdxDescriptor handle, size_t num_knobs, cufftdxKnobType* knobs_ptr, size_t size, long long int* values) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxSetOperatorInt64(cufftdxDescriptor handle, cufftdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxSetOperatorInt64s(cufftdxDescriptor handle, cufftdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetLTOIRSize(cufftdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetLTOIR(cufftdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetTraitStrSize(cufftdxDescriptor handle, cufftdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetTraitStr(cufftdxDescriptor handle, cufftdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetTraitInt64(cufftdxDescriptor handle, cufftdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetTraitInt64s(cufftdxDescriptor handle, cufftdxTraitType trait, size_t count, long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxGetTraitCommondxDataType(cufftdxDescriptor handle, cufftdxTraitType trait, commondxValueType* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxFinalizeCode(commondxCode code, cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cufftdxDestroyDescriptor(cufftdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* cufftdxOperatorTypeToStr(cufftdxOperatorType op) except?NULL nogil
cdef const char* cufftdxTraitTypeToStr(cufftdxTraitType op) except?NULL nogil
cdef commondxStatusType cusolverdxCreateDescriptor(cusolverdxDescriptor* handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxSetOptionStr(cusolverdxDescriptor handle, commondxOption opt, const char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxSetOperatorInt64(cusolverdxDescriptor handle, cusolverdxOperatorType op, long long int value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxSetOperatorInt64s(cusolverdxDescriptor handle, cusolverdxOperatorType op, size_t count, const long long int* array) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetLTOIRSize(cusolverdxDescriptor handle, size_t* lto_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetLTOIR(cusolverdxDescriptor handle, size_t size, void* lto) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetUniversalFATBINSize(cusolverdxDescriptor handle, size_t* fatbin_size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetUniversalFATBIN(cusolverdxDescriptor handle, size_t fatbin_size, void* fatbin) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetTraitStrSize(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t* size) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetTraitStr(cusolverdxDescriptor handle, cusolverdxTraitType trait, size_t size, char* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxGetTraitInt64(cusolverdxDescriptor handle, cusolverdxTraitType trait, long long int* value) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxFinalizeCode(commondxCode code, cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef commondxStatusType cusolverdxDestroyDescriptor(cusolverdxDescriptor handle) except?_COMMONDXSTATUSTYPE_INTERNAL_LOADING_ERROR nogil
cdef const char* cusolverdxOperatorTypeToStr(cusolverdxOperatorType op) except?NULL nogil
cdef const char* cusolverdxTraitTypeToStr(cusolverdxTraitType trait) except?NULL nogil
