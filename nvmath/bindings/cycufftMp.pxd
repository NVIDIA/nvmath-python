# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.2.6 to 11.4.0. Do not modify it directly.
# This layer exposes the C header to Cython as-is.

###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum libFormat_t "libFormat_t":
    LIB_FORMAT_CUFFT "LIB_FORMAT_CUFFT" = 0x0
    LIB_FORMAT_UNDEFINED "LIB_FORMAT_UNDEFINED" = 0x1

ctypedef enum cufftResult "cufftResult":
    CUFFT_SUCCESS "CUFFT_SUCCESS" = 0x0
    CUFFT_INVALID_PLAN "CUFFT_INVALID_PLAN" = 0x1
    CUFFT_ALLOC_FAILED "CUFFT_ALLOC_FAILED" = 0x2
    CUFFT_INVALID_TYPE "CUFFT_INVALID_TYPE" = 0x3
    CUFFT_INVALID_VALUE "CUFFT_INVALID_VALUE" = 0x4
    CUFFT_INTERNAL_ERROR "CUFFT_INTERNAL_ERROR" = 0x5
    CUFFT_EXEC_FAILED "CUFFT_EXEC_FAILED" = 0x6
    CUFFT_SETUP_FAILED "CUFFT_SETUP_FAILED" = 0x7
    CUFFT_INVALID_SIZE "CUFFT_INVALID_SIZE" = 0x8
    CUFFT_UNALIGNED_DATA "CUFFT_UNALIGNED_DATA" = 0x9
    CUFFT_INCOMPLETE_PARAMETER_LIST "CUFFT_INCOMPLETE_PARAMETER_LIST" = 0xA
    CUFFT_INVALID_DEVICE "CUFFT_INVALID_DEVICE" = 0xB
    CUFFT_PARSE_ERROR "CUFFT_PARSE_ERROR" = 0xC
    CUFFT_NO_WORKSPACE "CUFFT_NO_WORKSPACE" = 0xD
    CUFFT_NOT_IMPLEMENTED "CUFFT_NOT_IMPLEMENTED" = 0xE
    CUFFT_LICENSE_ERROR "CUFFT_LICENSE_ERROR" = 0x0F
    CUFFT_NOT_SUPPORTED "CUFFT_NOT_SUPPORTED" = 0x10
    _CUFFTRESULT_INTERNAL_LOADING_ERROR "_CUFFTRESULT_INTERNAL_LOADING_ERROR" = -42

ctypedef enum cufftType "cufftType":
    CUFFT_R2C "CUFFT_R2C" = 0x2a
    CUFFT_C2R "CUFFT_C2R" = 0x2c
    CUFFT_C2C "CUFFT_C2C" = 0x29
    CUFFT_D2Z "CUFFT_D2Z" = 0x6a
    CUFFT_Z2D "CUFFT_Z2D" = 0x6c
    CUFFT_Z2Z "CUFFT_Z2Z" = 0x69

ctypedef enum cufftCompatibility "cufftCompatibility":
    CUFFT_COMPATIBILITY_FFTW_PADDING "CUFFT_COMPATIBILITY_FFTW_PADDING" = 0x01

ctypedef enum cufftProperty "cufftProperty":
    NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT "NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT" = 0x1
    NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS "NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS" = 0x2

ctypedef enum cufftXtSubFormat "cufftXtSubFormat":
    CUFFT_XT_FORMAT_INPUT "CUFFT_XT_FORMAT_INPUT" = 0x00
    CUFFT_XT_FORMAT_OUTPUT "CUFFT_XT_FORMAT_OUTPUT" = 0x01
    CUFFT_XT_FORMAT_INPLACE "CUFFT_XT_FORMAT_INPLACE" = 0x02
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED "CUFFT_XT_FORMAT_INPLACE_SHUFFLED" = 0x03
    CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED "CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED" = 0x04
    CUFFT_XT_FORMAT_DISTRIBUTED_INPUT "CUFFT_XT_FORMAT_DISTRIBUTED_INPUT" = 0x05
    CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT "CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT" = 0x06
    CUFFT_FORMAT_UNDEFINED "CUFFT_FORMAT_UNDEFINED" = 0x07

ctypedef enum cufftXtCopyType "cufftXtCopyType":
    CUFFT_COPY_HOST_TO_DEVICE "CUFFT_COPY_HOST_TO_DEVICE" = 0x00
    CUFFT_COPY_DEVICE_TO_HOST "CUFFT_COPY_DEVICE_TO_HOST" = 0x01
    CUFFT_COPY_DEVICE_TO_DEVICE "CUFFT_COPY_DEVICE_TO_DEVICE" = 0x02
    CUFFT_COPY_UNDEFINED "CUFFT_COPY_UNDEFINED" = 0x03

ctypedef enum cufftXtQueryType "cufftXtQueryType":
    CUFFT_QUERY_1D_FACTORS "CUFFT_QUERY_1D_FACTORS" = 0x00
    CUFFT_QUERY_UNDEFINED "CUFFT_QUERY_UNDEFINED" = 0x01

ctypedef enum cufftXtWorkAreaPolicy "cufftXtWorkAreaPolicy":
    CUFFT_WORKAREA_MINIMAL "CUFFT_WORKAREA_MINIMAL" = 0
    CUFFT_WORKAREA_USER "CUFFT_WORKAREA_USER" = 1
    CUFFT_WORKAREA_PERFORMANCE "CUFFT_WORKAREA_PERFORMANCE" = 2

ctypedef enum cufftXtCallbackType "cufftXtCallbackType":
    CUFFT_CB_LD_COMPLEX "CUFFT_CB_LD_COMPLEX" = 0x0
    CUFFT_CB_LD_COMPLEX_DOUBLE "CUFFT_CB_LD_COMPLEX_DOUBLE" = 0x1
    CUFFT_CB_LD_REAL "CUFFT_CB_LD_REAL" = 0x2
    CUFFT_CB_LD_REAL_DOUBLE "CUFFT_CB_LD_REAL_DOUBLE" = 0x3
    CUFFT_CB_ST_COMPLEX "CUFFT_CB_ST_COMPLEX" = 0x4
    CUFFT_CB_ST_COMPLEX_DOUBLE "CUFFT_CB_ST_COMPLEX_DOUBLE" = 0x5
    CUFFT_CB_ST_REAL "CUFFT_CB_ST_REAL" = 0x6
    CUFFT_CB_ST_REAL_DOUBLE "CUFFT_CB_ST_REAL_DOUBLE" = 0x7
    CUFFT_CB_UNDEFINED "CUFFT_CB_UNDEFINED" = 0x8

ctypedef enum cufftMpCommType "cufftMpCommType":
    CUFFT_COMM_MPI "CUFFT_COMM_MPI" = 0x00
    CUFFT_COMM_NONE "CUFFT_COMM_NONE" = 0x01


# types
cdef extern from *:
    """
    #include <driver_types.h>
    #include <library_types.h>
    #include <cuComplex.h>

    #define MAX_CUDA_DESCRIPTOR_GPUS 64
    #define CUFFT_FORWARD -1
    #define CUFFT_INVERSE  1
    """
    ctypedef void* cudaStream_t 'cudaStream_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'
    ctypedef int cudaDataType 'cudaDataType'
    ctypedef int libraryPropertyType_t 'libraryPropertyType_t'
    ctypedef int libraryPropertyType 'libraryPropertyType'

    ctypedef struct cuComplex:
        pass
    ctypedef struct cuDoubleComplex:
        pass

    cdef const int MAX_CUDA_DESCRIPTOR_GPUS
    cdef const int cuFFTFORWARD
    cdef const int cuFFTINVERSE


ctypedef float cufftReal 'cufftReal'
ctypedef double cufftDoubleReal 'cufftDoubleReal'
ctypedef cuComplex cufftComplex 'cufftComplex'
ctypedef cuDoubleComplex cufftDoubleComplex 'cufftDoubleComplex'
ctypedef int cufftHandle 'cufftHandle'
ctypedef void* cufftReshapeHandle 'cufftReshapeHandle'
ctypedef struct cufftXt1dFactors 'cufftXt1dFactors':
    long long int size
    long long int stringCount
    long long int stringLength
    long long int substringLength
    long long int factor1
    long long int factor2
    long long int stringMask
    long long int substringMask
    long long int factor1Mask
    long long int factor2Mask
    int stringShift
    int substringShift
    int factor1Shift
    int factor2Shift
ctypedef cufftComplex (*cufftCallbackLoadC 'cufftCallbackLoadC')(
    void* dataIn,
    size_t offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftDoubleComplex (*cufftCallbackLoadZ 'cufftCallbackLoadZ')(
    void* dataIn,
    size_t offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftReal (*cufftCallbackLoadR 'cufftCallbackLoadR')(
    void* dataIn,
    size_t offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftDoubleReal (*cufftCallbackLoadD 'cufftCallbackLoadD')(
    void* dataIn,
    size_t offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftComplex (*cufftJITCallbackLoadC 'cufftJITCallbackLoadC')(
    void* dataIn,
    unsigned long long offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftDoubleComplex (*cufftJITCallbackLoadZ 'cufftJITCallbackLoadZ')(
    void* dataIn,
    unsigned long long offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftReal (*cufftJITCallbackLoadR 'cufftJITCallbackLoadR')(
    void* dataIn,
    unsigned long long offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef cufftDoubleReal (*cufftJITCallbackLoadD 'cufftJITCallbackLoadD')(
    void* dataIn,
    unsigned long long offset,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftCallbackStoreR 'cufftCallbackStoreR')(
    void* dataOut,
    size_t offset,
    cufftReal element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftJITCallbackStoreR 'cufftJITCallbackStoreR')(
    void* dataOut,
    unsigned long long offset,
    cufftReal element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftCallbackStoreD 'cufftCallbackStoreD')(
    void* dataOut,
    size_t offset,
    cufftDoubleReal element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftJITCallbackStoreD 'cufftJITCallbackStoreD')(
    void* dataOut,
    unsigned long long offset,
    cufftDoubleReal element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftCallbackStoreC 'cufftCallbackStoreC')(
    void* dataOut,
    size_t offset,
    cufftComplex element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftJITCallbackStoreC 'cufftJITCallbackStoreC')(
    void* dataOut,
    unsigned long long offset,
    cufftComplex element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftCallbackStoreZ 'cufftCallbackStoreZ')(
    void* dataOut,
    size_t offset,
    cufftDoubleComplex element,
    void* callerInfo,
    void* sharedPointer
)
ctypedef void (*cufftJITCallbackStoreZ 'cufftJITCallbackStoreZ')(
    void* dataOut,
    unsigned long long offset,
    cufftDoubleComplex element,
    void* callerInfo,
    void* sharedPointer
)

ctypedef struct cudaXtDesc_t 'cudaXtDesc_t':
    int version "version"
    int nGPUs "nGPUs"
    int GPUs "GPUs" [MAX_CUDA_DESCRIPTOR_GPUS]
    void* data "data" [MAX_CUDA_DESCRIPTOR_GPUS]
    size_t size "size" [MAX_CUDA_DESCRIPTOR_GPUS]
    void* cudaXtState "cudaXtState"

ctypedef cudaXtDesc_t cudaXtDesc 'cudaXtDesc'

ctypedef struct cudaLibXtDesc_t 'cudaLibXtDesc_t':
    int version "version"
    cudaXtDesc* descriptor "descriptor"
    libFormat_t library "library"
    int subFormat "subFormat"
    void* libDescriptor "libDescriptor"

ctypedef cudaLibXtDesc_t cudaLibXtDesc 'cudaLibXtDesc'


###############################################################################
# Functions
###############################################################################

cdef cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftPlanMany(cufftHandle* plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, long long int* onembed, long long int ostride, long long int odist, cufftType type, long long int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftEstimateMany(int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftCreate(cufftHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int* n, int* inembed, int istride, int idist, int* onembed, int ostride, int odist, cufftType type, int batch, size_t* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetSize(cufftHandle handle, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleComplex* odata, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata, cufftDoubleComplex* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata, cufftDoubleReal* odata) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftDestroy(cufftHandle plan) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetVersion(int* version) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetProperty(libraryPropertyType type, int* value) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftSetPlanPropertyInt64(cufftHandle plan, cufftProperty property, const long long int inputValueInt) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftGetPlanPropertyInt64(cufftHandle plan, cufftProperty property, long long int* returnPtrValue) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftResetPlanProperty(cufftHandle plan, cufftProperty property) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs, int* whichGPUs) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc** descriptor, cufftXtSubFormat format) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer, cufftXtCopyType type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtFree(cudaLibXtDesc* descriptor) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtSetWorkArea(cufftHandle plan, void** workArea) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtQueryPlan(cufftHandle plan, void* queryStruct, cufftXtQueryType queryType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType, size_t sharedSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtGetSizeMany(cufftHandle plan, int rank, long long int* n, long long int* inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int* onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t* workSize, cudaDataType executiontype) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc* input, cudaLibXtDesc* output, int direction) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy, size_t* workSize) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMpAttachComm(cufftHandle plan, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtSetDistribution(cufftHandle plan, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftXtSetSubformatDefault(cufftHandle plan, cufftXtSubFormat subformat_forward, cufftXtSubFormat subformat_inverse) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMpCreateReshape(cufftReshapeHandle* handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMpAttachReshapeComm(cufftReshapeHandle handle, cufftMpCommType comm_type, void* comm_handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMpGetReshapeSize(cufftReshapeHandle handle, size_t* workspace_size) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult __cufftMpMakeReshape_11_2(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_input, const long long int* strides_output) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMpExecReshapeAsync(cufftReshapeHandle handle, void* data_out, const void* data_in, void* workspace, cudaStream_t stream) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult cufftMpDestroyReshape(cufftReshapeHandle handle) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
cdef cufftResult __cufftMpMakeReshape_11_4(cufftReshapeHandle handle, size_t element_size, int rank, const long long int* lower_input, const long long int* upper_input, const long long int* strides_input, const long long int* lower_output, const long long int* upper_output, const long long int* strides_output, void* comm_handle, cufftMpCommType comm_type) except?_CUFFTRESULT_INTERNAL_LOADING_ERROR nogil
