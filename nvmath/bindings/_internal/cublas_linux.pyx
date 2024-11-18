# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated across versions from 11.0.3 to 12.6.2. Do not modify it directly.

from libc.stdint cimport intptr_t

from .utils cimport get_cublas_dso_version_suffix

from .utils import FunctionNotFoundError, NotSupportedError


###############################################################################
# Extern
###############################################################################

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_cublas_init = False
cdef void* __cuDriverGetVersion = NULL

cdef void* __cublasCreate_v2 = NULL
cdef void* __cublasDestroy_v2 = NULL
cdef void* __cublasGetVersion_v2 = NULL
cdef void* __cublasGetProperty = NULL
cdef void* __cublasGetCudartVersion = NULL
cdef void* __cublasSetWorkspace_v2 = NULL
cdef void* __cublasSetStream_v2 = NULL
cdef void* __cublasGetStream_v2 = NULL
cdef void* __cublasGetPointerMode_v2 = NULL
cdef void* __cublasSetPointerMode_v2 = NULL
cdef void* __cublasGetAtomicsMode = NULL
cdef void* __cublasSetAtomicsMode = NULL
cdef void* __cublasGetMathMode = NULL
cdef void* __cublasSetMathMode = NULL
cdef void* __cublasLoggerConfigure = NULL
cdef void* __cublasSetLoggerCallback = NULL
cdef void* __cublasGetLoggerCallback = NULL
cdef void* __cublasSetVector = NULL
cdef void* __cublasGetVector = NULL
cdef void* __cublasSetMatrix = NULL
cdef void* __cublasGetMatrix = NULL
cdef void* __cublasSetVectorAsync = NULL
cdef void* __cublasGetVectorAsync = NULL
cdef void* __cublasSetMatrixAsync = NULL
cdef void* __cublasGetMatrixAsync = NULL
cdef void* __cublasNrm2Ex = NULL
cdef void* __cublasSnrm2_v2 = NULL
cdef void* __cublasDnrm2_v2 = NULL
cdef void* __cublasScnrm2_v2 = NULL
cdef void* __cublasDznrm2_v2 = NULL
cdef void* __cublasDotEx = NULL
cdef void* __cublasDotcEx = NULL
cdef void* __cublasSdot_v2 = NULL
cdef void* __cublasDdot_v2 = NULL
cdef void* __cublasCdotu_v2 = NULL
cdef void* __cublasCdotc_v2 = NULL
cdef void* __cublasZdotu_v2 = NULL
cdef void* __cublasZdotc_v2 = NULL
cdef void* __cublasScalEx = NULL
cdef void* __cublasSscal_v2 = NULL
cdef void* __cublasDscal_v2 = NULL
cdef void* __cublasCscal_v2 = NULL
cdef void* __cublasCsscal_v2 = NULL
cdef void* __cublasZscal_v2 = NULL
cdef void* __cublasZdscal_v2 = NULL
cdef void* __cublasAxpyEx = NULL
cdef void* __cublasSaxpy_v2 = NULL
cdef void* __cublasDaxpy_v2 = NULL
cdef void* __cublasCaxpy_v2 = NULL
cdef void* __cublasZaxpy_v2 = NULL
cdef void* __cublasCopyEx = NULL
cdef void* __cublasScopy_v2 = NULL
cdef void* __cublasDcopy_v2 = NULL
cdef void* __cublasCcopy_v2 = NULL
cdef void* __cublasZcopy_v2 = NULL
cdef void* __cublasSswap_v2 = NULL
cdef void* __cublasDswap_v2 = NULL
cdef void* __cublasCswap_v2 = NULL
cdef void* __cublasZswap_v2 = NULL
cdef void* __cublasSwapEx = NULL
cdef void* __cublasIsamax_v2 = NULL
cdef void* __cublasIdamax_v2 = NULL
cdef void* __cublasIcamax_v2 = NULL
cdef void* __cublasIzamax_v2 = NULL
cdef void* __cublasIamaxEx = NULL
cdef void* __cublasIsamin_v2 = NULL
cdef void* __cublasIdamin_v2 = NULL
cdef void* __cublasIcamin_v2 = NULL
cdef void* __cublasIzamin_v2 = NULL
cdef void* __cublasIaminEx = NULL
cdef void* __cublasAsumEx = NULL
cdef void* __cublasSasum_v2 = NULL
cdef void* __cublasDasum_v2 = NULL
cdef void* __cublasScasum_v2 = NULL
cdef void* __cublasDzasum_v2 = NULL
cdef void* __cublasSrot_v2 = NULL
cdef void* __cublasDrot_v2 = NULL
cdef void* __cublasCrot_v2 = NULL
cdef void* __cublasCsrot_v2 = NULL
cdef void* __cublasZrot_v2 = NULL
cdef void* __cublasZdrot_v2 = NULL
cdef void* __cublasRotEx = NULL
cdef void* __cublasSrotg_v2 = NULL
cdef void* __cublasDrotg_v2 = NULL
cdef void* __cublasCrotg_v2 = NULL
cdef void* __cublasZrotg_v2 = NULL
cdef void* __cublasRotgEx = NULL
cdef void* __cublasSrotm_v2 = NULL
cdef void* __cublasDrotm_v2 = NULL
cdef void* __cublasRotmEx = NULL
cdef void* __cublasSrotmg_v2 = NULL
cdef void* __cublasDrotmg_v2 = NULL
cdef void* __cublasRotmgEx = NULL
cdef void* __cublasSgemv_v2 = NULL
cdef void* __cublasDgemv_v2 = NULL
cdef void* __cublasCgemv_v2 = NULL
cdef void* __cublasZgemv_v2 = NULL
cdef void* __cublasSgbmv_v2 = NULL
cdef void* __cublasDgbmv_v2 = NULL
cdef void* __cublasCgbmv_v2 = NULL
cdef void* __cublasZgbmv_v2 = NULL
cdef void* __cublasStrmv_v2 = NULL
cdef void* __cublasDtrmv_v2 = NULL
cdef void* __cublasCtrmv_v2 = NULL
cdef void* __cublasZtrmv_v2 = NULL
cdef void* __cublasStbmv_v2 = NULL
cdef void* __cublasDtbmv_v2 = NULL
cdef void* __cublasCtbmv_v2 = NULL
cdef void* __cublasZtbmv_v2 = NULL
cdef void* __cublasStpmv_v2 = NULL
cdef void* __cublasDtpmv_v2 = NULL
cdef void* __cublasCtpmv_v2 = NULL
cdef void* __cublasZtpmv_v2 = NULL
cdef void* __cublasStrsv_v2 = NULL
cdef void* __cublasDtrsv_v2 = NULL
cdef void* __cublasCtrsv_v2 = NULL
cdef void* __cublasZtrsv_v2 = NULL
cdef void* __cublasStpsv_v2 = NULL
cdef void* __cublasDtpsv_v2 = NULL
cdef void* __cublasCtpsv_v2 = NULL
cdef void* __cublasZtpsv_v2 = NULL
cdef void* __cublasStbsv_v2 = NULL
cdef void* __cublasDtbsv_v2 = NULL
cdef void* __cublasCtbsv_v2 = NULL
cdef void* __cublasZtbsv_v2 = NULL
cdef void* __cublasSsymv_v2 = NULL
cdef void* __cublasDsymv_v2 = NULL
cdef void* __cublasCsymv_v2 = NULL
cdef void* __cublasZsymv_v2 = NULL
cdef void* __cublasChemv_v2 = NULL
cdef void* __cublasZhemv_v2 = NULL
cdef void* __cublasSsbmv_v2 = NULL
cdef void* __cublasDsbmv_v2 = NULL
cdef void* __cublasChbmv_v2 = NULL
cdef void* __cublasZhbmv_v2 = NULL
cdef void* __cublasSspmv_v2 = NULL
cdef void* __cublasDspmv_v2 = NULL
cdef void* __cublasChpmv_v2 = NULL
cdef void* __cublasZhpmv_v2 = NULL
cdef void* __cublasSger_v2 = NULL
cdef void* __cublasDger_v2 = NULL
cdef void* __cublasCgeru_v2 = NULL
cdef void* __cublasCgerc_v2 = NULL
cdef void* __cublasZgeru_v2 = NULL
cdef void* __cublasZgerc_v2 = NULL
cdef void* __cublasSsyr_v2 = NULL
cdef void* __cublasDsyr_v2 = NULL
cdef void* __cublasCsyr_v2 = NULL
cdef void* __cublasZsyr_v2 = NULL
cdef void* __cublasCher_v2 = NULL
cdef void* __cublasZher_v2 = NULL
cdef void* __cublasSspr_v2 = NULL
cdef void* __cublasDspr_v2 = NULL
cdef void* __cublasChpr_v2 = NULL
cdef void* __cublasZhpr_v2 = NULL
cdef void* __cublasSsyr2_v2 = NULL
cdef void* __cublasDsyr2_v2 = NULL
cdef void* __cublasCsyr2_v2 = NULL
cdef void* __cublasZsyr2_v2 = NULL
cdef void* __cublasCher2_v2 = NULL
cdef void* __cublasZher2_v2 = NULL
cdef void* __cublasSspr2_v2 = NULL
cdef void* __cublasDspr2_v2 = NULL
cdef void* __cublasChpr2_v2 = NULL
cdef void* __cublasZhpr2_v2 = NULL
cdef void* __cublasSgemm_v2 = NULL
cdef void* __cublasDgemm_v2 = NULL
cdef void* __cublasCgemm_v2 = NULL
cdef void* __cublasCgemm3m = NULL
cdef void* __cublasCgemm3mEx = NULL
cdef void* __cublasZgemm_v2 = NULL
cdef void* __cublasZgemm3m = NULL
cdef void* __cublasSgemmEx = NULL
cdef void* __cublasGemmEx = NULL
cdef void* __cublasCgemmEx = NULL
cdef void* __cublasUint8gemmBias = NULL
cdef void* __cublasSsyrk_v2 = NULL
cdef void* __cublasDsyrk_v2 = NULL
cdef void* __cublasCsyrk_v2 = NULL
cdef void* __cublasZsyrk_v2 = NULL
cdef void* __cublasCsyrkEx = NULL
cdef void* __cublasCsyrk3mEx = NULL
cdef void* __cublasCherk_v2 = NULL
cdef void* __cublasZherk_v2 = NULL
cdef void* __cublasCherkEx = NULL
cdef void* __cublasCherk3mEx = NULL
cdef void* __cublasSsyr2k_v2 = NULL
cdef void* __cublasDsyr2k_v2 = NULL
cdef void* __cublasCsyr2k_v2 = NULL
cdef void* __cublasZsyr2k_v2 = NULL
cdef void* __cublasCher2k_v2 = NULL
cdef void* __cublasZher2k_v2 = NULL
cdef void* __cublasSsyrkx = NULL
cdef void* __cublasDsyrkx = NULL
cdef void* __cublasCsyrkx = NULL
cdef void* __cublasZsyrkx = NULL
cdef void* __cublasCherkx = NULL
cdef void* __cublasZherkx = NULL
cdef void* __cublasSsymm_v2 = NULL
cdef void* __cublasDsymm_v2 = NULL
cdef void* __cublasCsymm_v2 = NULL
cdef void* __cublasZsymm_v2 = NULL
cdef void* __cublasChemm_v2 = NULL
cdef void* __cublasZhemm_v2 = NULL
cdef void* __cublasStrsm_v2 = NULL
cdef void* __cublasDtrsm_v2 = NULL
cdef void* __cublasCtrsm_v2 = NULL
cdef void* __cublasZtrsm_v2 = NULL
cdef void* __cublasStrmm_v2 = NULL
cdef void* __cublasDtrmm_v2 = NULL
cdef void* __cublasCtrmm_v2 = NULL
cdef void* __cublasZtrmm_v2 = NULL
cdef void* __cublasSgemmBatched = NULL
cdef void* __cublasDgemmBatched = NULL
cdef void* __cublasCgemmBatched = NULL
cdef void* __cublasCgemm3mBatched = NULL
cdef void* __cublasZgemmBatched = NULL
cdef void* __cublasGemmBatchedEx = NULL
cdef void* __cublasGemmStridedBatchedEx = NULL
cdef void* __cublasSgemmStridedBatched = NULL
cdef void* __cublasDgemmStridedBatched = NULL
cdef void* __cublasCgemmStridedBatched = NULL
cdef void* __cublasCgemm3mStridedBatched = NULL
cdef void* __cublasZgemmStridedBatched = NULL
cdef void* __cublasSgeam = NULL
cdef void* __cublasDgeam = NULL
cdef void* __cublasCgeam = NULL
cdef void* __cublasZgeam = NULL
cdef void* __cublasSgetrfBatched = NULL
cdef void* __cublasDgetrfBatched = NULL
cdef void* __cublasCgetrfBatched = NULL
cdef void* __cublasZgetrfBatched = NULL
cdef void* __cublasSgetriBatched = NULL
cdef void* __cublasDgetriBatched = NULL
cdef void* __cublasCgetriBatched = NULL
cdef void* __cublasZgetriBatched = NULL
cdef void* __cublasSgetrsBatched = NULL
cdef void* __cublasDgetrsBatched = NULL
cdef void* __cublasCgetrsBatched = NULL
cdef void* __cublasZgetrsBatched = NULL
cdef void* __cublasStrsmBatched = NULL
cdef void* __cublasDtrsmBatched = NULL
cdef void* __cublasCtrsmBatched = NULL
cdef void* __cublasZtrsmBatched = NULL
cdef void* __cublasSmatinvBatched = NULL
cdef void* __cublasDmatinvBatched = NULL
cdef void* __cublasCmatinvBatched = NULL
cdef void* __cublasZmatinvBatched = NULL
cdef void* __cublasSgeqrfBatched = NULL
cdef void* __cublasDgeqrfBatched = NULL
cdef void* __cublasCgeqrfBatched = NULL
cdef void* __cublasZgeqrfBatched = NULL
cdef void* __cublasSgelsBatched = NULL
cdef void* __cublasDgelsBatched = NULL
cdef void* __cublasCgelsBatched = NULL
cdef void* __cublasZgelsBatched = NULL
cdef void* __cublasSdgmm = NULL
cdef void* __cublasDdgmm = NULL
cdef void* __cublasCdgmm = NULL
cdef void* __cublasZdgmm = NULL
cdef void* __cublasStpttr = NULL
cdef void* __cublasDtpttr = NULL
cdef void* __cublasCtpttr = NULL
cdef void* __cublasZtpttr = NULL
cdef void* __cublasStrttp = NULL
cdef void* __cublasDtrttp = NULL
cdef void* __cublasCtrttp = NULL
cdef void* __cublasZtrttp = NULL
cdef void* __cublasGetSmCountTarget = NULL
cdef void* __cublasSetSmCountTarget = NULL
cdef void* __cublasGetStatusName = NULL
cdef void* __cublasGetStatusString = NULL
cdef void* __cublasSgemvBatched = NULL
cdef void* __cublasDgemvBatched = NULL
cdef void* __cublasCgemvBatched = NULL
cdef void* __cublasZgemvBatched = NULL
cdef void* __cublasSgemvStridedBatched = NULL
cdef void* __cublasDgemvStridedBatched = NULL
cdef void* __cublasCgemvStridedBatched = NULL
cdef void* __cublasZgemvStridedBatched = NULL
cdef void* __cublasSetVector_64 = NULL
cdef void* __cublasGetVector_64 = NULL
cdef void* __cublasSetMatrix_64 = NULL
cdef void* __cublasGetMatrix_64 = NULL
cdef void* __cublasSetVectorAsync_64 = NULL
cdef void* __cublasGetVectorAsync_64 = NULL
cdef void* __cublasSetMatrixAsync_64 = NULL
cdef void* __cublasGetMatrixAsync_64 = NULL
cdef void* __cublasNrm2Ex_64 = NULL
cdef void* __cublasSnrm2_v2_64 = NULL
cdef void* __cublasDnrm2_v2_64 = NULL
cdef void* __cublasScnrm2_v2_64 = NULL
cdef void* __cublasDznrm2_v2_64 = NULL
cdef void* __cublasDotEx_64 = NULL
cdef void* __cublasDotcEx_64 = NULL
cdef void* __cublasSdot_v2_64 = NULL
cdef void* __cublasDdot_v2_64 = NULL
cdef void* __cublasCdotu_v2_64 = NULL
cdef void* __cublasCdotc_v2_64 = NULL
cdef void* __cublasZdotu_v2_64 = NULL
cdef void* __cublasZdotc_v2_64 = NULL
cdef void* __cublasScalEx_64 = NULL
cdef void* __cublasSscal_v2_64 = NULL
cdef void* __cublasDscal_v2_64 = NULL
cdef void* __cublasCscal_v2_64 = NULL
cdef void* __cublasCsscal_v2_64 = NULL
cdef void* __cublasZscal_v2_64 = NULL
cdef void* __cublasZdscal_v2_64 = NULL
cdef void* __cublasAxpyEx_64 = NULL
cdef void* __cublasSaxpy_v2_64 = NULL
cdef void* __cublasDaxpy_v2_64 = NULL
cdef void* __cublasCaxpy_v2_64 = NULL
cdef void* __cublasZaxpy_v2_64 = NULL
cdef void* __cublasCopyEx_64 = NULL
cdef void* __cublasScopy_v2_64 = NULL
cdef void* __cublasDcopy_v2_64 = NULL
cdef void* __cublasCcopy_v2_64 = NULL
cdef void* __cublasZcopy_v2_64 = NULL
cdef void* __cublasSswap_v2_64 = NULL
cdef void* __cublasDswap_v2_64 = NULL
cdef void* __cublasCswap_v2_64 = NULL
cdef void* __cublasZswap_v2_64 = NULL
cdef void* __cublasSwapEx_64 = NULL
cdef void* __cublasIsamax_v2_64 = NULL
cdef void* __cublasIdamax_v2_64 = NULL
cdef void* __cublasIcamax_v2_64 = NULL
cdef void* __cublasIzamax_v2_64 = NULL
cdef void* __cublasIamaxEx_64 = NULL
cdef void* __cublasIsamin_v2_64 = NULL
cdef void* __cublasIdamin_v2_64 = NULL
cdef void* __cublasIcamin_v2_64 = NULL
cdef void* __cublasIzamin_v2_64 = NULL
cdef void* __cublasIaminEx_64 = NULL
cdef void* __cublasAsumEx_64 = NULL
cdef void* __cublasSasum_v2_64 = NULL
cdef void* __cublasDasum_v2_64 = NULL
cdef void* __cublasScasum_v2_64 = NULL
cdef void* __cublasDzasum_v2_64 = NULL
cdef void* __cublasSrot_v2_64 = NULL
cdef void* __cublasDrot_v2_64 = NULL
cdef void* __cublasCrot_v2_64 = NULL
cdef void* __cublasCsrot_v2_64 = NULL
cdef void* __cublasZrot_v2_64 = NULL
cdef void* __cublasZdrot_v2_64 = NULL
cdef void* __cublasRotEx_64 = NULL
cdef void* __cublasSrotm_v2_64 = NULL
cdef void* __cublasDrotm_v2_64 = NULL
cdef void* __cublasRotmEx_64 = NULL
cdef void* __cublasSgemv_v2_64 = NULL
cdef void* __cublasDgemv_v2_64 = NULL
cdef void* __cublasCgemv_v2_64 = NULL
cdef void* __cublasZgemv_v2_64 = NULL
cdef void* __cublasSgbmv_v2_64 = NULL
cdef void* __cublasDgbmv_v2_64 = NULL
cdef void* __cublasCgbmv_v2_64 = NULL
cdef void* __cublasZgbmv_v2_64 = NULL
cdef void* __cublasStrmv_v2_64 = NULL
cdef void* __cublasDtrmv_v2_64 = NULL
cdef void* __cublasCtrmv_v2_64 = NULL
cdef void* __cublasZtrmv_v2_64 = NULL
cdef void* __cublasStbmv_v2_64 = NULL
cdef void* __cublasDtbmv_v2_64 = NULL
cdef void* __cublasCtbmv_v2_64 = NULL
cdef void* __cublasZtbmv_v2_64 = NULL
cdef void* __cublasStpmv_v2_64 = NULL
cdef void* __cublasDtpmv_v2_64 = NULL
cdef void* __cublasCtpmv_v2_64 = NULL
cdef void* __cublasZtpmv_v2_64 = NULL
cdef void* __cublasStrsv_v2_64 = NULL
cdef void* __cublasDtrsv_v2_64 = NULL
cdef void* __cublasCtrsv_v2_64 = NULL
cdef void* __cublasZtrsv_v2_64 = NULL
cdef void* __cublasStpsv_v2_64 = NULL
cdef void* __cublasDtpsv_v2_64 = NULL
cdef void* __cublasCtpsv_v2_64 = NULL
cdef void* __cublasZtpsv_v2_64 = NULL
cdef void* __cublasStbsv_v2_64 = NULL
cdef void* __cublasDtbsv_v2_64 = NULL
cdef void* __cublasCtbsv_v2_64 = NULL
cdef void* __cublasZtbsv_v2_64 = NULL
cdef void* __cublasSsymv_v2_64 = NULL
cdef void* __cublasDsymv_v2_64 = NULL
cdef void* __cublasCsymv_v2_64 = NULL
cdef void* __cublasZsymv_v2_64 = NULL
cdef void* __cublasChemv_v2_64 = NULL
cdef void* __cublasZhemv_v2_64 = NULL
cdef void* __cublasSsbmv_v2_64 = NULL
cdef void* __cublasDsbmv_v2_64 = NULL
cdef void* __cublasChbmv_v2_64 = NULL
cdef void* __cublasZhbmv_v2_64 = NULL
cdef void* __cublasSspmv_v2_64 = NULL
cdef void* __cublasDspmv_v2_64 = NULL
cdef void* __cublasChpmv_v2_64 = NULL
cdef void* __cublasZhpmv_v2_64 = NULL
cdef void* __cublasSger_v2_64 = NULL
cdef void* __cublasDger_v2_64 = NULL
cdef void* __cublasCgeru_v2_64 = NULL
cdef void* __cublasCgerc_v2_64 = NULL
cdef void* __cublasZgeru_v2_64 = NULL
cdef void* __cublasZgerc_v2_64 = NULL
cdef void* __cublasSsyr_v2_64 = NULL
cdef void* __cublasDsyr_v2_64 = NULL
cdef void* __cublasCsyr_v2_64 = NULL
cdef void* __cublasZsyr_v2_64 = NULL
cdef void* __cublasCher_v2_64 = NULL
cdef void* __cublasZher_v2_64 = NULL
cdef void* __cublasSspr_v2_64 = NULL
cdef void* __cublasDspr_v2_64 = NULL
cdef void* __cublasChpr_v2_64 = NULL
cdef void* __cublasZhpr_v2_64 = NULL
cdef void* __cublasSsyr2_v2_64 = NULL
cdef void* __cublasDsyr2_v2_64 = NULL
cdef void* __cublasCsyr2_v2_64 = NULL
cdef void* __cublasZsyr2_v2_64 = NULL
cdef void* __cublasCher2_v2_64 = NULL
cdef void* __cublasZher2_v2_64 = NULL
cdef void* __cublasSspr2_v2_64 = NULL
cdef void* __cublasDspr2_v2_64 = NULL
cdef void* __cublasChpr2_v2_64 = NULL
cdef void* __cublasZhpr2_v2_64 = NULL
cdef void* __cublasSgemvBatched_64 = NULL
cdef void* __cublasDgemvBatched_64 = NULL
cdef void* __cublasCgemvBatched_64 = NULL
cdef void* __cublasZgemvBatched_64 = NULL
cdef void* __cublasSgemvStridedBatched_64 = NULL
cdef void* __cublasDgemvStridedBatched_64 = NULL
cdef void* __cublasCgemvStridedBatched_64 = NULL
cdef void* __cublasZgemvStridedBatched_64 = NULL
cdef void* __cublasSgemm_v2_64 = NULL
cdef void* __cublasDgemm_v2_64 = NULL
cdef void* __cublasCgemm_v2_64 = NULL
cdef void* __cublasCgemm3m_64 = NULL
cdef void* __cublasCgemm3mEx_64 = NULL
cdef void* __cublasZgemm_v2_64 = NULL
cdef void* __cublasZgemm3m_64 = NULL
cdef void* __cublasSgemmEx_64 = NULL
cdef void* __cublasGemmEx_64 = NULL
cdef void* __cublasCgemmEx_64 = NULL
cdef void* __cublasSsyrk_v2_64 = NULL
cdef void* __cublasDsyrk_v2_64 = NULL
cdef void* __cublasCsyrk_v2_64 = NULL
cdef void* __cublasZsyrk_v2_64 = NULL
cdef void* __cublasCsyrkEx_64 = NULL
cdef void* __cublasCsyrk3mEx_64 = NULL
cdef void* __cublasCherk_v2_64 = NULL
cdef void* __cublasZherk_v2_64 = NULL
cdef void* __cublasCherkEx_64 = NULL
cdef void* __cublasCherk3mEx_64 = NULL
cdef void* __cublasSsyr2k_v2_64 = NULL
cdef void* __cublasDsyr2k_v2_64 = NULL
cdef void* __cublasCsyr2k_v2_64 = NULL
cdef void* __cublasZsyr2k_v2_64 = NULL
cdef void* __cublasCher2k_v2_64 = NULL
cdef void* __cublasZher2k_v2_64 = NULL
cdef void* __cublasSsyrkx_64 = NULL
cdef void* __cublasDsyrkx_64 = NULL
cdef void* __cublasCsyrkx_64 = NULL
cdef void* __cublasZsyrkx_64 = NULL
cdef void* __cublasCherkx_64 = NULL
cdef void* __cublasZherkx_64 = NULL
cdef void* __cublasSsymm_v2_64 = NULL
cdef void* __cublasDsymm_v2_64 = NULL
cdef void* __cublasCsymm_v2_64 = NULL
cdef void* __cublasZsymm_v2_64 = NULL
cdef void* __cublasChemm_v2_64 = NULL
cdef void* __cublasZhemm_v2_64 = NULL
cdef void* __cublasStrsm_v2_64 = NULL
cdef void* __cublasDtrsm_v2_64 = NULL
cdef void* __cublasCtrsm_v2_64 = NULL
cdef void* __cublasZtrsm_v2_64 = NULL
cdef void* __cublasStrmm_v2_64 = NULL
cdef void* __cublasDtrmm_v2_64 = NULL
cdef void* __cublasCtrmm_v2_64 = NULL
cdef void* __cublasZtrmm_v2_64 = NULL
cdef void* __cublasSgemmBatched_64 = NULL
cdef void* __cublasDgemmBatched_64 = NULL
cdef void* __cublasCgemmBatched_64 = NULL
cdef void* __cublasCgemm3mBatched_64 = NULL
cdef void* __cublasZgemmBatched_64 = NULL
cdef void* __cublasSgemmStridedBatched_64 = NULL
cdef void* __cublasDgemmStridedBatched_64 = NULL
cdef void* __cublasCgemmStridedBatched_64 = NULL
cdef void* __cublasCgemm3mStridedBatched_64 = NULL
cdef void* __cublasZgemmStridedBatched_64 = NULL
cdef void* __cublasGemmBatchedEx_64 = NULL
cdef void* __cublasGemmStridedBatchedEx_64 = NULL
cdef void* __cublasSgeam_64 = NULL
cdef void* __cublasDgeam_64 = NULL
cdef void* __cublasCgeam_64 = NULL
cdef void* __cublasZgeam_64 = NULL
cdef void* __cublasStrsmBatched_64 = NULL
cdef void* __cublasDtrsmBatched_64 = NULL
cdef void* __cublasCtrsmBatched_64 = NULL
cdef void* __cublasZtrsmBatched_64 = NULL
cdef void* __cublasSdgmm_64 = NULL
cdef void* __cublasDdgmm_64 = NULL
cdef void* __cublasCdgmm_64 = NULL
cdef void* __cublasZdgmm_64 = NULL


cdef void* load_library(const int driver_ver) except* with gil:
    cdef void* handle
    for suffix in get_cublas_dso_version_suffix(driver_ver):
        so_name = "libcublas.so" + (f".{suffix}" if suffix else suffix)
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            break
    else:
        err_msg = dlerror()
        raise RuntimeError(f'Failed to dlopen libcublas ({err_msg.decode()})')
    return handle


cdef int _check_or_init_cublas() except -1 nogil:
    global __py_cublas_init
    if __py_cublas_init:
        return 0

    # Load driver to check version
    cdef void* handle = NULL
    handle = dlopen('libcuda.so.1', RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise NotSupportedError(f'CUDA driver is not found ({err_msg.decode()})')
    global __cuDriverGetVersion
    if __cuDriverGetVersion == NULL:
        __cuDriverGetVersion = dlsym(handle, "cuDriverGetVersion")
    if __cuDriverGetVersion == NULL:
        with gil:
            raise RuntimeError('something went wrong')
    cdef int err, driver_ver
    err = (<int (*)(int*) nogil>__cuDriverGetVersion)(&driver_ver)
    if err != 0:
        with gil:
            raise RuntimeError('something went wrong')
    #dlclose(handle)
    handle = NULL

    # Load function
    global __cublasCreate_v2
    __cublasCreate_v2 = dlsym(RTLD_DEFAULT, 'cublasCreate_v2')
    if __cublasCreate_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCreate_v2 = dlsym(handle, 'cublasCreate_v2')

    global __cublasDestroy_v2
    __cublasDestroy_v2 = dlsym(RTLD_DEFAULT, 'cublasDestroy_v2')
    if __cublasDestroy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDestroy_v2 = dlsym(handle, 'cublasDestroy_v2')

    global __cublasGetVersion_v2
    __cublasGetVersion_v2 = dlsym(RTLD_DEFAULT, 'cublasGetVersion_v2')
    if __cublasGetVersion_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetVersion_v2 = dlsym(handle, 'cublasGetVersion_v2')

    global __cublasGetProperty
    __cublasGetProperty = dlsym(RTLD_DEFAULT, 'cublasGetProperty')
    if __cublasGetProperty == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetProperty = dlsym(handle, 'cublasGetProperty')

    global __cublasGetCudartVersion
    __cublasGetCudartVersion = dlsym(RTLD_DEFAULT, 'cublasGetCudartVersion')
    if __cublasGetCudartVersion == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetCudartVersion = dlsym(handle, 'cublasGetCudartVersion')

    global __cublasSetWorkspace_v2
    __cublasSetWorkspace_v2 = dlsym(RTLD_DEFAULT, 'cublasSetWorkspace_v2')
    if __cublasSetWorkspace_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetWorkspace_v2 = dlsym(handle, 'cublasSetWorkspace_v2')

    global __cublasSetStream_v2
    __cublasSetStream_v2 = dlsym(RTLD_DEFAULT, 'cublasSetStream_v2')
    if __cublasSetStream_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetStream_v2 = dlsym(handle, 'cublasSetStream_v2')

    global __cublasGetStream_v2
    __cublasGetStream_v2 = dlsym(RTLD_DEFAULT, 'cublasGetStream_v2')
    if __cublasGetStream_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetStream_v2 = dlsym(handle, 'cublasGetStream_v2')

    global __cublasGetPointerMode_v2
    __cublasGetPointerMode_v2 = dlsym(RTLD_DEFAULT, 'cublasGetPointerMode_v2')
    if __cublasGetPointerMode_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetPointerMode_v2 = dlsym(handle, 'cublasGetPointerMode_v2')

    global __cublasSetPointerMode_v2
    __cublasSetPointerMode_v2 = dlsym(RTLD_DEFAULT, 'cublasSetPointerMode_v2')
    if __cublasSetPointerMode_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetPointerMode_v2 = dlsym(handle, 'cublasSetPointerMode_v2')

    global __cublasGetAtomicsMode
    __cublasGetAtomicsMode = dlsym(RTLD_DEFAULT, 'cublasGetAtomicsMode')
    if __cublasGetAtomicsMode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetAtomicsMode = dlsym(handle, 'cublasGetAtomicsMode')

    global __cublasSetAtomicsMode
    __cublasSetAtomicsMode = dlsym(RTLD_DEFAULT, 'cublasSetAtomicsMode')
    if __cublasSetAtomicsMode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetAtomicsMode = dlsym(handle, 'cublasSetAtomicsMode')

    global __cublasGetMathMode
    __cublasGetMathMode = dlsym(RTLD_DEFAULT, 'cublasGetMathMode')
    if __cublasGetMathMode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetMathMode = dlsym(handle, 'cublasGetMathMode')

    global __cublasSetMathMode
    __cublasSetMathMode = dlsym(RTLD_DEFAULT, 'cublasSetMathMode')
    if __cublasSetMathMode == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetMathMode = dlsym(handle, 'cublasSetMathMode')

    global __cublasLoggerConfigure
    __cublasLoggerConfigure = dlsym(RTLD_DEFAULT, 'cublasLoggerConfigure')
    if __cublasLoggerConfigure == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasLoggerConfigure = dlsym(handle, 'cublasLoggerConfigure')

    global __cublasSetLoggerCallback
    __cublasSetLoggerCallback = dlsym(RTLD_DEFAULT, 'cublasSetLoggerCallback')
    if __cublasSetLoggerCallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetLoggerCallback = dlsym(handle, 'cublasSetLoggerCallback')

    global __cublasGetLoggerCallback
    __cublasGetLoggerCallback = dlsym(RTLD_DEFAULT, 'cublasGetLoggerCallback')
    if __cublasGetLoggerCallback == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetLoggerCallback = dlsym(handle, 'cublasGetLoggerCallback')

    global __cublasSetVector
    __cublasSetVector = dlsym(RTLD_DEFAULT, 'cublasSetVector')
    if __cublasSetVector == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetVector = dlsym(handle, 'cublasSetVector')

    global __cublasGetVector
    __cublasGetVector = dlsym(RTLD_DEFAULT, 'cublasGetVector')
    if __cublasGetVector == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetVector = dlsym(handle, 'cublasGetVector')

    global __cublasSetMatrix
    __cublasSetMatrix = dlsym(RTLD_DEFAULT, 'cublasSetMatrix')
    if __cublasSetMatrix == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetMatrix = dlsym(handle, 'cublasSetMatrix')

    global __cublasGetMatrix
    __cublasGetMatrix = dlsym(RTLD_DEFAULT, 'cublasGetMatrix')
    if __cublasGetMatrix == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetMatrix = dlsym(handle, 'cublasGetMatrix')

    global __cublasSetVectorAsync
    __cublasSetVectorAsync = dlsym(RTLD_DEFAULT, 'cublasSetVectorAsync')
    if __cublasSetVectorAsync == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetVectorAsync = dlsym(handle, 'cublasSetVectorAsync')

    global __cublasGetVectorAsync
    __cublasGetVectorAsync = dlsym(RTLD_DEFAULT, 'cublasGetVectorAsync')
    if __cublasGetVectorAsync == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetVectorAsync = dlsym(handle, 'cublasGetVectorAsync')

    global __cublasSetMatrixAsync
    __cublasSetMatrixAsync = dlsym(RTLD_DEFAULT, 'cublasSetMatrixAsync')
    if __cublasSetMatrixAsync == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetMatrixAsync = dlsym(handle, 'cublasSetMatrixAsync')

    global __cublasGetMatrixAsync
    __cublasGetMatrixAsync = dlsym(RTLD_DEFAULT, 'cublasGetMatrixAsync')
    if __cublasGetMatrixAsync == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetMatrixAsync = dlsym(handle, 'cublasGetMatrixAsync')

    global __cublasNrm2Ex
    __cublasNrm2Ex = dlsym(RTLD_DEFAULT, 'cublasNrm2Ex')
    if __cublasNrm2Ex == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasNrm2Ex = dlsym(handle, 'cublasNrm2Ex')

    global __cublasSnrm2_v2
    __cublasSnrm2_v2 = dlsym(RTLD_DEFAULT, 'cublasSnrm2_v2')
    if __cublasSnrm2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSnrm2_v2 = dlsym(handle, 'cublasSnrm2_v2')

    global __cublasDnrm2_v2
    __cublasDnrm2_v2 = dlsym(RTLD_DEFAULT, 'cublasDnrm2_v2')
    if __cublasDnrm2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDnrm2_v2 = dlsym(handle, 'cublasDnrm2_v2')

    global __cublasScnrm2_v2
    __cublasScnrm2_v2 = dlsym(RTLD_DEFAULT, 'cublasScnrm2_v2')
    if __cublasScnrm2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScnrm2_v2 = dlsym(handle, 'cublasScnrm2_v2')

    global __cublasDznrm2_v2
    __cublasDznrm2_v2 = dlsym(RTLD_DEFAULT, 'cublasDznrm2_v2')
    if __cublasDznrm2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDznrm2_v2 = dlsym(handle, 'cublasDznrm2_v2')

    global __cublasDotEx
    __cublasDotEx = dlsym(RTLD_DEFAULT, 'cublasDotEx')
    if __cublasDotEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDotEx = dlsym(handle, 'cublasDotEx')

    global __cublasDotcEx
    __cublasDotcEx = dlsym(RTLD_DEFAULT, 'cublasDotcEx')
    if __cublasDotcEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDotcEx = dlsym(handle, 'cublasDotcEx')

    global __cublasSdot_v2
    __cublasSdot_v2 = dlsym(RTLD_DEFAULT, 'cublasSdot_v2')
    if __cublasSdot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSdot_v2 = dlsym(handle, 'cublasSdot_v2')

    global __cublasDdot_v2
    __cublasDdot_v2 = dlsym(RTLD_DEFAULT, 'cublasDdot_v2')
    if __cublasDdot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDdot_v2 = dlsym(handle, 'cublasDdot_v2')

    global __cublasCdotu_v2
    __cublasCdotu_v2 = dlsym(RTLD_DEFAULT, 'cublasCdotu_v2')
    if __cublasCdotu_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCdotu_v2 = dlsym(handle, 'cublasCdotu_v2')

    global __cublasCdotc_v2
    __cublasCdotc_v2 = dlsym(RTLD_DEFAULT, 'cublasCdotc_v2')
    if __cublasCdotc_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCdotc_v2 = dlsym(handle, 'cublasCdotc_v2')

    global __cublasZdotu_v2
    __cublasZdotu_v2 = dlsym(RTLD_DEFAULT, 'cublasZdotu_v2')
    if __cublasZdotu_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdotu_v2 = dlsym(handle, 'cublasZdotu_v2')

    global __cublasZdotc_v2
    __cublasZdotc_v2 = dlsym(RTLD_DEFAULT, 'cublasZdotc_v2')
    if __cublasZdotc_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdotc_v2 = dlsym(handle, 'cublasZdotc_v2')

    global __cublasScalEx
    __cublasScalEx = dlsym(RTLD_DEFAULT, 'cublasScalEx')
    if __cublasScalEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScalEx = dlsym(handle, 'cublasScalEx')

    global __cublasSscal_v2
    __cublasSscal_v2 = dlsym(RTLD_DEFAULT, 'cublasSscal_v2')
    if __cublasSscal_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSscal_v2 = dlsym(handle, 'cublasSscal_v2')

    global __cublasDscal_v2
    __cublasDscal_v2 = dlsym(RTLD_DEFAULT, 'cublasDscal_v2')
    if __cublasDscal_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDscal_v2 = dlsym(handle, 'cublasDscal_v2')

    global __cublasCscal_v2
    __cublasCscal_v2 = dlsym(RTLD_DEFAULT, 'cublasCscal_v2')
    if __cublasCscal_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCscal_v2 = dlsym(handle, 'cublasCscal_v2')

    global __cublasCsscal_v2
    __cublasCsscal_v2 = dlsym(RTLD_DEFAULT, 'cublasCsscal_v2')
    if __cublasCsscal_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsscal_v2 = dlsym(handle, 'cublasCsscal_v2')

    global __cublasZscal_v2
    __cublasZscal_v2 = dlsym(RTLD_DEFAULT, 'cublasZscal_v2')
    if __cublasZscal_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZscal_v2 = dlsym(handle, 'cublasZscal_v2')

    global __cublasZdscal_v2
    __cublasZdscal_v2 = dlsym(RTLD_DEFAULT, 'cublasZdscal_v2')
    if __cublasZdscal_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdscal_v2 = dlsym(handle, 'cublasZdscal_v2')

    global __cublasAxpyEx
    __cublasAxpyEx = dlsym(RTLD_DEFAULT, 'cublasAxpyEx')
    if __cublasAxpyEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasAxpyEx = dlsym(handle, 'cublasAxpyEx')

    global __cublasSaxpy_v2
    __cublasSaxpy_v2 = dlsym(RTLD_DEFAULT, 'cublasSaxpy_v2')
    if __cublasSaxpy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSaxpy_v2 = dlsym(handle, 'cublasSaxpy_v2')

    global __cublasDaxpy_v2
    __cublasDaxpy_v2 = dlsym(RTLD_DEFAULT, 'cublasDaxpy_v2')
    if __cublasDaxpy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDaxpy_v2 = dlsym(handle, 'cublasDaxpy_v2')

    global __cublasCaxpy_v2
    __cublasCaxpy_v2 = dlsym(RTLD_DEFAULT, 'cublasCaxpy_v2')
    if __cublasCaxpy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCaxpy_v2 = dlsym(handle, 'cublasCaxpy_v2')

    global __cublasZaxpy_v2
    __cublasZaxpy_v2 = dlsym(RTLD_DEFAULT, 'cublasZaxpy_v2')
    if __cublasZaxpy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZaxpy_v2 = dlsym(handle, 'cublasZaxpy_v2')

    global __cublasCopyEx
    __cublasCopyEx = dlsym(RTLD_DEFAULT, 'cublasCopyEx')
    if __cublasCopyEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCopyEx = dlsym(handle, 'cublasCopyEx')

    global __cublasScopy_v2
    __cublasScopy_v2 = dlsym(RTLD_DEFAULT, 'cublasScopy_v2')
    if __cublasScopy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScopy_v2 = dlsym(handle, 'cublasScopy_v2')

    global __cublasDcopy_v2
    __cublasDcopy_v2 = dlsym(RTLD_DEFAULT, 'cublasDcopy_v2')
    if __cublasDcopy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDcopy_v2 = dlsym(handle, 'cublasDcopy_v2')

    global __cublasCcopy_v2
    __cublasCcopy_v2 = dlsym(RTLD_DEFAULT, 'cublasCcopy_v2')
    if __cublasCcopy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCcopy_v2 = dlsym(handle, 'cublasCcopy_v2')

    global __cublasZcopy_v2
    __cublasZcopy_v2 = dlsym(RTLD_DEFAULT, 'cublasZcopy_v2')
    if __cublasZcopy_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZcopy_v2 = dlsym(handle, 'cublasZcopy_v2')

    global __cublasSswap_v2
    __cublasSswap_v2 = dlsym(RTLD_DEFAULT, 'cublasSswap_v2')
    if __cublasSswap_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSswap_v2 = dlsym(handle, 'cublasSswap_v2')

    global __cublasDswap_v2
    __cublasDswap_v2 = dlsym(RTLD_DEFAULT, 'cublasDswap_v2')
    if __cublasDswap_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDswap_v2 = dlsym(handle, 'cublasDswap_v2')

    global __cublasCswap_v2
    __cublasCswap_v2 = dlsym(RTLD_DEFAULT, 'cublasCswap_v2')
    if __cublasCswap_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCswap_v2 = dlsym(handle, 'cublasCswap_v2')

    global __cublasZswap_v2
    __cublasZswap_v2 = dlsym(RTLD_DEFAULT, 'cublasZswap_v2')
    if __cublasZswap_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZswap_v2 = dlsym(handle, 'cublasZswap_v2')

    global __cublasSwapEx
    __cublasSwapEx = dlsym(RTLD_DEFAULT, 'cublasSwapEx')
    if __cublasSwapEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSwapEx = dlsym(handle, 'cublasSwapEx')

    global __cublasIsamax_v2
    __cublasIsamax_v2 = dlsym(RTLD_DEFAULT, 'cublasIsamax_v2')
    if __cublasIsamax_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIsamax_v2 = dlsym(handle, 'cublasIsamax_v2')

    global __cublasIdamax_v2
    __cublasIdamax_v2 = dlsym(RTLD_DEFAULT, 'cublasIdamax_v2')
    if __cublasIdamax_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIdamax_v2 = dlsym(handle, 'cublasIdamax_v2')

    global __cublasIcamax_v2
    __cublasIcamax_v2 = dlsym(RTLD_DEFAULT, 'cublasIcamax_v2')
    if __cublasIcamax_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIcamax_v2 = dlsym(handle, 'cublasIcamax_v2')

    global __cublasIzamax_v2
    __cublasIzamax_v2 = dlsym(RTLD_DEFAULT, 'cublasIzamax_v2')
    if __cublasIzamax_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIzamax_v2 = dlsym(handle, 'cublasIzamax_v2')

    global __cublasIamaxEx
    __cublasIamaxEx = dlsym(RTLD_DEFAULT, 'cublasIamaxEx')
    if __cublasIamaxEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIamaxEx = dlsym(handle, 'cublasIamaxEx')

    global __cublasIsamin_v2
    __cublasIsamin_v2 = dlsym(RTLD_DEFAULT, 'cublasIsamin_v2')
    if __cublasIsamin_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIsamin_v2 = dlsym(handle, 'cublasIsamin_v2')

    global __cublasIdamin_v2
    __cublasIdamin_v2 = dlsym(RTLD_DEFAULT, 'cublasIdamin_v2')
    if __cublasIdamin_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIdamin_v2 = dlsym(handle, 'cublasIdamin_v2')

    global __cublasIcamin_v2
    __cublasIcamin_v2 = dlsym(RTLD_DEFAULT, 'cublasIcamin_v2')
    if __cublasIcamin_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIcamin_v2 = dlsym(handle, 'cublasIcamin_v2')

    global __cublasIzamin_v2
    __cublasIzamin_v2 = dlsym(RTLD_DEFAULT, 'cublasIzamin_v2')
    if __cublasIzamin_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIzamin_v2 = dlsym(handle, 'cublasIzamin_v2')

    global __cublasIaminEx
    __cublasIaminEx = dlsym(RTLD_DEFAULT, 'cublasIaminEx')
    if __cublasIaminEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIaminEx = dlsym(handle, 'cublasIaminEx')

    global __cublasAsumEx
    __cublasAsumEx = dlsym(RTLD_DEFAULT, 'cublasAsumEx')
    if __cublasAsumEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasAsumEx = dlsym(handle, 'cublasAsumEx')

    global __cublasSasum_v2
    __cublasSasum_v2 = dlsym(RTLD_DEFAULT, 'cublasSasum_v2')
    if __cublasSasum_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSasum_v2 = dlsym(handle, 'cublasSasum_v2')

    global __cublasDasum_v2
    __cublasDasum_v2 = dlsym(RTLD_DEFAULT, 'cublasDasum_v2')
    if __cublasDasum_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDasum_v2 = dlsym(handle, 'cublasDasum_v2')

    global __cublasScasum_v2
    __cublasScasum_v2 = dlsym(RTLD_DEFAULT, 'cublasScasum_v2')
    if __cublasScasum_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScasum_v2 = dlsym(handle, 'cublasScasum_v2')

    global __cublasDzasum_v2
    __cublasDzasum_v2 = dlsym(RTLD_DEFAULT, 'cublasDzasum_v2')
    if __cublasDzasum_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDzasum_v2 = dlsym(handle, 'cublasDzasum_v2')

    global __cublasSrot_v2
    __cublasSrot_v2 = dlsym(RTLD_DEFAULT, 'cublasSrot_v2')
    if __cublasSrot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSrot_v2 = dlsym(handle, 'cublasSrot_v2')

    global __cublasDrot_v2
    __cublasDrot_v2 = dlsym(RTLD_DEFAULT, 'cublasDrot_v2')
    if __cublasDrot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDrot_v2 = dlsym(handle, 'cublasDrot_v2')

    global __cublasCrot_v2
    __cublasCrot_v2 = dlsym(RTLD_DEFAULT, 'cublasCrot_v2')
    if __cublasCrot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCrot_v2 = dlsym(handle, 'cublasCrot_v2')

    global __cublasCsrot_v2
    __cublasCsrot_v2 = dlsym(RTLD_DEFAULT, 'cublasCsrot_v2')
    if __cublasCsrot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsrot_v2 = dlsym(handle, 'cublasCsrot_v2')

    global __cublasZrot_v2
    __cublasZrot_v2 = dlsym(RTLD_DEFAULT, 'cublasZrot_v2')
    if __cublasZrot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZrot_v2 = dlsym(handle, 'cublasZrot_v2')

    global __cublasZdrot_v2
    __cublasZdrot_v2 = dlsym(RTLD_DEFAULT, 'cublasZdrot_v2')
    if __cublasZdrot_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdrot_v2 = dlsym(handle, 'cublasZdrot_v2')

    global __cublasRotEx
    __cublasRotEx = dlsym(RTLD_DEFAULT, 'cublasRotEx')
    if __cublasRotEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasRotEx = dlsym(handle, 'cublasRotEx')

    global __cublasSrotg_v2
    __cublasSrotg_v2 = dlsym(RTLD_DEFAULT, 'cublasSrotg_v2')
    if __cublasSrotg_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSrotg_v2 = dlsym(handle, 'cublasSrotg_v2')

    global __cublasDrotg_v2
    __cublasDrotg_v2 = dlsym(RTLD_DEFAULT, 'cublasDrotg_v2')
    if __cublasDrotg_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDrotg_v2 = dlsym(handle, 'cublasDrotg_v2')

    global __cublasCrotg_v2
    __cublasCrotg_v2 = dlsym(RTLD_DEFAULT, 'cublasCrotg_v2')
    if __cublasCrotg_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCrotg_v2 = dlsym(handle, 'cublasCrotg_v2')

    global __cublasZrotg_v2
    __cublasZrotg_v2 = dlsym(RTLD_DEFAULT, 'cublasZrotg_v2')
    if __cublasZrotg_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZrotg_v2 = dlsym(handle, 'cublasZrotg_v2')

    global __cublasRotgEx
    __cublasRotgEx = dlsym(RTLD_DEFAULT, 'cublasRotgEx')
    if __cublasRotgEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasRotgEx = dlsym(handle, 'cublasRotgEx')

    global __cublasSrotm_v2
    __cublasSrotm_v2 = dlsym(RTLD_DEFAULT, 'cublasSrotm_v2')
    if __cublasSrotm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSrotm_v2 = dlsym(handle, 'cublasSrotm_v2')

    global __cublasDrotm_v2
    __cublasDrotm_v2 = dlsym(RTLD_DEFAULT, 'cublasDrotm_v2')
    if __cublasDrotm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDrotm_v2 = dlsym(handle, 'cublasDrotm_v2')

    global __cublasRotmEx
    __cublasRotmEx = dlsym(RTLD_DEFAULT, 'cublasRotmEx')
    if __cublasRotmEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasRotmEx = dlsym(handle, 'cublasRotmEx')

    global __cublasSrotmg_v2
    __cublasSrotmg_v2 = dlsym(RTLD_DEFAULT, 'cublasSrotmg_v2')
    if __cublasSrotmg_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSrotmg_v2 = dlsym(handle, 'cublasSrotmg_v2')

    global __cublasDrotmg_v2
    __cublasDrotmg_v2 = dlsym(RTLD_DEFAULT, 'cublasDrotmg_v2')
    if __cublasDrotmg_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDrotmg_v2 = dlsym(handle, 'cublasDrotmg_v2')

    global __cublasRotmgEx
    __cublasRotmgEx = dlsym(RTLD_DEFAULT, 'cublasRotmgEx')
    if __cublasRotmgEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasRotmgEx = dlsym(handle, 'cublasRotmgEx')

    global __cublasSgemv_v2
    __cublasSgemv_v2 = dlsym(RTLD_DEFAULT, 'cublasSgemv_v2')
    if __cublasSgemv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemv_v2 = dlsym(handle, 'cublasSgemv_v2')

    global __cublasDgemv_v2
    __cublasDgemv_v2 = dlsym(RTLD_DEFAULT, 'cublasDgemv_v2')
    if __cublasDgemv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemv_v2 = dlsym(handle, 'cublasDgemv_v2')

    global __cublasCgemv_v2
    __cublasCgemv_v2 = dlsym(RTLD_DEFAULT, 'cublasCgemv_v2')
    if __cublasCgemv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemv_v2 = dlsym(handle, 'cublasCgemv_v2')

    global __cublasZgemv_v2
    __cublasZgemv_v2 = dlsym(RTLD_DEFAULT, 'cublasZgemv_v2')
    if __cublasZgemv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemv_v2 = dlsym(handle, 'cublasZgemv_v2')

    global __cublasSgbmv_v2
    __cublasSgbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasSgbmv_v2')
    if __cublasSgbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgbmv_v2 = dlsym(handle, 'cublasSgbmv_v2')

    global __cublasDgbmv_v2
    __cublasDgbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasDgbmv_v2')
    if __cublasDgbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgbmv_v2 = dlsym(handle, 'cublasDgbmv_v2')

    global __cublasCgbmv_v2
    __cublasCgbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasCgbmv_v2')
    if __cublasCgbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgbmv_v2 = dlsym(handle, 'cublasCgbmv_v2')

    global __cublasZgbmv_v2
    __cublasZgbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasZgbmv_v2')
    if __cublasZgbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgbmv_v2 = dlsym(handle, 'cublasZgbmv_v2')

    global __cublasStrmv_v2
    __cublasStrmv_v2 = dlsym(RTLD_DEFAULT, 'cublasStrmv_v2')
    if __cublasStrmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrmv_v2 = dlsym(handle, 'cublasStrmv_v2')

    global __cublasDtrmv_v2
    __cublasDtrmv_v2 = dlsym(RTLD_DEFAULT, 'cublasDtrmv_v2')
    if __cublasDtrmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrmv_v2 = dlsym(handle, 'cublasDtrmv_v2')

    global __cublasCtrmv_v2
    __cublasCtrmv_v2 = dlsym(RTLD_DEFAULT, 'cublasCtrmv_v2')
    if __cublasCtrmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrmv_v2 = dlsym(handle, 'cublasCtrmv_v2')

    global __cublasZtrmv_v2
    __cublasZtrmv_v2 = dlsym(RTLD_DEFAULT, 'cublasZtrmv_v2')
    if __cublasZtrmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrmv_v2 = dlsym(handle, 'cublasZtrmv_v2')

    global __cublasStbmv_v2
    __cublasStbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasStbmv_v2')
    if __cublasStbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStbmv_v2 = dlsym(handle, 'cublasStbmv_v2')

    global __cublasDtbmv_v2
    __cublasDtbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasDtbmv_v2')
    if __cublasDtbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtbmv_v2 = dlsym(handle, 'cublasDtbmv_v2')

    global __cublasCtbmv_v2
    __cublasCtbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasCtbmv_v2')
    if __cublasCtbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtbmv_v2 = dlsym(handle, 'cublasCtbmv_v2')

    global __cublasZtbmv_v2
    __cublasZtbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasZtbmv_v2')
    if __cublasZtbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtbmv_v2 = dlsym(handle, 'cublasZtbmv_v2')

    global __cublasStpmv_v2
    __cublasStpmv_v2 = dlsym(RTLD_DEFAULT, 'cublasStpmv_v2')
    if __cublasStpmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStpmv_v2 = dlsym(handle, 'cublasStpmv_v2')

    global __cublasDtpmv_v2
    __cublasDtpmv_v2 = dlsym(RTLD_DEFAULT, 'cublasDtpmv_v2')
    if __cublasDtpmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtpmv_v2 = dlsym(handle, 'cublasDtpmv_v2')

    global __cublasCtpmv_v2
    __cublasCtpmv_v2 = dlsym(RTLD_DEFAULT, 'cublasCtpmv_v2')
    if __cublasCtpmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtpmv_v2 = dlsym(handle, 'cublasCtpmv_v2')

    global __cublasZtpmv_v2
    __cublasZtpmv_v2 = dlsym(RTLD_DEFAULT, 'cublasZtpmv_v2')
    if __cublasZtpmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtpmv_v2 = dlsym(handle, 'cublasZtpmv_v2')

    global __cublasStrsv_v2
    __cublasStrsv_v2 = dlsym(RTLD_DEFAULT, 'cublasStrsv_v2')
    if __cublasStrsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrsv_v2 = dlsym(handle, 'cublasStrsv_v2')

    global __cublasDtrsv_v2
    __cublasDtrsv_v2 = dlsym(RTLD_DEFAULT, 'cublasDtrsv_v2')
    if __cublasDtrsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrsv_v2 = dlsym(handle, 'cublasDtrsv_v2')

    global __cublasCtrsv_v2
    __cublasCtrsv_v2 = dlsym(RTLD_DEFAULT, 'cublasCtrsv_v2')
    if __cublasCtrsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrsv_v2 = dlsym(handle, 'cublasCtrsv_v2')

    global __cublasZtrsv_v2
    __cublasZtrsv_v2 = dlsym(RTLD_DEFAULT, 'cublasZtrsv_v2')
    if __cublasZtrsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrsv_v2 = dlsym(handle, 'cublasZtrsv_v2')

    global __cublasStpsv_v2
    __cublasStpsv_v2 = dlsym(RTLD_DEFAULT, 'cublasStpsv_v2')
    if __cublasStpsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStpsv_v2 = dlsym(handle, 'cublasStpsv_v2')

    global __cublasDtpsv_v2
    __cublasDtpsv_v2 = dlsym(RTLD_DEFAULT, 'cublasDtpsv_v2')
    if __cublasDtpsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtpsv_v2 = dlsym(handle, 'cublasDtpsv_v2')

    global __cublasCtpsv_v2
    __cublasCtpsv_v2 = dlsym(RTLD_DEFAULT, 'cublasCtpsv_v2')
    if __cublasCtpsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtpsv_v2 = dlsym(handle, 'cublasCtpsv_v2')

    global __cublasZtpsv_v2
    __cublasZtpsv_v2 = dlsym(RTLD_DEFAULT, 'cublasZtpsv_v2')
    if __cublasZtpsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtpsv_v2 = dlsym(handle, 'cublasZtpsv_v2')

    global __cublasStbsv_v2
    __cublasStbsv_v2 = dlsym(RTLD_DEFAULT, 'cublasStbsv_v2')
    if __cublasStbsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStbsv_v2 = dlsym(handle, 'cublasStbsv_v2')

    global __cublasDtbsv_v2
    __cublasDtbsv_v2 = dlsym(RTLD_DEFAULT, 'cublasDtbsv_v2')
    if __cublasDtbsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtbsv_v2 = dlsym(handle, 'cublasDtbsv_v2')

    global __cublasCtbsv_v2
    __cublasCtbsv_v2 = dlsym(RTLD_DEFAULT, 'cublasCtbsv_v2')
    if __cublasCtbsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtbsv_v2 = dlsym(handle, 'cublasCtbsv_v2')

    global __cublasZtbsv_v2
    __cublasZtbsv_v2 = dlsym(RTLD_DEFAULT, 'cublasZtbsv_v2')
    if __cublasZtbsv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtbsv_v2 = dlsym(handle, 'cublasZtbsv_v2')

    global __cublasSsymv_v2
    __cublasSsymv_v2 = dlsym(RTLD_DEFAULT, 'cublasSsymv_v2')
    if __cublasSsymv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsymv_v2 = dlsym(handle, 'cublasSsymv_v2')

    global __cublasDsymv_v2
    __cublasDsymv_v2 = dlsym(RTLD_DEFAULT, 'cublasDsymv_v2')
    if __cublasDsymv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsymv_v2 = dlsym(handle, 'cublasDsymv_v2')

    global __cublasCsymv_v2
    __cublasCsymv_v2 = dlsym(RTLD_DEFAULT, 'cublasCsymv_v2')
    if __cublasCsymv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsymv_v2 = dlsym(handle, 'cublasCsymv_v2')

    global __cublasZsymv_v2
    __cublasZsymv_v2 = dlsym(RTLD_DEFAULT, 'cublasZsymv_v2')
    if __cublasZsymv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsymv_v2 = dlsym(handle, 'cublasZsymv_v2')

    global __cublasChemv_v2
    __cublasChemv_v2 = dlsym(RTLD_DEFAULT, 'cublasChemv_v2')
    if __cublasChemv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChemv_v2 = dlsym(handle, 'cublasChemv_v2')

    global __cublasZhemv_v2
    __cublasZhemv_v2 = dlsym(RTLD_DEFAULT, 'cublasZhemv_v2')
    if __cublasZhemv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhemv_v2 = dlsym(handle, 'cublasZhemv_v2')

    global __cublasSsbmv_v2
    __cublasSsbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasSsbmv_v2')
    if __cublasSsbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsbmv_v2 = dlsym(handle, 'cublasSsbmv_v2')

    global __cublasDsbmv_v2
    __cublasDsbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasDsbmv_v2')
    if __cublasDsbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsbmv_v2 = dlsym(handle, 'cublasDsbmv_v2')

    global __cublasChbmv_v2
    __cublasChbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasChbmv_v2')
    if __cublasChbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChbmv_v2 = dlsym(handle, 'cublasChbmv_v2')

    global __cublasZhbmv_v2
    __cublasZhbmv_v2 = dlsym(RTLD_DEFAULT, 'cublasZhbmv_v2')
    if __cublasZhbmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhbmv_v2 = dlsym(handle, 'cublasZhbmv_v2')

    global __cublasSspmv_v2
    __cublasSspmv_v2 = dlsym(RTLD_DEFAULT, 'cublasSspmv_v2')
    if __cublasSspmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSspmv_v2 = dlsym(handle, 'cublasSspmv_v2')

    global __cublasDspmv_v2
    __cublasDspmv_v2 = dlsym(RTLD_DEFAULT, 'cublasDspmv_v2')
    if __cublasDspmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDspmv_v2 = dlsym(handle, 'cublasDspmv_v2')

    global __cublasChpmv_v2
    __cublasChpmv_v2 = dlsym(RTLD_DEFAULT, 'cublasChpmv_v2')
    if __cublasChpmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChpmv_v2 = dlsym(handle, 'cublasChpmv_v2')

    global __cublasZhpmv_v2
    __cublasZhpmv_v2 = dlsym(RTLD_DEFAULT, 'cublasZhpmv_v2')
    if __cublasZhpmv_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhpmv_v2 = dlsym(handle, 'cublasZhpmv_v2')

    global __cublasSger_v2
    __cublasSger_v2 = dlsym(RTLD_DEFAULT, 'cublasSger_v2')
    if __cublasSger_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSger_v2 = dlsym(handle, 'cublasSger_v2')

    global __cublasDger_v2
    __cublasDger_v2 = dlsym(RTLD_DEFAULT, 'cublasDger_v2')
    if __cublasDger_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDger_v2 = dlsym(handle, 'cublasDger_v2')

    global __cublasCgeru_v2
    __cublasCgeru_v2 = dlsym(RTLD_DEFAULT, 'cublasCgeru_v2')
    if __cublasCgeru_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgeru_v2 = dlsym(handle, 'cublasCgeru_v2')

    global __cublasCgerc_v2
    __cublasCgerc_v2 = dlsym(RTLD_DEFAULT, 'cublasCgerc_v2')
    if __cublasCgerc_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgerc_v2 = dlsym(handle, 'cublasCgerc_v2')

    global __cublasZgeru_v2
    __cublasZgeru_v2 = dlsym(RTLD_DEFAULT, 'cublasZgeru_v2')
    if __cublasZgeru_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgeru_v2 = dlsym(handle, 'cublasZgeru_v2')

    global __cublasZgerc_v2
    __cublasZgerc_v2 = dlsym(RTLD_DEFAULT, 'cublasZgerc_v2')
    if __cublasZgerc_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgerc_v2 = dlsym(handle, 'cublasZgerc_v2')

    global __cublasSsyr_v2
    __cublasSsyr_v2 = dlsym(RTLD_DEFAULT, 'cublasSsyr_v2')
    if __cublasSsyr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyr_v2 = dlsym(handle, 'cublasSsyr_v2')

    global __cublasDsyr_v2
    __cublasDsyr_v2 = dlsym(RTLD_DEFAULT, 'cublasDsyr_v2')
    if __cublasDsyr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyr_v2 = dlsym(handle, 'cublasDsyr_v2')

    global __cublasCsyr_v2
    __cublasCsyr_v2 = dlsym(RTLD_DEFAULT, 'cublasCsyr_v2')
    if __cublasCsyr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyr_v2 = dlsym(handle, 'cublasCsyr_v2')

    global __cublasZsyr_v2
    __cublasZsyr_v2 = dlsym(RTLD_DEFAULT, 'cublasZsyr_v2')
    if __cublasZsyr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyr_v2 = dlsym(handle, 'cublasZsyr_v2')

    global __cublasCher_v2
    __cublasCher_v2 = dlsym(RTLD_DEFAULT, 'cublasCher_v2')
    if __cublasCher_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCher_v2 = dlsym(handle, 'cublasCher_v2')

    global __cublasZher_v2
    __cublasZher_v2 = dlsym(RTLD_DEFAULT, 'cublasZher_v2')
    if __cublasZher_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZher_v2 = dlsym(handle, 'cublasZher_v2')

    global __cublasSspr_v2
    __cublasSspr_v2 = dlsym(RTLD_DEFAULT, 'cublasSspr_v2')
    if __cublasSspr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSspr_v2 = dlsym(handle, 'cublasSspr_v2')

    global __cublasDspr_v2
    __cublasDspr_v2 = dlsym(RTLD_DEFAULT, 'cublasDspr_v2')
    if __cublasDspr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDspr_v2 = dlsym(handle, 'cublasDspr_v2')

    global __cublasChpr_v2
    __cublasChpr_v2 = dlsym(RTLD_DEFAULT, 'cublasChpr_v2')
    if __cublasChpr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChpr_v2 = dlsym(handle, 'cublasChpr_v2')

    global __cublasZhpr_v2
    __cublasZhpr_v2 = dlsym(RTLD_DEFAULT, 'cublasZhpr_v2')
    if __cublasZhpr_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhpr_v2 = dlsym(handle, 'cublasZhpr_v2')

    global __cublasSsyr2_v2
    __cublasSsyr2_v2 = dlsym(RTLD_DEFAULT, 'cublasSsyr2_v2')
    if __cublasSsyr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyr2_v2 = dlsym(handle, 'cublasSsyr2_v2')

    global __cublasDsyr2_v2
    __cublasDsyr2_v2 = dlsym(RTLD_DEFAULT, 'cublasDsyr2_v2')
    if __cublasDsyr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyr2_v2 = dlsym(handle, 'cublasDsyr2_v2')

    global __cublasCsyr2_v2
    __cublasCsyr2_v2 = dlsym(RTLD_DEFAULT, 'cublasCsyr2_v2')
    if __cublasCsyr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyr2_v2 = dlsym(handle, 'cublasCsyr2_v2')

    global __cublasZsyr2_v2
    __cublasZsyr2_v2 = dlsym(RTLD_DEFAULT, 'cublasZsyr2_v2')
    if __cublasZsyr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyr2_v2 = dlsym(handle, 'cublasZsyr2_v2')

    global __cublasCher2_v2
    __cublasCher2_v2 = dlsym(RTLD_DEFAULT, 'cublasCher2_v2')
    if __cublasCher2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCher2_v2 = dlsym(handle, 'cublasCher2_v2')

    global __cublasZher2_v2
    __cublasZher2_v2 = dlsym(RTLD_DEFAULT, 'cublasZher2_v2')
    if __cublasZher2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZher2_v2 = dlsym(handle, 'cublasZher2_v2')

    global __cublasSspr2_v2
    __cublasSspr2_v2 = dlsym(RTLD_DEFAULT, 'cublasSspr2_v2')
    if __cublasSspr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSspr2_v2 = dlsym(handle, 'cublasSspr2_v2')

    global __cublasDspr2_v2
    __cublasDspr2_v2 = dlsym(RTLD_DEFAULT, 'cublasDspr2_v2')
    if __cublasDspr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDspr2_v2 = dlsym(handle, 'cublasDspr2_v2')

    global __cublasChpr2_v2
    __cublasChpr2_v2 = dlsym(RTLD_DEFAULT, 'cublasChpr2_v2')
    if __cublasChpr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChpr2_v2 = dlsym(handle, 'cublasChpr2_v2')

    global __cublasZhpr2_v2
    __cublasZhpr2_v2 = dlsym(RTLD_DEFAULT, 'cublasZhpr2_v2')
    if __cublasZhpr2_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhpr2_v2 = dlsym(handle, 'cublasZhpr2_v2')

    global __cublasSgemm_v2
    __cublasSgemm_v2 = dlsym(RTLD_DEFAULT, 'cublasSgemm_v2')
    if __cublasSgemm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemm_v2 = dlsym(handle, 'cublasSgemm_v2')

    global __cublasDgemm_v2
    __cublasDgemm_v2 = dlsym(RTLD_DEFAULT, 'cublasDgemm_v2')
    if __cublasDgemm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemm_v2 = dlsym(handle, 'cublasDgemm_v2')

    global __cublasCgemm_v2
    __cublasCgemm_v2 = dlsym(RTLD_DEFAULT, 'cublasCgemm_v2')
    if __cublasCgemm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm_v2 = dlsym(handle, 'cublasCgemm_v2')

    global __cublasCgemm3m
    __cublasCgemm3m = dlsym(RTLD_DEFAULT, 'cublasCgemm3m')
    if __cublasCgemm3m == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3m = dlsym(handle, 'cublasCgemm3m')

    global __cublasCgemm3mEx
    __cublasCgemm3mEx = dlsym(RTLD_DEFAULT, 'cublasCgemm3mEx')
    if __cublasCgemm3mEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3mEx = dlsym(handle, 'cublasCgemm3mEx')

    global __cublasZgemm_v2
    __cublasZgemm_v2 = dlsym(RTLD_DEFAULT, 'cublasZgemm_v2')
    if __cublasZgemm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemm_v2 = dlsym(handle, 'cublasZgemm_v2')

    global __cublasZgemm3m
    __cublasZgemm3m = dlsym(RTLD_DEFAULT, 'cublasZgemm3m')
    if __cublasZgemm3m == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemm3m = dlsym(handle, 'cublasZgemm3m')

    global __cublasSgemmEx
    __cublasSgemmEx = dlsym(RTLD_DEFAULT, 'cublasSgemmEx')
    if __cublasSgemmEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemmEx = dlsym(handle, 'cublasSgemmEx')

    global __cublasGemmEx
    __cublasGemmEx = dlsym(RTLD_DEFAULT, 'cublasGemmEx')
    if __cublasGemmEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGemmEx = dlsym(handle, 'cublasGemmEx')

    global __cublasCgemmEx
    __cublasCgemmEx = dlsym(RTLD_DEFAULT, 'cublasCgemmEx')
    if __cublasCgemmEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemmEx = dlsym(handle, 'cublasCgemmEx')

    global __cublasUint8gemmBias
    __cublasUint8gemmBias = dlsym(RTLD_DEFAULT, 'cublasUint8gemmBias')
    if __cublasUint8gemmBias == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasUint8gemmBias = dlsym(handle, 'cublasUint8gemmBias')

    global __cublasSsyrk_v2
    __cublasSsyrk_v2 = dlsym(RTLD_DEFAULT, 'cublasSsyrk_v2')
    if __cublasSsyrk_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyrk_v2 = dlsym(handle, 'cublasSsyrk_v2')

    global __cublasDsyrk_v2
    __cublasDsyrk_v2 = dlsym(RTLD_DEFAULT, 'cublasDsyrk_v2')
    if __cublasDsyrk_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyrk_v2 = dlsym(handle, 'cublasDsyrk_v2')

    global __cublasCsyrk_v2
    __cublasCsyrk_v2 = dlsym(RTLD_DEFAULT, 'cublasCsyrk_v2')
    if __cublasCsyrk_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrk_v2 = dlsym(handle, 'cublasCsyrk_v2')

    global __cublasZsyrk_v2
    __cublasZsyrk_v2 = dlsym(RTLD_DEFAULT, 'cublasZsyrk_v2')
    if __cublasZsyrk_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyrk_v2 = dlsym(handle, 'cublasZsyrk_v2')

    global __cublasCsyrkEx
    __cublasCsyrkEx = dlsym(RTLD_DEFAULT, 'cublasCsyrkEx')
    if __cublasCsyrkEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrkEx = dlsym(handle, 'cublasCsyrkEx')

    global __cublasCsyrk3mEx
    __cublasCsyrk3mEx = dlsym(RTLD_DEFAULT, 'cublasCsyrk3mEx')
    if __cublasCsyrk3mEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrk3mEx = dlsym(handle, 'cublasCsyrk3mEx')

    global __cublasCherk_v2
    __cublasCherk_v2 = dlsym(RTLD_DEFAULT, 'cublasCherk_v2')
    if __cublasCherk_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherk_v2 = dlsym(handle, 'cublasCherk_v2')

    global __cublasZherk_v2
    __cublasZherk_v2 = dlsym(RTLD_DEFAULT, 'cublasZherk_v2')
    if __cublasZherk_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZherk_v2 = dlsym(handle, 'cublasZherk_v2')

    global __cublasCherkEx
    __cublasCherkEx = dlsym(RTLD_DEFAULT, 'cublasCherkEx')
    if __cublasCherkEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherkEx = dlsym(handle, 'cublasCherkEx')

    global __cublasCherk3mEx
    __cublasCherk3mEx = dlsym(RTLD_DEFAULT, 'cublasCherk3mEx')
    if __cublasCherk3mEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherk3mEx = dlsym(handle, 'cublasCherk3mEx')

    global __cublasSsyr2k_v2
    __cublasSsyr2k_v2 = dlsym(RTLD_DEFAULT, 'cublasSsyr2k_v2')
    if __cublasSsyr2k_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyr2k_v2 = dlsym(handle, 'cublasSsyr2k_v2')

    global __cublasDsyr2k_v2
    __cublasDsyr2k_v2 = dlsym(RTLD_DEFAULT, 'cublasDsyr2k_v2')
    if __cublasDsyr2k_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyr2k_v2 = dlsym(handle, 'cublasDsyr2k_v2')

    global __cublasCsyr2k_v2
    __cublasCsyr2k_v2 = dlsym(RTLD_DEFAULT, 'cublasCsyr2k_v2')
    if __cublasCsyr2k_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyr2k_v2 = dlsym(handle, 'cublasCsyr2k_v2')

    global __cublasZsyr2k_v2
    __cublasZsyr2k_v2 = dlsym(RTLD_DEFAULT, 'cublasZsyr2k_v2')
    if __cublasZsyr2k_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyr2k_v2 = dlsym(handle, 'cublasZsyr2k_v2')

    global __cublasCher2k_v2
    __cublasCher2k_v2 = dlsym(RTLD_DEFAULT, 'cublasCher2k_v2')
    if __cublasCher2k_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCher2k_v2 = dlsym(handle, 'cublasCher2k_v2')

    global __cublasZher2k_v2
    __cublasZher2k_v2 = dlsym(RTLD_DEFAULT, 'cublasZher2k_v2')
    if __cublasZher2k_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZher2k_v2 = dlsym(handle, 'cublasZher2k_v2')

    global __cublasSsyrkx
    __cublasSsyrkx = dlsym(RTLD_DEFAULT, 'cublasSsyrkx')
    if __cublasSsyrkx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyrkx = dlsym(handle, 'cublasSsyrkx')

    global __cublasDsyrkx
    __cublasDsyrkx = dlsym(RTLD_DEFAULT, 'cublasDsyrkx')
    if __cublasDsyrkx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyrkx = dlsym(handle, 'cublasDsyrkx')

    global __cublasCsyrkx
    __cublasCsyrkx = dlsym(RTLD_DEFAULT, 'cublasCsyrkx')
    if __cublasCsyrkx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrkx = dlsym(handle, 'cublasCsyrkx')

    global __cublasZsyrkx
    __cublasZsyrkx = dlsym(RTLD_DEFAULT, 'cublasZsyrkx')
    if __cublasZsyrkx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyrkx = dlsym(handle, 'cublasZsyrkx')

    global __cublasCherkx
    __cublasCherkx = dlsym(RTLD_DEFAULT, 'cublasCherkx')
    if __cublasCherkx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherkx = dlsym(handle, 'cublasCherkx')

    global __cublasZherkx
    __cublasZherkx = dlsym(RTLD_DEFAULT, 'cublasZherkx')
    if __cublasZherkx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZherkx = dlsym(handle, 'cublasZherkx')

    global __cublasSsymm_v2
    __cublasSsymm_v2 = dlsym(RTLD_DEFAULT, 'cublasSsymm_v2')
    if __cublasSsymm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsymm_v2 = dlsym(handle, 'cublasSsymm_v2')

    global __cublasDsymm_v2
    __cublasDsymm_v2 = dlsym(RTLD_DEFAULT, 'cublasDsymm_v2')
    if __cublasDsymm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsymm_v2 = dlsym(handle, 'cublasDsymm_v2')

    global __cublasCsymm_v2
    __cublasCsymm_v2 = dlsym(RTLD_DEFAULT, 'cublasCsymm_v2')
    if __cublasCsymm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsymm_v2 = dlsym(handle, 'cublasCsymm_v2')

    global __cublasZsymm_v2
    __cublasZsymm_v2 = dlsym(RTLD_DEFAULT, 'cublasZsymm_v2')
    if __cublasZsymm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsymm_v2 = dlsym(handle, 'cublasZsymm_v2')

    global __cublasChemm_v2
    __cublasChemm_v2 = dlsym(RTLD_DEFAULT, 'cublasChemm_v2')
    if __cublasChemm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChemm_v2 = dlsym(handle, 'cublasChemm_v2')

    global __cublasZhemm_v2
    __cublasZhemm_v2 = dlsym(RTLD_DEFAULT, 'cublasZhemm_v2')
    if __cublasZhemm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhemm_v2 = dlsym(handle, 'cublasZhemm_v2')

    global __cublasStrsm_v2
    __cublasStrsm_v2 = dlsym(RTLD_DEFAULT, 'cublasStrsm_v2')
    if __cublasStrsm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrsm_v2 = dlsym(handle, 'cublasStrsm_v2')

    global __cublasDtrsm_v2
    __cublasDtrsm_v2 = dlsym(RTLD_DEFAULT, 'cublasDtrsm_v2')
    if __cublasDtrsm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrsm_v2 = dlsym(handle, 'cublasDtrsm_v2')

    global __cublasCtrsm_v2
    __cublasCtrsm_v2 = dlsym(RTLD_DEFAULT, 'cublasCtrsm_v2')
    if __cublasCtrsm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrsm_v2 = dlsym(handle, 'cublasCtrsm_v2')

    global __cublasZtrsm_v2
    __cublasZtrsm_v2 = dlsym(RTLD_DEFAULT, 'cublasZtrsm_v2')
    if __cublasZtrsm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrsm_v2 = dlsym(handle, 'cublasZtrsm_v2')

    global __cublasStrmm_v2
    __cublasStrmm_v2 = dlsym(RTLD_DEFAULT, 'cublasStrmm_v2')
    if __cublasStrmm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrmm_v2 = dlsym(handle, 'cublasStrmm_v2')

    global __cublasDtrmm_v2
    __cublasDtrmm_v2 = dlsym(RTLD_DEFAULT, 'cublasDtrmm_v2')
    if __cublasDtrmm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrmm_v2 = dlsym(handle, 'cublasDtrmm_v2')

    global __cublasCtrmm_v2
    __cublasCtrmm_v2 = dlsym(RTLD_DEFAULT, 'cublasCtrmm_v2')
    if __cublasCtrmm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrmm_v2 = dlsym(handle, 'cublasCtrmm_v2')

    global __cublasZtrmm_v2
    __cublasZtrmm_v2 = dlsym(RTLD_DEFAULT, 'cublasZtrmm_v2')
    if __cublasZtrmm_v2 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrmm_v2 = dlsym(handle, 'cublasZtrmm_v2')

    global __cublasSgemmBatched
    __cublasSgemmBatched = dlsym(RTLD_DEFAULT, 'cublasSgemmBatched')
    if __cublasSgemmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemmBatched = dlsym(handle, 'cublasSgemmBatched')

    global __cublasDgemmBatched
    __cublasDgemmBatched = dlsym(RTLD_DEFAULT, 'cublasDgemmBatched')
    if __cublasDgemmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemmBatched = dlsym(handle, 'cublasDgemmBatched')

    global __cublasCgemmBatched
    __cublasCgemmBatched = dlsym(RTLD_DEFAULT, 'cublasCgemmBatched')
    if __cublasCgemmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemmBatched = dlsym(handle, 'cublasCgemmBatched')

    global __cublasCgemm3mBatched
    __cublasCgemm3mBatched = dlsym(RTLD_DEFAULT, 'cublasCgemm3mBatched')
    if __cublasCgemm3mBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3mBatched = dlsym(handle, 'cublasCgemm3mBatched')

    global __cublasZgemmBatched
    __cublasZgemmBatched = dlsym(RTLD_DEFAULT, 'cublasZgemmBatched')
    if __cublasZgemmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemmBatched = dlsym(handle, 'cublasZgemmBatched')

    global __cublasGemmBatchedEx
    __cublasGemmBatchedEx = dlsym(RTLD_DEFAULT, 'cublasGemmBatchedEx')
    if __cublasGemmBatchedEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGemmBatchedEx = dlsym(handle, 'cublasGemmBatchedEx')

    global __cublasGemmStridedBatchedEx
    __cublasGemmStridedBatchedEx = dlsym(RTLD_DEFAULT, 'cublasGemmStridedBatchedEx')
    if __cublasGemmStridedBatchedEx == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGemmStridedBatchedEx = dlsym(handle, 'cublasGemmStridedBatchedEx')

    global __cublasSgemmStridedBatched
    __cublasSgemmStridedBatched = dlsym(RTLD_DEFAULT, 'cublasSgemmStridedBatched')
    if __cublasSgemmStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemmStridedBatched = dlsym(handle, 'cublasSgemmStridedBatched')

    global __cublasDgemmStridedBatched
    __cublasDgemmStridedBatched = dlsym(RTLD_DEFAULT, 'cublasDgemmStridedBatched')
    if __cublasDgemmStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemmStridedBatched = dlsym(handle, 'cublasDgemmStridedBatched')

    global __cublasCgemmStridedBatched
    __cublasCgemmStridedBatched = dlsym(RTLD_DEFAULT, 'cublasCgemmStridedBatched')
    if __cublasCgemmStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemmStridedBatched = dlsym(handle, 'cublasCgemmStridedBatched')

    global __cublasCgemm3mStridedBatched
    __cublasCgemm3mStridedBatched = dlsym(RTLD_DEFAULT, 'cublasCgemm3mStridedBatched')
    if __cublasCgemm3mStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3mStridedBatched = dlsym(handle, 'cublasCgemm3mStridedBatched')

    global __cublasZgemmStridedBatched
    __cublasZgemmStridedBatched = dlsym(RTLD_DEFAULT, 'cublasZgemmStridedBatched')
    if __cublasZgemmStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemmStridedBatched = dlsym(handle, 'cublasZgemmStridedBatched')

    global __cublasSgeam
    __cublasSgeam = dlsym(RTLD_DEFAULT, 'cublasSgeam')
    if __cublasSgeam == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgeam = dlsym(handle, 'cublasSgeam')

    global __cublasDgeam
    __cublasDgeam = dlsym(RTLD_DEFAULT, 'cublasDgeam')
    if __cublasDgeam == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgeam = dlsym(handle, 'cublasDgeam')

    global __cublasCgeam
    __cublasCgeam = dlsym(RTLD_DEFAULT, 'cublasCgeam')
    if __cublasCgeam == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgeam = dlsym(handle, 'cublasCgeam')

    global __cublasZgeam
    __cublasZgeam = dlsym(RTLD_DEFAULT, 'cublasZgeam')
    if __cublasZgeam == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgeam = dlsym(handle, 'cublasZgeam')

    global __cublasSgetrfBatched
    __cublasSgetrfBatched = dlsym(RTLD_DEFAULT, 'cublasSgetrfBatched')
    if __cublasSgetrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgetrfBatched = dlsym(handle, 'cublasSgetrfBatched')

    global __cublasDgetrfBatched
    __cublasDgetrfBatched = dlsym(RTLD_DEFAULT, 'cublasDgetrfBatched')
    if __cublasDgetrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgetrfBatched = dlsym(handle, 'cublasDgetrfBatched')

    global __cublasCgetrfBatched
    __cublasCgetrfBatched = dlsym(RTLD_DEFAULT, 'cublasCgetrfBatched')
    if __cublasCgetrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgetrfBatched = dlsym(handle, 'cublasCgetrfBatched')

    global __cublasZgetrfBatched
    __cublasZgetrfBatched = dlsym(RTLD_DEFAULT, 'cublasZgetrfBatched')
    if __cublasZgetrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgetrfBatched = dlsym(handle, 'cublasZgetrfBatched')

    global __cublasSgetriBatched
    __cublasSgetriBatched = dlsym(RTLD_DEFAULT, 'cublasSgetriBatched')
    if __cublasSgetriBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgetriBatched = dlsym(handle, 'cublasSgetriBatched')

    global __cublasDgetriBatched
    __cublasDgetriBatched = dlsym(RTLD_DEFAULT, 'cublasDgetriBatched')
    if __cublasDgetriBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgetriBatched = dlsym(handle, 'cublasDgetriBatched')

    global __cublasCgetriBatched
    __cublasCgetriBatched = dlsym(RTLD_DEFAULT, 'cublasCgetriBatched')
    if __cublasCgetriBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgetriBatched = dlsym(handle, 'cublasCgetriBatched')

    global __cublasZgetriBatched
    __cublasZgetriBatched = dlsym(RTLD_DEFAULT, 'cublasZgetriBatched')
    if __cublasZgetriBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgetriBatched = dlsym(handle, 'cublasZgetriBatched')

    global __cublasSgetrsBatched
    __cublasSgetrsBatched = dlsym(RTLD_DEFAULT, 'cublasSgetrsBatched')
    if __cublasSgetrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgetrsBatched = dlsym(handle, 'cublasSgetrsBatched')

    global __cublasDgetrsBatched
    __cublasDgetrsBatched = dlsym(RTLD_DEFAULT, 'cublasDgetrsBatched')
    if __cublasDgetrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgetrsBatched = dlsym(handle, 'cublasDgetrsBatched')

    global __cublasCgetrsBatched
    __cublasCgetrsBatched = dlsym(RTLD_DEFAULT, 'cublasCgetrsBatched')
    if __cublasCgetrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgetrsBatched = dlsym(handle, 'cublasCgetrsBatched')

    global __cublasZgetrsBatched
    __cublasZgetrsBatched = dlsym(RTLD_DEFAULT, 'cublasZgetrsBatched')
    if __cublasZgetrsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgetrsBatched = dlsym(handle, 'cublasZgetrsBatched')

    global __cublasStrsmBatched
    __cublasStrsmBatched = dlsym(RTLD_DEFAULT, 'cublasStrsmBatched')
    if __cublasStrsmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrsmBatched = dlsym(handle, 'cublasStrsmBatched')

    global __cublasDtrsmBatched
    __cublasDtrsmBatched = dlsym(RTLD_DEFAULT, 'cublasDtrsmBatched')
    if __cublasDtrsmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrsmBatched = dlsym(handle, 'cublasDtrsmBatched')

    global __cublasCtrsmBatched
    __cublasCtrsmBatched = dlsym(RTLD_DEFAULT, 'cublasCtrsmBatched')
    if __cublasCtrsmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrsmBatched = dlsym(handle, 'cublasCtrsmBatched')

    global __cublasZtrsmBatched
    __cublasZtrsmBatched = dlsym(RTLD_DEFAULT, 'cublasZtrsmBatched')
    if __cublasZtrsmBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrsmBatched = dlsym(handle, 'cublasZtrsmBatched')

    global __cublasSmatinvBatched
    __cublasSmatinvBatched = dlsym(RTLD_DEFAULT, 'cublasSmatinvBatched')
    if __cublasSmatinvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSmatinvBatched = dlsym(handle, 'cublasSmatinvBatched')

    global __cublasDmatinvBatched
    __cublasDmatinvBatched = dlsym(RTLD_DEFAULT, 'cublasDmatinvBatched')
    if __cublasDmatinvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDmatinvBatched = dlsym(handle, 'cublasDmatinvBatched')

    global __cublasCmatinvBatched
    __cublasCmatinvBatched = dlsym(RTLD_DEFAULT, 'cublasCmatinvBatched')
    if __cublasCmatinvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCmatinvBatched = dlsym(handle, 'cublasCmatinvBatched')

    global __cublasZmatinvBatched
    __cublasZmatinvBatched = dlsym(RTLD_DEFAULT, 'cublasZmatinvBatched')
    if __cublasZmatinvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZmatinvBatched = dlsym(handle, 'cublasZmatinvBatched')

    global __cublasSgeqrfBatched
    __cublasSgeqrfBatched = dlsym(RTLD_DEFAULT, 'cublasSgeqrfBatched')
    if __cublasSgeqrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgeqrfBatched = dlsym(handle, 'cublasSgeqrfBatched')

    global __cublasDgeqrfBatched
    __cublasDgeqrfBatched = dlsym(RTLD_DEFAULT, 'cublasDgeqrfBatched')
    if __cublasDgeqrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgeqrfBatched = dlsym(handle, 'cublasDgeqrfBatched')

    global __cublasCgeqrfBatched
    __cublasCgeqrfBatched = dlsym(RTLD_DEFAULT, 'cublasCgeqrfBatched')
    if __cublasCgeqrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgeqrfBatched = dlsym(handle, 'cublasCgeqrfBatched')

    global __cublasZgeqrfBatched
    __cublasZgeqrfBatched = dlsym(RTLD_DEFAULT, 'cublasZgeqrfBatched')
    if __cublasZgeqrfBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgeqrfBatched = dlsym(handle, 'cublasZgeqrfBatched')

    global __cublasSgelsBatched
    __cublasSgelsBatched = dlsym(RTLD_DEFAULT, 'cublasSgelsBatched')
    if __cublasSgelsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgelsBatched = dlsym(handle, 'cublasSgelsBatched')

    global __cublasDgelsBatched
    __cublasDgelsBatched = dlsym(RTLD_DEFAULT, 'cublasDgelsBatched')
    if __cublasDgelsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgelsBatched = dlsym(handle, 'cublasDgelsBatched')

    global __cublasCgelsBatched
    __cublasCgelsBatched = dlsym(RTLD_DEFAULT, 'cublasCgelsBatched')
    if __cublasCgelsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgelsBatched = dlsym(handle, 'cublasCgelsBatched')

    global __cublasZgelsBatched
    __cublasZgelsBatched = dlsym(RTLD_DEFAULT, 'cublasZgelsBatched')
    if __cublasZgelsBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgelsBatched = dlsym(handle, 'cublasZgelsBatched')

    global __cublasSdgmm
    __cublasSdgmm = dlsym(RTLD_DEFAULT, 'cublasSdgmm')
    if __cublasSdgmm == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSdgmm = dlsym(handle, 'cublasSdgmm')

    global __cublasDdgmm
    __cublasDdgmm = dlsym(RTLD_DEFAULT, 'cublasDdgmm')
    if __cublasDdgmm == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDdgmm = dlsym(handle, 'cublasDdgmm')

    global __cublasCdgmm
    __cublasCdgmm = dlsym(RTLD_DEFAULT, 'cublasCdgmm')
    if __cublasCdgmm == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCdgmm = dlsym(handle, 'cublasCdgmm')

    global __cublasZdgmm
    __cublasZdgmm = dlsym(RTLD_DEFAULT, 'cublasZdgmm')
    if __cublasZdgmm == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdgmm = dlsym(handle, 'cublasZdgmm')

    global __cublasStpttr
    __cublasStpttr = dlsym(RTLD_DEFAULT, 'cublasStpttr')
    if __cublasStpttr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStpttr = dlsym(handle, 'cublasStpttr')

    global __cublasDtpttr
    __cublasDtpttr = dlsym(RTLD_DEFAULT, 'cublasDtpttr')
    if __cublasDtpttr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtpttr = dlsym(handle, 'cublasDtpttr')

    global __cublasCtpttr
    __cublasCtpttr = dlsym(RTLD_DEFAULT, 'cublasCtpttr')
    if __cublasCtpttr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtpttr = dlsym(handle, 'cublasCtpttr')

    global __cublasZtpttr
    __cublasZtpttr = dlsym(RTLD_DEFAULT, 'cublasZtpttr')
    if __cublasZtpttr == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtpttr = dlsym(handle, 'cublasZtpttr')

    global __cublasStrttp
    __cublasStrttp = dlsym(RTLD_DEFAULT, 'cublasStrttp')
    if __cublasStrttp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrttp = dlsym(handle, 'cublasStrttp')

    global __cublasDtrttp
    __cublasDtrttp = dlsym(RTLD_DEFAULT, 'cublasDtrttp')
    if __cublasDtrttp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrttp = dlsym(handle, 'cublasDtrttp')

    global __cublasCtrttp
    __cublasCtrttp = dlsym(RTLD_DEFAULT, 'cublasCtrttp')
    if __cublasCtrttp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrttp = dlsym(handle, 'cublasCtrttp')

    global __cublasZtrttp
    __cublasZtrttp = dlsym(RTLD_DEFAULT, 'cublasZtrttp')
    if __cublasZtrttp == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrttp = dlsym(handle, 'cublasZtrttp')

    global __cublasGetSmCountTarget
    __cublasGetSmCountTarget = dlsym(RTLD_DEFAULT, 'cublasGetSmCountTarget')
    if __cublasGetSmCountTarget == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetSmCountTarget = dlsym(handle, 'cublasGetSmCountTarget')

    global __cublasSetSmCountTarget
    __cublasSetSmCountTarget = dlsym(RTLD_DEFAULT, 'cublasSetSmCountTarget')
    if __cublasSetSmCountTarget == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetSmCountTarget = dlsym(handle, 'cublasSetSmCountTarget')

    global __cublasGetStatusName
    __cublasGetStatusName = dlsym(RTLD_DEFAULT, 'cublasGetStatusName')
    if __cublasGetStatusName == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetStatusName = dlsym(handle, 'cublasGetStatusName')

    global __cublasGetStatusString
    __cublasGetStatusString = dlsym(RTLD_DEFAULT, 'cublasGetStatusString')
    if __cublasGetStatusString == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetStatusString = dlsym(handle, 'cublasGetStatusString')

    global __cublasSgemvBatched
    __cublasSgemvBatched = dlsym(RTLD_DEFAULT, 'cublasSgemvBatched')
    if __cublasSgemvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemvBatched = dlsym(handle, 'cublasSgemvBatched')

    global __cublasDgemvBatched
    __cublasDgemvBatched = dlsym(RTLD_DEFAULT, 'cublasDgemvBatched')
    if __cublasDgemvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemvBatched = dlsym(handle, 'cublasDgemvBatched')

    global __cublasCgemvBatched
    __cublasCgemvBatched = dlsym(RTLD_DEFAULT, 'cublasCgemvBatched')
    if __cublasCgemvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemvBatched = dlsym(handle, 'cublasCgemvBatched')

    global __cublasZgemvBatched
    __cublasZgemvBatched = dlsym(RTLD_DEFAULT, 'cublasZgemvBatched')
    if __cublasZgemvBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemvBatched = dlsym(handle, 'cublasZgemvBatched')

    global __cublasSgemvStridedBatched
    __cublasSgemvStridedBatched = dlsym(RTLD_DEFAULT, 'cublasSgemvStridedBatched')
    if __cublasSgemvStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemvStridedBatched = dlsym(handle, 'cublasSgemvStridedBatched')

    global __cublasDgemvStridedBatched
    __cublasDgemvStridedBatched = dlsym(RTLD_DEFAULT, 'cublasDgemvStridedBatched')
    if __cublasDgemvStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemvStridedBatched = dlsym(handle, 'cublasDgemvStridedBatched')

    global __cublasCgemvStridedBatched
    __cublasCgemvStridedBatched = dlsym(RTLD_DEFAULT, 'cublasCgemvStridedBatched')
    if __cublasCgemvStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemvStridedBatched = dlsym(handle, 'cublasCgemvStridedBatched')

    global __cublasZgemvStridedBatched
    __cublasZgemvStridedBatched = dlsym(RTLD_DEFAULT, 'cublasZgemvStridedBatched')
    if __cublasZgemvStridedBatched == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemvStridedBatched = dlsym(handle, 'cublasZgemvStridedBatched')

    global __cublasSetVector_64
    __cublasSetVector_64 = dlsym(RTLD_DEFAULT, 'cublasSetVector_64')
    if __cublasSetVector_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetVector_64 = dlsym(handle, 'cublasSetVector_64')

    global __cublasGetVector_64
    __cublasGetVector_64 = dlsym(RTLD_DEFAULT, 'cublasGetVector_64')
    if __cublasGetVector_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetVector_64 = dlsym(handle, 'cublasGetVector_64')

    global __cublasSetMatrix_64
    __cublasSetMatrix_64 = dlsym(RTLD_DEFAULT, 'cublasSetMatrix_64')
    if __cublasSetMatrix_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetMatrix_64 = dlsym(handle, 'cublasSetMatrix_64')

    global __cublasGetMatrix_64
    __cublasGetMatrix_64 = dlsym(RTLD_DEFAULT, 'cublasGetMatrix_64')
    if __cublasGetMatrix_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetMatrix_64 = dlsym(handle, 'cublasGetMatrix_64')

    global __cublasSetVectorAsync_64
    __cublasSetVectorAsync_64 = dlsym(RTLD_DEFAULT, 'cublasSetVectorAsync_64')
    if __cublasSetVectorAsync_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetVectorAsync_64 = dlsym(handle, 'cublasSetVectorAsync_64')

    global __cublasGetVectorAsync_64
    __cublasGetVectorAsync_64 = dlsym(RTLD_DEFAULT, 'cublasGetVectorAsync_64')
    if __cublasGetVectorAsync_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetVectorAsync_64 = dlsym(handle, 'cublasGetVectorAsync_64')

    global __cublasSetMatrixAsync_64
    __cublasSetMatrixAsync_64 = dlsym(RTLD_DEFAULT, 'cublasSetMatrixAsync_64')
    if __cublasSetMatrixAsync_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSetMatrixAsync_64 = dlsym(handle, 'cublasSetMatrixAsync_64')

    global __cublasGetMatrixAsync_64
    __cublasGetMatrixAsync_64 = dlsym(RTLD_DEFAULT, 'cublasGetMatrixAsync_64')
    if __cublasGetMatrixAsync_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGetMatrixAsync_64 = dlsym(handle, 'cublasGetMatrixAsync_64')

    global __cublasNrm2Ex_64
    __cublasNrm2Ex_64 = dlsym(RTLD_DEFAULT, 'cublasNrm2Ex_64')
    if __cublasNrm2Ex_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasNrm2Ex_64 = dlsym(handle, 'cublasNrm2Ex_64')

    global __cublasSnrm2_v2_64
    __cublasSnrm2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSnrm2_v2_64')
    if __cublasSnrm2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSnrm2_v2_64 = dlsym(handle, 'cublasSnrm2_v2_64')

    global __cublasDnrm2_v2_64
    __cublasDnrm2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDnrm2_v2_64')
    if __cublasDnrm2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDnrm2_v2_64 = dlsym(handle, 'cublasDnrm2_v2_64')

    global __cublasScnrm2_v2_64
    __cublasScnrm2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasScnrm2_v2_64')
    if __cublasScnrm2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScnrm2_v2_64 = dlsym(handle, 'cublasScnrm2_v2_64')

    global __cublasDznrm2_v2_64
    __cublasDznrm2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDznrm2_v2_64')
    if __cublasDznrm2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDznrm2_v2_64 = dlsym(handle, 'cublasDznrm2_v2_64')

    global __cublasDotEx_64
    __cublasDotEx_64 = dlsym(RTLD_DEFAULT, 'cublasDotEx_64')
    if __cublasDotEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDotEx_64 = dlsym(handle, 'cublasDotEx_64')

    global __cublasDotcEx_64
    __cublasDotcEx_64 = dlsym(RTLD_DEFAULT, 'cublasDotcEx_64')
    if __cublasDotcEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDotcEx_64 = dlsym(handle, 'cublasDotcEx_64')

    global __cublasSdot_v2_64
    __cublasSdot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSdot_v2_64')
    if __cublasSdot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSdot_v2_64 = dlsym(handle, 'cublasSdot_v2_64')

    global __cublasDdot_v2_64
    __cublasDdot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDdot_v2_64')
    if __cublasDdot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDdot_v2_64 = dlsym(handle, 'cublasDdot_v2_64')

    global __cublasCdotu_v2_64
    __cublasCdotu_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCdotu_v2_64')
    if __cublasCdotu_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCdotu_v2_64 = dlsym(handle, 'cublasCdotu_v2_64')

    global __cublasCdotc_v2_64
    __cublasCdotc_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCdotc_v2_64')
    if __cublasCdotc_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCdotc_v2_64 = dlsym(handle, 'cublasCdotc_v2_64')

    global __cublasZdotu_v2_64
    __cublasZdotu_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZdotu_v2_64')
    if __cublasZdotu_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdotu_v2_64 = dlsym(handle, 'cublasZdotu_v2_64')

    global __cublasZdotc_v2_64
    __cublasZdotc_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZdotc_v2_64')
    if __cublasZdotc_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdotc_v2_64 = dlsym(handle, 'cublasZdotc_v2_64')

    global __cublasScalEx_64
    __cublasScalEx_64 = dlsym(RTLD_DEFAULT, 'cublasScalEx_64')
    if __cublasScalEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScalEx_64 = dlsym(handle, 'cublasScalEx_64')

    global __cublasSscal_v2_64
    __cublasSscal_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSscal_v2_64')
    if __cublasSscal_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSscal_v2_64 = dlsym(handle, 'cublasSscal_v2_64')

    global __cublasDscal_v2_64
    __cublasDscal_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDscal_v2_64')
    if __cublasDscal_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDscal_v2_64 = dlsym(handle, 'cublasDscal_v2_64')

    global __cublasCscal_v2_64
    __cublasCscal_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCscal_v2_64')
    if __cublasCscal_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCscal_v2_64 = dlsym(handle, 'cublasCscal_v2_64')

    global __cublasCsscal_v2_64
    __cublasCsscal_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsscal_v2_64')
    if __cublasCsscal_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsscal_v2_64 = dlsym(handle, 'cublasCsscal_v2_64')

    global __cublasZscal_v2_64
    __cublasZscal_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZscal_v2_64')
    if __cublasZscal_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZscal_v2_64 = dlsym(handle, 'cublasZscal_v2_64')

    global __cublasZdscal_v2_64
    __cublasZdscal_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZdscal_v2_64')
    if __cublasZdscal_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdscal_v2_64 = dlsym(handle, 'cublasZdscal_v2_64')

    global __cublasAxpyEx_64
    __cublasAxpyEx_64 = dlsym(RTLD_DEFAULT, 'cublasAxpyEx_64')
    if __cublasAxpyEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasAxpyEx_64 = dlsym(handle, 'cublasAxpyEx_64')

    global __cublasSaxpy_v2_64
    __cublasSaxpy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSaxpy_v2_64')
    if __cublasSaxpy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSaxpy_v2_64 = dlsym(handle, 'cublasSaxpy_v2_64')

    global __cublasDaxpy_v2_64
    __cublasDaxpy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDaxpy_v2_64')
    if __cublasDaxpy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDaxpy_v2_64 = dlsym(handle, 'cublasDaxpy_v2_64')

    global __cublasCaxpy_v2_64
    __cublasCaxpy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCaxpy_v2_64')
    if __cublasCaxpy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCaxpy_v2_64 = dlsym(handle, 'cublasCaxpy_v2_64')

    global __cublasZaxpy_v2_64
    __cublasZaxpy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZaxpy_v2_64')
    if __cublasZaxpy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZaxpy_v2_64 = dlsym(handle, 'cublasZaxpy_v2_64')

    global __cublasCopyEx_64
    __cublasCopyEx_64 = dlsym(RTLD_DEFAULT, 'cublasCopyEx_64')
    if __cublasCopyEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCopyEx_64 = dlsym(handle, 'cublasCopyEx_64')

    global __cublasScopy_v2_64
    __cublasScopy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasScopy_v2_64')
    if __cublasScopy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScopy_v2_64 = dlsym(handle, 'cublasScopy_v2_64')

    global __cublasDcopy_v2_64
    __cublasDcopy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDcopy_v2_64')
    if __cublasDcopy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDcopy_v2_64 = dlsym(handle, 'cublasDcopy_v2_64')

    global __cublasCcopy_v2_64
    __cublasCcopy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCcopy_v2_64')
    if __cublasCcopy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCcopy_v2_64 = dlsym(handle, 'cublasCcopy_v2_64')

    global __cublasZcopy_v2_64
    __cublasZcopy_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZcopy_v2_64')
    if __cublasZcopy_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZcopy_v2_64 = dlsym(handle, 'cublasZcopy_v2_64')

    global __cublasSswap_v2_64
    __cublasSswap_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSswap_v2_64')
    if __cublasSswap_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSswap_v2_64 = dlsym(handle, 'cublasSswap_v2_64')

    global __cublasDswap_v2_64
    __cublasDswap_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDswap_v2_64')
    if __cublasDswap_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDswap_v2_64 = dlsym(handle, 'cublasDswap_v2_64')

    global __cublasCswap_v2_64
    __cublasCswap_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCswap_v2_64')
    if __cublasCswap_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCswap_v2_64 = dlsym(handle, 'cublasCswap_v2_64')

    global __cublasZswap_v2_64
    __cublasZswap_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZswap_v2_64')
    if __cublasZswap_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZswap_v2_64 = dlsym(handle, 'cublasZswap_v2_64')

    global __cublasSwapEx_64
    __cublasSwapEx_64 = dlsym(RTLD_DEFAULT, 'cublasSwapEx_64')
    if __cublasSwapEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSwapEx_64 = dlsym(handle, 'cublasSwapEx_64')

    global __cublasIsamax_v2_64
    __cublasIsamax_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIsamax_v2_64')
    if __cublasIsamax_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIsamax_v2_64 = dlsym(handle, 'cublasIsamax_v2_64')

    global __cublasIdamax_v2_64
    __cublasIdamax_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIdamax_v2_64')
    if __cublasIdamax_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIdamax_v2_64 = dlsym(handle, 'cublasIdamax_v2_64')

    global __cublasIcamax_v2_64
    __cublasIcamax_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIcamax_v2_64')
    if __cublasIcamax_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIcamax_v2_64 = dlsym(handle, 'cublasIcamax_v2_64')

    global __cublasIzamax_v2_64
    __cublasIzamax_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIzamax_v2_64')
    if __cublasIzamax_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIzamax_v2_64 = dlsym(handle, 'cublasIzamax_v2_64')

    global __cublasIamaxEx_64
    __cublasIamaxEx_64 = dlsym(RTLD_DEFAULT, 'cublasIamaxEx_64')
    if __cublasIamaxEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIamaxEx_64 = dlsym(handle, 'cublasIamaxEx_64')

    global __cublasIsamin_v2_64
    __cublasIsamin_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIsamin_v2_64')
    if __cublasIsamin_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIsamin_v2_64 = dlsym(handle, 'cublasIsamin_v2_64')

    global __cublasIdamin_v2_64
    __cublasIdamin_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIdamin_v2_64')
    if __cublasIdamin_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIdamin_v2_64 = dlsym(handle, 'cublasIdamin_v2_64')

    global __cublasIcamin_v2_64
    __cublasIcamin_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIcamin_v2_64')
    if __cublasIcamin_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIcamin_v2_64 = dlsym(handle, 'cublasIcamin_v2_64')

    global __cublasIzamin_v2_64
    __cublasIzamin_v2_64 = dlsym(RTLD_DEFAULT, 'cublasIzamin_v2_64')
    if __cublasIzamin_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIzamin_v2_64 = dlsym(handle, 'cublasIzamin_v2_64')

    global __cublasIaminEx_64
    __cublasIaminEx_64 = dlsym(RTLD_DEFAULT, 'cublasIaminEx_64')
    if __cublasIaminEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasIaminEx_64 = dlsym(handle, 'cublasIaminEx_64')

    global __cublasAsumEx_64
    __cublasAsumEx_64 = dlsym(RTLD_DEFAULT, 'cublasAsumEx_64')
    if __cublasAsumEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasAsumEx_64 = dlsym(handle, 'cublasAsumEx_64')

    global __cublasSasum_v2_64
    __cublasSasum_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSasum_v2_64')
    if __cublasSasum_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSasum_v2_64 = dlsym(handle, 'cublasSasum_v2_64')

    global __cublasDasum_v2_64
    __cublasDasum_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDasum_v2_64')
    if __cublasDasum_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDasum_v2_64 = dlsym(handle, 'cublasDasum_v2_64')

    global __cublasScasum_v2_64
    __cublasScasum_v2_64 = dlsym(RTLD_DEFAULT, 'cublasScasum_v2_64')
    if __cublasScasum_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasScasum_v2_64 = dlsym(handle, 'cublasScasum_v2_64')

    global __cublasDzasum_v2_64
    __cublasDzasum_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDzasum_v2_64')
    if __cublasDzasum_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDzasum_v2_64 = dlsym(handle, 'cublasDzasum_v2_64')

    global __cublasSrot_v2_64
    __cublasSrot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSrot_v2_64')
    if __cublasSrot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSrot_v2_64 = dlsym(handle, 'cublasSrot_v2_64')

    global __cublasDrot_v2_64
    __cublasDrot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDrot_v2_64')
    if __cublasDrot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDrot_v2_64 = dlsym(handle, 'cublasDrot_v2_64')

    global __cublasCrot_v2_64
    __cublasCrot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCrot_v2_64')
    if __cublasCrot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCrot_v2_64 = dlsym(handle, 'cublasCrot_v2_64')

    global __cublasCsrot_v2_64
    __cublasCsrot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsrot_v2_64')
    if __cublasCsrot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsrot_v2_64 = dlsym(handle, 'cublasCsrot_v2_64')

    global __cublasZrot_v2_64
    __cublasZrot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZrot_v2_64')
    if __cublasZrot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZrot_v2_64 = dlsym(handle, 'cublasZrot_v2_64')

    global __cublasZdrot_v2_64
    __cublasZdrot_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZdrot_v2_64')
    if __cublasZdrot_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdrot_v2_64 = dlsym(handle, 'cublasZdrot_v2_64')

    global __cublasRotEx_64
    __cublasRotEx_64 = dlsym(RTLD_DEFAULT, 'cublasRotEx_64')
    if __cublasRotEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasRotEx_64 = dlsym(handle, 'cublasRotEx_64')

    global __cublasSrotm_v2_64
    __cublasSrotm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSrotm_v2_64')
    if __cublasSrotm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSrotm_v2_64 = dlsym(handle, 'cublasSrotm_v2_64')

    global __cublasDrotm_v2_64
    __cublasDrotm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDrotm_v2_64')
    if __cublasDrotm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDrotm_v2_64 = dlsym(handle, 'cublasDrotm_v2_64')

    global __cublasRotmEx_64
    __cublasRotmEx_64 = dlsym(RTLD_DEFAULT, 'cublasRotmEx_64')
    if __cublasRotmEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasRotmEx_64 = dlsym(handle, 'cublasRotmEx_64')

    global __cublasSgemv_v2_64
    __cublasSgemv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSgemv_v2_64')
    if __cublasSgemv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemv_v2_64 = dlsym(handle, 'cublasSgemv_v2_64')

    global __cublasDgemv_v2_64
    __cublasDgemv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDgemv_v2_64')
    if __cublasDgemv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemv_v2_64 = dlsym(handle, 'cublasDgemv_v2_64')

    global __cublasCgemv_v2_64
    __cublasCgemv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCgemv_v2_64')
    if __cublasCgemv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemv_v2_64 = dlsym(handle, 'cublasCgemv_v2_64')

    global __cublasZgemv_v2_64
    __cublasZgemv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZgemv_v2_64')
    if __cublasZgemv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemv_v2_64 = dlsym(handle, 'cublasZgemv_v2_64')

    global __cublasSgbmv_v2_64
    __cublasSgbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSgbmv_v2_64')
    if __cublasSgbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgbmv_v2_64 = dlsym(handle, 'cublasSgbmv_v2_64')

    global __cublasDgbmv_v2_64
    __cublasDgbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDgbmv_v2_64')
    if __cublasDgbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgbmv_v2_64 = dlsym(handle, 'cublasDgbmv_v2_64')

    global __cublasCgbmv_v2_64
    __cublasCgbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCgbmv_v2_64')
    if __cublasCgbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgbmv_v2_64 = dlsym(handle, 'cublasCgbmv_v2_64')

    global __cublasZgbmv_v2_64
    __cublasZgbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZgbmv_v2_64')
    if __cublasZgbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgbmv_v2_64 = dlsym(handle, 'cublasZgbmv_v2_64')

    global __cublasStrmv_v2_64
    __cublasStrmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStrmv_v2_64')
    if __cublasStrmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrmv_v2_64 = dlsym(handle, 'cublasStrmv_v2_64')

    global __cublasDtrmv_v2_64
    __cublasDtrmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtrmv_v2_64')
    if __cublasDtrmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrmv_v2_64 = dlsym(handle, 'cublasDtrmv_v2_64')

    global __cublasCtrmv_v2_64
    __cublasCtrmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtrmv_v2_64')
    if __cublasCtrmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrmv_v2_64 = dlsym(handle, 'cublasCtrmv_v2_64')

    global __cublasZtrmv_v2_64
    __cublasZtrmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtrmv_v2_64')
    if __cublasZtrmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrmv_v2_64 = dlsym(handle, 'cublasZtrmv_v2_64')

    global __cublasStbmv_v2_64
    __cublasStbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStbmv_v2_64')
    if __cublasStbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStbmv_v2_64 = dlsym(handle, 'cublasStbmv_v2_64')

    global __cublasDtbmv_v2_64
    __cublasDtbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtbmv_v2_64')
    if __cublasDtbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtbmv_v2_64 = dlsym(handle, 'cublasDtbmv_v2_64')

    global __cublasCtbmv_v2_64
    __cublasCtbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtbmv_v2_64')
    if __cublasCtbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtbmv_v2_64 = dlsym(handle, 'cublasCtbmv_v2_64')

    global __cublasZtbmv_v2_64
    __cublasZtbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtbmv_v2_64')
    if __cublasZtbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtbmv_v2_64 = dlsym(handle, 'cublasZtbmv_v2_64')

    global __cublasStpmv_v2_64
    __cublasStpmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStpmv_v2_64')
    if __cublasStpmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStpmv_v2_64 = dlsym(handle, 'cublasStpmv_v2_64')

    global __cublasDtpmv_v2_64
    __cublasDtpmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtpmv_v2_64')
    if __cublasDtpmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtpmv_v2_64 = dlsym(handle, 'cublasDtpmv_v2_64')

    global __cublasCtpmv_v2_64
    __cublasCtpmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtpmv_v2_64')
    if __cublasCtpmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtpmv_v2_64 = dlsym(handle, 'cublasCtpmv_v2_64')

    global __cublasZtpmv_v2_64
    __cublasZtpmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtpmv_v2_64')
    if __cublasZtpmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtpmv_v2_64 = dlsym(handle, 'cublasZtpmv_v2_64')

    global __cublasStrsv_v2_64
    __cublasStrsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStrsv_v2_64')
    if __cublasStrsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrsv_v2_64 = dlsym(handle, 'cublasStrsv_v2_64')

    global __cublasDtrsv_v2_64
    __cublasDtrsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtrsv_v2_64')
    if __cublasDtrsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrsv_v2_64 = dlsym(handle, 'cublasDtrsv_v2_64')

    global __cublasCtrsv_v2_64
    __cublasCtrsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtrsv_v2_64')
    if __cublasCtrsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrsv_v2_64 = dlsym(handle, 'cublasCtrsv_v2_64')

    global __cublasZtrsv_v2_64
    __cublasZtrsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtrsv_v2_64')
    if __cublasZtrsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrsv_v2_64 = dlsym(handle, 'cublasZtrsv_v2_64')

    global __cublasStpsv_v2_64
    __cublasStpsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStpsv_v2_64')
    if __cublasStpsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStpsv_v2_64 = dlsym(handle, 'cublasStpsv_v2_64')

    global __cublasDtpsv_v2_64
    __cublasDtpsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtpsv_v2_64')
    if __cublasDtpsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtpsv_v2_64 = dlsym(handle, 'cublasDtpsv_v2_64')

    global __cublasCtpsv_v2_64
    __cublasCtpsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtpsv_v2_64')
    if __cublasCtpsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtpsv_v2_64 = dlsym(handle, 'cublasCtpsv_v2_64')

    global __cublasZtpsv_v2_64
    __cublasZtpsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtpsv_v2_64')
    if __cublasZtpsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtpsv_v2_64 = dlsym(handle, 'cublasZtpsv_v2_64')

    global __cublasStbsv_v2_64
    __cublasStbsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStbsv_v2_64')
    if __cublasStbsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStbsv_v2_64 = dlsym(handle, 'cublasStbsv_v2_64')

    global __cublasDtbsv_v2_64
    __cublasDtbsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtbsv_v2_64')
    if __cublasDtbsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtbsv_v2_64 = dlsym(handle, 'cublasDtbsv_v2_64')

    global __cublasCtbsv_v2_64
    __cublasCtbsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtbsv_v2_64')
    if __cublasCtbsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtbsv_v2_64 = dlsym(handle, 'cublasCtbsv_v2_64')

    global __cublasZtbsv_v2_64
    __cublasZtbsv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtbsv_v2_64')
    if __cublasZtbsv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtbsv_v2_64 = dlsym(handle, 'cublasZtbsv_v2_64')

    global __cublasSsymv_v2_64
    __cublasSsymv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsymv_v2_64')
    if __cublasSsymv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsymv_v2_64 = dlsym(handle, 'cublasSsymv_v2_64')

    global __cublasDsymv_v2_64
    __cublasDsymv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsymv_v2_64')
    if __cublasDsymv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsymv_v2_64 = dlsym(handle, 'cublasDsymv_v2_64')

    global __cublasCsymv_v2_64
    __cublasCsymv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsymv_v2_64')
    if __cublasCsymv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsymv_v2_64 = dlsym(handle, 'cublasCsymv_v2_64')

    global __cublasZsymv_v2_64
    __cublasZsymv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZsymv_v2_64')
    if __cublasZsymv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsymv_v2_64 = dlsym(handle, 'cublasZsymv_v2_64')

    global __cublasChemv_v2_64
    __cublasChemv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasChemv_v2_64')
    if __cublasChemv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChemv_v2_64 = dlsym(handle, 'cublasChemv_v2_64')

    global __cublasZhemv_v2_64
    __cublasZhemv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZhemv_v2_64')
    if __cublasZhemv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhemv_v2_64 = dlsym(handle, 'cublasZhemv_v2_64')

    global __cublasSsbmv_v2_64
    __cublasSsbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsbmv_v2_64')
    if __cublasSsbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsbmv_v2_64 = dlsym(handle, 'cublasSsbmv_v2_64')

    global __cublasDsbmv_v2_64
    __cublasDsbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsbmv_v2_64')
    if __cublasDsbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsbmv_v2_64 = dlsym(handle, 'cublasDsbmv_v2_64')

    global __cublasChbmv_v2_64
    __cublasChbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasChbmv_v2_64')
    if __cublasChbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChbmv_v2_64 = dlsym(handle, 'cublasChbmv_v2_64')

    global __cublasZhbmv_v2_64
    __cublasZhbmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZhbmv_v2_64')
    if __cublasZhbmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhbmv_v2_64 = dlsym(handle, 'cublasZhbmv_v2_64')

    global __cublasSspmv_v2_64
    __cublasSspmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSspmv_v2_64')
    if __cublasSspmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSspmv_v2_64 = dlsym(handle, 'cublasSspmv_v2_64')

    global __cublasDspmv_v2_64
    __cublasDspmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDspmv_v2_64')
    if __cublasDspmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDspmv_v2_64 = dlsym(handle, 'cublasDspmv_v2_64')

    global __cublasChpmv_v2_64
    __cublasChpmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasChpmv_v2_64')
    if __cublasChpmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChpmv_v2_64 = dlsym(handle, 'cublasChpmv_v2_64')

    global __cublasZhpmv_v2_64
    __cublasZhpmv_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZhpmv_v2_64')
    if __cublasZhpmv_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhpmv_v2_64 = dlsym(handle, 'cublasZhpmv_v2_64')

    global __cublasSger_v2_64
    __cublasSger_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSger_v2_64')
    if __cublasSger_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSger_v2_64 = dlsym(handle, 'cublasSger_v2_64')

    global __cublasDger_v2_64
    __cublasDger_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDger_v2_64')
    if __cublasDger_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDger_v2_64 = dlsym(handle, 'cublasDger_v2_64')

    global __cublasCgeru_v2_64
    __cublasCgeru_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCgeru_v2_64')
    if __cublasCgeru_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgeru_v2_64 = dlsym(handle, 'cublasCgeru_v2_64')

    global __cublasCgerc_v2_64
    __cublasCgerc_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCgerc_v2_64')
    if __cublasCgerc_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgerc_v2_64 = dlsym(handle, 'cublasCgerc_v2_64')

    global __cublasZgeru_v2_64
    __cublasZgeru_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZgeru_v2_64')
    if __cublasZgeru_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgeru_v2_64 = dlsym(handle, 'cublasZgeru_v2_64')

    global __cublasZgerc_v2_64
    __cublasZgerc_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZgerc_v2_64')
    if __cublasZgerc_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgerc_v2_64 = dlsym(handle, 'cublasZgerc_v2_64')

    global __cublasSsyr_v2_64
    __cublasSsyr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsyr_v2_64')
    if __cublasSsyr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyr_v2_64 = dlsym(handle, 'cublasSsyr_v2_64')

    global __cublasDsyr_v2_64
    __cublasDsyr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsyr_v2_64')
    if __cublasDsyr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyr_v2_64 = dlsym(handle, 'cublasDsyr_v2_64')

    global __cublasCsyr_v2_64
    __cublasCsyr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsyr_v2_64')
    if __cublasCsyr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyr_v2_64 = dlsym(handle, 'cublasCsyr_v2_64')

    global __cublasZsyr_v2_64
    __cublasZsyr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZsyr_v2_64')
    if __cublasZsyr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyr_v2_64 = dlsym(handle, 'cublasZsyr_v2_64')

    global __cublasCher_v2_64
    __cublasCher_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCher_v2_64')
    if __cublasCher_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCher_v2_64 = dlsym(handle, 'cublasCher_v2_64')

    global __cublasZher_v2_64
    __cublasZher_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZher_v2_64')
    if __cublasZher_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZher_v2_64 = dlsym(handle, 'cublasZher_v2_64')

    global __cublasSspr_v2_64
    __cublasSspr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSspr_v2_64')
    if __cublasSspr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSspr_v2_64 = dlsym(handle, 'cublasSspr_v2_64')

    global __cublasDspr_v2_64
    __cublasDspr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDspr_v2_64')
    if __cublasDspr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDspr_v2_64 = dlsym(handle, 'cublasDspr_v2_64')

    global __cublasChpr_v2_64
    __cublasChpr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasChpr_v2_64')
    if __cublasChpr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChpr_v2_64 = dlsym(handle, 'cublasChpr_v2_64')

    global __cublasZhpr_v2_64
    __cublasZhpr_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZhpr_v2_64')
    if __cublasZhpr_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhpr_v2_64 = dlsym(handle, 'cublasZhpr_v2_64')

    global __cublasSsyr2_v2_64
    __cublasSsyr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsyr2_v2_64')
    if __cublasSsyr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyr2_v2_64 = dlsym(handle, 'cublasSsyr2_v2_64')

    global __cublasDsyr2_v2_64
    __cublasDsyr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsyr2_v2_64')
    if __cublasDsyr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyr2_v2_64 = dlsym(handle, 'cublasDsyr2_v2_64')

    global __cublasCsyr2_v2_64
    __cublasCsyr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsyr2_v2_64')
    if __cublasCsyr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyr2_v2_64 = dlsym(handle, 'cublasCsyr2_v2_64')

    global __cublasZsyr2_v2_64
    __cublasZsyr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZsyr2_v2_64')
    if __cublasZsyr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyr2_v2_64 = dlsym(handle, 'cublasZsyr2_v2_64')

    global __cublasCher2_v2_64
    __cublasCher2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCher2_v2_64')
    if __cublasCher2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCher2_v2_64 = dlsym(handle, 'cublasCher2_v2_64')

    global __cublasZher2_v2_64
    __cublasZher2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZher2_v2_64')
    if __cublasZher2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZher2_v2_64 = dlsym(handle, 'cublasZher2_v2_64')

    global __cublasSspr2_v2_64
    __cublasSspr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSspr2_v2_64')
    if __cublasSspr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSspr2_v2_64 = dlsym(handle, 'cublasSspr2_v2_64')

    global __cublasDspr2_v2_64
    __cublasDspr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDspr2_v2_64')
    if __cublasDspr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDspr2_v2_64 = dlsym(handle, 'cublasDspr2_v2_64')

    global __cublasChpr2_v2_64
    __cublasChpr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasChpr2_v2_64')
    if __cublasChpr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChpr2_v2_64 = dlsym(handle, 'cublasChpr2_v2_64')

    global __cublasZhpr2_v2_64
    __cublasZhpr2_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZhpr2_v2_64')
    if __cublasZhpr2_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhpr2_v2_64 = dlsym(handle, 'cublasZhpr2_v2_64')

    global __cublasSgemvBatched_64
    __cublasSgemvBatched_64 = dlsym(RTLD_DEFAULT, 'cublasSgemvBatched_64')
    if __cublasSgemvBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemvBatched_64 = dlsym(handle, 'cublasSgemvBatched_64')

    global __cublasDgemvBatched_64
    __cublasDgemvBatched_64 = dlsym(RTLD_DEFAULT, 'cublasDgemvBatched_64')
    if __cublasDgemvBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemvBatched_64 = dlsym(handle, 'cublasDgemvBatched_64')

    global __cublasCgemvBatched_64
    __cublasCgemvBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCgemvBatched_64')
    if __cublasCgemvBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemvBatched_64 = dlsym(handle, 'cublasCgemvBatched_64')

    global __cublasZgemvBatched_64
    __cublasZgemvBatched_64 = dlsym(RTLD_DEFAULT, 'cublasZgemvBatched_64')
    if __cublasZgemvBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemvBatched_64 = dlsym(handle, 'cublasZgemvBatched_64')

    global __cublasSgemvStridedBatched_64
    __cublasSgemvStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasSgemvStridedBatched_64')
    if __cublasSgemvStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemvStridedBatched_64 = dlsym(handle, 'cublasSgemvStridedBatched_64')

    global __cublasDgemvStridedBatched_64
    __cublasDgemvStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasDgemvStridedBatched_64')
    if __cublasDgemvStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemvStridedBatched_64 = dlsym(handle, 'cublasDgemvStridedBatched_64')

    global __cublasCgemvStridedBatched_64
    __cublasCgemvStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCgemvStridedBatched_64')
    if __cublasCgemvStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemvStridedBatched_64 = dlsym(handle, 'cublasCgemvStridedBatched_64')

    global __cublasZgemvStridedBatched_64
    __cublasZgemvStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasZgemvStridedBatched_64')
    if __cublasZgemvStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemvStridedBatched_64 = dlsym(handle, 'cublasZgemvStridedBatched_64')

    global __cublasSgemm_v2_64
    __cublasSgemm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSgemm_v2_64')
    if __cublasSgemm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemm_v2_64 = dlsym(handle, 'cublasSgemm_v2_64')

    global __cublasDgemm_v2_64
    __cublasDgemm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDgemm_v2_64')
    if __cublasDgemm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemm_v2_64 = dlsym(handle, 'cublasDgemm_v2_64')

    global __cublasCgemm_v2_64
    __cublasCgemm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCgemm_v2_64')
    if __cublasCgemm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm_v2_64 = dlsym(handle, 'cublasCgemm_v2_64')

    global __cublasCgemm3m_64
    __cublasCgemm3m_64 = dlsym(RTLD_DEFAULT, 'cublasCgemm3m_64')
    if __cublasCgemm3m_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3m_64 = dlsym(handle, 'cublasCgemm3m_64')

    global __cublasCgemm3mEx_64
    __cublasCgemm3mEx_64 = dlsym(RTLD_DEFAULT, 'cublasCgemm3mEx_64')
    if __cublasCgemm3mEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3mEx_64 = dlsym(handle, 'cublasCgemm3mEx_64')

    global __cublasZgemm_v2_64
    __cublasZgemm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZgemm_v2_64')
    if __cublasZgemm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemm_v2_64 = dlsym(handle, 'cublasZgemm_v2_64')

    global __cublasZgemm3m_64
    __cublasZgemm3m_64 = dlsym(RTLD_DEFAULT, 'cublasZgemm3m_64')
    if __cublasZgemm3m_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemm3m_64 = dlsym(handle, 'cublasZgemm3m_64')

    global __cublasSgemmEx_64
    __cublasSgemmEx_64 = dlsym(RTLD_DEFAULT, 'cublasSgemmEx_64')
    if __cublasSgemmEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemmEx_64 = dlsym(handle, 'cublasSgemmEx_64')

    global __cublasGemmEx_64
    __cublasGemmEx_64 = dlsym(RTLD_DEFAULT, 'cublasGemmEx_64')
    if __cublasGemmEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGemmEx_64 = dlsym(handle, 'cublasGemmEx_64')

    global __cublasCgemmEx_64
    __cublasCgemmEx_64 = dlsym(RTLD_DEFAULT, 'cublasCgemmEx_64')
    if __cublasCgemmEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemmEx_64 = dlsym(handle, 'cublasCgemmEx_64')

    global __cublasSsyrk_v2_64
    __cublasSsyrk_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsyrk_v2_64')
    if __cublasSsyrk_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyrk_v2_64 = dlsym(handle, 'cublasSsyrk_v2_64')

    global __cublasDsyrk_v2_64
    __cublasDsyrk_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsyrk_v2_64')
    if __cublasDsyrk_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyrk_v2_64 = dlsym(handle, 'cublasDsyrk_v2_64')

    global __cublasCsyrk_v2_64
    __cublasCsyrk_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsyrk_v2_64')
    if __cublasCsyrk_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrk_v2_64 = dlsym(handle, 'cublasCsyrk_v2_64')

    global __cublasZsyrk_v2_64
    __cublasZsyrk_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZsyrk_v2_64')
    if __cublasZsyrk_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyrk_v2_64 = dlsym(handle, 'cublasZsyrk_v2_64')

    global __cublasCsyrkEx_64
    __cublasCsyrkEx_64 = dlsym(RTLD_DEFAULT, 'cublasCsyrkEx_64')
    if __cublasCsyrkEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrkEx_64 = dlsym(handle, 'cublasCsyrkEx_64')

    global __cublasCsyrk3mEx_64
    __cublasCsyrk3mEx_64 = dlsym(RTLD_DEFAULT, 'cublasCsyrk3mEx_64')
    if __cublasCsyrk3mEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrk3mEx_64 = dlsym(handle, 'cublasCsyrk3mEx_64')

    global __cublasCherk_v2_64
    __cublasCherk_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCherk_v2_64')
    if __cublasCherk_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherk_v2_64 = dlsym(handle, 'cublasCherk_v2_64')

    global __cublasZherk_v2_64
    __cublasZherk_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZherk_v2_64')
    if __cublasZherk_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZherk_v2_64 = dlsym(handle, 'cublasZherk_v2_64')

    global __cublasCherkEx_64
    __cublasCherkEx_64 = dlsym(RTLD_DEFAULT, 'cublasCherkEx_64')
    if __cublasCherkEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherkEx_64 = dlsym(handle, 'cublasCherkEx_64')

    global __cublasCherk3mEx_64
    __cublasCherk3mEx_64 = dlsym(RTLD_DEFAULT, 'cublasCherk3mEx_64')
    if __cublasCherk3mEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherk3mEx_64 = dlsym(handle, 'cublasCherk3mEx_64')

    global __cublasSsyr2k_v2_64
    __cublasSsyr2k_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsyr2k_v2_64')
    if __cublasSsyr2k_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyr2k_v2_64 = dlsym(handle, 'cublasSsyr2k_v2_64')

    global __cublasDsyr2k_v2_64
    __cublasDsyr2k_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsyr2k_v2_64')
    if __cublasDsyr2k_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyr2k_v2_64 = dlsym(handle, 'cublasDsyr2k_v2_64')

    global __cublasCsyr2k_v2_64
    __cublasCsyr2k_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsyr2k_v2_64')
    if __cublasCsyr2k_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyr2k_v2_64 = dlsym(handle, 'cublasCsyr2k_v2_64')

    global __cublasZsyr2k_v2_64
    __cublasZsyr2k_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZsyr2k_v2_64')
    if __cublasZsyr2k_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyr2k_v2_64 = dlsym(handle, 'cublasZsyr2k_v2_64')

    global __cublasCher2k_v2_64
    __cublasCher2k_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCher2k_v2_64')
    if __cublasCher2k_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCher2k_v2_64 = dlsym(handle, 'cublasCher2k_v2_64')

    global __cublasZher2k_v2_64
    __cublasZher2k_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZher2k_v2_64')
    if __cublasZher2k_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZher2k_v2_64 = dlsym(handle, 'cublasZher2k_v2_64')

    global __cublasSsyrkx_64
    __cublasSsyrkx_64 = dlsym(RTLD_DEFAULT, 'cublasSsyrkx_64')
    if __cublasSsyrkx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsyrkx_64 = dlsym(handle, 'cublasSsyrkx_64')

    global __cublasDsyrkx_64
    __cublasDsyrkx_64 = dlsym(RTLD_DEFAULT, 'cublasDsyrkx_64')
    if __cublasDsyrkx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsyrkx_64 = dlsym(handle, 'cublasDsyrkx_64')

    global __cublasCsyrkx_64
    __cublasCsyrkx_64 = dlsym(RTLD_DEFAULT, 'cublasCsyrkx_64')
    if __cublasCsyrkx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsyrkx_64 = dlsym(handle, 'cublasCsyrkx_64')

    global __cublasZsyrkx_64
    __cublasZsyrkx_64 = dlsym(RTLD_DEFAULT, 'cublasZsyrkx_64')
    if __cublasZsyrkx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsyrkx_64 = dlsym(handle, 'cublasZsyrkx_64')

    global __cublasCherkx_64
    __cublasCherkx_64 = dlsym(RTLD_DEFAULT, 'cublasCherkx_64')
    if __cublasCherkx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCherkx_64 = dlsym(handle, 'cublasCherkx_64')

    global __cublasZherkx_64
    __cublasZherkx_64 = dlsym(RTLD_DEFAULT, 'cublasZherkx_64')
    if __cublasZherkx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZherkx_64 = dlsym(handle, 'cublasZherkx_64')

    global __cublasSsymm_v2_64
    __cublasSsymm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasSsymm_v2_64')
    if __cublasSsymm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSsymm_v2_64 = dlsym(handle, 'cublasSsymm_v2_64')

    global __cublasDsymm_v2_64
    __cublasDsymm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDsymm_v2_64')
    if __cublasDsymm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDsymm_v2_64 = dlsym(handle, 'cublasDsymm_v2_64')

    global __cublasCsymm_v2_64
    __cublasCsymm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCsymm_v2_64')
    if __cublasCsymm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCsymm_v2_64 = dlsym(handle, 'cublasCsymm_v2_64')

    global __cublasZsymm_v2_64
    __cublasZsymm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZsymm_v2_64')
    if __cublasZsymm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZsymm_v2_64 = dlsym(handle, 'cublasZsymm_v2_64')

    global __cublasChemm_v2_64
    __cublasChemm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasChemm_v2_64')
    if __cublasChemm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasChemm_v2_64 = dlsym(handle, 'cublasChemm_v2_64')

    global __cublasZhemm_v2_64
    __cublasZhemm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZhemm_v2_64')
    if __cublasZhemm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZhemm_v2_64 = dlsym(handle, 'cublasZhemm_v2_64')

    global __cublasStrsm_v2_64
    __cublasStrsm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStrsm_v2_64')
    if __cublasStrsm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrsm_v2_64 = dlsym(handle, 'cublasStrsm_v2_64')

    global __cublasDtrsm_v2_64
    __cublasDtrsm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtrsm_v2_64')
    if __cublasDtrsm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrsm_v2_64 = dlsym(handle, 'cublasDtrsm_v2_64')

    global __cublasCtrsm_v2_64
    __cublasCtrsm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtrsm_v2_64')
    if __cublasCtrsm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrsm_v2_64 = dlsym(handle, 'cublasCtrsm_v2_64')

    global __cublasZtrsm_v2_64
    __cublasZtrsm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtrsm_v2_64')
    if __cublasZtrsm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrsm_v2_64 = dlsym(handle, 'cublasZtrsm_v2_64')

    global __cublasStrmm_v2_64
    __cublasStrmm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasStrmm_v2_64')
    if __cublasStrmm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrmm_v2_64 = dlsym(handle, 'cublasStrmm_v2_64')

    global __cublasDtrmm_v2_64
    __cublasDtrmm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasDtrmm_v2_64')
    if __cublasDtrmm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrmm_v2_64 = dlsym(handle, 'cublasDtrmm_v2_64')

    global __cublasCtrmm_v2_64
    __cublasCtrmm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasCtrmm_v2_64')
    if __cublasCtrmm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrmm_v2_64 = dlsym(handle, 'cublasCtrmm_v2_64')

    global __cublasZtrmm_v2_64
    __cublasZtrmm_v2_64 = dlsym(RTLD_DEFAULT, 'cublasZtrmm_v2_64')
    if __cublasZtrmm_v2_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrmm_v2_64 = dlsym(handle, 'cublasZtrmm_v2_64')

    global __cublasSgemmBatched_64
    __cublasSgemmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasSgemmBatched_64')
    if __cublasSgemmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemmBatched_64 = dlsym(handle, 'cublasSgemmBatched_64')

    global __cublasDgemmBatched_64
    __cublasDgemmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasDgemmBatched_64')
    if __cublasDgemmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemmBatched_64 = dlsym(handle, 'cublasDgemmBatched_64')

    global __cublasCgemmBatched_64
    __cublasCgemmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCgemmBatched_64')
    if __cublasCgemmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemmBatched_64 = dlsym(handle, 'cublasCgemmBatched_64')

    global __cublasCgemm3mBatched_64
    __cublasCgemm3mBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCgemm3mBatched_64')
    if __cublasCgemm3mBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3mBatched_64 = dlsym(handle, 'cublasCgemm3mBatched_64')

    global __cublasZgemmBatched_64
    __cublasZgemmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasZgemmBatched_64')
    if __cublasZgemmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemmBatched_64 = dlsym(handle, 'cublasZgemmBatched_64')

    global __cublasSgemmStridedBatched_64
    __cublasSgemmStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasSgemmStridedBatched_64')
    if __cublasSgemmStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgemmStridedBatched_64 = dlsym(handle, 'cublasSgemmStridedBatched_64')

    global __cublasDgemmStridedBatched_64
    __cublasDgemmStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasDgemmStridedBatched_64')
    if __cublasDgemmStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgemmStridedBatched_64 = dlsym(handle, 'cublasDgemmStridedBatched_64')

    global __cublasCgemmStridedBatched_64
    __cublasCgemmStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCgemmStridedBatched_64')
    if __cublasCgemmStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemmStridedBatched_64 = dlsym(handle, 'cublasCgemmStridedBatched_64')

    global __cublasCgemm3mStridedBatched_64
    __cublasCgemm3mStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCgemm3mStridedBatched_64')
    if __cublasCgemm3mStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgemm3mStridedBatched_64 = dlsym(handle, 'cublasCgemm3mStridedBatched_64')

    global __cublasZgemmStridedBatched_64
    __cublasZgemmStridedBatched_64 = dlsym(RTLD_DEFAULT, 'cublasZgemmStridedBatched_64')
    if __cublasZgemmStridedBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgemmStridedBatched_64 = dlsym(handle, 'cublasZgemmStridedBatched_64')

    global __cublasGemmBatchedEx_64
    __cublasGemmBatchedEx_64 = dlsym(RTLD_DEFAULT, 'cublasGemmBatchedEx_64')
    if __cublasGemmBatchedEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGemmBatchedEx_64 = dlsym(handle, 'cublasGemmBatchedEx_64')

    global __cublasGemmStridedBatchedEx_64
    __cublasGemmStridedBatchedEx_64 = dlsym(RTLD_DEFAULT, 'cublasGemmStridedBatchedEx_64')
    if __cublasGemmStridedBatchedEx_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasGemmStridedBatchedEx_64 = dlsym(handle, 'cublasGemmStridedBatchedEx_64')

    global __cublasSgeam_64
    __cublasSgeam_64 = dlsym(RTLD_DEFAULT, 'cublasSgeam_64')
    if __cublasSgeam_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSgeam_64 = dlsym(handle, 'cublasSgeam_64')

    global __cublasDgeam_64
    __cublasDgeam_64 = dlsym(RTLD_DEFAULT, 'cublasDgeam_64')
    if __cublasDgeam_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDgeam_64 = dlsym(handle, 'cublasDgeam_64')

    global __cublasCgeam_64
    __cublasCgeam_64 = dlsym(RTLD_DEFAULT, 'cublasCgeam_64')
    if __cublasCgeam_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCgeam_64 = dlsym(handle, 'cublasCgeam_64')

    global __cublasZgeam_64
    __cublasZgeam_64 = dlsym(RTLD_DEFAULT, 'cublasZgeam_64')
    if __cublasZgeam_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZgeam_64 = dlsym(handle, 'cublasZgeam_64')

    global __cublasStrsmBatched_64
    __cublasStrsmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasStrsmBatched_64')
    if __cublasStrsmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasStrsmBatched_64 = dlsym(handle, 'cublasStrsmBatched_64')

    global __cublasDtrsmBatched_64
    __cublasDtrsmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasDtrsmBatched_64')
    if __cublasDtrsmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDtrsmBatched_64 = dlsym(handle, 'cublasDtrsmBatched_64')

    global __cublasCtrsmBatched_64
    __cublasCtrsmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasCtrsmBatched_64')
    if __cublasCtrsmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCtrsmBatched_64 = dlsym(handle, 'cublasCtrsmBatched_64')

    global __cublasZtrsmBatched_64
    __cublasZtrsmBatched_64 = dlsym(RTLD_DEFAULT, 'cublasZtrsmBatched_64')
    if __cublasZtrsmBatched_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZtrsmBatched_64 = dlsym(handle, 'cublasZtrsmBatched_64')

    global __cublasSdgmm_64
    __cublasSdgmm_64 = dlsym(RTLD_DEFAULT, 'cublasSdgmm_64')
    if __cublasSdgmm_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasSdgmm_64 = dlsym(handle, 'cublasSdgmm_64')

    global __cublasDdgmm_64
    __cublasDdgmm_64 = dlsym(RTLD_DEFAULT, 'cublasDdgmm_64')
    if __cublasDdgmm_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasDdgmm_64 = dlsym(handle, 'cublasDdgmm_64')

    global __cublasCdgmm_64
    __cublasCdgmm_64 = dlsym(RTLD_DEFAULT, 'cublasCdgmm_64')
    if __cublasCdgmm_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasCdgmm_64 = dlsym(handle, 'cublasCdgmm_64')

    global __cublasZdgmm_64
    __cublasZdgmm_64 = dlsym(RTLD_DEFAULT, 'cublasZdgmm_64')
    if __cublasZdgmm_64 == NULL:
        if handle == NULL:
            handle = load_library(driver_ver)
        __cublasZdgmm_64 = dlsym(handle, 'cublasZdgmm_64')

    __py_cublas_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_cublas()
    cdef dict data = {}

    global __cublasCreate_v2
    data["__cublasCreate_v2"] = <intptr_t>__cublasCreate_v2

    global __cublasDestroy_v2
    data["__cublasDestroy_v2"] = <intptr_t>__cublasDestroy_v2

    global __cublasGetVersion_v2
    data["__cublasGetVersion_v2"] = <intptr_t>__cublasGetVersion_v2

    global __cublasGetProperty
    data["__cublasGetProperty"] = <intptr_t>__cublasGetProperty

    global __cublasGetCudartVersion
    data["__cublasGetCudartVersion"] = <intptr_t>__cublasGetCudartVersion

    global __cublasSetWorkspace_v2
    data["__cublasSetWorkspace_v2"] = <intptr_t>__cublasSetWorkspace_v2

    global __cublasSetStream_v2
    data["__cublasSetStream_v2"] = <intptr_t>__cublasSetStream_v2

    global __cublasGetStream_v2
    data["__cublasGetStream_v2"] = <intptr_t>__cublasGetStream_v2

    global __cublasGetPointerMode_v2
    data["__cublasGetPointerMode_v2"] = <intptr_t>__cublasGetPointerMode_v2

    global __cublasSetPointerMode_v2
    data["__cublasSetPointerMode_v2"] = <intptr_t>__cublasSetPointerMode_v2

    global __cublasGetAtomicsMode
    data["__cublasGetAtomicsMode"] = <intptr_t>__cublasGetAtomicsMode

    global __cublasSetAtomicsMode
    data["__cublasSetAtomicsMode"] = <intptr_t>__cublasSetAtomicsMode

    global __cublasGetMathMode
    data["__cublasGetMathMode"] = <intptr_t>__cublasGetMathMode

    global __cublasSetMathMode
    data["__cublasSetMathMode"] = <intptr_t>__cublasSetMathMode

    global __cublasLoggerConfigure
    data["__cublasLoggerConfigure"] = <intptr_t>__cublasLoggerConfigure

    global __cublasSetLoggerCallback
    data["__cublasSetLoggerCallback"] = <intptr_t>__cublasSetLoggerCallback

    global __cublasGetLoggerCallback
    data["__cublasGetLoggerCallback"] = <intptr_t>__cublasGetLoggerCallback

    global __cublasSetVector
    data["__cublasSetVector"] = <intptr_t>__cublasSetVector

    global __cublasGetVector
    data["__cublasGetVector"] = <intptr_t>__cublasGetVector

    global __cublasSetMatrix
    data["__cublasSetMatrix"] = <intptr_t>__cublasSetMatrix

    global __cublasGetMatrix
    data["__cublasGetMatrix"] = <intptr_t>__cublasGetMatrix

    global __cublasSetVectorAsync
    data["__cublasSetVectorAsync"] = <intptr_t>__cublasSetVectorAsync

    global __cublasGetVectorAsync
    data["__cublasGetVectorAsync"] = <intptr_t>__cublasGetVectorAsync

    global __cublasSetMatrixAsync
    data["__cublasSetMatrixAsync"] = <intptr_t>__cublasSetMatrixAsync

    global __cublasGetMatrixAsync
    data["__cublasGetMatrixAsync"] = <intptr_t>__cublasGetMatrixAsync

    global __cublasNrm2Ex
    data["__cublasNrm2Ex"] = <intptr_t>__cublasNrm2Ex

    global __cublasSnrm2_v2
    data["__cublasSnrm2_v2"] = <intptr_t>__cublasSnrm2_v2

    global __cublasDnrm2_v2
    data["__cublasDnrm2_v2"] = <intptr_t>__cublasDnrm2_v2

    global __cublasScnrm2_v2
    data["__cublasScnrm2_v2"] = <intptr_t>__cublasScnrm2_v2

    global __cublasDznrm2_v2
    data["__cublasDznrm2_v2"] = <intptr_t>__cublasDznrm2_v2

    global __cublasDotEx
    data["__cublasDotEx"] = <intptr_t>__cublasDotEx

    global __cublasDotcEx
    data["__cublasDotcEx"] = <intptr_t>__cublasDotcEx

    global __cublasSdot_v2
    data["__cublasSdot_v2"] = <intptr_t>__cublasSdot_v2

    global __cublasDdot_v2
    data["__cublasDdot_v2"] = <intptr_t>__cublasDdot_v2

    global __cublasCdotu_v2
    data["__cublasCdotu_v2"] = <intptr_t>__cublasCdotu_v2

    global __cublasCdotc_v2
    data["__cublasCdotc_v2"] = <intptr_t>__cublasCdotc_v2

    global __cublasZdotu_v2
    data["__cublasZdotu_v2"] = <intptr_t>__cublasZdotu_v2

    global __cublasZdotc_v2
    data["__cublasZdotc_v2"] = <intptr_t>__cublasZdotc_v2

    global __cublasScalEx
    data["__cublasScalEx"] = <intptr_t>__cublasScalEx

    global __cublasSscal_v2
    data["__cublasSscal_v2"] = <intptr_t>__cublasSscal_v2

    global __cublasDscal_v2
    data["__cublasDscal_v2"] = <intptr_t>__cublasDscal_v2

    global __cublasCscal_v2
    data["__cublasCscal_v2"] = <intptr_t>__cublasCscal_v2

    global __cublasCsscal_v2
    data["__cublasCsscal_v2"] = <intptr_t>__cublasCsscal_v2

    global __cublasZscal_v2
    data["__cublasZscal_v2"] = <intptr_t>__cublasZscal_v2

    global __cublasZdscal_v2
    data["__cublasZdscal_v2"] = <intptr_t>__cublasZdscal_v2

    global __cublasAxpyEx
    data["__cublasAxpyEx"] = <intptr_t>__cublasAxpyEx

    global __cublasSaxpy_v2
    data["__cublasSaxpy_v2"] = <intptr_t>__cublasSaxpy_v2

    global __cublasDaxpy_v2
    data["__cublasDaxpy_v2"] = <intptr_t>__cublasDaxpy_v2

    global __cublasCaxpy_v2
    data["__cublasCaxpy_v2"] = <intptr_t>__cublasCaxpy_v2

    global __cublasZaxpy_v2
    data["__cublasZaxpy_v2"] = <intptr_t>__cublasZaxpy_v2

    global __cublasCopyEx
    data["__cublasCopyEx"] = <intptr_t>__cublasCopyEx

    global __cublasScopy_v2
    data["__cublasScopy_v2"] = <intptr_t>__cublasScopy_v2

    global __cublasDcopy_v2
    data["__cublasDcopy_v2"] = <intptr_t>__cublasDcopy_v2

    global __cublasCcopy_v2
    data["__cublasCcopy_v2"] = <intptr_t>__cublasCcopy_v2

    global __cublasZcopy_v2
    data["__cublasZcopy_v2"] = <intptr_t>__cublasZcopy_v2

    global __cublasSswap_v2
    data["__cublasSswap_v2"] = <intptr_t>__cublasSswap_v2

    global __cublasDswap_v2
    data["__cublasDswap_v2"] = <intptr_t>__cublasDswap_v2

    global __cublasCswap_v2
    data["__cublasCswap_v2"] = <intptr_t>__cublasCswap_v2

    global __cublasZswap_v2
    data["__cublasZswap_v2"] = <intptr_t>__cublasZswap_v2

    global __cublasSwapEx
    data["__cublasSwapEx"] = <intptr_t>__cublasSwapEx

    global __cublasIsamax_v2
    data["__cublasIsamax_v2"] = <intptr_t>__cublasIsamax_v2

    global __cublasIdamax_v2
    data["__cublasIdamax_v2"] = <intptr_t>__cublasIdamax_v2

    global __cublasIcamax_v2
    data["__cublasIcamax_v2"] = <intptr_t>__cublasIcamax_v2

    global __cublasIzamax_v2
    data["__cublasIzamax_v2"] = <intptr_t>__cublasIzamax_v2

    global __cublasIamaxEx
    data["__cublasIamaxEx"] = <intptr_t>__cublasIamaxEx

    global __cublasIsamin_v2
    data["__cublasIsamin_v2"] = <intptr_t>__cublasIsamin_v2

    global __cublasIdamin_v2
    data["__cublasIdamin_v2"] = <intptr_t>__cublasIdamin_v2

    global __cublasIcamin_v2
    data["__cublasIcamin_v2"] = <intptr_t>__cublasIcamin_v2

    global __cublasIzamin_v2
    data["__cublasIzamin_v2"] = <intptr_t>__cublasIzamin_v2

    global __cublasIaminEx
    data["__cublasIaminEx"] = <intptr_t>__cublasIaminEx

    global __cublasAsumEx
    data["__cublasAsumEx"] = <intptr_t>__cublasAsumEx

    global __cublasSasum_v2
    data["__cublasSasum_v2"] = <intptr_t>__cublasSasum_v2

    global __cublasDasum_v2
    data["__cublasDasum_v2"] = <intptr_t>__cublasDasum_v2

    global __cublasScasum_v2
    data["__cublasScasum_v2"] = <intptr_t>__cublasScasum_v2

    global __cublasDzasum_v2
    data["__cublasDzasum_v2"] = <intptr_t>__cublasDzasum_v2

    global __cublasSrot_v2
    data["__cublasSrot_v2"] = <intptr_t>__cublasSrot_v2

    global __cublasDrot_v2
    data["__cublasDrot_v2"] = <intptr_t>__cublasDrot_v2

    global __cublasCrot_v2
    data["__cublasCrot_v2"] = <intptr_t>__cublasCrot_v2

    global __cublasCsrot_v2
    data["__cublasCsrot_v2"] = <intptr_t>__cublasCsrot_v2

    global __cublasZrot_v2
    data["__cublasZrot_v2"] = <intptr_t>__cublasZrot_v2

    global __cublasZdrot_v2
    data["__cublasZdrot_v2"] = <intptr_t>__cublasZdrot_v2

    global __cublasRotEx
    data["__cublasRotEx"] = <intptr_t>__cublasRotEx

    global __cublasSrotg_v2
    data["__cublasSrotg_v2"] = <intptr_t>__cublasSrotg_v2

    global __cublasDrotg_v2
    data["__cublasDrotg_v2"] = <intptr_t>__cublasDrotg_v2

    global __cublasCrotg_v2
    data["__cublasCrotg_v2"] = <intptr_t>__cublasCrotg_v2

    global __cublasZrotg_v2
    data["__cublasZrotg_v2"] = <intptr_t>__cublasZrotg_v2

    global __cublasRotgEx
    data["__cublasRotgEx"] = <intptr_t>__cublasRotgEx

    global __cublasSrotm_v2
    data["__cublasSrotm_v2"] = <intptr_t>__cublasSrotm_v2

    global __cublasDrotm_v2
    data["__cublasDrotm_v2"] = <intptr_t>__cublasDrotm_v2

    global __cublasRotmEx
    data["__cublasRotmEx"] = <intptr_t>__cublasRotmEx

    global __cublasSrotmg_v2
    data["__cublasSrotmg_v2"] = <intptr_t>__cublasSrotmg_v2

    global __cublasDrotmg_v2
    data["__cublasDrotmg_v2"] = <intptr_t>__cublasDrotmg_v2

    global __cublasRotmgEx
    data["__cublasRotmgEx"] = <intptr_t>__cublasRotmgEx

    global __cublasSgemv_v2
    data["__cublasSgemv_v2"] = <intptr_t>__cublasSgemv_v2

    global __cublasDgemv_v2
    data["__cublasDgemv_v2"] = <intptr_t>__cublasDgemv_v2

    global __cublasCgemv_v2
    data["__cublasCgemv_v2"] = <intptr_t>__cublasCgemv_v2

    global __cublasZgemv_v2
    data["__cublasZgemv_v2"] = <intptr_t>__cublasZgemv_v2

    global __cublasSgbmv_v2
    data["__cublasSgbmv_v2"] = <intptr_t>__cublasSgbmv_v2

    global __cublasDgbmv_v2
    data["__cublasDgbmv_v2"] = <intptr_t>__cublasDgbmv_v2

    global __cublasCgbmv_v2
    data["__cublasCgbmv_v2"] = <intptr_t>__cublasCgbmv_v2

    global __cublasZgbmv_v2
    data["__cublasZgbmv_v2"] = <intptr_t>__cublasZgbmv_v2

    global __cublasStrmv_v2
    data["__cublasStrmv_v2"] = <intptr_t>__cublasStrmv_v2

    global __cublasDtrmv_v2
    data["__cublasDtrmv_v2"] = <intptr_t>__cublasDtrmv_v2

    global __cublasCtrmv_v2
    data["__cublasCtrmv_v2"] = <intptr_t>__cublasCtrmv_v2

    global __cublasZtrmv_v2
    data["__cublasZtrmv_v2"] = <intptr_t>__cublasZtrmv_v2

    global __cublasStbmv_v2
    data["__cublasStbmv_v2"] = <intptr_t>__cublasStbmv_v2

    global __cublasDtbmv_v2
    data["__cublasDtbmv_v2"] = <intptr_t>__cublasDtbmv_v2

    global __cublasCtbmv_v2
    data["__cublasCtbmv_v2"] = <intptr_t>__cublasCtbmv_v2

    global __cublasZtbmv_v2
    data["__cublasZtbmv_v2"] = <intptr_t>__cublasZtbmv_v2

    global __cublasStpmv_v2
    data["__cublasStpmv_v2"] = <intptr_t>__cublasStpmv_v2

    global __cublasDtpmv_v2
    data["__cublasDtpmv_v2"] = <intptr_t>__cublasDtpmv_v2

    global __cublasCtpmv_v2
    data["__cublasCtpmv_v2"] = <intptr_t>__cublasCtpmv_v2

    global __cublasZtpmv_v2
    data["__cublasZtpmv_v2"] = <intptr_t>__cublasZtpmv_v2

    global __cublasStrsv_v2
    data["__cublasStrsv_v2"] = <intptr_t>__cublasStrsv_v2

    global __cublasDtrsv_v2
    data["__cublasDtrsv_v2"] = <intptr_t>__cublasDtrsv_v2

    global __cublasCtrsv_v2
    data["__cublasCtrsv_v2"] = <intptr_t>__cublasCtrsv_v2

    global __cublasZtrsv_v2
    data["__cublasZtrsv_v2"] = <intptr_t>__cublasZtrsv_v2

    global __cublasStpsv_v2
    data["__cublasStpsv_v2"] = <intptr_t>__cublasStpsv_v2

    global __cublasDtpsv_v2
    data["__cublasDtpsv_v2"] = <intptr_t>__cublasDtpsv_v2

    global __cublasCtpsv_v2
    data["__cublasCtpsv_v2"] = <intptr_t>__cublasCtpsv_v2

    global __cublasZtpsv_v2
    data["__cublasZtpsv_v2"] = <intptr_t>__cublasZtpsv_v2

    global __cublasStbsv_v2
    data["__cublasStbsv_v2"] = <intptr_t>__cublasStbsv_v2

    global __cublasDtbsv_v2
    data["__cublasDtbsv_v2"] = <intptr_t>__cublasDtbsv_v2

    global __cublasCtbsv_v2
    data["__cublasCtbsv_v2"] = <intptr_t>__cublasCtbsv_v2

    global __cublasZtbsv_v2
    data["__cublasZtbsv_v2"] = <intptr_t>__cublasZtbsv_v2

    global __cublasSsymv_v2
    data["__cublasSsymv_v2"] = <intptr_t>__cublasSsymv_v2

    global __cublasDsymv_v2
    data["__cublasDsymv_v2"] = <intptr_t>__cublasDsymv_v2

    global __cublasCsymv_v2
    data["__cublasCsymv_v2"] = <intptr_t>__cublasCsymv_v2

    global __cublasZsymv_v2
    data["__cublasZsymv_v2"] = <intptr_t>__cublasZsymv_v2

    global __cublasChemv_v2
    data["__cublasChemv_v2"] = <intptr_t>__cublasChemv_v2

    global __cublasZhemv_v2
    data["__cublasZhemv_v2"] = <intptr_t>__cublasZhemv_v2

    global __cublasSsbmv_v2
    data["__cublasSsbmv_v2"] = <intptr_t>__cublasSsbmv_v2

    global __cublasDsbmv_v2
    data["__cublasDsbmv_v2"] = <intptr_t>__cublasDsbmv_v2

    global __cublasChbmv_v2
    data["__cublasChbmv_v2"] = <intptr_t>__cublasChbmv_v2

    global __cublasZhbmv_v2
    data["__cublasZhbmv_v2"] = <intptr_t>__cublasZhbmv_v2

    global __cublasSspmv_v2
    data["__cublasSspmv_v2"] = <intptr_t>__cublasSspmv_v2

    global __cublasDspmv_v2
    data["__cublasDspmv_v2"] = <intptr_t>__cublasDspmv_v2

    global __cublasChpmv_v2
    data["__cublasChpmv_v2"] = <intptr_t>__cublasChpmv_v2

    global __cublasZhpmv_v2
    data["__cublasZhpmv_v2"] = <intptr_t>__cublasZhpmv_v2

    global __cublasSger_v2
    data["__cublasSger_v2"] = <intptr_t>__cublasSger_v2

    global __cublasDger_v2
    data["__cublasDger_v2"] = <intptr_t>__cublasDger_v2

    global __cublasCgeru_v2
    data["__cublasCgeru_v2"] = <intptr_t>__cublasCgeru_v2

    global __cublasCgerc_v2
    data["__cublasCgerc_v2"] = <intptr_t>__cublasCgerc_v2

    global __cublasZgeru_v2
    data["__cublasZgeru_v2"] = <intptr_t>__cublasZgeru_v2

    global __cublasZgerc_v2
    data["__cublasZgerc_v2"] = <intptr_t>__cublasZgerc_v2

    global __cublasSsyr_v2
    data["__cublasSsyr_v2"] = <intptr_t>__cublasSsyr_v2

    global __cublasDsyr_v2
    data["__cublasDsyr_v2"] = <intptr_t>__cublasDsyr_v2

    global __cublasCsyr_v2
    data["__cublasCsyr_v2"] = <intptr_t>__cublasCsyr_v2

    global __cublasZsyr_v2
    data["__cublasZsyr_v2"] = <intptr_t>__cublasZsyr_v2

    global __cublasCher_v2
    data["__cublasCher_v2"] = <intptr_t>__cublasCher_v2

    global __cublasZher_v2
    data["__cublasZher_v2"] = <intptr_t>__cublasZher_v2

    global __cublasSspr_v2
    data["__cublasSspr_v2"] = <intptr_t>__cublasSspr_v2

    global __cublasDspr_v2
    data["__cublasDspr_v2"] = <intptr_t>__cublasDspr_v2

    global __cublasChpr_v2
    data["__cublasChpr_v2"] = <intptr_t>__cublasChpr_v2

    global __cublasZhpr_v2
    data["__cublasZhpr_v2"] = <intptr_t>__cublasZhpr_v2

    global __cublasSsyr2_v2
    data["__cublasSsyr2_v2"] = <intptr_t>__cublasSsyr2_v2

    global __cublasDsyr2_v2
    data["__cublasDsyr2_v2"] = <intptr_t>__cublasDsyr2_v2

    global __cublasCsyr2_v2
    data["__cublasCsyr2_v2"] = <intptr_t>__cublasCsyr2_v2

    global __cublasZsyr2_v2
    data["__cublasZsyr2_v2"] = <intptr_t>__cublasZsyr2_v2

    global __cublasCher2_v2
    data["__cublasCher2_v2"] = <intptr_t>__cublasCher2_v2

    global __cublasZher2_v2
    data["__cublasZher2_v2"] = <intptr_t>__cublasZher2_v2

    global __cublasSspr2_v2
    data["__cublasSspr2_v2"] = <intptr_t>__cublasSspr2_v2

    global __cublasDspr2_v2
    data["__cublasDspr2_v2"] = <intptr_t>__cublasDspr2_v2

    global __cublasChpr2_v2
    data["__cublasChpr2_v2"] = <intptr_t>__cublasChpr2_v2

    global __cublasZhpr2_v2
    data["__cublasZhpr2_v2"] = <intptr_t>__cublasZhpr2_v2

    global __cublasSgemm_v2
    data["__cublasSgemm_v2"] = <intptr_t>__cublasSgemm_v2

    global __cublasDgemm_v2
    data["__cublasDgemm_v2"] = <intptr_t>__cublasDgemm_v2

    global __cublasCgemm_v2
    data["__cublasCgemm_v2"] = <intptr_t>__cublasCgemm_v2

    global __cublasCgemm3m
    data["__cublasCgemm3m"] = <intptr_t>__cublasCgemm3m

    global __cublasCgemm3mEx
    data["__cublasCgemm3mEx"] = <intptr_t>__cublasCgemm3mEx

    global __cublasZgemm_v2
    data["__cublasZgemm_v2"] = <intptr_t>__cublasZgemm_v2

    global __cublasZgemm3m
    data["__cublasZgemm3m"] = <intptr_t>__cublasZgemm3m

    global __cublasSgemmEx
    data["__cublasSgemmEx"] = <intptr_t>__cublasSgemmEx

    global __cublasGemmEx
    data["__cublasGemmEx"] = <intptr_t>__cublasGemmEx

    global __cublasCgemmEx
    data["__cublasCgemmEx"] = <intptr_t>__cublasCgemmEx

    global __cublasUint8gemmBias
    data["__cublasUint8gemmBias"] = <intptr_t>__cublasUint8gemmBias

    global __cublasSsyrk_v2
    data["__cublasSsyrk_v2"] = <intptr_t>__cublasSsyrk_v2

    global __cublasDsyrk_v2
    data["__cublasDsyrk_v2"] = <intptr_t>__cublasDsyrk_v2

    global __cublasCsyrk_v2
    data["__cublasCsyrk_v2"] = <intptr_t>__cublasCsyrk_v2

    global __cublasZsyrk_v2
    data["__cublasZsyrk_v2"] = <intptr_t>__cublasZsyrk_v2

    global __cublasCsyrkEx
    data["__cublasCsyrkEx"] = <intptr_t>__cublasCsyrkEx

    global __cublasCsyrk3mEx
    data["__cublasCsyrk3mEx"] = <intptr_t>__cublasCsyrk3mEx

    global __cublasCherk_v2
    data["__cublasCherk_v2"] = <intptr_t>__cublasCherk_v2

    global __cublasZherk_v2
    data["__cublasZherk_v2"] = <intptr_t>__cublasZherk_v2

    global __cublasCherkEx
    data["__cublasCherkEx"] = <intptr_t>__cublasCherkEx

    global __cublasCherk3mEx
    data["__cublasCherk3mEx"] = <intptr_t>__cublasCherk3mEx

    global __cublasSsyr2k_v2
    data["__cublasSsyr2k_v2"] = <intptr_t>__cublasSsyr2k_v2

    global __cublasDsyr2k_v2
    data["__cublasDsyr2k_v2"] = <intptr_t>__cublasDsyr2k_v2

    global __cublasCsyr2k_v2
    data["__cublasCsyr2k_v2"] = <intptr_t>__cublasCsyr2k_v2

    global __cublasZsyr2k_v2
    data["__cublasZsyr2k_v2"] = <intptr_t>__cublasZsyr2k_v2

    global __cublasCher2k_v2
    data["__cublasCher2k_v2"] = <intptr_t>__cublasCher2k_v2

    global __cublasZher2k_v2
    data["__cublasZher2k_v2"] = <intptr_t>__cublasZher2k_v2

    global __cublasSsyrkx
    data["__cublasSsyrkx"] = <intptr_t>__cublasSsyrkx

    global __cublasDsyrkx
    data["__cublasDsyrkx"] = <intptr_t>__cublasDsyrkx

    global __cublasCsyrkx
    data["__cublasCsyrkx"] = <intptr_t>__cublasCsyrkx

    global __cublasZsyrkx
    data["__cublasZsyrkx"] = <intptr_t>__cublasZsyrkx

    global __cublasCherkx
    data["__cublasCherkx"] = <intptr_t>__cublasCherkx

    global __cublasZherkx
    data["__cublasZherkx"] = <intptr_t>__cublasZherkx

    global __cublasSsymm_v2
    data["__cublasSsymm_v2"] = <intptr_t>__cublasSsymm_v2

    global __cublasDsymm_v2
    data["__cublasDsymm_v2"] = <intptr_t>__cublasDsymm_v2

    global __cublasCsymm_v2
    data["__cublasCsymm_v2"] = <intptr_t>__cublasCsymm_v2

    global __cublasZsymm_v2
    data["__cublasZsymm_v2"] = <intptr_t>__cublasZsymm_v2

    global __cublasChemm_v2
    data["__cublasChemm_v2"] = <intptr_t>__cublasChemm_v2

    global __cublasZhemm_v2
    data["__cublasZhemm_v2"] = <intptr_t>__cublasZhemm_v2

    global __cublasStrsm_v2
    data["__cublasStrsm_v2"] = <intptr_t>__cublasStrsm_v2

    global __cublasDtrsm_v2
    data["__cublasDtrsm_v2"] = <intptr_t>__cublasDtrsm_v2

    global __cublasCtrsm_v2
    data["__cublasCtrsm_v2"] = <intptr_t>__cublasCtrsm_v2

    global __cublasZtrsm_v2
    data["__cublasZtrsm_v2"] = <intptr_t>__cublasZtrsm_v2

    global __cublasStrmm_v2
    data["__cublasStrmm_v2"] = <intptr_t>__cublasStrmm_v2

    global __cublasDtrmm_v2
    data["__cublasDtrmm_v2"] = <intptr_t>__cublasDtrmm_v2

    global __cublasCtrmm_v2
    data["__cublasCtrmm_v2"] = <intptr_t>__cublasCtrmm_v2

    global __cublasZtrmm_v2
    data["__cublasZtrmm_v2"] = <intptr_t>__cublasZtrmm_v2

    global __cublasSgemmBatched
    data["__cublasSgemmBatched"] = <intptr_t>__cublasSgemmBatched

    global __cublasDgemmBatched
    data["__cublasDgemmBatched"] = <intptr_t>__cublasDgemmBatched

    global __cublasCgemmBatched
    data["__cublasCgemmBatched"] = <intptr_t>__cublasCgemmBatched

    global __cublasCgemm3mBatched
    data["__cublasCgemm3mBatched"] = <intptr_t>__cublasCgemm3mBatched

    global __cublasZgemmBatched
    data["__cublasZgemmBatched"] = <intptr_t>__cublasZgemmBatched

    global __cublasGemmBatchedEx
    data["__cublasGemmBatchedEx"] = <intptr_t>__cublasGemmBatchedEx

    global __cublasGemmStridedBatchedEx
    data["__cublasGemmStridedBatchedEx"] = <intptr_t>__cublasGemmStridedBatchedEx

    global __cublasSgemmStridedBatched
    data["__cublasSgemmStridedBatched"] = <intptr_t>__cublasSgemmStridedBatched

    global __cublasDgemmStridedBatched
    data["__cublasDgemmStridedBatched"] = <intptr_t>__cublasDgemmStridedBatched

    global __cublasCgemmStridedBatched
    data["__cublasCgemmStridedBatched"] = <intptr_t>__cublasCgemmStridedBatched

    global __cublasCgemm3mStridedBatched
    data["__cublasCgemm3mStridedBatched"] = <intptr_t>__cublasCgemm3mStridedBatched

    global __cublasZgemmStridedBatched
    data["__cublasZgemmStridedBatched"] = <intptr_t>__cublasZgemmStridedBatched

    global __cublasSgeam
    data["__cublasSgeam"] = <intptr_t>__cublasSgeam

    global __cublasDgeam
    data["__cublasDgeam"] = <intptr_t>__cublasDgeam

    global __cublasCgeam
    data["__cublasCgeam"] = <intptr_t>__cublasCgeam

    global __cublasZgeam
    data["__cublasZgeam"] = <intptr_t>__cublasZgeam

    global __cublasSgetrfBatched
    data["__cublasSgetrfBatched"] = <intptr_t>__cublasSgetrfBatched

    global __cublasDgetrfBatched
    data["__cublasDgetrfBatched"] = <intptr_t>__cublasDgetrfBatched

    global __cublasCgetrfBatched
    data["__cublasCgetrfBatched"] = <intptr_t>__cublasCgetrfBatched

    global __cublasZgetrfBatched
    data["__cublasZgetrfBatched"] = <intptr_t>__cublasZgetrfBatched

    global __cublasSgetriBatched
    data["__cublasSgetriBatched"] = <intptr_t>__cublasSgetriBatched

    global __cublasDgetriBatched
    data["__cublasDgetriBatched"] = <intptr_t>__cublasDgetriBatched

    global __cublasCgetriBatched
    data["__cublasCgetriBatched"] = <intptr_t>__cublasCgetriBatched

    global __cublasZgetriBatched
    data["__cublasZgetriBatched"] = <intptr_t>__cublasZgetriBatched

    global __cublasSgetrsBatched
    data["__cublasSgetrsBatched"] = <intptr_t>__cublasSgetrsBatched

    global __cublasDgetrsBatched
    data["__cublasDgetrsBatched"] = <intptr_t>__cublasDgetrsBatched

    global __cublasCgetrsBatched
    data["__cublasCgetrsBatched"] = <intptr_t>__cublasCgetrsBatched

    global __cublasZgetrsBatched
    data["__cublasZgetrsBatched"] = <intptr_t>__cublasZgetrsBatched

    global __cublasStrsmBatched
    data["__cublasStrsmBatched"] = <intptr_t>__cublasStrsmBatched

    global __cublasDtrsmBatched
    data["__cublasDtrsmBatched"] = <intptr_t>__cublasDtrsmBatched

    global __cublasCtrsmBatched
    data["__cublasCtrsmBatched"] = <intptr_t>__cublasCtrsmBatched

    global __cublasZtrsmBatched
    data["__cublasZtrsmBatched"] = <intptr_t>__cublasZtrsmBatched

    global __cublasSmatinvBatched
    data["__cublasSmatinvBatched"] = <intptr_t>__cublasSmatinvBatched

    global __cublasDmatinvBatched
    data["__cublasDmatinvBatched"] = <intptr_t>__cublasDmatinvBatched

    global __cublasCmatinvBatched
    data["__cublasCmatinvBatched"] = <intptr_t>__cublasCmatinvBatched

    global __cublasZmatinvBatched
    data["__cublasZmatinvBatched"] = <intptr_t>__cublasZmatinvBatched

    global __cublasSgeqrfBatched
    data["__cublasSgeqrfBatched"] = <intptr_t>__cublasSgeqrfBatched

    global __cublasDgeqrfBatched
    data["__cublasDgeqrfBatched"] = <intptr_t>__cublasDgeqrfBatched

    global __cublasCgeqrfBatched
    data["__cublasCgeqrfBatched"] = <intptr_t>__cublasCgeqrfBatched

    global __cublasZgeqrfBatched
    data["__cublasZgeqrfBatched"] = <intptr_t>__cublasZgeqrfBatched

    global __cublasSgelsBatched
    data["__cublasSgelsBatched"] = <intptr_t>__cublasSgelsBatched

    global __cublasDgelsBatched
    data["__cublasDgelsBatched"] = <intptr_t>__cublasDgelsBatched

    global __cublasCgelsBatched
    data["__cublasCgelsBatched"] = <intptr_t>__cublasCgelsBatched

    global __cublasZgelsBatched
    data["__cublasZgelsBatched"] = <intptr_t>__cublasZgelsBatched

    global __cublasSdgmm
    data["__cublasSdgmm"] = <intptr_t>__cublasSdgmm

    global __cublasDdgmm
    data["__cublasDdgmm"] = <intptr_t>__cublasDdgmm

    global __cublasCdgmm
    data["__cublasCdgmm"] = <intptr_t>__cublasCdgmm

    global __cublasZdgmm
    data["__cublasZdgmm"] = <intptr_t>__cublasZdgmm

    global __cublasStpttr
    data["__cublasStpttr"] = <intptr_t>__cublasStpttr

    global __cublasDtpttr
    data["__cublasDtpttr"] = <intptr_t>__cublasDtpttr

    global __cublasCtpttr
    data["__cublasCtpttr"] = <intptr_t>__cublasCtpttr

    global __cublasZtpttr
    data["__cublasZtpttr"] = <intptr_t>__cublasZtpttr

    global __cublasStrttp
    data["__cublasStrttp"] = <intptr_t>__cublasStrttp

    global __cublasDtrttp
    data["__cublasDtrttp"] = <intptr_t>__cublasDtrttp

    global __cublasCtrttp
    data["__cublasCtrttp"] = <intptr_t>__cublasCtrttp

    global __cublasZtrttp
    data["__cublasZtrttp"] = <intptr_t>__cublasZtrttp

    global __cublasGetSmCountTarget
    data["__cublasGetSmCountTarget"] = <intptr_t>__cublasGetSmCountTarget

    global __cublasSetSmCountTarget
    data["__cublasSetSmCountTarget"] = <intptr_t>__cublasSetSmCountTarget

    global __cublasGetStatusName
    data["__cublasGetStatusName"] = <intptr_t>__cublasGetStatusName

    global __cublasGetStatusString
    data["__cublasGetStatusString"] = <intptr_t>__cublasGetStatusString

    global __cublasSgemvBatched
    data["__cublasSgemvBatched"] = <intptr_t>__cublasSgemvBatched

    global __cublasDgemvBatched
    data["__cublasDgemvBatched"] = <intptr_t>__cublasDgemvBatched

    global __cublasCgemvBatched
    data["__cublasCgemvBatched"] = <intptr_t>__cublasCgemvBatched

    global __cublasZgemvBatched
    data["__cublasZgemvBatched"] = <intptr_t>__cublasZgemvBatched

    global __cublasSgemvStridedBatched
    data["__cublasSgemvStridedBatched"] = <intptr_t>__cublasSgemvStridedBatched

    global __cublasDgemvStridedBatched
    data["__cublasDgemvStridedBatched"] = <intptr_t>__cublasDgemvStridedBatched

    global __cublasCgemvStridedBatched
    data["__cublasCgemvStridedBatched"] = <intptr_t>__cublasCgemvStridedBatched

    global __cublasZgemvStridedBatched
    data["__cublasZgemvStridedBatched"] = <intptr_t>__cublasZgemvStridedBatched

    global __cublasSetVector_64
    data["__cublasSetVector_64"] = <intptr_t>__cublasSetVector_64

    global __cublasGetVector_64
    data["__cublasGetVector_64"] = <intptr_t>__cublasGetVector_64

    global __cublasSetMatrix_64
    data["__cublasSetMatrix_64"] = <intptr_t>__cublasSetMatrix_64

    global __cublasGetMatrix_64
    data["__cublasGetMatrix_64"] = <intptr_t>__cublasGetMatrix_64

    global __cublasSetVectorAsync_64
    data["__cublasSetVectorAsync_64"] = <intptr_t>__cublasSetVectorAsync_64

    global __cublasGetVectorAsync_64
    data["__cublasGetVectorAsync_64"] = <intptr_t>__cublasGetVectorAsync_64

    global __cublasSetMatrixAsync_64
    data["__cublasSetMatrixAsync_64"] = <intptr_t>__cublasSetMatrixAsync_64

    global __cublasGetMatrixAsync_64
    data["__cublasGetMatrixAsync_64"] = <intptr_t>__cublasGetMatrixAsync_64

    global __cublasNrm2Ex_64
    data["__cublasNrm2Ex_64"] = <intptr_t>__cublasNrm2Ex_64

    global __cublasSnrm2_v2_64
    data["__cublasSnrm2_v2_64"] = <intptr_t>__cublasSnrm2_v2_64

    global __cublasDnrm2_v2_64
    data["__cublasDnrm2_v2_64"] = <intptr_t>__cublasDnrm2_v2_64

    global __cublasScnrm2_v2_64
    data["__cublasScnrm2_v2_64"] = <intptr_t>__cublasScnrm2_v2_64

    global __cublasDznrm2_v2_64
    data["__cublasDznrm2_v2_64"] = <intptr_t>__cublasDznrm2_v2_64

    global __cublasDotEx_64
    data["__cublasDotEx_64"] = <intptr_t>__cublasDotEx_64

    global __cublasDotcEx_64
    data["__cublasDotcEx_64"] = <intptr_t>__cublasDotcEx_64

    global __cublasSdot_v2_64
    data["__cublasSdot_v2_64"] = <intptr_t>__cublasSdot_v2_64

    global __cublasDdot_v2_64
    data["__cublasDdot_v2_64"] = <intptr_t>__cublasDdot_v2_64

    global __cublasCdotu_v2_64
    data["__cublasCdotu_v2_64"] = <intptr_t>__cublasCdotu_v2_64

    global __cublasCdotc_v2_64
    data["__cublasCdotc_v2_64"] = <intptr_t>__cublasCdotc_v2_64

    global __cublasZdotu_v2_64
    data["__cublasZdotu_v2_64"] = <intptr_t>__cublasZdotu_v2_64

    global __cublasZdotc_v2_64
    data["__cublasZdotc_v2_64"] = <intptr_t>__cublasZdotc_v2_64

    global __cublasScalEx_64
    data["__cublasScalEx_64"] = <intptr_t>__cublasScalEx_64

    global __cublasSscal_v2_64
    data["__cublasSscal_v2_64"] = <intptr_t>__cublasSscal_v2_64

    global __cublasDscal_v2_64
    data["__cublasDscal_v2_64"] = <intptr_t>__cublasDscal_v2_64

    global __cublasCscal_v2_64
    data["__cublasCscal_v2_64"] = <intptr_t>__cublasCscal_v2_64

    global __cublasCsscal_v2_64
    data["__cublasCsscal_v2_64"] = <intptr_t>__cublasCsscal_v2_64

    global __cublasZscal_v2_64
    data["__cublasZscal_v2_64"] = <intptr_t>__cublasZscal_v2_64

    global __cublasZdscal_v2_64
    data["__cublasZdscal_v2_64"] = <intptr_t>__cublasZdscal_v2_64

    global __cublasAxpyEx_64
    data["__cublasAxpyEx_64"] = <intptr_t>__cublasAxpyEx_64

    global __cublasSaxpy_v2_64
    data["__cublasSaxpy_v2_64"] = <intptr_t>__cublasSaxpy_v2_64

    global __cublasDaxpy_v2_64
    data["__cublasDaxpy_v2_64"] = <intptr_t>__cublasDaxpy_v2_64

    global __cublasCaxpy_v2_64
    data["__cublasCaxpy_v2_64"] = <intptr_t>__cublasCaxpy_v2_64

    global __cublasZaxpy_v2_64
    data["__cublasZaxpy_v2_64"] = <intptr_t>__cublasZaxpy_v2_64

    global __cublasCopyEx_64
    data["__cublasCopyEx_64"] = <intptr_t>__cublasCopyEx_64

    global __cublasScopy_v2_64
    data["__cublasScopy_v2_64"] = <intptr_t>__cublasScopy_v2_64

    global __cublasDcopy_v2_64
    data["__cublasDcopy_v2_64"] = <intptr_t>__cublasDcopy_v2_64

    global __cublasCcopy_v2_64
    data["__cublasCcopy_v2_64"] = <intptr_t>__cublasCcopy_v2_64

    global __cublasZcopy_v2_64
    data["__cublasZcopy_v2_64"] = <intptr_t>__cublasZcopy_v2_64

    global __cublasSswap_v2_64
    data["__cublasSswap_v2_64"] = <intptr_t>__cublasSswap_v2_64

    global __cublasDswap_v2_64
    data["__cublasDswap_v2_64"] = <intptr_t>__cublasDswap_v2_64

    global __cublasCswap_v2_64
    data["__cublasCswap_v2_64"] = <intptr_t>__cublasCswap_v2_64

    global __cublasZswap_v2_64
    data["__cublasZswap_v2_64"] = <intptr_t>__cublasZswap_v2_64

    global __cublasSwapEx_64
    data["__cublasSwapEx_64"] = <intptr_t>__cublasSwapEx_64

    global __cublasIsamax_v2_64
    data["__cublasIsamax_v2_64"] = <intptr_t>__cublasIsamax_v2_64

    global __cublasIdamax_v2_64
    data["__cublasIdamax_v2_64"] = <intptr_t>__cublasIdamax_v2_64

    global __cublasIcamax_v2_64
    data["__cublasIcamax_v2_64"] = <intptr_t>__cublasIcamax_v2_64

    global __cublasIzamax_v2_64
    data["__cublasIzamax_v2_64"] = <intptr_t>__cublasIzamax_v2_64

    global __cublasIamaxEx_64
    data["__cublasIamaxEx_64"] = <intptr_t>__cublasIamaxEx_64

    global __cublasIsamin_v2_64
    data["__cublasIsamin_v2_64"] = <intptr_t>__cublasIsamin_v2_64

    global __cublasIdamin_v2_64
    data["__cublasIdamin_v2_64"] = <intptr_t>__cublasIdamin_v2_64

    global __cublasIcamin_v2_64
    data["__cublasIcamin_v2_64"] = <intptr_t>__cublasIcamin_v2_64

    global __cublasIzamin_v2_64
    data["__cublasIzamin_v2_64"] = <intptr_t>__cublasIzamin_v2_64

    global __cublasIaminEx_64
    data["__cublasIaminEx_64"] = <intptr_t>__cublasIaminEx_64

    global __cublasAsumEx_64
    data["__cublasAsumEx_64"] = <intptr_t>__cublasAsumEx_64

    global __cublasSasum_v2_64
    data["__cublasSasum_v2_64"] = <intptr_t>__cublasSasum_v2_64

    global __cublasDasum_v2_64
    data["__cublasDasum_v2_64"] = <intptr_t>__cublasDasum_v2_64

    global __cublasScasum_v2_64
    data["__cublasScasum_v2_64"] = <intptr_t>__cublasScasum_v2_64

    global __cublasDzasum_v2_64
    data["__cublasDzasum_v2_64"] = <intptr_t>__cublasDzasum_v2_64

    global __cublasSrot_v2_64
    data["__cublasSrot_v2_64"] = <intptr_t>__cublasSrot_v2_64

    global __cublasDrot_v2_64
    data["__cublasDrot_v2_64"] = <intptr_t>__cublasDrot_v2_64

    global __cublasCrot_v2_64
    data["__cublasCrot_v2_64"] = <intptr_t>__cublasCrot_v2_64

    global __cublasCsrot_v2_64
    data["__cublasCsrot_v2_64"] = <intptr_t>__cublasCsrot_v2_64

    global __cublasZrot_v2_64
    data["__cublasZrot_v2_64"] = <intptr_t>__cublasZrot_v2_64

    global __cublasZdrot_v2_64
    data["__cublasZdrot_v2_64"] = <intptr_t>__cublasZdrot_v2_64

    global __cublasRotEx_64
    data["__cublasRotEx_64"] = <intptr_t>__cublasRotEx_64

    global __cublasSrotm_v2_64
    data["__cublasSrotm_v2_64"] = <intptr_t>__cublasSrotm_v2_64

    global __cublasDrotm_v2_64
    data["__cublasDrotm_v2_64"] = <intptr_t>__cublasDrotm_v2_64

    global __cublasRotmEx_64
    data["__cublasRotmEx_64"] = <intptr_t>__cublasRotmEx_64

    global __cublasSgemv_v2_64
    data["__cublasSgemv_v2_64"] = <intptr_t>__cublasSgemv_v2_64

    global __cublasDgemv_v2_64
    data["__cublasDgemv_v2_64"] = <intptr_t>__cublasDgemv_v2_64

    global __cublasCgemv_v2_64
    data["__cublasCgemv_v2_64"] = <intptr_t>__cublasCgemv_v2_64

    global __cublasZgemv_v2_64
    data["__cublasZgemv_v2_64"] = <intptr_t>__cublasZgemv_v2_64

    global __cublasSgbmv_v2_64
    data["__cublasSgbmv_v2_64"] = <intptr_t>__cublasSgbmv_v2_64

    global __cublasDgbmv_v2_64
    data["__cublasDgbmv_v2_64"] = <intptr_t>__cublasDgbmv_v2_64

    global __cublasCgbmv_v2_64
    data["__cublasCgbmv_v2_64"] = <intptr_t>__cublasCgbmv_v2_64

    global __cublasZgbmv_v2_64
    data["__cublasZgbmv_v2_64"] = <intptr_t>__cublasZgbmv_v2_64

    global __cublasStrmv_v2_64
    data["__cublasStrmv_v2_64"] = <intptr_t>__cublasStrmv_v2_64

    global __cublasDtrmv_v2_64
    data["__cublasDtrmv_v2_64"] = <intptr_t>__cublasDtrmv_v2_64

    global __cublasCtrmv_v2_64
    data["__cublasCtrmv_v2_64"] = <intptr_t>__cublasCtrmv_v2_64

    global __cublasZtrmv_v2_64
    data["__cublasZtrmv_v2_64"] = <intptr_t>__cublasZtrmv_v2_64

    global __cublasStbmv_v2_64
    data["__cublasStbmv_v2_64"] = <intptr_t>__cublasStbmv_v2_64

    global __cublasDtbmv_v2_64
    data["__cublasDtbmv_v2_64"] = <intptr_t>__cublasDtbmv_v2_64

    global __cublasCtbmv_v2_64
    data["__cublasCtbmv_v2_64"] = <intptr_t>__cublasCtbmv_v2_64

    global __cublasZtbmv_v2_64
    data["__cublasZtbmv_v2_64"] = <intptr_t>__cublasZtbmv_v2_64

    global __cublasStpmv_v2_64
    data["__cublasStpmv_v2_64"] = <intptr_t>__cublasStpmv_v2_64

    global __cublasDtpmv_v2_64
    data["__cublasDtpmv_v2_64"] = <intptr_t>__cublasDtpmv_v2_64

    global __cublasCtpmv_v2_64
    data["__cublasCtpmv_v2_64"] = <intptr_t>__cublasCtpmv_v2_64

    global __cublasZtpmv_v2_64
    data["__cublasZtpmv_v2_64"] = <intptr_t>__cublasZtpmv_v2_64

    global __cublasStrsv_v2_64
    data["__cublasStrsv_v2_64"] = <intptr_t>__cublasStrsv_v2_64

    global __cublasDtrsv_v2_64
    data["__cublasDtrsv_v2_64"] = <intptr_t>__cublasDtrsv_v2_64

    global __cublasCtrsv_v2_64
    data["__cublasCtrsv_v2_64"] = <intptr_t>__cublasCtrsv_v2_64

    global __cublasZtrsv_v2_64
    data["__cublasZtrsv_v2_64"] = <intptr_t>__cublasZtrsv_v2_64

    global __cublasStpsv_v2_64
    data["__cublasStpsv_v2_64"] = <intptr_t>__cublasStpsv_v2_64

    global __cublasDtpsv_v2_64
    data["__cublasDtpsv_v2_64"] = <intptr_t>__cublasDtpsv_v2_64

    global __cublasCtpsv_v2_64
    data["__cublasCtpsv_v2_64"] = <intptr_t>__cublasCtpsv_v2_64

    global __cublasZtpsv_v2_64
    data["__cublasZtpsv_v2_64"] = <intptr_t>__cublasZtpsv_v2_64

    global __cublasStbsv_v2_64
    data["__cublasStbsv_v2_64"] = <intptr_t>__cublasStbsv_v2_64

    global __cublasDtbsv_v2_64
    data["__cublasDtbsv_v2_64"] = <intptr_t>__cublasDtbsv_v2_64

    global __cublasCtbsv_v2_64
    data["__cublasCtbsv_v2_64"] = <intptr_t>__cublasCtbsv_v2_64

    global __cublasZtbsv_v2_64
    data["__cublasZtbsv_v2_64"] = <intptr_t>__cublasZtbsv_v2_64

    global __cublasSsymv_v2_64
    data["__cublasSsymv_v2_64"] = <intptr_t>__cublasSsymv_v2_64

    global __cublasDsymv_v2_64
    data["__cublasDsymv_v2_64"] = <intptr_t>__cublasDsymv_v2_64

    global __cublasCsymv_v2_64
    data["__cublasCsymv_v2_64"] = <intptr_t>__cublasCsymv_v2_64

    global __cublasZsymv_v2_64
    data["__cublasZsymv_v2_64"] = <intptr_t>__cublasZsymv_v2_64

    global __cublasChemv_v2_64
    data["__cublasChemv_v2_64"] = <intptr_t>__cublasChemv_v2_64

    global __cublasZhemv_v2_64
    data["__cublasZhemv_v2_64"] = <intptr_t>__cublasZhemv_v2_64

    global __cublasSsbmv_v2_64
    data["__cublasSsbmv_v2_64"] = <intptr_t>__cublasSsbmv_v2_64

    global __cublasDsbmv_v2_64
    data["__cublasDsbmv_v2_64"] = <intptr_t>__cublasDsbmv_v2_64

    global __cublasChbmv_v2_64
    data["__cublasChbmv_v2_64"] = <intptr_t>__cublasChbmv_v2_64

    global __cublasZhbmv_v2_64
    data["__cublasZhbmv_v2_64"] = <intptr_t>__cublasZhbmv_v2_64

    global __cublasSspmv_v2_64
    data["__cublasSspmv_v2_64"] = <intptr_t>__cublasSspmv_v2_64

    global __cublasDspmv_v2_64
    data["__cublasDspmv_v2_64"] = <intptr_t>__cublasDspmv_v2_64

    global __cublasChpmv_v2_64
    data["__cublasChpmv_v2_64"] = <intptr_t>__cublasChpmv_v2_64

    global __cublasZhpmv_v2_64
    data["__cublasZhpmv_v2_64"] = <intptr_t>__cublasZhpmv_v2_64

    global __cublasSger_v2_64
    data["__cublasSger_v2_64"] = <intptr_t>__cublasSger_v2_64

    global __cublasDger_v2_64
    data["__cublasDger_v2_64"] = <intptr_t>__cublasDger_v2_64

    global __cublasCgeru_v2_64
    data["__cublasCgeru_v2_64"] = <intptr_t>__cublasCgeru_v2_64

    global __cublasCgerc_v2_64
    data["__cublasCgerc_v2_64"] = <intptr_t>__cublasCgerc_v2_64

    global __cublasZgeru_v2_64
    data["__cublasZgeru_v2_64"] = <intptr_t>__cublasZgeru_v2_64

    global __cublasZgerc_v2_64
    data["__cublasZgerc_v2_64"] = <intptr_t>__cublasZgerc_v2_64

    global __cublasSsyr_v2_64
    data["__cublasSsyr_v2_64"] = <intptr_t>__cublasSsyr_v2_64

    global __cublasDsyr_v2_64
    data["__cublasDsyr_v2_64"] = <intptr_t>__cublasDsyr_v2_64

    global __cublasCsyr_v2_64
    data["__cublasCsyr_v2_64"] = <intptr_t>__cublasCsyr_v2_64

    global __cublasZsyr_v2_64
    data["__cublasZsyr_v2_64"] = <intptr_t>__cublasZsyr_v2_64

    global __cublasCher_v2_64
    data["__cublasCher_v2_64"] = <intptr_t>__cublasCher_v2_64

    global __cublasZher_v2_64
    data["__cublasZher_v2_64"] = <intptr_t>__cublasZher_v2_64

    global __cublasSspr_v2_64
    data["__cublasSspr_v2_64"] = <intptr_t>__cublasSspr_v2_64

    global __cublasDspr_v2_64
    data["__cublasDspr_v2_64"] = <intptr_t>__cublasDspr_v2_64

    global __cublasChpr_v2_64
    data["__cublasChpr_v2_64"] = <intptr_t>__cublasChpr_v2_64

    global __cublasZhpr_v2_64
    data["__cublasZhpr_v2_64"] = <intptr_t>__cublasZhpr_v2_64

    global __cublasSsyr2_v2_64
    data["__cublasSsyr2_v2_64"] = <intptr_t>__cublasSsyr2_v2_64

    global __cublasDsyr2_v2_64
    data["__cublasDsyr2_v2_64"] = <intptr_t>__cublasDsyr2_v2_64

    global __cublasCsyr2_v2_64
    data["__cublasCsyr2_v2_64"] = <intptr_t>__cublasCsyr2_v2_64

    global __cublasZsyr2_v2_64
    data["__cublasZsyr2_v2_64"] = <intptr_t>__cublasZsyr2_v2_64

    global __cublasCher2_v2_64
    data["__cublasCher2_v2_64"] = <intptr_t>__cublasCher2_v2_64

    global __cublasZher2_v2_64
    data["__cublasZher2_v2_64"] = <intptr_t>__cublasZher2_v2_64

    global __cublasSspr2_v2_64
    data["__cublasSspr2_v2_64"] = <intptr_t>__cublasSspr2_v2_64

    global __cublasDspr2_v2_64
    data["__cublasDspr2_v2_64"] = <intptr_t>__cublasDspr2_v2_64

    global __cublasChpr2_v2_64
    data["__cublasChpr2_v2_64"] = <intptr_t>__cublasChpr2_v2_64

    global __cublasZhpr2_v2_64
    data["__cublasZhpr2_v2_64"] = <intptr_t>__cublasZhpr2_v2_64

    global __cublasSgemvBatched_64
    data["__cublasSgemvBatched_64"] = <intptr_t>__cublasSgemvBatched_64

    global __cublasDgemvBatched_64
    data["__cublasDgemvBatched_64"] = <intptr_t>__cublasDgemvBatched_64

    global __cublasCgemvBatched_64
    data["__cublasCgemvBatched_64"] = <intptr_t>__cublasCgemvBatched_64

    global __cublasZgemvBatched_64
    data["__cublasZgemvBatched_64"] = <intptr_t>__cublasZgemvBatched_64

    global __cublasSgemvStridedBatched_64
    data["__cublasSgemvStridedBatched_64"] = <intptr_t>__cublasSgemvStridedBatched_64

    global __cublasDgemvStridedBatched_64
    data["__cublasDgemvStridedBatched_64"] = <intptr_t>__cublasDgemvStridedBatched_64

    global __cublasCgemvStridedBatched_64
    data["__cublasCgemvStridedBatched_64"] = <intptr_t>__cublasCgemvStridedBatched_64

    global __cublasZgemvStridedBatched_64
    data["__cublasZgemvStridedBatched_64"] = <intptr_t>__cublasZgemvStridedBatched_64

    global __cublasSgemm_v2_64
    data["__cublasSgemm_v2_64"] = <intptr_t>__cublasSgemm_v2_64

    global __cublasDgemm_v2_64
    data["__cublasDgemm_v2_64"] = <intptr_t>__cublasDgemm_v2_64

    global __cublasCgemm_v2_64
    data["__cublasCgemm_v2_64"] = <intptr_t>__cublasCgemm_v2_64

    global __cublasCgemm3m_64
    data["__cublasCgemm3m_64"] = <intptr_t>__cublasCgemm3m_64

    global __cublasCgemm3mEx_64
    data["__cublasCgemm3mEx_64"] = <intptr_t>__cublasCgemm3mEx_64

    global __cublasZgemm_v2_64
    data["__cublasZgemm_v2_64"] = <intptr_t>__cublasZgemm_v2_64

    global __cublasZgemm3m_64
    data["__cublasZgemm3m_64"] = <intptr_t>__cublasZgemm3m_64

    global __cublasSgemmEx_64
    data["__cublasSgemmEx_64"] = <intptr_t>__cublasSgemmEx_64

    global __cublasGemmEx_64
    data["__cublasGemmEx_64"] = <intptr_t>__cublasGemmEx_64

    global __cublasCgemmEx_64
    data["__cublasCgemmEx_64"] = <intptr_t>__cublasCgemmEx_64

    global __cublasSsyrk_v2_64
    data["__cublasSsyrk_v2_64"] = <intptr_t>__cublasSsyrk_v2_64

    global __cublasDsyrk_v2_64
    data["__cublasDsyrk_v2_64"] = <intptr_t>__cublasDsyrk_v2_64

    global __cublasCsyrk_v2_64
    data["__cublasCsyrk_v2_64"] = <intptr_t>__cublasCsyrk_v2_64

    global __cublasZsyrk_v2_64
    data["__cublasZsyrk_v2_64"] = <intptr_t>__cublasZsyrk_v2_64

    global __cublasCsyrkEx_64
    data["__cublasCsyrkEx_64"] = <intptr_t>__cublasCsyrkEx_64

    global __cublasCsyrk3mEx_64
    data["__cublasCsyrk3mEx_64"] = <intptr_t>__cublasCsyrk3mEx_64

    global __cublasCherk_v2_64
    data["__cublasCherk_v2_64"] = <intptr_t>__cublasCherk_v2_64

    global __cublasZherk_v2_64
    data["__cublasZherk_v2_64"] = <intptr_t>__cublasZherk_v2_64

    global __cublasCherkEx_64
    data["__cublasCherkEx_64"] = <intptr_t>__cublasCherkEx_64

    global __cublasCherk3mEx_64
    data["__cublasCherk3mEx_64"] = <intptr_t>__cublasCherk3mEx_64

    global __cublasSsyr2k_v2_64
    data["__cublasSsyr2k_v2_64"] = <intptr_t>__cublasSsyr2k_v2_64

    global __cublasDsyr2k_v2_64
    data["__cublasDsyr2k_v2_64"] = <intptr_t>__cublasDsyr2k_v2_64

    global __cublasCsyr2k_v2_64
    data["__cublasCsyr2k_v2_64"] = <intptr_t>__cublasCsyr2k_v2_64

    global __cublasZsyr2k_v2_64
    data["__cublasZsyr2k_v2_64"] = <intptr_t>__cublasZsyr2k_v2_64

    global __cublasCher2k_v2_64
    data["__cublasCher2k_v2_64"] = <intptr_t>__cublasCher2k_v2_64

    global __cublasZher2k_v2_64
    data["__cublasZher2k_v2_64"] = <intptr_t>__cublasZher2k_v2_64

    global __cublasSsyrkx_64
    data["__cublasSsyrkx_64"] = <intptr_t>__cublasSsyrkx_64

    global __cublasDsyrkx_64
    data["__cublasDsyrkx_64"] = <intptr_t>__cublasDsyrkx_64

    global __cublasCsyrkx_64
    data["__cublasCsyrkx_64"] = <intptr_t>__cublasCsyrkx_64

    global __cublasZsyrkx_64
    data["__cublasZsyrkx_64"] = <intptr_t>__cublasZsyrkx_64

    global __cublasCherkx_64
    data["__cublasCherkx_64"] = <intptr_t>__cublasCherkx_64

    global __cublasZherkx_64
    data["__cublasZherkx_64"] = <intptr_t>__cublasZherkx_64

    global __cublasSsymm_v2_64
    data["__cublasSsymm_v2_64"] = <intptr_t>__cublasSsymm_v2_64

    global __cublasDsymm_v2_64
    data["__cublasDsymm_v2_64"] = <intptr_t>__cublasDsymm_v2_64

    global __cublasCsymm_v2_64
    data["__cublasCsymm_v2_64"] = <intptr_t>__cublasCsymm_v2_64

    global __cublasZsymm_v2_64
    data["__cublasZsymm_v2_64"] = <intptr_t>__cublasZsymm_v2_64

    global __cublasChemm_v2_64
    data["__cublasChemm_v2_64"] = <intptr_t>__cublasChemm_v2_64

    global __cublasZhemm_v2_64
    data["__cublasZhemm_v2_64"] = <intptr_t>__cublasZhemm_v2_64

    global __cublasStrsm_v2_64
    data["__cublasStrsm_v2_64"] = <intptr_t>__cublasStrsm_v2_64

    global __cublasDtrsm_v2_64
    data["__cublasDtrsm_v2_64"] = <intptr_t>__cublasDtrsm_v2_64

    global __cublasCtrsm_v2_64
    data["__cublasCtrsm_v2_64"] = <intptr_t>__cublasCtrsm_v2_64

    global __cublasZtrsm_v2_64
    data["__cublasZtrsm_v2_64"] = <intptr_t>__cublasZtrsm_v2_64

    global __cublasStrmm_v2_64
    data["__cublasStrmm_v2_64"] = <intptr_t>__cublasStrmm_v2_64

    global __cublasDtrmm_v2_64
    data["__cublasDtrmm_v2_64"] = <intptr_t>__cublasDtrmm_v2_64

    global __cublasCtrmm_v2_64
    data["__cublasCtrmm_v2_64"] = <intptr_t>__cublasCtrmm_v2_64

    global __cublasZtrmm_v2_64
    data["__cublasZtrmm_v2_64"] = <intptr_t>__cublasZtrmm_v2_64

    global __cublasSgemmBatched_64
    data["__cublasSgemmBatched_64"] = <intptr_t>__cublasSgemmBatched_64

    global __cublasDgemmBatched_64
    data["__cublasDgemmBatched_64"] = <intptr_t>__cublasDgemmBatched_64

    global __cublasCgemmBatched_64
    data["__cublasCgemmBatched_64"] = <intptr_t>__cublasCgemmBatched_64

    global __cublasCgemm3mBatched_64
    data["__cublasCgemm3mBatched_64"] = <intptr_t>__cublasCgemm3mBatched_64

    global __cublasZgemmBatched_64
    data["__cublasZgemmBatched_64"] = <intptr_t>__cublasZgemmBatched_64

    global __cublasSgemmStridedBatched_64
    data["__cublasSgemmStridedBatched_64"] = <intptr_t>__cublasSgemmStridedBatched_64

    global __cublasDgemmStridedBatched_64
    data["__cublasDgemmStridedBatched_64"] = <intptr_t>__cublasDgemmStridedBatched_64

    global __cublasCgemmStridedBatched_64
    data["__cublasCgemmStridedBatched_64"] = <intptr_t>__cublasCgemmStridedBatched_64

    global __cublasCgemm3mStridedBatched_64
    data["__cublasCgemm3mStridedBatched_64"] = <intptr_t>__cublasCgemm3mStridedBatched_64

    global __cublasZgemmStridedBatched_64
    data["__cublasZgemmStridedBatched_64"] = <intptr_t>__cublasZgemmStridedBatched_64

    global __cublasGemmBatchedEx_64
    data["__cublasGemmBatchedEx_64"] = <intptr_t>__cublasGemmBatchedEx_64

    global __cublasGemmStridedBatchedEx_64
    data["__cublasGemmStridedBatchedEx_64"] = <intptr_t>__cublasGemmStridedBatchedEx_64

    global __cublasSgeam_64
    data["__cublasSgeam_64"] = <intptr_t>__cublasSgeam_64

    global __cublasDgeam_64
    data["__cublasDgeam_64"] = <intptr_t>__cublasDgeam_64

    global __cublasCgeam_64
    data["__cublasCgeam_64"] = <intptr_t>__cublasCgeam_64

    global __cublasZgeam_64
    data["__cublasZgeam_64"] = <intptr_t>__cublasZgeam_64

    global __cublasStrsmBatched_64
    data["__cublasStrsmBatched_64"] = <intptr_t>__cublasStrsmBatched_64

    global __cublasDtrsmBatched_64
    data["__cublasDtrsmBatched_64"] = <intptr_t>__cublasDtrsmBatched_64

    global __cublasCtrsmBatched_64
    data["__cublasCtrsmBatched_64"] = <intptr_t>__cublasCtrsmBatched_64

    global __cublasZtrsmBatched_64
    data["__cublasZtrsmBatched_64"] = <intptr_t>__cublasZtrsmBatched_64

    global __cublasSdgmm_64
    data["__cublasSdgmm_64"] = <intptr_t>__cublasSdgmm_64

    global __cublasDdgmm_64
    data["__cublasDdgmm_64"] = <intptr_t>__cublasDdgmm_64

    global __cublasCdgmm_64
    data["__cublasCdgmm_64"] = <intptr_t>__cublasCdgmm_64

    global __cublasZdgmm_64
    data["__cublasZdgmm_64"] = <intptr_t>__cublasZdgmm_64

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


###############################################################################
# Wrapper functions
###############################################################################

cdef cublasStatus_t _cublasCreate(cublasHandle_t* handle) except* nogil:
    global __cublasCreate_v2
    _check_or_init_cublas()
    if __cublasCreate_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCreate_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t*) nogil>__cublasCreate_v2)(
        handle)


cdef cublasStatus_t _cublasDestroy(cublasHandle_t handle) except* nogil:
    global __cublasDestroy_v2
    _check_or_init_cublas()
    if __cublasDestroy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDestroy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t) nogil>__cublasDestroy_v2)(
        handle)


cdef cublasStatus_t _cublasGetVersion(cublasHandle_t handle, int* version) except* nogil:
    global __cublasGetVersion_v2
    _check_or_init_cublas()
    if __cublasGetVersion_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVersion_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) nogil>__cublasGetVersion_v2)(
        handle, version)


cdef cublasStatus_t _cublasGetProperty(libraryPropertyType type, int* value) except* nogil:
    global __cublasGetProperty
    _check_or_init_cublas()
    if __cublasGetProperty == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetProperty is not found")
    return (<cublasStatus_t (*)(libraryPropertyType, int*) nogil>__cublasGetProperty)(
        type, value)


cdef size_t _cublasGetCudartVersion() except* nogil:
    global __cublasGetCudartVersion
    _check_or_init_cublas()
    if __cublasGetCudartVersion == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetCudartVersion is not found")
    return (<size_t (*)() nogil>__cublasGetCudartVersion)(
        )


cdef cublasStatus_t _cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t workspaceSizeInBytes) except* nogil:
    global __cublasSetWorkspace_v2
    _check_or_init_cublas()
    if __cublasSetWorkspace_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetWorkspace_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, void*, size_t) nogil>__cublasSetWorkspace_v2)(
        handle, workspace, workspaceSizeInBytes)


cdef cublasStatus_t _cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) except* nogil:
    global __cublasSetStream_v2
    _check_or_init_cublas()
    if __cublasSetStream_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetStream_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaStream_t) nogil>__cublasSetStream_v2)(
        handle, streamId)


cdef cublasStatus_t _cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId) except* nogil:
    global __cublasGetStream_v2
    _check_or_init_cublas()
    if __cublasGetStream_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetStream_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cudaStream_t*) nogil>__cublasGetStream_v2)(
        handle, streamId)


cdef cublasStatus_t _cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) except* nogil:
    global __cublasGetPointerMode_v2
    _check_or_init_cublas()
    if __cublasGetPointerMode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetPointerMode_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t*) nogil>__cublasGetPointerMode_v2)(
        handle, mode)


cdef cublasStatus_t _cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) except* nogil:
    global __cublasSetPointerMode_v2
    _check_or_init_cublas()
    if __cublasSetPointerMode_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetPointerMode_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t) nogil>__cublasSetPointerMode_v2)(
        handle, mode)


cdef cublasStatus_t _cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode) except* nogil:
    global __cublasGetAtomicsMode
    _check_or_init_cublas()
    if __cublasGetAtomicsMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetAtomicsMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasAtomicsMode_t*) nogil>__cublasGetAtomicsMode)(
        handle, mode)


cdef cublasStatus_t _cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) except* nogil:
    global __cublasSetAtomicsMode
    _check_or_init_cublas()
    if __cublasSetAtomicsMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetAtomicsMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasAtomicsMode_t) nogil>__cublasSetAtomicsMode)(
        handle, mode)


cdef cublasStatus_t _cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) except* nogil:
    global __cublasGetMathMode
    _check_or_init_cublas()
    if __cublasGetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMathMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasMath_t*) nogil>__cublasGetMathMode)(
        handle, mode)


cdef cublasStatus_t _cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) except* nogil:
    global __cublasSetMathMode
    _check_or_init_cublas()
    if __cublasSetMathMode == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMathMode is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasMath_t) nogil>__cublasSetMathMode)(
        handle, mode)


cdef cublasStatus_t _cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) except* nogil:
    global __cublasLoggerConfigure
    _check_or_init_cublas()
    if __cublasLoggerConfigure == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasLoggerConfigure is not found")
    return (<cublasStatus_t (*)(int, int, int, const char*) nogil>__cublasLoggerConfigure)(
        logIsOn, logToStdOut, logToStdErr, logFileName)


cdef cublasStatus_t _cublasSetLoggerCallback(cublasLogCallback userCallback) except* nogil:
    global __cublasSetLoggerCallback
    _check_or_init_cublas()
    if __cublasSetLoggerCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetLoggerCallback is not found")
    return (<cublasStatus_t (*)(cublasLogCallback) nogil>__cublasSetLoggerCallback)(
        userCallback)


cdef cublasStatus_t _cublasGetLoggerCallback(cublasLogCallback* userCallback) except* nogil:
    global __cublasGetLoggerCallback
    _check_or_init_cublas()
    if __cublasGetLoggerCallback == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetLoggerCallback is not found")
    return (<cublasStatus_t (*)(cublasLogCallback*) nogil>__cublasGetLoggerCallback)(
        userCallback)


cdef cublasStatus_t _cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy) except* nogil:
    global __cublasSetVector
    _check_or_init_cublas()
    if __cublasSetVector == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVector is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int) nogil>__cublasSetVector)(
        n, elemSize, x, incx, devicePtr, incy)


cdef cublasStatus_t _cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) except* nogil:
    global __cublasGetVector
    _check_or_init_cublas()
    if __cublasGetVector == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVector is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int) nogil>__cublasGetVector)(
        n, elemSize, x, incx, y, incy)


cdef cublasStatus_t _cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except* nogil:
    global __cublasSetMatrix
    _check_or_init_cublas()
    if __cublasSetMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrix is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int) nogil>__cublasSetMatrix)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb) except* nogil:
    global __cublasGetMatrix
    _check_or_init_cublas()
    if __cublasGetMatrix == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrix is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int) nogil>__cublasGetMatrix)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream) except* nogil:
    global __cublasSetVectorAsync
    _check_or_init_cublas()
    if __cublasSetVectorAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVectorAsync is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int, cudaStream_t) nogil>__cublasSetVectorAsync)(
        n, elemSize, hostPtr, incx, devicePtr, incy, stream)


cdef cublasStatus_t _cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream) except* nogil:
    global __cublasGetVectorAsync
    _check_or_init_cublas()
    if __cublasGetVectorAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVectorAsync is not found")
    return (<cublasStatus_t (*)(int, int, const void*, int, void*, int, cudaStream_t) nogil>__cublasGetVectorAsync)(
        n, elemSize, devicePtr, incx, hostPtr, incy, stream)


cdef cublasStatus_t _cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except* nogil:
    global __cublasSetMatrixAsync
    _check_or_init_cublas()
    if __cublasSetMatrixAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrixAsync is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int, cudaStream_t) nogil>__cublasSetMatrixAsync)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream) except* nogil:
    global __cublasGetMatrixAsync
    _check_or_init_cublas()
    if __cublasGetMatrixAsync == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrixAsync is not found")
    return (<cublasStatus_t (*)(int, int, int, const void*, int, void*, int, cudaStream_t) nogil>__cublasGetMatrixAsync)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    global __cublasNrm2Ex
    _check_or_init_cublas()
    if __cublasNrm2Ex == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasNrm2Ex is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) nogil>__cublasNrm2Ex)(
        handle, n, x, xType, incx, result, resultType, executionType)


cdef cublasStatus_t _cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) except* nogil:
    global __cublasSnrm2_v2
    _check_or_init_cublas()
    if __cublasSnrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSnrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*) nogil>__cublasSnrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result) except* nogil:
    global __cublasDnrm2_v2
    _check_or_init_cublas()
    if __cublasDnrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDnrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*) nogil>__cublasDnrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScnrm2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except* nogil:
    global __cublasScnrm2_v2
    _check_or_init_cublas()
    if __cublasScnrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScnrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, float*) nogil>__cublasScnrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDznrm2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except* nogil:
    global __cublasDznrm2_v2
    _check_or_init_cublas()
    if __cublasDznrm2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDznrm2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, double*) nogil>__cublasDznrm2_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    global __cublasDotEx
    _check_or_init_cublas()
    if __cublasDotEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) nogil>__cublasDotEx)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    global __cublasDotcEx
    _check_or_init_cublas()
    if __cublasDotcEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotcEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) nogil>__cublasDotcEx)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasSdot(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result) except* nogil:
    global __cublasSdot_v2
    _check_or_init_cublas()
    if __cublasSdot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, const float*, int, float*) nogil>__cublasSdot_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasDdot(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result) except* nogil:
    global __cublasDdot_v2
    _check_or_init_cublas()
    if __cublasDdot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, const double*, int, double*) nogil>__cublasDdot_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotu(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except* nogil:
    global __cublasCdotu_v2
    _check_or_init_cublas()
    if __cublasCdotu_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotu_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, const cuComplex*, int, cuComplex*) nogil>__cublasCdotu_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotc(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result) except* nogil:
    global __cublasCdotc_v2
    _check_or_init_cublas()
    if __cublasCdotc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, const cuComplex*, int, cuComplex*) nogil>__cublasCdotc_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotu(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except* nogil:
    global __cublasZdotu_v2
    _check_or_init_cublas()
    if __cublasZdotu_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotu_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*) nogil>__cublasZdotu_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotc(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result) except* nogil:
    global __cublasZdotc_v2
    _check_or_init_cublas()
    if __cublasZdotc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*) nogil>__cublasZdotc_v2)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType) except* nogil:
    global __cublasScalEx
    _check_or_init_cublas()
    if __cublasScalEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScalEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, void*, cudaDataType, int, cudaDataType) nogil>__cublasScalEx)(
        handle, n, alpha, alphaType, x, xType, incx, executionType)


cdef cublasStatus_t _cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) except* nogil:
    global __cublasSscal_v2
    _check_or_init_cublas()
    if __cublasSscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, float*, int) nogil>__cublasSscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) except* nogil:
    global __cublasDscal_v2
    _check_or_init_cublas()
    if __cublasDscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, double*, int) nogil>__cublasDscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCscal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx) except* nogil:
    global __cublasCscal_v2
    _check_or_init_cublas()
    if __cublasCscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, cuComplex*, int) nogil>__cublasCscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCsscal(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx) except* nogil:
    global __cublasCsscal_v2
    _check_or_init_cublas()
    if __cublasCsscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, cuComplex*, int) nogil>__cublasCsscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZscal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZscal_v2
    _check_or_init_cublas()
    if __cublasZscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZdscal(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZdscal_v2
    _check_or_init_cublas()
    if __cublasZdscal_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdscal_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, cuDoubleComplex*, int) nogil>__cublasZdscal_v2)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype) except* nogil:
    global __cublasAxpyEx
    _check_or_init_cublas()
    if __cublasAxpyEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAxpyEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, const void*, cudaDataType, int, void*, cudaDataType, int, cudaDataType) nogil>__cublasAxpyEx)(
        handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)


cdef cublasStatus_t _cublasSaxpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) except* nogil:
    global __cublasSaxpy_v2
    _check_or_init_cublas()
    if __cublasSaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, const float*, int, float*, int) nogil>__cublasSaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasDaxpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) except* nogil:
    global __cublasDaxpy_v2
    _check_or_init_cublas()
    if __cublasDaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, const double*, int, double*, int) nogil>__cublasDaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCaxpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy) except* nogil:
    global __cublasCaxpy_v2
    _check_or_init_cublas()
    if __cublasCaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) nogil>__cublasCaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZaxpy_v2
    _check_or_init_cublas()
    if __cublasZaxpy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZaxpy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZaxpy_v2)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCopyEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except* nogil:
    global __cublasCopyEx
    _check_or_init_cublas()
    if __cublasCopyEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCopyEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, void*, cudaDataType, int) nogil>__cublasCopyEx)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasScopy(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy) except* nogil:
    global __cublasScopy_v2
    _check_or_init_cublas()
    if __cublasScopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*, int) nogil>__cublasScopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDcopy(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy) except* nogil:
    global __cublasDcopy_v2
    _check_or_init_cublas()
    if __cublasDcopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDcopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*, int) nogil>__cublasDcopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCcopy(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy) except* nogil:
    global __cublasCcopy_v2
    _check_or_init_cublas()
    if __cublasCcopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCcopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCcopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZcopy(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZcopy_v2
    _check_or_init_cublas()
    if __cublasZcopy_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZcopy_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZcopy_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) except* nogil:
    global __cublasSswap_v2
    _check_or_init_cublas()
    if __cublasSswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float*, int, float*, int) nogil>__cublasSswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) except* nogil:
    global __cublasDswap_v2
    _check_or_init_cublas()
    if __cublasDswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double*, int, double*, int) nogil>__cublasDswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCswap(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy) except* nogil:
    global __cublasCswap_v2
    _check_or_init_cublas()
    if __cublasCswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex*, int, cuComplex*, int) nogil>__cublasCswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZswap_v2
    _check_or_init_cublas()
    if __cublasZswap_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZswap_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZswap_v2)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSwapEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy) except* nogil:
    global __cublasSwapEx
    _check_or_init_cublas()
    if __cublasSwapEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSwapEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, void*, cudaDataType, int, void*, cudaDataType, int) nogil>__cublasSwapEx)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) except* nogil:
    global __cublasIsamax_v2
    _check_or_init_cublas()
    if __cublasIsamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, int*) nogil>__cublasIsamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result) except* nogil:
    global __cublasIdamax_v2
    _check_or_init_cublas()
    if __cublasIdamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, int*) nogil>__cublasIdamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except* nogil:
    global __cublasIcamax_v2
    _check_or_init_cublas()
    if __cublasIcamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, int*) nogil>__cublasIcamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except* nogil:
    global __cublasIzamax_v2
    _check_or_init_cublas()
    if __cublasIzamax_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamax_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, int*) nogil>__cublasIzamax_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIamaxEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except* nogil:
    global __cublasIamaxEx
    _check_or_init_cublas()
    if __cublasIamaxEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIamaxEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, int*) nogil>__cublasIamaxEx)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) except* nogil:
    global __cublasIsamin_v2
    _check_or_init_cublas()
    if __cublasIsamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, int*) nogil>__cublasIsamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamin(cublasHandle_t handle, int n, const double* x, int incx, int* result) except* nogil:
    global __cublasIdamin_v2
    _check_or_init_cublas()
    if __cublasIdamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, int*) nogil>__cublasIdamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamin(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) except* nogil:
    global __cublasIcamin_v2
    _check_or_init_cublas()
    if __cublasIcamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, int*) nogil>__cublasIcamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamin(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) except* nogil:
    global __cublasIzamin_v2
    _check_or_init_cublas()
    if __cublasIzamin_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamin_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, int*) nogil>__cublasIzamin_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIaminEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, int* result) except* nogil:
    global __cublasIaminEx
    _check_or_init_cublas()
    if __cublasIaminEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIaminEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, int*) nogil>__cublasIaminEx)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasAsumEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executiontype) except* nogil:
    global __cublasAsumEx
    _check_or_init_cublas()
    if __cublasAsumEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAsumEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const void*, cudaDataType, int, void*, cudaDataType, cudaDataType) nogil>__cublasAsumEx)(
        handle, n, x, xType, incx, result, resultType, executiontype)


cdef cublasStatus_t _cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) except* nogil:
    global __cublasSasum_v2
    _check_or_init_cublas()
    if __cublasSasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*) nogil>__cublasSasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result) except* nogil:
    global __cublasDasum_v2
    _check_or_init_cublas()
    if __cublasDasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*) nogil>__cublasDasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScasum(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result) except* nogil:
    global __cublasScasum_v2
    _check_or_init_cublas()
    if __cublasScasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex*, int, float*) nogil>__cublasScasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDzasum(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result) except* nogil:
    global __cublasDzasum_v2
    _check_or_init_cublas()
    if __cublasDzasum_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDzasum_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex*, int, double*) nogil>__cublasDzasum_v2)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasSrot(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s) except* nogil:
    global __cublasSrot_v2
    _check_or_init_cublas()
    if __cublasSrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float*, int, float*, int, const float*, const float*) nogil>__cublasSrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasDrot(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s) except* nogil:
    global __cublasDrot_v2
    _check_or_init_cublas()
    if __cublasDrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double*, int, double*, int, const double*, const double*) nogil>__cublasDrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s) except* nogil:
    global __cublasCrot_v2
    _check_or_init_cublas()
    if __cublasCrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex*, int, cuComplex*, int, const float*, const cuComplex*) nogil>__cublasCrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCsrot(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s) except* nogil:
    global __cublasCsrot_v2
    _check_or_init_cublas()
    if __cublasCsrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex*, int, cuComplex*, int, const float*, const float*) nogil>__cublasCsrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s) except* nogil:
    global __cublasZrot_v2
    _check_or_init_cublas()
    if __cublasZrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, const double*, const cuDoubleComplex*) nogil>__cublasZrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s) except* nogil:
    global __cublasZdrot_v2
    _check_or_init_cublas()
    if __cublasZdrot_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdrot_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex*, int, cuDoubleComplex*, int, const double*, const double*) nogil>__cublasZdrot_v2)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except* nogil:
    global __cublasRotEx
    _check_or_init_cublas()
    if __cublasRotEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, void*, cudaDataType, int, void*, cudaDataType, int, const void*, const void*, cudaDataType, cudaDataType) nogil>__cublasRotEx)(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)


cdef cublasStatus_t _cublasSrotg(cublasHandle_t handle, float* a, float* b, float* c, float* s) except* nogil:
    global __cublasSrotg_v2
    _check_or_init_cublas()
    if __cublasSrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, float*, float*, float*, float*) nogil>__cublasSrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasDrotg(cublasHandle_t handle, double* a, double* b, double* c, double* s) except* nogil:
    global __cublasDrotg_v2
    _check_or_init_cublas()
    if __cublasDrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, double*, double*, double*, double*) nogil>__cublasDrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasCrotg(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s) except* nogil:
    global __cublasCrotg_v2
    _check_or_init_cublas()
    if __cublasCrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cuComplex*, cuComplex*, float*, cuComplex*) nogil>__cublasCrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasZrotg(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s) except* nogil:
    global __cublasZrotg_v2
    _check_or_init_cublas()
    if __cublasZrotg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZrotg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cuDoubleComplex*, cuDoubleComplex*, double*, cuDoubleComplex*) nogil>__cublasZrotg_v2)(
        handle, a, b, c, s)


cdef cublasStatus_t _cublasRotgEx(cublasHandle_t handle, void* a, void* b, cudaDataType abType, void* c, void* s, cudaDataType csType, cudaDataType executiontype) except* nogil:
    global __cublasRotgEx
    _check_or_init_cublas()
    if __cublasRotgEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotgEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, void*, void*, cudaDataType, void*, void*, cudaDataType, cudaDataType) nogil>__cublasRotgEx)(
        handle, a, b, abType, c, s, csType, executiontype)


cdef cublasStatus_t _cublasSrotm(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param) except* nogil:
    global __cublasSrotm_v2
    _check_or_init_cublas()
    if __cublasSrotm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float*, int, float*, int, const float*) nogil>__cublasSrotm_v2)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasDrotm(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param) except* nogil:
    global __cublasDrotm_v2
    _check_or_init_cublas()
    if __cublasDrotm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double*, int, double*, int, const double*) nogil>__cublasDrotm_v2)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasRotmEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil:
    global __cublasRotmEx
    _check_or_init_cublas()
    if __cublasRotmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, void*, cudaDataType, int, void*, cudaDataType, int, const void*, cudaDataType, cudaDataType) nogil>__cublasRotmEx)(
        handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)


cdef cublasStatus_t _cublasSrotmg(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param) except* nogil:
    global __cublasSrotmg_v2
    _check_or_init_cublas()
    if __cublasSrotmg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotmg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, float*, float*, float*, const float*, float*) nogil>__cublasSrotmg_v2)(
        handle, d1, d2, x1, y1, param)


cdef cublasStatus_t _cublasDrotmg(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param) except* nogil:
    global __cublasDrotmg_v2
    _check_or_init_cublas()
    if __cublasDrotmg_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotmg_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, double*, double*, double*, const double*, double*) nogil>__cublasDrotmg_v2)(
        handle, d1, d2, x1, y1, param)


cdef cublasStatus_t _cublasRotmgEx(cublasHandle_t handle, void* d1, cudaDataType d1Type, void* d2, cudaDataType d2Type, void* x1, cudaDataType x1Type, const void* y1, cudaDataType y1Type, void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil:
    global __cublasRotmgEx
    _check_or_init_cublas()
    if __cublasRotmgEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotmgEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, void*, cudaDataType, void*, cudaDataType, void*, cudaDataType, const void*, cudaDataType, void*, cudaDataType, cudaDataType) nogil>__cublasRotmgEx)(
        handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype)


cdef cublasStatus_t _cublasSgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    global __cublasSgemv_v2
    _check_or_init_cublas()
    if __cublasSgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    global __cublasDgemv_v2
    _check_or_init_cublas()
    if __cublasDgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    global __cublasCgemv_v2
    _check_or_init_cublas()
    if __cublasCgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZgemv_v2
    _check_or_init_cublas()
    if __cublasZgemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZgemv_v2)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    global __cublasSgbmv_v2
    _check_or_init_cublas()
    if __cublasSgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    global __cublasDgbmv_v2
    _check_or_init_cublas()
    if __cublasDgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    global __cublasCgbmv_v2
    _check_or_init_cublas()
    if __cublasCgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZgbmv_v2
    _check_or_init_cublas()
    if __cublasZgbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZgbmv_v2)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except* nogil:
    global __cublasStrmv_v2
    _check_or_init_cublas()
    if __cublasStrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, int, float*, int) nogil>__cublasStrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except* nogil:
    global __cublasDtrmv_v2
    _check_or_init_cublas()
    if __cublasDtrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, int, double*, int) nogil>__cublasDtrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    global __cublasCtrmv_v2
    _check_or_init_cublas()
    if __cublasCtrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCtrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZtrmv_v2
    _check_or_init_cublas()
    if __cublasZtrmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZtrmv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except* nogil:
    global __cublasStbmv_v2
    _check_or_init_cublas()
    if __cublasStbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, int, float*, int) nogil>__cublasStbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except* nogil:
    global __cublasDtbmv_v2
    _check_or_init_cublas()
    if __cublasDtbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, int, double*, int) nogil>__cublasDtbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    global __cublasCtbmv_v2
    _check_or_init_cublas()
    if __cublasCtbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCtbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZtbmv_v2
    _check_or_init_cublas()
    if __cublasZtbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZtbmv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except* nogil:
    global __cublasStpmv_v2
    _check_or_init_cublas()
    if __cublasStpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, float*, int) nogil>__cublasStpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except* nogil:
    global __cublasDtpmv_v2
    _check_or_init_cublas()
    if __cublasDtpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, double*, int) nogil>__cublasDtpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except* nogil:
    global __cublasCtpmv_v2
    _check_or_init_cublas()
    if __cublasCtpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, cuComplex*, int) nogil>__cublasCtpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZtpmv_v2
    _check_or_init_cublas()
    if __cublasZtpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZtpmv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) except* nogil:
    global __cublasStrsv_v2
    _check_or_init_cublas()
    if __cublasStrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, int, float*, int) nogil>__cublasStrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx) except* nogil:
    global __cublasDtrsv_v2
    _check_or_init_cublas()
    if __cublasDtrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, int, double*, int) nogil>__cublasDtrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    global __cublasCtrsv_v2
    _check_or_init_cublas()
    if __cublasCtrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCtrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZtrsv_v2
    _check_or_init_cublas()
    if __cublasZtrsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZtrsv_v2)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx) except* nogil:
    global __cublasStpsv_v2
    _check_or_init_cublas()
    if __cublasStpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const float*, float*, int) nogil>__cublasStpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx) except* nogil:
    global __cublasDtpsv_v2
    _check_or_init_cublas()
    if __cublasDtpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const double*, double*, int) nogil>__cublasDtpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx) except* nogil:
    global __cublasCtpsv_v2
    _check_or_init_cublas()
    if __cublasCtpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuComplex*, cuComplex*, int) nogil>__cublasCtpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZtpsv_v2
    _check_or_init_cublas()
    if __cublasZtpsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZtpsv_v2)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx) except* nogil:
    global __cublasStbsv_v2
    _check_or_init_cublas()
    if __cublasStbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, int, float*, int) nogil>__cublasStbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx) except* nogil:
    global __cublasDtbsv_v2
    _check_or_init_cublas()
    if __cublasDtbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, int, double*, int) nogil>__cublasDtbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx) except* nogil:
    global __cublasCtbsv_v2
    _check_or_init_cublas()
    if __cublasCtbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCtbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) except* nogil:
    global __cublasZtbsv_v2
    _check_or_init_cublas()
    if __cublasZtbsv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbsv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZtbsv_v2)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    global __cublasSsymv_v2
    _check_or_init_cublas()
    if __cublasSsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    global __cublasDsymv_v2
    _check_or_init_cublas()
    if __cublasDsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    global __cublasCsymv_v2
    _check_or_init_cublas()
    if __cublasCsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZsymv_v2
    _check_or_init_cublas()
    if __cublasZsymv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZsymv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    global __cublasChemv_v2
    _check_or_init_cublas()
    if __cublasChemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasChemv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZhemv_v2
    _check_or_init_cublas()
    if __cublasZhemv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZhemv_v2)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    global __cublasSsbmv_v2
    _check_or_init_cublas()
    if __cublasSsbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSsbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    global __cublasDsbmv_v2
    _check_or_init_cublas()
    if __cublasDsbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDsbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    global __cublasChbmv_v2
    _check_or_init_cublas()
    if __cublasChbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasChbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZhbmv_v2
    _check_or_init_cublas()
    if __cublasZhbmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhbmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZhbmv_v2)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy) except* nogil:
    global __cublasSspmv_v2
    _check_or_init_cublas()
    if __cublasSspmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, const float*, int, const float*, float*, int) nogil>__cublasSspmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy) except* nogil:
    global __cublasDspmv_v2
    _check_or_init_cublas()
    if __cublasDspmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, const double*, int, const double*, double*, int) nogil>__cublasDspmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy) except* nogil:
    global __cublasChpmv_v2
    _check_or_init_cublas()
    if __cublasChpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasChpmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy) except* nogil:
    global __cublasZhpmv_v2
    _check_or_init_cublas()
    if __cublasZhpmv_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpmv_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZhpmv_v2)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSger(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except* nogil:
    global __cublasSger_v2
    _check_or_init_cublas()
    if __cublasSger_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSger_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const float*, const float*, int, const float*, int, float*, int) nogil>__cublasSger_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDger(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except* nogil:
    global __cublasDger_v2
    _check_or_init_cublas()
    if __cublasDger_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDger_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const double*, const double*, int, const double*, int, double*, int) nogil>__cublasDger_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgeru(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    global __cublasCgeru_v2
    _check_or_init_cublas()
    if __cublasCgeru_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeru_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCgeru_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgerc(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    global __cublasCgerc_v2
    _check_or_init_cublas()
    if __cublasCgerc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgerc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCgerc_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgeru(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZgeru_v2
    _check_or_init_cublas()
    if __cublasZgeru_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeru_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZgeru_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgerc(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZgerc_v2
    _check_or_init_cublas()
    if __cublasZgerc_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgerc_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZgerc_v2)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda) except* nogil:
    global __cublasSsyr_v2
    _check_or_init_cublas()
    if __cublasSsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, float*, int) nogil>__cublasSsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* A, int lda) except* nogil:
    global __cublasDsyr_v2
    _check_or_init_cublas()
    if __cublasDsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, double*, int) nogil>__cublasDsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except* nogil:
    global __cublasCsyr_v2
    _check_or_init_cublas()
    if __cublasCsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) nogil>__cublasCsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZsyr_v2
    _check_or_init_cublas()
    if __cublasZsyr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZsyr_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda) except* nogil:
    global __cublasCher_v2
    _check_or_init_cublas()
    if __cublasCher_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const cuComplex*, int, cuComplex*, int) nogil>__cublasCher_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZher_v2
    _check_or_init_cublas()
    if __cublasZher_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZher_v2)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP) except* nogil:
    global __cublasSspr_v2
    _check_or_init_cublas()
    if __cublasSspr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, float*) nogil>__cublasSspr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP) except* nogil:
    global __cublasDspr_v2
    _check_or_init_cublas()
    if __cublasDspr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, double*) nogil>__cublasDspr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP) except* nogil:
    global __cublasChpr_v2
    _check_or_init_cublas()
    if __cublasChpr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const cuComplex*, int, cuComplex*) nogil>__cublasChpr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP) except* nogil:
    global __cublasZhpr_v2
    _check_or_init_cublas()
    if __cublasZhpr_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const cuDoubleComplex*, int, cuDoubleComplex*) nogil>__cublasZhpr_v2)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda) except* nogil:
    global __cublasSsyr2_v2
    _check_or_init_cublas()
    if __cublasSsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, const float*, int, float*, int) nogil>__cublasSsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda) except* nogil:
    global __cublasDsyr2_v2
    _check_or_init_cublas()
    if __cublasDsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, const double*, int, double*, int) nogil>__cublasDsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    global __cublasCsyr2_v2
    _check_or_init_cublas()
    if __cublasCsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZsyr2_v2
    _check_or_init_cublas()
    if __cublasZsyr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZsyr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda) except* nogil:
    global __cublasCher2_v2
    _check_or_init_cublas()
    if __cublasCher2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCher2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZher2_v2
    _check_or_init_cublas()
    if __cublasZher2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZher2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP) except* nogil:
    global __cublasSspr2_v2
    _check_or_init_cublas()
    if __cublasSspr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, const float*, int, const float*, int, float*) nogil>__cublasSspr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP) except* nogil:
    global __cublasDspr2_v2
    _check_or_init_cublas()
    if __cublasDspr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, const double*, int, const double*, int, double*) nogil>__cublasDspr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP) except* nogil:
    global __cublasChpr2_v2
    _check_or_init_cublas()
    if __cublasChpr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*) nogil>__cublasChpr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP) except* nogil:
    global __cublasZhpr2_v2
    _check_or_init_cublas()
    if __cublasZhpr2_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr2_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*) nogil>__cublasZhpr2_v2)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    global __cublasSgemm_v2
    _check_or_init_cublas()
    if __cublasSgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    global __cublasDgemm_v2
    _check_or_init_cublas()
    if __cublasDgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCgemm_v2
    _check_or_init_cublas()
    if __cublasCgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCgemm3m
    _check_or_init_cublas()
    if __cublasCgemm3m == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3m is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCgemm3m)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasCgemm3mEx
    _check_or_init_cublas()
    if __cublasCgemm3mEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const void*, cudaDataType, int, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) nogil>__cublasCgemm3mEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasZgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZgemm_v2
    _check_or_init_cublas()
    if __cublasZgemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZgemm_v2)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZgemm3m
    _check_or_init_cublas()
    if __cublasZgemm3m == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm3m is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZgemm3m)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasSgemmEx
    _check_or_init_cublas()
    if __cublasSgemmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const void*, cudaDataType, int, const void*, cudaDataType, int, const float*, void*, cudaDataType, int) nogil>__cublasSgemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    global __cublasGemmEx
    _check_or_init_cublas()
    if __cublasGemmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t) nogil>__cublasGemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)


cdef cublasStatus_t _cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasCgemmEx
    _check_or_init_cublas()
    if __cublasCgemmEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const void*, cudaDataType, int, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) nogil>__cublasCgemmEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc, int m, int n, int k, const unsigned char* A, int A_bias, int lda, const unsigned char* B, int B_bias, int ldb, unsigned char* C, int C_bias, int ldc, int C_mult, int C_shift) except* nogil:
    global __cublasUint8gemmBias
    _check_or_init_cublas()
    if __cublasUint8gemmBias == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasUint8gemmBias is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, int, int, int, const unsigned char*, int, int, const unsigned char*, int, int, unsigned char*, int, int, int, int) nogil>__cublasUint8gemmBias)(
        handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)


cdef cublasStatus_t _cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) except* nogil:
    global __cublasSsyrk_v2
    _check_or_init_cublas()
    if __cublasSsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, float*, int) nogil>__cublasSsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) except* nogil:
    global __cublasDsyrk_v2
    _check_or_init_cublas()
    if __cublasDsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, double*, int) nogil>__cublasDsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCsyrk_v2
    _check_or_init_cublas()
    if __cublasCsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZsyrk_v2
    _check_or_init_cublas()
    if __cublasZsyrk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZsyrk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasCsyrkEx
    _check_or_init_cublas()
    if __cublasCsyrkEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) nogil>__cublasCsyrkEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasCsyrk3mEx
    _check_or_init_cublas()
    if __cublasCsyrk3mEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk3mEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const void*, cudaDataType, int, const cuComplex*, void*, cudaDataType, int) nogil>__cublasCsyrk3mEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCherk_v2
    _check_or_init_cublas()
    if __cublasCherk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const cuComplex*, int, const float*, cuComplex*, int) nogil>__cublasCherk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZherk_v2
    _check_or_init_cublas()
    if __cublasZherk_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherk_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int) nogil>__cublasZherk_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasCherkEx
    _check_or_init_cublas()
    if __cublasCherkEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const void*, cudaDataType, int, const float*, void*, cudaDataType, int) nogil>__cublasCherkEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc) except* nogil:
    global __cublasCherk3mEx
    _check_or_init_cublas()
    if __cublasCherk3mEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk3mEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const void*, cudaDataType, int, const float*, void*, cudaDataType, int) nogil>__cublasCherk3mEx)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    global __cublasSsyr2k_v2
    _check_or_init_cublas()
    if __cublasSsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    global __cublasDsyr2k_v2
    _check_or_init_cublas()
    if __cublasDsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCsyr2k_v2
    _check_or_init_cublas()
    if __cublasCsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZsyr2k_v2
    _check_or_init_cublas()
    if __cublasZsyr2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZsyr2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCher2k_v2
    _check_or_init_cublas()
    if __cublasCher2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const float*, cuComplex*, int) nogil>__cublasCher2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZher2k_v2
    _check_or_init_cublas()
    if __cublasZher2k_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2k_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int) nogil>__cublasZher2k_v2)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    global __cublasSsyrkx
    _check_or_init_cublas()
    if __cublasSsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    global __cublasDsyrkx
    _check_or_init_cublas()
    if __cublasDsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCsyrkx
    _check_or_init_cublas()
    if __cublasCsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZsyrkx
    _check_or_init_cublas()
    if __cublasZsyrkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZsyrkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCherkx
    _check_or_init_cublas()
    if __cublasCherkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const float*, cuComplex*, int) nogil>__cublasCherkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZherkx
    _check_or_init_cublas()
    if __cublasZherkx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherkx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int) nogil>__cublasZherkx)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) except* nogil:
    global __cublasSsymm_v2
    _check_or_init_cublas()
    if __cublasSsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) nogil>__cublasSsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) except* nogil:
    global __cublasDsymm_v2
    _check_or_init_cublas()
    if __cublasDsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const double*, const double*, int, const double*, int, const double*, double*, int) nogil>__cublasDsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasCsymm_v2
    _check_or_init_cublas()
    if __cublasCsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasCsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZsymm_v2
    _check_or_init_cublas()
    if __cublasZsymm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZsymm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) except* nogil:
    global __cublasChemm_v2
    _check_or_init_cublas()
    if __cublasChemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, const cuComplex*, cuComplex*, int) nogil>__cublasChemm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZhemm_v2
    _check_or_init_cublas()
    if __cublasZhemm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZhemm_v2)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasStrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb) except* nogil:
    global __cublasStrsm_v2
    _check_or_init_cublas()
    if __cublasStrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, float*, int) nogil>__cublasStrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb) except* nogil:
    global __cublasDtrsm_v2
    _check_or_init_cublas()
    if __cublasDtrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, double*, int) nogil>__cublasDtrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb) except* nogil:
    global __cublasCtrsm_v2
    _check_or_init_cublas()
    if __cublasCtrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) nogil>__cublasCtrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb) except* nogil:
    global __cublasZtrsm_v2
    _check_or_init_cublas()
    if __cublasZtrsm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZtrsm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasStrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc) except* nogil:
    global __cublasStrmm_v2
    _check_or_init_cublas()
    if __cublasStrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, const float*, int, float*, int) nogil>__cublasStrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc) except* nogil:
    global __cublasDtrmm_v2
    _check_or_init_cublas()
    if __cublasDtrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, const double*, int, double*, int) nogil>__cublasDtrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc) except* nogil:
    global __cublasCtrmm_v2
    _check_or_init_cublas()
    if __cublasCtrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCtrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZtrmm_v2
    _check_or_init_cublas()
    if __cublasZtrmm_v2 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmm_v2 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZtrmm_v2)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount) except* nogil:
    global __cublasSgemmBatched
    _check_or_init_cublas()
    if __cublasSgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float* const*, int, const float* const*, int, const float*, float* const*, int, int) nogil>__cublasSgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount) except* nogil:
    global __cublasDgemmBatched
    _check_or_init_cublas()
    if __cublasDgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double* const*, int, const double* const*, int, const double*, double* const*, int, int) nogil>__cublasDgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except* nogil:
    global __cublasCgemmBatched
    _check_or_init_cublas()
    if __cublasCgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex* const*, int, const cuComplex* const*, int, const cuComplex*, cuComplex* const*, int, int) nogil>__cublasCgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount) except* nogil:
    global __cublasCgemm3mBatched
    _check_or_init_cublas()
    if __cublasCgemm3mBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex* const*, int, const cuComplex* const*, int, const cuComplex*, cuComplex* const*, int, int) nogil>__cublasCgemm3mBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount) except* nogil:
    global __cublasZgemmBatched
    _check_or_init_cublas()
    if __cublasZgemmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex* const*, int, const cuDoubleComplex* const*, int, const cuDoubleComplex*, cuDoubleComplex* const*, int, int) nogil>__cublasZgemmBatched)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    global __cublasGemmBatchedEx
    _check_or_init_cublas()
    if __cublasGemmBatchedEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmBatchedEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void* const*, cudaDataType, int, const void* const*, cudaDataType, int, const void*, void* const*, cudaDataType, int, int, cublasComputeType_t, cublasGemmAlgo_t) nogil>__cublasGemmBatchedEx)(
        handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)


cdef cublasStatus_t _cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    global __cublasGemmStridedBatchedEx
    _check_or_init_cublas()
    if __cublasGemmStridedBatchedEx == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmStridedBatchedEx is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType, int, long long int, const void*, cudaDataType, int, long long int, const void*, void*, cudaDataType, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t) nogil>__cublasGemmStridedBatchedEx)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)


cdef cublasStatus_t _cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount) except* nogil:
    global __cublasSgemmStridedBatched
    _check_or_init_cublas()
    if __cublasSgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, long long int, const float*, int, long long int, const float*, float*, int, long long int, int) nogil>__cublasSgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount) except* nogil:
    global __cublasDgemmStridedBatched
    _check_or_init_cublas()
    if __cublasDgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double*, const double*, int, long long int, const double*, int, long long int, const double*, double*, int, long long int, int) nogil>__cublasDgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except* nogil:
    global __cublasCgemmStridedBatched
    _check_or_init_cublas()
    if __cublasCgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, long long int, const cuComplex*, int, long long int, const cuComplex*, cuComplex*, int, long long int, int) nogil>__cublasCgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemm3mStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount) except* nogil:
    global __cublasCgemm3mStridedBatched
    _check_or_init_cublas()
    if __cublasCgemm3mStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex*, const cuComplex*, int, long long int, const cuComplex*, int, long long int, const cuComplex*, cuComplex*, int, long long int, int) nogil>__cublasCgemm3mStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount) except* nogil:
    global __cublasZgemmStridedBatched
    _check_or_init_cublas()
    if __cublasZgemmStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, cuDoubleComplex*, int, long long int, int) nogil>__cublasZgemmStridedBatched)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) except* nogil:
    global __cublasSgeam
    _check_or_init_cublas()
    if __cublasSgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const float*, const float*, int, const float*, const float*, int, float*, int) nogil>__cublasSgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) except* nogil:
    global __cublasDgeam
    _check_or_init_cublas()
    if __cublasDgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const double*, const double*, int, const double*, const double*, int, double*, int) nogil>__cublasDgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc) except* nogil:
    global __cublasCgeam
    _check_or_init_cublas()
    if __cublasCgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, const cuComplex*, int, cuComplex*, int) nogil>__cublasCgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZgeam
    _check_or_init_cublas()
    if __cublasZgeam == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeam is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, const cuDoubleComplex*, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZgeam)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    global __cublasSgetrfBatched
    _check_or_init_cublas()
    if __cublasSgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, float* const*, int, int*, int*, int) nogil>__cublasSgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    global __cublasDgetrfBatched
    _check_or_init_cublas()
    if __cublasDgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, double* const*, int, int*, int*, int) nogil>__cublasDgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    global __cublasCgetrfBatched
    _check_or_init_cublas()
    if __cublasCgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuComplex* const*, int, int*, int*, int) nogil>__cublasCgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize) except* nogil:
    global __cublasZgetrfBatched
    _check_or_init_cublas()
    if __cublasZgetrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgetrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex* const*, int, int*, int*, int) nogil>__cublasZgetrfBatched)(
        handle, n, A, lda, P, info, batchSize)


cdef cublasStatus_t _cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize) except* nogil:
    global __cublasSgetriBatched
    _check_or_init_cublas()
    if __cublasSgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float* const*, int, const int*, float* const*, int, int*, int) nogil>__cublasSgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize) except* nogil:
    global __cublasDgetriBatched
    _check_or_init_cublas()
    if __cublasDgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double* const*, int, const int*, double* const*, int, int*, int) nogil>__cublasDgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize) except* nogil:
    global __cublasCgetriBatched
    _check_or_init_cublas()
    if __cublasCgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex* const*, int, const int*, cuComplex* const*, int, int*, int) nogil>__cublasCgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize) except* nogil:
    global __cublasZgetriBatched
    _check_or_init_cublas()
    if __cublasZgetriBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgetriBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex* const*, int, const int*, cuDoubleComplex* const*, int, int*, int) nogil>__cublasZgetriBatched)(
        handle, n, A, lda, P, C, ldc, info, batchSize)


cdef cublasStatus_t _cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    global __cublasSgetrsBatched
    _check_or_init_cublas()
    if __cublasSgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float* const*, int, const int*, float* const*, int, int*, int) nogil>__cublasSgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    global __cublasDgetrsBatched
    _check_or_init_cublas()
    if __cublasDgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double* const*, int, const int*, double* const*, int, int*, int) nogil>__cublasDgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    global __cublasCgetrsBatched
    _check_or_init_cublas()
    if __cublasCgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex* const*, int, const int*, cuComplex* const*, int, int*, int) nogil>__cublasCgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize) except* nogil:
    global __cublasZgetrsBatched
    _check_or_init_cublas()
    if __cublasZgetrsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgetrsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex* const*, int, const int*, cuDoubleComplex* const*, int, int*, int) nogil>__cublasZgetrsBatched)(
        handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)


cdef cublasStatus_t _cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount) except* nogil:
    global __cublasStrsmBatched
    _check_or_init_cublas()
    if __cublasStrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const float*, const float* const*, int, float* const*, int, int) nogil>__cublasStrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount) except* nogil:
    global __cublasDtrsmBatched
    _check_or_init_cublas()
    if __cublasDtrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const double*, const double* const*, int, double* const*, int, int) nogil>__cublasDtrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount) except* nogil:
    global __cublasCtrsmBatched
    _check_or_init_cublas()
    if __cublasCtrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*, const cuComplex* const*, int, cuComplex* const*, int, int) nogil>__cublasCtrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount) except* nogil:
    global __cublasZtrsmBatched
    _check_or_init_cublas()
    if __cublasZtrsmBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsmBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const cuDoubleComplex*, const cuDoubleComplex* const*, int, cuDoubleComplex* const*, int, int) nogil>__cublasZtrsmBatched)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasSmatinvBatched(cublasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    global __cublasSmatinvBatched
    _check_or_init_cublas()
    if __cublasSmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const float* const*, int, float* const*, int, int*, int) nogil>__cublasSmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasDmatinvBatched(cublasHandle_t handle, int n, const double* const A[], int lda, double* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    global __cublasDmatinvBatched
    _check_or_init_cublas()
    if __cublasDmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const double* const*, int, double* const*, int, int*, int) nogil>__cublasDmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, cuComplex* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    global __cublasCmatinvBatched
    _check_or_init_cublas()
    if __cublasCmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuComplex* const*, int, cuComplex* const*, int, int*, int) nogil>__cublasCmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const Ainv[], int lda_inv, int* info, int batchSize) except* nogil:
    global __cublasZmatinvBatched
    _check_or_init_cublas()
    if __cublasZmatinvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZmatinvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex* const*, int, cuDoubleComplex* const*, int, int*, int) nogil>__cublasZmatinvBatched)(
        handle, n, A, lda, Ainv, lda_inv, info, batchSize)


cdef cublasStatus_t _cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize) except* nogil:
    global __cublasSgeqrfBatched
    _check_or_init_cublas()
    if __cublasSgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, float* const*, int, float* const*, int*, int) nogil>__cublasSgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize) except* nogil:
    global __cublasDgeqrfBatched
    _check_or_init_cublas()
    if __cublasDgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, double* const*, int, double* const*, int*, int) nogil>__cublasDgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize) except* nogil:
    global __cublasCgeqrfBatched
    _check_or_init_cublas()
    if __cublasCgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, cuComplex* const*, int, cuComplex* const*, int*, int) nogil>__cublasCgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize) except* nogil:
    global __cublasZgeqrfBatched
    _check_or_init_cublas()
    if __cublasZgeqrfBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeqrfBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int, int, cuDoubleComplex* const*, int, cuDoubleComplex* const*, int*, int) nogil>__cublasZgeqrfBatched)(
        handle, m, n, Aarray, lda, TauArray, info, batchSize)


cdef cublasStatus_t _cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    global __cublasSgelsBatched
    _check_or_init_cublas()
    if __cublasSgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, float* const*, int, float* const*, int, int*, int*, int) nogil>__cublasSgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    global __cublasDgelsBatched
    _check_or_init_cublas()
    if __cublasDgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, double* const*, int, double* const*, int, int*, int*, int) nogil>__cublasDgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    global __cublasCgelsBatched
    _check_or_init_cublas()
    if __cublasCgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, cuComplex* const*, int, cuComplex* const*, int, int*, int*, int) nogil>__cublasCgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize) except* nogil:
    global __cublasZgelsBatched
    _check_or_init_cublas()
    if __cublasZgelsBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgelsBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, cuDoubleComplex* const*, int, cuDoubleComplex* const*, int, int*, int*, int) nogil>__cublasZgelsBatched)(
        handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)


cdef cublasStatus_t _cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc) except* nogil:
    global __cublasSdgmm
    _check_or_init_cublas()
    if __cublasSdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const float*, int, const float*, int, float*, int) nogil>__cublasSdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc) except* nogil:
    global __cublasDdgmm
    _check_or_init_cublas()
    if __cublasDdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const double*, int, const double*, int, double*, int) nogil>__cublasDdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc) except* nogil:
    global __cublasCdgmm
    _check_or_init_cublas()
    if __cublasCdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const cuComplex*, int, const cuComplex*, int, cuComplex*, int) nogil>__cublasCdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc) except* nogil:
    global __cublasZdgmm
    _check_or_init_cublas()
    if __cublasZdgmm == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdgmm is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const cuDoubleComplex*, int, const cuDoubleComplex*, int, cuDoubleComplex*, int) nogil>__cublasZdgmm)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* AP, float* A, int lda) except* nogil:
    global __cublasStpttr
    _check_or_init_cublas()
    if __cublasStpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, float*, int) nogil>__cublasStpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* AP, double* A, int lda) except* nogil:
    global __cublasDtpttr
    _check_or_init_cublas()
    if __cublasDtpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, double*, int) nogil>__cublasDtpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* AP, cuComplex* A, int lda) except* nogil:
    global __cublasCtpttr
    _check_or_init_cublas()
    if __cublasCtpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, cuComplex*, int) nogil>__cublasCtpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* AP, cuDoubleComplex* A, int lda) except* nogil:
    global __cublasZtpttr
    _check_or_init_cublas()
    if __cublasZtpttr == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpttr is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, cuDoubleComplex*, int) nogil>__cublasZtpttr)(
        handle, uplo, n, AP, A, lda)


cdef cublasStatus_t _cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* A, int lda, float* AP) except* nogil:
    global __cublasStrttp
    _check_or_init_cublas()
    if __cublasStrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float*, int, float*) nogil>__cublasStrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* A, int lda, double* AP) except* nogil:
    global __cublasDtrttp
    _check_or_init_cublas()
    if __cublasDtrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double*, int, double*) nogil>__cublasDtrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* A, int lda, cuComplex* AP) except* nogil:
    global __cublasCtrttp
    _check_or_init_cublas()
    if __cublasCtrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex*, int, cuComplex*) nogil>__cublasCtrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* AP) except* nogil:
    global __cublasZtrttp
    _check_or_init_cublas()
    if __cublasZtrttp == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrttp is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex*, int, cuDoubleComplex*) nogil>__cublasZtrttp)(
        handle, uplo, n, A, lda, AP)


cdef cublasStatus_t _cublasGetSmCountTarget(cublasHandle_t handle, int* smCountTarget) except* nogil:
    global __cublasGetSmCountTarget
    _check_or_init_cublas()
    if __cublasGetSmCountTarget == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetSmCountTarget is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int*) nogil>__cublasGetSmCountTarget)(
        handle, smCountTarget)


cdef cublasStatus_t _cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) except* nogil:
    global __cublasSetSmCountTarget
    _check_or_init_cublas()
    if __cublasSetSmCountTarget == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetSmCountTarget is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int) nogil>__cublasSetSmCountTarget)(
        handle, smCountTarget)


cdef const char* _cublasGetStatusName(cublasStatus_t status) except* nogil:
    global __cublasGetStatusName
    _check_or_init_cublas()
    if __cublasGetStatusName == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetStatusName is not found")
    return (<const char* (*)(cublasStatus_t) nogil>__cublasGetStatusName)(
        status)


cdef const char* _cublasGetStatusString(cublasStatus_t status) except* nogil:
    global __cublasGetStatusString
    _check_or_init_cublas()
    if __cublasGetStatusString == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetStatusString is not found")
    return (<const char* (*)(cublasStatus_t) nogil>__cublasGetStatusString)(
        status)


cdef cublasStatus_t _cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount) except* nogil:
    global __cublasSgemvBatched
    _check_or_init_cublas()
    if __cublasSgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float* const*, int, const float* const*, int, const float*, float* const*, int, int) nogil>__cublasSgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double* beta, double* const yarray[], int incy, int batchCount) except* nogil:
    global __cublasDgemvBatched
    _check_or_init_cublas()
    if __cublasDgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double* const*, int, const double* const*, int, const double*, double* const*, int, int) nogil>__cublasDgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasCgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const xarray[], int incx, const cuComplex* beta, cuComplex* const yarray[], int incy, int batchCount) except* nogil:
    global __cublasCgemvBatched
    _check_or_init_cublas()
    if __cublasCgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex* const*, int, const cuComplex* const*, int, const cuComplex*, cuComplex* const*, int, int) nogil>__cublasCgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasZgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const xarray[], int incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int incy, int batchCount) except* nogil:
    global __cublasZgemvBatched
    _check_or_init_cublas()
    if __cublasZgemvBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex* const*, int, const cuDoubleComplex* const*, int, const cuDoubleComplex*, cuDoubleComplex* const*, int, int) nogil>__cublasZgemvBatched)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, long long int strideA, const float* x, int incx, long long int stridex, const float* beta, float* y, int incy, long long int stridey, int batchCount) except* nogil:
    global __cublasSgemvStridedBatched
    _check_or_init_cublas()
    if __cublasSgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float*, const float*, int, long long int, const float*, int, long long int, const float*, float*, int, long long int, int) nogil>__cublasSgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, long long int strideA, const double* x, int incx, long long int stridex, const double* beta, double* y, int incy, long long int stridey, int batchCount) except* nogil:
    global __cublasDgemvStridedBatched
    _check_or_init_cublas()
    if __cublasDgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double*, const double*, int, long long int, const double*, int, long long int, const double*, double*, int, long long int, int) nogil>__cublasDgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasCgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* x, int incx, long long int stridex, const cuComplex* beta, cuComplex* y, int incy, long long int stridey, int batchCount) except* nogil:
    global __cublasCgemvStridedBatched
    _check_or_init_cublas()
    if __cublasCgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex*, const cuComplex*, int, long long int, const cuComplex*, int, long long int, const cuComplex*, cuComplex*, int, long long int, int) nogil>__cublasCgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasZgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* x, int incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy, long long int stridey, int batchCount) except* nogil:
    global __cublasZgemvStridedBatched
    _check_or_init_cublas()
    if __cublasZgemvStridedBatched == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvStridedBatched is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, int, long long int, const cuDoubleComplex*, cuDoubleComplex*, int, long long int, int) nogil>__cublasZgemvStridedBatched)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasSetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* devicePtr, int64_t incy) except* nogil:
    global __cublasSetVector_64
    _check_or_init_cublas()
    if __cublasSetVector_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVector_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t) nogil>__cublasSetVector_64)(
        n, elemSize, x, incx, devicePtr, incy)


cdef cublasStatus_t _cublasGetVector_64(int64_t n, int64_t elemSize, const void* x, int64_t incx, void* y, int64_t incy) except* nogil:
    global __cublasGetVector_64
    _check_or_init_cublas()
    if __cublasGetVector_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVector_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t) nogil>__cublasGetVector_64)(
        n, elemSize, x, incx, y, incy)


cdef cublasStatus_t _cublasSetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except* nogil:
    global __cublasSetMatrix_64
    _check_or_init_cublas()
    if __cublasSetMatrix_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrix_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t) nogil>__cublasSetMatrix_64)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasGetMatrix_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb) except* nogil:
    global __cublasGetMatrix_64
    _check_or_init_cublas()
    if __cublasGetMatrix_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrix_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t) nogil>__cublasGetMatrix_64)(
        rows, cols, elemSize, A, lda, B, ldb)


cdef cublasStatus_t _cublasSetVectorAsync_64(int64_t n, int64_t elemSize, const void* hostPtr, int64_t incx, void* devicePtr, int64_t incy, cudaStream_t stream) except* nogil:
    global __cublasSetVectorAsync_64
    _check_or_init_cublas()
    if __cublasSetVectorAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetVectorAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) nogil>__cublasSetVectorAsync_64)(
        n, elemSize, hostPtr, incx, devicePtr, incy, stream)


cdef cublasStatus_t _cublasGetVectorAsync_64(int64_t n, int64_t elemSize, const void* devicePtr, int64_t incx, void* hostPtr, int64_t incy, cudaStream_t stream) except* nogil:
    global __cublasGetVectorAsync_64
    _check_or_init_cublas()
    if __cublasGetVectorAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetVectorAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) nogil>__cublasGetVectorAsync_64)(
        n, elemSize, devicePtr, incx, hostPtr, incy, stream)


cdef cublasStatus_t _cublasSetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except* nogil:
    global __cublasSetMatrixAsync_64
    _check_or_init_cublas()
    if __cublasSetMatrixAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSetMatrixAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) nogil>__cublasSetMatrixAsync_64)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasGetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void* A, int64_t lda, void* B, int64_t ldb, cudaStream_t stream) except* nogil:
    global __cublasGetMatrixAsync_64
    _check_or_init_cublas()
    if __cublasGetMatrixAsync_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGetMatrixAsync_64 is not found")
    return (<cublasStatus_t (*)(int64_t, int64_t, int64_t, const void*, int64_t, void*, int64_t, cudaStream_t) nogil>__cublasGetMatrixAsync_64)(
        rows, cols, elemSize, A, lda, B, ldb, stream)


cdef cublasStatus_t _cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    global __cublasNrm2Ex_64
    _check_or_init_cublas()
    if __cublasNrm2Ex_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasNrm2Ex_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) nogil>__cublasNrm2Ex_64)(
        handle, n, x, xType, incx, result, resultType, executionType)


cdef cublasStatus_t _cublasSnrm2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except* nogil:
    global __cublasSnrm2_v2_64
    _check_or_init_cublas()
    if __cublasSnrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSnrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, float*) nogil>__cublasSnrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDnrm2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except* nogil:
    global __cublasDnrm2_v2_64
    _check_or_init_cublas()
    if __cublasDnrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDnrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, double*) nogil>__cublasDnrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScnrm2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except* nogil:
    global __cublasScnrm2_v2_64
    _check_or_init_cublas()
    if __cublasScnrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScnrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, float*) nogil>__cublasScnrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDznrm2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except* nogil:
    global __cublasDznrm2_v2_64
    _check_or_init_cublas()
    if __cublasDznrm2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDznrm2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, double*) nogil>__cublasDznrm2_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDotEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    global __cublasDotEx_64
    _check_or_init_cublas()
    if __cublasDotEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) nogil>__cublasDotEx_64)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType) except* nogil:
    global __cublasDotcEx_64
    _check_or_init_cublas()
    if __cublasDotcEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDotcEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) nogil>__cublasDotcEx_64)(
        handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)


cdef cublasStatus_t _cublasSdot_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result) except* nogil:
    global __cublasSdot_v2_64
    _check_or_init_cublas()
    if __cublasSdot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, const float*, int64_t, float*) nogil>__cublasSdot_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasDdot_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result) except* nogil:
    global __cublasDdot_v2_64
    _check_or_init_cublas()
    if __cublasDdot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, const double*, int64_t, double*) nogil>__cublasDdot_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotu_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except* nogil:
    global __cublasCdotu_v2_64
    _check_or_init_cublas()
    if __cublasCdotu_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotu_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*) nogil>__cublasCdotu_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasCdotc_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result) except* nogil:
    global __cublasCdotc_v2_64
    _check_or_init_cublas()
    if __cublasCdotc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdotc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*) nogil>__cublasCdotc_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotu_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except* nogil:
    global __cublasZdotu_v2_64
    _check_or_init_cublas()
    if __cublasZdotu_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotu_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*) nogil>__cublasZdotu_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasZdotc_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result) except* nogil:
    global __cublasZdotc_v2_64
    _check_or_init_cublas()
    if __cublasZdotc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdotc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*) nogil>__cublasZdotc_v2_64)(
        handle, n, x, incx, y, incy, result)


cdef cublasStatus_t _cublasScalEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int64_t incx, cudaDataType executionType) except* nogil:
    global __cublasScalEx_64
    _check_or_init_cublas()
    if __cublasScalEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScalEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, void*, cudaDataType, int64_t, cudaDataType) nogil>__cublasScalEx_64)(
        handle, n, alpha, alphaType, x, xType, incx, executionType)


cdef cublasStatus_t _cublasSscal_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx) except* nogil:
    global __cublasSscal_v2_64
    _check_or_init_cublas()
    if __cublasSscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, float*, int64_t) nogil>__cublasSscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasDscal_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx) except* nogil:
    global __cublasDscal_v2_64
    _check_or_init_cublas()
    if __cublasDscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, double*, int64_t) nogil>__cublasDscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCscal_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCscal_v2_64
    _check_or_init_cublas()
    if __cublasCscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasCsscal_64(cublasHandle_t handle, int64_t n, const float* alpha, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCsscal_v2_64
    _check_or_init_cublas()
    if __cublasCsscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, cuComplex*, int64_t) nogil>__cublasCsscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZscal_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZscal_v2_64
    _check_or_init_cublas()
    if __cublasZscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasZdscal_64(cublasHandle_t handle, int64_t n, const double* alpha, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZdscal_v2_64
    _check_or_init_cublas()
    if __cublasZdscal_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdscal_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, cuDoubleComplex*, int64_t) nogil>__cublasZdscal_v2_64)(
        handle, n, alpha, x, incx)


cdef cublasStatus_t _cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, cudaDataType executiontype) except* nogil:
    global __cublasAxpyEx_64
    _check_or_init_cublas()
    if __cublasAxpyEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAxpyEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, const void*, cudaDataType, int64_t, void*, cudaDataType, int64_t, cudaDataType) nogil>__cublasAxpyEx_64)(
        handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)


cdef cublasStatus_t _cublasSaxpy_64(cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy) except* nogil:
    global __cublasSaxpy_v2_64
    _check_or_init_cublas()
    if __cublasSaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, const float*, int64_t, float*, int64_t) nogil>__cublasSaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasDaxpy_64(cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy) except* nogil:
    global __cublasDaxpy_v2_64
    _check_or_init_cublas()
    if __cublasDaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, const double*, int64_t, double*, int64_t) nogil>__cublasDaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCaxpy_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil:
    global __cublasCaxpy_v2_64
    _check_or_init_cublas()
    if __cublasCaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasZaxpy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZaxpy_v2_64
    _check_or_init_cublas()
    if __cublasZaxpy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZaxpy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZaxpy_v2_64)(
        handle, n, alpha, x, incx, y, incy)


cdef cublasStatus_t _cublasCopyEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except* nogil:
    global __cublasCopyEx_64
    _check_or_init_cublas()
    if __cublasCopyEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCopyEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, int64_t) nogil>__cublasCopyEx_64)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasScopy_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy) except* nogil:
    global __cublasScopy_v2_64
    _check_or_init_cublas()
    if __cublasScopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasScopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDcopy_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy) except* nogil:
    global __cublasDcopy_v2_64
    _check_or_init_cublas()
    if __cublasDcopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDcopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDcopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCcopy_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil:
    global __cublasCcopy_v2_64
    _check_or_init_cublas()
    if __cublasCcopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCcopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCcopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZcopy_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZcopy_v2_64
    _check_or_init_cublas()
    if __cublasZcopy_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZcopy_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZcopy_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSswap_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy) except* nogil:
    global __cublasSswap_v2_64
    _check_or_init_cublas()
    if __cublasSswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, float*, int64_t, float*, int64_t) nogil>__cublasSswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasDswap_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy) except* nogil:
    global __cublasDswap_v2_64
    _check_or_init_cublas()
    if __cublasDswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, double*, int64_t, double*, int64_t) nogil>__cublasDswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasCswap_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy) except* nogil:
    global __cublasCswap_v2_64
    _check_or_init_cublas()
    if __cublasCswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasZswap_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZswap_v2_64
    _check_or_init_cublas()
    if __cublasZswap_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZswap_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZswap_v2_64)(
        handle, n, x, incx, y, incy)


cdef cublasStatus_t _cublasSwapEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy) except* nogil:
    global __cublasSwapEx_64
    _check_or_init_cublas()
    if __cublasSwapEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSwapEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, void*, cudaDataType, int64_t, void*, cudaDataType, int64_t) nogil>__cublasSwapEx_64)(
        handle, n, x, xType, incx, y, yType, incy)


cdef cublasStatus_t _cublasIsamax_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIsamax_v2_64
    _check_or_init_cublas()
    if __cublasIsamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, int64_t*) nogil>__cublasIsamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamax_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIdamax_v2_64
    _check_or_init_cublas()
    if __cublasIdamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, int64_t*) nogil>__cublasIdamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamax_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIcamax_v2_64
    _check_or_init_cublas()
    if __cublasIcamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, int64_t*) nogil>__cublasIcamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamax_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIzamax_v2_64
    _check_or_init_cublas()
    if __cublasIzamax_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamax_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, int64_t*) nogil>__cublasIzamax_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIamaxEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except* nogil:
    global __cublasIamaxEx_64
    _check_or_init_cublas()
    if __cublasIamaxEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIamaxEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, int64_t*) nogil>__cublasIamaxEx_64)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasIsamin_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIsamin_v2_64
    _check_or_init_cublas()
    if __cublasIsamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIsamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, int64_t*) nogil>__cublasIsamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIdamin_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIdamin_v2_64
    _check_or_init_cublas()
    if __cublasIdamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIdamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, int64_t*) nogil>__cublasIdamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIcamin_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIcamin_v2_64
    _check_or_init_cublas()
    if __cublasIcamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIcamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, int64_t*) nogil>__cublasIcamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIzamin_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result) except* nogil:
    global __cublasIzamin_v2_64
    _check_or_init_cublas()
    if __cublasIzamin_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIzamin_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, int64_t*) nogil>__cublasIzamin_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasIaminEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, int64_t* result) except* nogil:
    global __cublasIaminEx_64
    _check_or_init_cublas()
    if __cublasIaminEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasIaminEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, int64_t*) nogil>__cublasIaminEx_64)(
        handle, n, x, xType, incx, result)


cdef cublasStatus_t _cublasAsumEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executiontype) except* nogil:
    global __cublasAsumEx_64
    _check_or_init_cublas()
    if __cublasAsumEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasAsumEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const void*, cudaDataType, int64_t, void*, cudaDataType, cudaDataType) nogil>__cublasAsumEx_64)(
        handle, n, x, xType, incx, result, resultType, executiontype)


cdef cublasStatus_t _cublasSasum_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result) except* nogil:
    global __cublasSasum_v2_64
    _check_or_init_cublas()
    if __cublasSasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const float*, int64_t, float*) nogil>__cublasSasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDasum_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result) except* nogil:
    global __cublasDasum_v2_64
    _check_or_init_cublas()
    if __cublasDasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const double*, int64_t, double*) nogil>__cublasDasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasScasum_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result) except* nogil:
    global __cublasScasum_v2_64
    _check_or_init_cublas()
    if __cublasScasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasScasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuComplex*, int64_t, float*) nogil>__cublasScasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasDzasum_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result) except* nogil:
    global __cublasDzasum_v2_64
    _check_or_init_cublas()
    if __cublasDzasum_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDzasum_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, const cuDoubleComplex*, int64_t, double*) nogil>__cublasDzasum_v2_64)(
        handle, n, x, incx, result)


cdef cublasStatus_t _cublasSrot_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s) except* nogil:
    global __cublasSrot_v2_64
    _check_or_init_cublas()
    if __cublasSrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, float*, int64_t, float*, int64_t, const float*, const float*) nogil>__cublasSrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasDrot_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s) except* nogil:
    global __cublasDrot_v2_64
    _check_or_init_cublas()
    if __cublasDrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, double*, int64_t, double*, int64_t, const double*, const double*) nogil>__cublasDrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const cuComplex* s) except* nogil:
    global __cublasCrot_v2_64
    _check_or_init_cublas()
    if __cublasCrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuComplex*, int64_t, cuComplex*, int64_t, const float*, const cuComplex*) nogil>__cublasCrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasCsrot_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const float* s) except* nogil:
    global __cublasCsrot_v2_64
    _check_or_init_cublas()
    if __cublasCsrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuComplex*, int64_t, cuComplex*, int64_t, const float*, const float*) nogil>__cublasCsrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const cuDoubleComplex* s) except* nogil:
    global __cublasZrot_v2_64
    _check_or_init_cublas()
    if __cublasZrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t, const double*, const cuDoubleComplex*) nogil>__cublasZrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasZdrot_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const double* s) except* nogil:
    global __cublasZdrot_v2_64
    _check_or_init_cublas()
    if __cublasZdrot_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdrot_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t, const double*, const double*) nogil>__cublasZdrot_v2_64)(
        handle, n, x, incx, y, incy, c, s)


cdef cublasStatus_t _cublasRotEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype) except* nogil:
    global __cublasRotEx_64
    _check_or_init_cublas()
    if __cublasRotEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, void*, cudaDataType, int64_t, void*, cudaDataType, int64_t, const void*, const void*, cudaDataType, cudaDataType) nogil>__cublasRotEx_64)(
        handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)


cdef cublasStatus_t _cublasSrotm_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param) except* nogil:
    global __cublasSrotm_v2_64
    _check_or_init_cublas()
    if __cublasSrotm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSrotm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, float*, int64_t, float*, int64_t, const float*) nogil>__cublasSrotm_v2_64)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasDrotm_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param) except* nogil:
    global __cublasDrotm_v2_64
    _check_or_init_cublas()
    if __cublasDrotm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDrotm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, double*, int64_t, double*, int64_t, const double*) nogil>__cublasDrotm_v2_64)(
        handle, n, x, incx, y, incy, param)


cdef cublasStatus_t _cublasRotmEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* param, cudaDataType paramType, cudaDataType executiontype) except* nogil:
    global __cublasRotmEx_64
    _check_or_init_cublas()
    if __cublasRotmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasRotmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, void*, cudaDataType, int64_t, void*, cudaDataType, int64_t, const void*, cudaDataType, cudaDataType) nogil>__cublasRotmEx_64)(
        handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)


cdef cublasStatus_t _cublasSgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    global __cublasSgemv_v2_64
    _check_or_init_cublas()
    if __cublasSgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    global __cublasDgemv_v2_64
    _check_or_init_cublas()
    if __cublasDgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    global __cublasCgemv_v2_64
    _check_or_init_cublas()
    if __cublasCgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgemv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZgemv_v2_64
    _check_or_init_cublas()
    if __cublasZgemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZgemv_v2_64)(
        handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    global __cublasSgbmv_v2_64
    _check_or_init_cublas()
    if __cublasSgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    global __cublasDgbmv_v2_64
    _check_or_init_cublas()
    if __cublasDgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    global __cublasCgbmv_v2_64
    _check_or_init_cublas()
    if __cublasCgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZgbmv_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZgbmv_v2_64
    _check_or_init_cublas()
    if __cublasZgbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZgbmv_v2_64)(
        handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasStrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    global __cublasStrmv_v2_64
    _check_or_init_cublas()
    if __cublasStrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasStrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    global __cublasDtrmv_v2_64
    _check_or_init_cublas()
    if __cublasDtrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDtrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCtrmv_v2_64
    _check_or_init_cublas()
    if __cublasCtrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCtrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZtrmv_v2_64
    _check_or_init_cublas()
    if __cublasZtrmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZtrmv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    global __cublasStbmv_v2_64
    _check_or_init_cublas()
    if __cublasStbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasStbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    global __cublasDtbmv_v2_64
    _check_or_init_cublas()
    if __cublasDtbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDtbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCtbmv_v2_64
    _check_or_init_cublas()
    if __cublasCtbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCtbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZtbmv_v2_64
    _check_or_init_cublas()
    if __cublasZtbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZtbmv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasStpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except* nogil:
    global __cublasStpmv_v2_64
    _check_or_init_cublas()
    if __cublasStpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, float*, int64_t) nogil>__cublasStpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except* nogil:
    global __cublasDtpmv_v2_64
    _check_or_init_cublas()
    if __cublasDtpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, double*, int64_t) nogil>__cublasDtpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCtpmv_v2_64
    _check_or_init_cublas()
    if __cublasCtpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCtpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZtpmv_v2_64
    _check_or_init_cublas()
    if __cublasZtpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZtpmv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    global __cublasStrsv_v2_64
    _check_or_init_cublas()
    if __cublasStrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasStrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasDtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    global __cublasDtrsv_v2_64
    _check_or_init_cublas()
    if __cublasDtrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDtrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasCtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCtrsv_v2_64
    _check_or_init_cublas()
    if __cublasCtrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCtrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasZtrsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZtrsv_v2_64
    _check_or_init_cublas()
    if __cublasZtrsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZtrsv_v2_64)(
        handle, uplo, trans, diag, n, A, lda, x, incx)


cdef cublasStatus_t _cublasStpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx) except* nogil:
    global __cublasStpsv_v2_64
    _check_or_init_cublas()
    if __cublasStpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const float*, float*, int64_t) nogil>__cublasStpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasDtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx) except* nogil:
    global __cublasDtpsv_v2_64
    _check_or_init_cublas()
    if __cublasDtpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const double*, double*, int64_t) nogil>__cublasDtpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasCtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCtpsv_v2_64
    _check_or_init_cublas()
    if __cublasCtpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCtpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasZtpsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZtpsv_v2_64
    _check_or_init_cublas()
    if __cublasZtpsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtpsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZtpsv_v2_64)(
        handle, uplo, trans, diag, n, AP, x, incx)


cdef cublasStatus_t _cublasStbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx) except* nogil:
    global __cublasStbsv_v2_64
    _check_or_init_cublas()
    if __cublasStbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasStbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasDtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx) except* nogil:
    global __cublasDtbsv_v2_64
    _check_or_init_cublas()
    if __cublasDtbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDtbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasCtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx) except* nogil:
    global __cublasCtbsv_v2_64
    _check_or_init_cublas()
    if __cublasCtbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCtbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasZtbsv_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx) except* nogil:
    global __cublasZtbsv_v2_64
    _check_or_init_cublas()
    if __cublasZtbsv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtbsv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZtbsv_v2_64)(
        handle, uplo, trans, diag, n, k, A, lda, x, incx)


cdef cublasStatus_t _cublasSsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    global __cublasSsymv_v2_64
    _check_or_init_cublas()
    if __cublasSsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    global __cublasDsymv_v2_64
    _check_or_init_cublas()
    if __cublasDsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasCsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    global __cublasCsymv_v2_64
    _check_or_init_cublas()
    if __cublasCsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZsymv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZsymv_v2_64
    _check_or_init_cublas()
    if __cublasZsymv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZsymv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    global __cublasChemv_v2_64
    _check_or_init_cublas()
    if __cublasChemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasChemv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhemv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZhemv_v2_64
    _check_or_init_cublas()
    if __cublasZhemv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZhemv_v2_64)(
        handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    global __cublasSsbmv_v2_64
    _check_or_init_cublas()
    if __cublasSsbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSsbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDsbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    global __cublasDsbmv_v2_64
    _check_or_init_cublas()
    if __cublasDsbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDsbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    global __cublasChbmv_v2_64
    _check_or_init_cublas()
    if __cublasChbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasChbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhbmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZhbmv_v2_64
    _check_or_init_cublas()
    if __cublasZhbmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhbmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZhbmv_v2_64)(
        handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy) except* nogil:
    global __cublasSspmv_v2_64
    _check_or_init_cublas()
    if __cublasSspmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSspmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasDspmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy) except* nogil:
    global __cublasDspmv_v2_64
    _check_or_init_cublas()
    if __cublasDspmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDspmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasChpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy) except* nogil:
    global __cublasChpmv_v2_64
    _check_or_init_cublas()
    if __cublasChpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasChpmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasZhpmv_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy) except* nogil:
    global __cublasZhpmv_v2_64
    _check_or_init_cublas()
    if __cublasZhpmv_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpmv_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZhpmv_v2_64)(
        handle, uplo, n, alpha, AP, x, incx, beta, y, incy)


cdef cublasStatus_t _cublasSger_64(cublasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except* nogil:
    global __cublasSger_v2_64
    _check_or_init_cublas()
    if __cublasSger_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSger_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasSger_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDger_64(cublasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except* nogil:
    global __cublasDger_v2_64
    _check_or_init_cublas()
    if __cublasDger_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDger_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDger_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    global __cublasCgeru_v2_64
    _check_or_init_cublas()
    if __cublasCgeru_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeru_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCgeru_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    global __cublasCgerc_v2_64
    _check_or_init_cublas()
    if __cublasCgerc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgerc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCgerc_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgeru_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    global __cublasZgeru_v2_64
    _check_or_init_cublas()
    if __cublasZgeru_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeru_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZgeru_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZgerc_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    global __cublasZgerc_v2_64
    _check_or_init_cublas()
    if __cublasZgerc_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgerc_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZgerc_v2_64)(
        handle, m, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* A, int64_t lda) except* nogil:
    global __cublasSsyr_v2_64
    _check_or_init_cublas()
    if __cublasSsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, float*, int64_t) nogil>__cublasSsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasDsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* A, int64_t lda) except* nogil:
    global __cublasDsyr_v2_64
    _check_or_init_cublas()
    if __cublasDsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, double*, int64_t) nogil>__cublasDsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except* nogil:
    global __cublasCsyr_v2_64
    _check_or_init_cublas()
    if __cublasCsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZsyr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except* nogil:
    global __cublasZsyr_v2_64
    _check_or_init_cublas()
    if __cublasZsyr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZsyr_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasCher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda) except* nogil:
    global __cublasCher_v2_64
    _check_or_init_cublas()
    if __cublasCher_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCher_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasZher_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda) except* nogil:
    global __cublasZher_v2_64
    _check_or_init_cublas()
    if __cublasZher_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZher_v2_64)(
        handle, uplo, n, alpha, x, incx, A, lda)


cdef cublasStatus_t _cublasSspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP) except* nogil:
    global __cublasSspr_v2_64
    _check_or_init_cublas()
    if __cublasSspr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, float*) nogil>__cublasSspr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasDspr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP) except* nogil:
    global __cublasDspr_v2_64
    _check_or_init_cublas()
    if __cublasDspr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, double*) nogil>__cublasDspr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasChpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* AP) except* nogil:
    global __cublasChpr_v2_64
    _check_or_init_cublas()
    if __cublasChpr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const cuComplex*, int64_t, cuComplex*) nogil>__cublasChpr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasZhpr_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* AP) except* nogil:
    global __cublasZhpr_v2_64
    _check_or_init_cublas()
    if __cublasZhpr_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const cuDoubleComplex*, int64_t, cuDoubleComplex*) nogil>__cublasZhpr_v2_64)(
        handle, uplo, n, alpha, x, incx, AP)


cdef cublasStatus_t _cublasSsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda) except* nogil:
    global __cublasSsyr2_v2_64
    _check_or_init_cublas()
    if __cublasSsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasSsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasDsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda) except* nogil:
    global __cublasDsyr2_v2_64
    _check_or_init_cublas()
    if __cublasDsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    global __cublasCsyr2_v2_64
    _check_or_init_cublas()
    if __cublasCsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZsyr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    global __cublasZsyr2_v2_64
    _check_or_init_cublas()
    if __cublasZsyr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZsyr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasCher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda) except* nogil:
    global __cublasCher2_v2_64
    _check_or_init_cublas()
    if __cublasCher2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCher2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasZher2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda) except* nogil:
    global __cublasZher2_v2_64
    _check_or_init_cublas()
    if __cublasZher2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZher2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, A, lda)


cdef cublasStatus_t _cublasSspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP) except* nogil:
    global __cublasSspr2_v2_64
    _check_or_init_cublas()
    if __cublasSspr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSspr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*) nogil>__cublasSspr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasDspr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP) except* nogil:
    global __cublasDspr2_v2_64
    _check_or_init_cublas()
    if __cublasDspr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDspr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*) nogil>__cublasDspr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasChpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* AP) except* nogil:
    global __cublasChpr2_v2_64
    _check_or_init_cublas()
    if __cublasChpr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChpr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*) nogil>__cublasChpr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasZhpr2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* AP) except* nogil:
    global __cublasZhpr2_v2_64
    _check_or_init_cublas()
    if __cublasZhpr2_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhpr2_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*) nogil>__cublasZhpr2_v2_64)(
        handle, uplo, n, alpha, x, incx, y, incy, AP)


cdef cublasStatus_t _cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float* beta, float* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    global __cublasSgemvBatched_64
    _check_or_init_cublas()
    if __cublasSgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const float*, const float* const*, int64_t, const float* const*, int64_t, const float*, float* const*, int64_t, int64_t) nogil>__cublasSgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double* beta, double* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    global __cublasDgemvBatched_64
    _check_or_init_cublas()
    if __cublasDgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const double*, const double* const*, int64_t, const double* const*, int64_t, const double*, double* const*, int64_t, int64_t) nogil>__cublasDgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex* beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    global __cublasCgemvBatched_64
    _check_or_init_cublas()
    if __cublasCgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, const cuComplex* const*, int64_t, const cuComplex*, cuComplex* const*, int64_t, int64_t) nogil>__cublasCgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount) except* nogil:
    global __cublasZgemvBatched_64
    _check_or_init_cublas()
    if __cublasZgemvBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex*, cuDoubleComplex* const*, int64_t, int64_t) nogil>__cublasZgemvBatched_64)(
        handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)


cdef cublasStatus_t _cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* x, int64_t incx, long long int stridex, const float* beta, float* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    global __cublasSgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasSgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, long long int, const float*, int64_t, long long int, const float*, float*, int64_t, long long int, int64_t) nogil>__cublasSgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* x, int64_t incx, long long int stridex, const double* beta, double* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    global __cublasDgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasDgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, long long int, const double*, int64_t, long long int, const double*, double*, int64_t, long long int, int64_t) nogil>__cublasDgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* x, int64_t incx, long long int stridex, const cuComplex* beta, cuComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    global __cublasCgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasCgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, long long int, const cuComplex*, int64_t, long long int, const cuComplex*, cuComplex*, int64_t, long long int, int64_t) nogil>__cublasCgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* x, int64_t incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy, long long int stridey, int64_t batchCount) except* nogil:
    global __cublasZgemvStridedBatched_64
    _check_or_init_cublas()
    if __cublasZgemvStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemvStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, cuDoubleComplex*, int64_t, long long int, int64_t) nogil>__cublasZgemvStridedBatched_64)(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)


cdef cublasStatus_t _cublasSgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    global __cublasSgemm_v2_64
    _check_or_init_cublas()
    if __cublasSgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    global __cublasDgemm_v2_64
    _check_or_init_cublas()
    if __cublasDgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCgemm_v2_64
    _check_or_init_cublas()
    if __cublasCgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCgemm3m_64
    _check_or_init_cublas()
    if __cublasCgemm3m_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3m_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCgemm3m_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCgemm3mEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasCgemm3mEx_64
    _check_or_init_cublas()
    if __cublasCgemm3mEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) nogil>__cublasCgemm3mEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasZgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZgemm_v2_64
    _check_or_init_cublas()
    if __cublasZgemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZgemm_v2_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZgemm3m_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZgemm3m_64
    _check_or_init_cublas()
    if __cublasZgemm3m_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemm3m_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZgemm3m_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasSgemmEx_64
    _check_or_init_cublas()
    if __cublasSgemmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const float*, void*, cudaDataType, int64_t) nogil>__cublasSgemmEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    global __cublasGemmEx_64
    _check_or_init_cublas()
    if __cublasGemmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const void*, void*, cudaDataType, int64_t, cublasComputeType_t, cublasGemmAlgo_t) nogil>__cublasGemmEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)


cdef cublasStatus_t _cublasCgemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasCgemmEx_64
    _check_or_init_cublas()
    if __cublasCgemmEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) nogil>__cublasCgemmEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasSsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* beta, float* C, int64_t ldc) except* nogil:
    global __cublasSsyrk_v2_64
    _check_or_init_cublas()
    if __cublasSsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* beta, double* C, int64_t ldc) except* nogil:
    global __cublasDsyrk_v2_64
    _check_or_init_cublas()
    if __cublasDsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCsyrk_v2_64
    _check_or_init_cublas()
    if __cublasCsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZsyrk_v2_64
    _check_or_init_cublas()
    if __cublasZsyrk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZsyrk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasCsyrkEx_64
    _check_or_init_cublas()
    if __cublasCsyrkEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) nogil>__cublasCsyrkEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCsyrk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const void* A, cudaDataType Atype, int64_t lda, const cuComplex* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasCsyrk3mEx_64
    _check_or_init_cublas()
    if __cublasCsyrk3mEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrk3mEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const void*, cudaDataType, int64_t, const cuComplex*, void*, cudaDataType, int64_t) nogil>__cublasCsyrk3mEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const cuComplex* A, int64_t lda, const float* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCherk_v2_64
    _check_or_init_cublas()
    if __cublasCherk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const cuComplex*, int64_t, const float*, cuComplex*, int64_t) nogil>__cublasCherk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasZherk_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const cuDoubleComplex* A, int64_t lda, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZherk_v2_64
    _check_or_init_cublas()
    if __cublasZherk_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherk_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const cuDoubleComplex*, int64_t, const double*, cuDoubleComplex*, int64_t) nogil>__cublasZherk_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)


cdef cublasStatus_t _cublasCherkEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasCherkEx_64
    _check_or_init_cublas()
    if __cublasCherkEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const void*, cudaDataType, int64_t, const float*, void*, cudaDataType, int64_t) nogil>__cublasCherkEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasCherk3mEx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype, int64_t lda, const float* beta, void* C, cudaDataType Ctype, int64_t ldc) except* nogil:
    global __cublasCherk3mEx_64
    _check_or_init_cublas()
    if __cublasCherk3mEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherk3mEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const void*, cudaDataType, int64_t, const float*, void*, cudaDataType, int64_t) nogil>__cublasCherk3mEx_64)(
        handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)


cdef cublasStatus_t _cublasSsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    global __cublasSsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasSsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    global __cublasDsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasDsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasCsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyr2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZsyr2k_v2_64
    _check_or_init_cublas()
    if __cublasZsyr2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyr2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZsyr2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCher2k_v2_64
    _check_or_init_cublas()
    if __cublasCher2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCher2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const float*, cuComplex*, int64_t) nogil>__cublasCher2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZher2k_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZher2k_v2_64
    _check_or_init_cublas()
    if __cublasZher2k_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZher2k_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const double*, cuDoubleComplex*, int64_t) nogil>__cublasZher2k_v2_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    global __cublasSsyrkx_64
    _check_or_init_cublas()
    if __cublasSsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    global __cublasDsyrkx_64
    _check_or_init_cublas()
    if __cublasDsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCsyrkx_64
    _check_or_init_cublas()
    if __cublasCsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZsyrkx_64
    _check_or_init_cublas()
    if __cublasZsyrkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsyrkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZsyrkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCherkx_64
    _check_or_init_cublas()
    if __cublasCherkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCherkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const float*, cuComplex*, int64_t) nogil>__cublasCherkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZherkx_64
    _check_or_init_cublas()
    if __cublasZherkx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZherkx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const double*, cuDoubleComplex*, int64_t) nogil>__cublasZherkx_64)(
        handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasSsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc) except* nogil:
    global __cublasSsymm_v2_64
    _check_or_init_cublas()
    if __cublasSsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, const float*, float*, int64_t) nogil>__cublasSsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasDsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc) except* nogil:
    global __cublasDsymm_v2_64
    _check_or_init_cublas()
    if __cublasDsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, const double*, double*, int64_t) nogil>__cublasDsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasCsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCsymm_v2_64
    _check_or_init_cublas()
    if __cublasCsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasCsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZsymm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZsymm_v2_64
    _check_or_init_cublas()
    if __cublasZsymm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZsymm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZsymm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasChemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasChemm_v2_64
    _check_or_init_cublas()
    if __cublasChemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasChemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, const cuComplex*, cuComplex*, int64_t) nogil>__cublasChemm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasZhemm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZhemm_v2_64
    _check_or_init_cublas()
    if __cublasZhemm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZhemm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, cuDoubleComplex*, int64_t) nogil>__cublasZhemm_v2_64)(
        handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)


cdef cublasStatus_t _cublasStrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, float* B, int64_t ldb) except* nogil:
    global __cublasStrsm_v2_64
    _check_or_init_cublas()
    if __cublasStrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, const float*, int64_t, float*, int64_t) nogil>__cublasStrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasDtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, double* B, int64_t ldb) except* nogil:
    global __cublasDtrsm_v2_64
    _check_or_init_cublas()
    if __cublasDtrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, const double*, int64_t, double*, int64_t) nogil>__cublasDtrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasCtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, cuComplex* B, int64_t ldb) except* nogil:
    global __cublasCtrsm_v2_64
    _check_or_init_cublas()
    if __cublasCtrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCtrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasZtrsm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* B, int64_t ldb) except* nogil:
    global __cublasZtrsm_v2_64
    _check_or_init_cublas()
    if __cublasZtrsm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZtrsm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)


cdef cublasStatus_t _cublasStrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc) except* nogil:
    global __cublasStrmm_v2_64
    _check_or_init_cublas()
    if __cublasStrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasStrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* C, int64_t ldc) except* nogil:
    global __cublasDtrmm_v2_64
    _check_or_init_cublas()
    if __cublasDtrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDtrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCtrmm_v2_64
    _check_or_init_cublas()
    if __cublasCtrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCtrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZtrmm_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZtrmm_v2_64
    _check_or_init_cublas()
    if __cublasZtrmm_v2_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrmm_v2_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZtrmm_v2_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)


cdef cublasStatus_t _cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float* beta, float* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    global __cublasSgemmBatched_64
    _check_or_init_cublas()
    if __cublasSgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const float* const*, int64_t, const float* const*, int64_t, const float*, float* const*, int64_t, int64_t) nogil>__cublasSgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double* beta, double* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    global __cublasDgemmBatched_64
    _check_or_init_cublas()
    if __cublasDgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const double*, const double* const*, int64_t, const double* const*, int64_t, const double*, double* const*, int64_t, int64_t) nogil>__cublasDgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    global __cublasCgemmBatched_64
    _check_or_init_cublas()
    if __cublasCgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, const cuComplex* const*, int64_t, const cuComplex*, cuComplex* const*, int64_t, int64_t) nogil>__cublasCgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasCgemm3mBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    global __cublasCgemm3mBatched_64
    _check_or_init_cublas()
    if __cublasCgemm3mBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, const cuComplex* const*, int64_t, const cuComplex*, cuComplex* const*, int64_t, int64_t) nogil>__cublasCgemm3mBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount) except* nogil:
    global __cublasZgemmBatched_64
    _check_or_init_cublas()
    if __cublasZgemmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex* const*, int64_t, const cuDoubleComplex*, cuDoubleComplex* const*, int64_t, int64_t) nogil>__cublasZgemmBatched_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)


cdef cublasStatus_t _cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    global __cublasSgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasSgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const float*, const float*, int64_t, long long int, const float*, int64_t, long long int, const float*, float*, int64_t, long long int, int64_t) nogil>__cublasSgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* B, int64_t ldb, long long int strideB, const double* beta, double* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    global __cublasDgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasDgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const double*, const double*, int64_t, long long int, const double*, int64_t, long long int, const double*, double*, int64_t, long long int, int64_t) nogil>__cublasDgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    global __cublasCgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasCgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, long long int, const cuComplex*, int64_t, long long int, const cuComplex*, cuComplex*, int64_t, long long int, int64_t) nogil>__cublasCgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasCgemm3mStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    global __cublasCgemm3mStridedBatched_64
    _check_or_init_cublas()
    if __cublasCgemm3mStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgemm3mStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, long long int, const cuComplex*, int64_t, long long int, const cuComplex*, cuComplex*, int64_t, long long int, int64_t) nogil>__cublasCgemm3mStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* B, int64_t ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc, long long int strideC, int64_t batchCount) except* nogil:
    global __cublasZgemmStridedBatched_64
    _check_or_init_cublas()
    if __cublasZgemmStridedBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgemmStridedBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, int64_t, long long int, const cuDoubleComplex*, cuDoubleComplex*, int64_t, long long int, int64_t) nogil>__cublasZgemmStridedBatched_64)(
        handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)


cdef cublasStatus_t _cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    global __cublasGemmBatchedEx_64
    _check_or_init_cublas()
    if __cublasGemmBatchedEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmBatchedEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void* const*, cudaDataType, int64_t, const void* const*, cudaDataType, int64_t, const void*, void* const*, cudaDataType, int64_t, int64_t, cublasComputeType_t, cublasGemmAlgo_t) nogil>__cublasGemmBatchedEx_64)(
        handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)


cdef cublasStatus_t _cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, long long int strideA, const void* B, cudaDataType Btype, int64_t ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, long long int strideC, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) except* nogil:
    global __cublasGemmStridedBatchedEx_64
    _check_or_init_cublas()
    if __cublasGemmStridedBatchedEx_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasGemmStridedBatchedEx_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const void*, const void*, cudaDataType, int64_t, long long int, const void*, cudaDataType, int64_t, long long int, const void*, void*, cudaDataType, int64_t, long long int, int64_t, cublasComputeType_t, cublasGemmAlgo_t) nogil>__cublasGemmStridedBatchedEx_64)(
        handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)


cdef cublasStatus_t _cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* beta, const float* B, int64_t ldb, float* C, int64_t ldc) except* nogil:
    global __cublasSgeam_64
    _check_or_init_cublas()
    if __cublasSgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const float*, const float*, int64_t, const float*, const float*, int64_t, float*, int64_t) nogil>__cublasSgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* beta, const double* B, int64_t ldb, double* C, int64_t ldc) except* nogil:
    global __cublasDgeam_64
    _check_or_init_cublas()
    if __cublasDgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const double*, const double*, int64_t, const double*, const double*, int64_t, double*, int64_t) nogil>__cublasDgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCgeam_64
    _check_or_init_cublas()
    if __cublasCgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const cuComplex*, const cuComplex*, int64_t, const cuComplex*, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZgeam_64
    _check_or_init_cublas()
    if __cublasZgeam_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZgeam_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZgeam_64)(
        handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)


cdef cublasStatus_t _cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    global __cublasStrsmBatched_64
    _check_or_init_cublas()
    if __cublasStrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasStrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const float*, const float* const*, int64_t, float* const*, int64_t, int64_t) nogil>__cublasStrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    global __cublasDtrsmBatched_64
    _check_or_init_cublas()
    if __cublasDtrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDtrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const double*, const double* const*, int64_t, double* const*, int64_t, int64_t) nogil>__cublasDtrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    global __cublasCtrsmBatched_64
    _check_or_init_cublas()
    if __cublasCtrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCtrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuComplex*, const cuComplex* const*, int64_t, cuComplex* const*, int64_t, int64_t) nogil>__cublasCtrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount) except* nogil:
    global __cublasZtrsmBatched_64
    _check_or_init_cublas()
    if __cublasZtrsmBatched_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZtrsmBatched_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int64_t, int64_t, const cuDoubleComplex*, const cuDoubleComplex* const*, int64_t, cuDoubleComplex* const*, int64_t, int64_t) nogil>__cublasZtrsmBatched_64)(
        handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)


cdef cublasStatus_t _cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float* A, int64_t lda, const float* x, int64_t incx, float* C, int64_t ldc) except* nogil:
    global __cublasSdgmm_64
    _check_or_init_cublas()
    if __cublasSdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasSdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, float*, int64_t) nogil>__cublasSdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double* A, int64_t lda, const double* x, int64_t incx, double* C, int64_t ldc) except* nogil:
    global __cublasDdgmm_64
    _check_or_init_cublas()
    if __cublasDdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasDdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const double*, int64_t, const double*, int64_t, double*, int64_t) nogil>__cublasDdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, cuComplex* C, int64_t ldc) except* nogil:
    global __cublasCdgmm_64
    _check_or_init_cublas()
    if __cublasCdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasCdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const cuComplex*, int64_t, const cuComplex*, int64_t, cuComplex*, int64_t) nogil>__cublasCdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)


cdef cublasStatus_t _cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* C, int64_t ldc) except* nogil:
    global __cublasZdgmm_64
    _check_or_init_cublas()
    if __cublasZdgmm_64 == NULL:
        with gil:
            raise FunctionNotFoundError("function cublasZdgmm_64 is not found")
    return (<cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int64_t, int64_t, const cuDoubleComplex*, int64_t, const cuDoubleComplex*, int64_t, cuDoubleComplex*, int64_t) nogil>__cublasZdgmm_64)(
        handle, mode, m, n, A, lda, x, incx, C, ldc)
