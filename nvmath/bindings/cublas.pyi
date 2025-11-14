# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import _cython_3_1_4
import enum
from typing import Any, Callable, ClassVar

__pyx_capi__: dict
__test__: dict
asum_ex: _cython_3_1_4.cython_function_or_method
asum_ex_64: _cython_3_1_4.cython_function_or_method
axpy_ex: _cython_3_1_4.cython_function_or_method
axpy_ex_64: _cython_3_1_4.cython_function_or_method
caxpy: _cython_3_1_4.cython_function_or_method
caxpy_64: _cython_3_1_4.cython_function_or_method
ccopy: _cython_3_1_4.cython_function_or_method
ccopy_64: _cython_3_1_4.cython_function_or_method
cdgmm: _cython_3_1_4.cython_function_or_method
cdgmm_64: _cython_3_1_4.cython_function_or_method
cdgmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
cdotc: _cython_3_1_4.cython_function_or_method
cdotc_64: _cython_3_1_4.cython_function_or_method
cdotu: _cython_3_1_4.cython_function_or_method
cdotu_64: _cython_3_1_4.cython_function_or_method
cgbmv: _cython_3_1_4.cython_function_or_method
cgbmv_64: _cython_3_1_4.cython_function_or_method
cgeam: _cython_3_1_4.cython_function_or_method
cgeam_64: _cython_3_1_4.cython_function_or_method
cgels_batched: _cython_3_1_4.cython_function_or_method
cgemm: _cython_3_1_4.cython_function_or_method
cgemm3m: _cython_3_1_4.cython_function_or_method
cgemm3m_64: _cython_3_1_4.cython_function_or_method
cgemm3m_batched: _cython_3_1_4.cython_function_or_method
cgemm3m_batched_64: _cython_3_1_4.cython_function_or_method
cgemm3m_ex: _cython_3_1_4.cython_function_or_method
cgemm3m_ex_64: _cython_3_1_4.cython_function_or_method
cgemm3m_strided_batched: _cython_3_1_4.cython_function_or_method
cgemm3m_strided_batched_64: _cython_3_1_4.cython_function_or_method
cgemm_64: _cython_3_1_4.cython_function_or_method
cgemm_batched: _cython_3_1_4.cython_function_or_method
cgemm_batched_64: _cython_3_1_4.cython_function_or_method
cgemm_ex: _cython_3_1_4.cython_function_or_method
cgemm_ex_64: _cython_3_1_4.cython_function_or_method
cgemm_strided_batched: _cython_3_1_4.cython_function_or_method
cgemm_strided_batched_64: _cython_3_1_4.cython_function_or_method
cgemv: _cython_3_1_4.cython_function_or_method
cgemv_64: _cython_3_1_4.cython_function_or_method
cgemv_batched: _cython_3_1_4.cython_function_or_method
cgemv_batched_64: _cython_3_1_4.cython_function_or_method
cgemv_strided_batched: _cython_3_1_4.cython_function_or_method
cgemv_strided_batched_64: _cython_3_1_4.cython_function_or_method
cgeqrf_batched: _cython_3_1_4.cython_function_or_method
cgerc: _cython_3_1_4.cython_function_or_method
cgerc_64: _cython_3_1_4.cython_function_or_method
cgeru: _cython_3_1_4.cython_function_or_method
cgeru_64: _cython_3_1_4.cython_function_or_method
cgetrf_batched: _cython_3_1_4.cython_function_or_method
cgetri_batched: _cython_3_1_4.cython_function_or_method
cgetrs_batched: _cython_3_1_4.cython_function_or_method
chbmv: _cython_3_1_4.cython_function_or_method
chbmv_64: _cython_3_1_4.cython_function_or_method
check_status: _cython_3_1_4.cython_function_or_method
chemm: _cython_3_1_4.cython_function_or_method
chemm_64: _cython_3_1_4.cython_function_or_method
chemm_strided_batched_64: _cython_3_1_4.cython_function_or_method
chemv: _cython_3_1_4.cython_function_or_method
chemv_64: _cython_3_1_4.cython_function_or_method
cher: _cython_3_1_4.cython_function_or_method
cher2: _cython_3_1_4.cython_function_or_method
cher2_64: _cython_3_1_4.cython_function_or_method
cher2k: _cython_3_1_4.cython_function_or_method
cher2k_64: _cython_3_1_4.cython_function_or_method
cher_64: _cython_3_1_4.cython_function_or_method
cherk: _cython_3_1_4.cython_function_or_method
cherk3m_ex: _cython_3_1_4.cython_function_or_method
cherk3m_ex_64: _cython_3_1_4.cython_function_or_method
cherk_64: _cython_3_1_4.cython_function_or_method
cherk_ex: _cython_3_1_4.cython_function_or_method
cherk_ex_64: _cython_3_1_4.cython_function_or_method
cherkx: _cython_3_1_4.cython_function_or_method
cherkx_64: _cython_3_1_4.cython_function_or_method
chpmv: _cython_3_1_4.cython_function_or_method
chpmv_64: _cython_3_1_4.cython_function_or_method
chpr: _cython_3_1_4.cython_function_or_method
chpr2: _cython_3_1_4.cython_function_or_method
chpr2_64: _cython_3_1_4.cython_function_or_method
chpr_64: _cython_3_1_4.cython_function_or_method
cmatinv_batched: _cython_3_1_4.cython_function_or_method
copy_ex: _cython_3_1_4.cython_function_or_method
copy_ex_64: _cython_3_1_4.cython_function_or_method
create: _cython_3_1_4.cython_function_or_method
crot: _cython_3_1_4.cython_function_or_method
crot_64: _cython_3_1_4.cython_function_or_method
crotg: _cython_3_1_4.cython_function_or_method
cscal: _cython_3_1_4.cython_function_or_method
cscal_64: _cython_3_1_4.cython_function_or_method
csrot: _cython_3_1_4.cython_function_or_method
csrot_64: _cython_3_1_4.cython_function_or_method
csscal: _cython_3_1_4.cython_function_or_method
csscal_64: _cython_3_1_4.cython_function_or_method
cswap: _cython_3_1_4.cython_function_or_method
cswap_64: _cython_3_1_4.cython_function_or_method
csymm: _cython_3_1_4.cython_function_or_method
csymm_64: _cython_3_1_4.cython_function_or_method
csymm_strided_batched_64: _cython_3_1_4.cython_function_or_method
csymv: _cython_3_1_4.cython_function_or_method
csymv_64: _cython_3_1_4.cython_function_or_method
csyr: _cython_3_1_4.cython_function_or_method
csyr2: _cython_3_1_4.cython_function_or_method
csyr2_64: _cython_3_1_4.cython_function_or_method
csyr2k: _cython_3_1_4.cython_function_or_method
csyr2k_64: _cython_3_1_4.cython_function_or_method
csyr_64: _cython_3_1_4.cython_function_or_method
csyrk: _cython_3_1_4.cython_function_or_method
csyrk3m_ex: _cython_3_1_4.cython_function_or_method
csyrk3m_ex_64: _cython_3_1_4.cython_function_or_method
csyrk_64: _cython_3_1_4.cython_function_or_method
csyrk_ex: _cython_3_1_4.cython_function_or_method
csyrk_ex_64: _cython_3_1_4.cython_function_or_method
csyrkx: _cython_3_1_4.cython_function_or_method
csyrkx_64: _cython_3_1_4.cython_function_or_method
ctbmv: _cython_3_1_4.cython_function_or_method
ctbmv_64: _cython_3_1_4.cython_function_or_method
ctbsv: _cython_3_1_4.cython_function_or_method
ctbsv_64: _cython_3_1_4.cython_function_or_method
ctpmv: _cython_3_1_4.cython_function_or_method
ctpmv_64: _cython_3_1_4.cython_function_or_method
ctpsv: _cython_3_1_4.cython_function_or_method
ctpsv_64: _cython_3_1_4.cython_function_or_method
ctpttr: _cython_3_1_4.cython_function_or_method
ctrmm: _cython_3_1_4.cython_function_or_method
ctrmm_64: _cython_3_1_4.cython_function_or_method
ctrmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
ctrmv: _cython_3_1_4.cython_function_or_method
ctrmv_64: _cython_3_1_4.cython_function_or_method
ctrsm: _cython_3_1_4.cython_function_or_method
ctrsm_64: _cython_3_1_4.cython_function_or_method
ctrsm_batched: _cython_3_1_4.cython_function_or_method
ctrsm_batched_64: _cython_3_1_4.cython_function_or_method
ctrsv: _cython_3_1_4.cython_function_or_method
ctrsv_64: _cython_3_1_4.cython_function_or_method
ctrttp: _cython_3_1_4.cython_function_or_method
dasum: _cython_3_1_4.cython_function_or_method
dasum_64: _cython_3_1_4.cython_function_or_method
daxpy: _cython_3_1_4.cython_function_or_method
daxpy_64: _cython_3_1_4.cython_function_or_method
dcopy: _cython_3_1_4.cython_function_or_method
dcopy_64: _cython_3_1_4.cython_function_or_method
ddgmm: _cython_3_1_4.cython_function_or_method
ddgmm_64: _cython_3_1_4.cython_function_or_method
ddgmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
ddot: _cython_3_1_4.cython_function_or_method
ddot_64: _cython_3_1_4.cython_function_or_method
destroy: _cython_3_1_4.cython_function_or_method
dgbmv: _cython_3_1_4.cython_function_or_method
dgbmv_64: _cython_3_1_4.cython_function_or_method
dgeam: _cython_3_1_4.cython_function_or_method
dgeam_64: _cython_3_1_4.cython_function_or_method
dgels_batched: _cython_3_1_4.cython_function_or_method
dgemm: _cython_3_1_4.cython_function_or_method
dgemm_64: _cython_3_1_4.cython_function_or_method
dgemm_batched: _cython_3_1_4.cython_function_or_method
dgemm_batched_64: _cython_3_1_4.cython_function_or_method
dgemm_grouped_batched: _cython_3_1_4.cython_function_or_method
dgemm_grouped_batched_64: _cython_3_1_4.cython_function_or_method
dgemm_strided_batched: _cython_3_1_4.cython_function_or_method
dgemm_strided_batched_64: _cython_3_1_4.cython_function_or_method
dgemv: _cython_3_1_4.cython_function_or_method
dgemv_64: _cython_3_1_4.cython_function_or_method
dgemv_batched: _cython_3_1_4.cython_function_or_method
dgemv_batched_64: _cython_3_1_4.cython_function_or_method
dgemv_strided_batched: _cython_3_1_4.cython_function_or_method
dgemv_strided_batched_64: _cython_3_1_4.cython_function_or_method
dgeqrf_batched: _cython_3_1_4.cython_function_or_method
dger: _cython_3_1_4.cython_function_or_method
dger_64: _cython_3_1_4.cython_function_or_method
dgetrf_batched: _cython_3_1_4.cython_function_or_method
dgetri_batched: _cython_3_1_4.cython_function_or_method
dgetrs_batched: _cython_3_1_4.cython_function_or_method
dmatinv_batched: _cython_3_1_4.cython_function_or_method
dnrm2: _cython_3_1_4.cython_function_or_method
dnrm2_64: _cython_3_1_4.cython_function_or_method
dot_ex: _cython_3_1_4.cython_function_or_method
dot_ex_64: _cython_3_1_4.cython_function_or_method
dotc_ex: _cython_3_1_4.cython_function_or_method
dotc_ex_64: _cython_3_1_4.cython_function_or_method
drot: _cython_3_1_4.cython_function_or_method
drot_64: _cython_3_1_4.cython_function_or_method
drotg: _cython_3_1_4.cython_function_or_method
drotm: _cython_3_1_4.cython_function_or_method
drotm_64: _cython_3_1_4.cython_function_or_method
drotmg: _cython_3_1_4.cython_function_or_method
dsbmv: _cython_3_1_4.cython_function_or_method
dsbmv_64: _cython_3_1_4.cython_function_or_method
dscal: _cython_3_1_4.cython_function_or_method
dscal_64: _cython_3_1_4.cython_function_or_method
dspmv: _cython_3_1_4.cython_function_or_method
dspmv_64: _cython_3_1_4.cython_function_or_method
dspr: _cython_3_1_4.cython_function_or_method
dspr2: _cython_3_1_4.cython_function_or_method
dspr2_64: _cython_3_1_4.cython_function_or_method
dspr_64: _cython_3_1_4.cython_function_or_method
dswap: _cython_3_1_4.cython_function_or_method
dswap_64: _cython_3_1_4.cython_function_or_method
dsymm: _cython_3_1_4.cython_function_or_method
dsymm_64: _cython_3_1_4.cython_function_or_method
dsymm_strided_batched_64: _cython_3_1_4.cython_function_or_method
dsymv: _cython_3_1_4.cython_function_or_method
dsymv_64: _cython_3_1_4.cython_function_or_method
dsyr: _cython_3_1_4.cython_function_or_method
dsyr2: _cython_3_1_4.cython_function_or_method
dsyr2_64: _cython_3_1_4.cython_function_or_method
dsyr2k: _cython_3_1_4.cython_function_or_method
dsyr2k_64: _cython_3_1_4.cython_function_or_method
dsyr_64: _cython_3_1_4.cython_function_or_method
dsyrk: _cython_3_1_4.cython_function_or_method
dsyrk_64: _cython_3_1_4.cython_function_or_method
dsyrkx: _cython_3_1_4.cython_function_or_method
dsyrkx_64: _cython_3_1_4.cython_function_or_method
dtbmv: _cython_3_1_4.cython_function_or_method
dtbmv_64: _cython_3_1_4.cython_function_or_method
dtbsv: _cython_3_1_4.cython_function_or_method
dtbsv_64: _cython_3_1_4.cython_function_or_method
dtpmv: _cython_3_1_4.cython_function_or_method
dtpmv_64: _cython_3_1_4.cython_function_or_method
dtpsv: _cython_3_1_4.cython_function_or_method
dtpsv_64: _cython_3_1_4.cython_function_or_method
dtpttr: _cython_3_1_4.cython_function_or_method
dtrmm: _cython_3_1_4.cython_function_or_method
dtrmm_64: _cython_3_1_4.cython_function_or_method
dtrmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
dtrmv: _cython_3_1_4.cython_function_or_method
dtrmv_64: _cython_3_1_4.cython_function_or_method
dtrsm: _cython_3_1_4.cython_function_or_method
dtrsm_64: _cython_3_1_4.cython_function_or_method
dtrsm_batched: _cython_3_1_4.cython_function_or_method
dtrsm_batched_64: _cython_3_1_4.cython_function_or_method
dtrsv: _cython_3_1_4.cython_function_or_method
dtrsv_64: _cython_3_1_4.cython_function_or_method
dtrttp: _cython_3_1_4.cython_function_or_method
dzasum: _cython_3_1_4.cython_function_or_method
dzasum_64: _cython_3_1_4.cython_function_or_method
dznrm2: _cython_3_1_4.cython_function_or_method
dznrm2_64: _cython_3_1_4.cython_function_or_method
gemm_batched_ex: _cython_3_1_4.cython_function_or_method
gemm_batched_ex_64: _cython_3_1_4.cython_function_or_method
gemm_ex: _cython_3_1_4.cython_function_or_method
gemm_ex_64: _cython_3_1_4.cython_function_or_method
gemm_grouped_batched_ex: _cython_3_1_4.cython_function_or_method
gemm_grouped_batched_ex_64: _cython_3_1_4.cython_function_or_method
gemm_strided_batched_ex: _cython_3_1_4.cython_function_or_method
gemm_strided_batched_ex_64: _cython_3_1_4.cython_function_or_method
get_atomics_mode: _cython_3_1_4.cython_function_or_method
get_cudart_version: _cython_3_1_4.cython_function_or_method
get_emulation_strategy: _cython_3_1_4.cython_function_or_method
get_math_mode: _cython_3_1_4.cython_function_or_method
get_matrix: _cython_3_1_4.cython_function_or_method
get_matrix_64: _cython_3_1_4.cython_function_or_method
get_matrix_async: _cython_3_1_4.cython_function_or_method
get_matrix_async_64: _cython_3_1_4.cython_function_or_method
get_pointer_mode: _cython_3_1_4.cython_function_or_method
get_property: _cython_3_1_4.cython_function_or_method
get_sm_count_target: _cython_3_1_4.cython_function_or_method
get_status_name: _cython_3_1_4.cython_function_or_method
get_status_string: _cython_3_1_4.cython_function_or_method
get_stream: _cython_3_1_4.cython_function_or_method
get_vector: _cython_3_1_4.cython_function_or_method
get_vector_64: _cython_3_1_4.cython_function_or_method
get_vector_async: _cython_3_1_4.cython_function_or_method
get_vector_async_64: _cython_3_1_4.cython_function_or_method
get_version: _cython_3_1_4.cython_function_or_method
iamax_ex: _cython_3_1_4.cython_function_or_method
iamax_ex_64: _cython_3_1_4.cython_function_or_method
iamin_ex: _cython_3_1_4.cython_function_or_method
iamin_ex_64: _cython_3_1_4.cython_function_or_method
icamax: _cython_3_1_4.cython_function_or_method
icamax_64: _cython_3_1_4.cython_function_or_method
icamin: _cython_3_1_4.cython_function_or_method
icamin_64: _cython_3_1_4.cython_function_or_method
idamax: _cython_3_1_4.cython_function_or_method
idamax_64: _cython_3_1_4.cython_function_or_method
idamin: _cython_3_1_4.cython_function_or_method
idamin_64: _cython_3_1_4.cython_function_or_method
isamax: _cython_3_1_4.cython_function_or_method
isamax_64: _cython_3_1_4.cython_function_or_method
isamin: _cython_3_1_4.cython_function_or_method
isamin_64: _cython_3_1_4.cython_function_or_method
izamax: _cython_3_1_4.cython_function_or_method
izamax_64: _cython_3_1_4.cython_function_or_method
izamin: _cython_3_1_4.cython_function_or_method
izamin_64: _cython_3_1_4.cython_function_or_method
logger_configure: _cython_3_1_4.cython_function_or_method
nrm2_ex: _cython_3_1_4.cython_function_or_method
nrm2ex_64: _cython_3_1_4.cython_function_or_method
rot_ex: _cython_3_1_4.cython_function_or_method
rot_ex_64: _cython_3_1_4.cython_function_or_method
rotg_ex: _cython_3_1_4.cython_function_or_method
rotm_ex: _cython_3_1_4.cython_function_or_method
rotm_ex_64: _cython_3_1_4.cython_function_or_method
rotmg_ex: _cython_3_1_4.cython_function_or_method
sasum: _cython_3_1_4.cython_function_or_method
sasum_64: _cython_3_1_4.cython_function_or_method
saxpy: _cython_3_1_4.cython_function_or_method
saxpy_64: _cython_3_1_4.cython_function_or_method
scal_ex: _cython_3_1_4.cython_function_or_method
scal_ex_64: _cython_3_1_4.cython_function_or_method
scasum: _cython_3_1_4.cython_function_or_method
scasum_64: _cython_3_1_4.cython_function_or_method
scnrm2: _cython_3_1_4.cython_function_or_method
scnrm2_64: _cython_3_1_4.cython_function_or_method
scopy: _cython_3_1_4.cython_function_or_method
scopy_64: _cython_3_1_4.cython_function_or_method
sdgmm: _cython_3_1_4.cython_function_or_method
sdgmm_64: _cython_3_1_4.cython_function_or_method
sdgmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
sdot: _cython_3_1_4.cython_function_or_method
sdot_64: _cython_3_1_4.cython_function_or_method
set_atomics_mode: _cython_3_1_4.cython_function_or_method
set_emulation_strategy: _cython_3_1_4.cython_function_or_method
set_math_mode: _cython_3_1_4.cython_function_or_method
set_matrix: _cython_3_1_4.cython_function_or_method
set_matrix_64: _cython_3_1_4.cython_function_or_method
set_matrix_async: _cython_3_1_4.cython_function_or_method
set_matrix_async_64: _cython_3_1_4.cython_function_or_method
set_pointer_mode: _cython_3_1_4.cython_function_or_method
set_sm_count_target: _cython_3_1_4.cython_function_or_method
set_stream: _cython_3_1_4.cython_function_or_method
set_vector: _cython_3_1_4.cython_function_or_method
set_vector_64: _cython_3_1_4.cython_function_or_method
set_vector_async: _cython_3_1_4.cython_function_or_method
set_vector_async_64: _cython_3_1_4.cython_function_or_method
set_workspace: _cython_3_1_4.cython_function_or_method
sgbmv: _cython_3_1_4.cython_function_or_method
sgbmv_64: _cython_3_1_4.cython_function_or_method
sgeam: _cython_3_1_4.cython_function_or_method
sgeam_64: _cython_3_1_4.cython_function_or_method
sgels_batched: _cython_3_1_4.cython_function_or_method
sgemm: _cython_3_1_4.cython_function_or_method
sgemm_64: _cython_3_1_4.cython_function_or_method
sgemm_batched: _cython_3_1_4.cython_function_or_method
sgemm_batched_64: _cython_3_1_4.cython_function_or_method
sgemm_ex: _cython_3_1_4.cython_function_or_method
sgemm_ex_64: _cython_3_1_4.cython_function_or_method
sgemm_grouped_batched: _cython_3_1_4.cython_function_or_method
sgemm_grouped_batched_64: _cython_3_1_4.cython_function_or_method
sgemm_strided_batched: _cython_3_1_4.cython_function_or_method
sgemm_strided_batched_64: _cython_3_1_4.cython_function_or_method
sgemv: _cython_3_1_4.cython_function_or_method
sgemv_64: _cython_3_1_4.cython_function_or_method
sgemv_batched: _cython_3_1_4.cython_function_or_method
sgemv_batched_64: _cython_3_1_4.cython_function_or_method
sgemv_strided_batched: _cython_3_1_4.cython_function_or_method
sgemv_strided_batched_64: _cython_3_1_4.cython_function_or_method
sgeqrf_batched: _cython_3_1_4.cython_function_or_method
sger: _cython_3_1_4.cython_function_or_method
sger_64: _cython_3_1_4.cython_function_or_method
sgetrf_batched: _cython_3_1_4.cython_function_or_method
sgetri_batched: _cython_3_1_4.cython_function_or_method
sgetrs_batched: _cython_3_1_4.cython_function_or_method
smatinv_batched: _cython_3_1_4.cython_function_or_method
snrm2: _cython_3_1_4.cython_function_or_method
snrm2_64: _cython_3_1_4.cython_function_or_method
srot: _cython_3_1_4.cython_function_or_method
srot_64: _cython_3_1_4.cython_function_or_method
srotg: _cython_3_1_4.cython_function_or_method
srotm: _cython_3_1_4.cython_function_or_method
srotm_64: _cython_3_1_4.cython_function_or_method
srotmg: _cython_3_1_4.cython_function_or_method
ssbmv: _cython_3_1_4.cython_function_or_method
ssbmv_64: _cython_3_1_4.cython_function_or_method
sscal: _cython_3_1_4.cython_function_or_method
sscal_64: _cython_3_1_4.cython_function_or_method
sspmv: _cython_3_1_4.cython_function_or_method
sspmv_64: _cython_3_1_4.cython_function_or_method
sspr: _cython_3_1_4.cython_function_or_method
sspr2: _cython_3_1_4.cython_function_or_method
sspr2_64: _cython_3_1_4.cython_function_or_method
sspr_64: _cython_3_1_4.cython_function_or_method
sswap: _cython_3_1_4.cython_function_or_method
sswap_64: _cython_3_1_4.cython_function_or_method
ssymm: _cython_3_1_4.cython_function_or_method
ssymm_64: _cython_3_1_4.cython_function_or_method
ssymm_strided_batched_64: _cython_3_1_4.cython_function_or_method
ssymv: _cython_3_1_4.cython_function_or_method
ssymv_64: _cython_3_1_4.cython_function_or_method
ssyr: _cython_3_1_4.cython_function_or_method
ssyr2: _cython_3_1_4.cython_function_or_method
ssyr2_64: _cython_3_1_4.cython_function_or_method
ssyr2k: _cython_3_1_4.cython_function_or_method
ssyr2k_64: _cython_3_1_4.cython_function_or_method
ssyr_64: _cython_3_1_4.cython_function_or_method
ssyrk: _cython_3_1_4.cython_function_or_method
ssyrk_64: _cython_3_1_4.cython_function_or_method
ssyrkx: _cython_3_1_4.cython_function_or_method
ssyrkx_64: _cython_3_1_4.cython_function_or_method
stbmv: _cython_3_1_4.cython_function_or_method
stbmv_64: _cython_3_1_4.cython_function_or_method
stbsv: _cython_3_1_4.cython_function_or_method
stbsv_64: _cython_3_1_4.cython_function_or_method
stpmv: _cython_3_1_4.cython_function_or_method
stpmv_64: _cython_3_1_4.cython_function_or_method
stpsv: _cython_3_1_4.cython_function_or_method
stpsv_64: _cython_3_1_4.cython_function_or_method
stpttr: _cython_3_1_4.cython_function_or_method
strmm: _cython_3_1_4.cython_function_or_method
strmm_64: _cython_3_1_4.cython_function_or_method
strmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
strmv: _cython_3_1_4.cython_function_or_method
strmv_64: _cython_3_1_4.cython_function_or_method
strsm: _cython_3_1_4.cython_function_or_method
strsm_64: _cython_3_1_4.cython_function_or_method
strsm_batched: _cython_3_1_4.cython_function_or_method
strsm_batched_64: _cython_3_1_4.cython_function_or_method
strsv: _cython_3_1_4.cython_function_or_method
strsv_64: _cython_3_1_4.cython_function_or_method
strttp: _cython_3_1_4.cython_function_or_method
swap_ex: _cython_3_1_4.cython_function_or_method
swap_ex_64: _cython_3_1_4.cython_function_or_method
uint8gemm_bias: _cython_3_1_4.cython_function_or_method
zaxpy: _cython_3_1_4.cython_function_or_method
zaxpy_64: _cython_3_1_4.cython_function_or_method
zcopy: _cython_3_1_4.cython_function_or_method
zcopy_64: _cython_3_1_4.cython_function_or_method
zdgmm: _cython_3_1_4.cython_function_or_method
zdgmm_64: _cython_3_1_4.cython_function_or_method
zdgmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
zdotc: _cython_3_1_4.cython_function_or_method
zdotc_64: _cython_3_1_4.cython_function_or_method
zdotu: _cython_3_1_4.cython_function_or_method
zdotu_64: _cython_3_1_4.cython_function_or_method
zdrot: _cython_3_1_4.cython_function_or_method
zdrot_64: _cython_3_1_4.cython_function_or_method
zdscal: _cython_3_1_4.cython_function_or_method
zdscal_64: _cython_3_1_4.cython_function_or_method
zgbmv: _cython_3_1_4.cython_function_or_method
zgbmv_64: _cython_3_1_4.cython_function_or_method
zgeam: _cython_3_1_4.cython_function_or_method
zgeam_64: _cython_3_1_4.cython_function_or_method
zgels_batched: _cython_3_1_4.cython_function_or_method
zgemm: _cython_3_1_4.cython_function_or_method
zgemm3m: _cython_3_1_4.cython_function_or_method
zgemm3m_64: _cython_3_1_4.cython_function_or_method
zgemm_64: _cython_3_1_4.cython_function_or_method
zgemm_batched: _cython_3_1_4.cython_function_or_method
zgemm_batched_64: _cython_3_1_4.cython_function_or_method
zgemm_strided_batched: _cython_3_1_4.cython_function_or_method
zgemm_strided_batched_64: _cython_3_1_4.cython_function_or_method
zgemv: _cython_3_1_4.cython_function_or_method
zgemv_64: _cython_3_1_4.cython_function_or_method
zgemv_batched: _cython_3_1_4.cython_function_or_method
zgemv_batched_64: _cython_3_1_4.cython_function_or_method
zgemv_strided_batched: _cython_3_1_4.cython_function_or_method
zgemv_strided_batched_64: _cython_3_1_4.cython_function_or_method
zgeqrf_batched: _cython_3_1_4.cython_function_or_method
zgerc: _cython_3_1_4.cython_function_or_method
zgerc_64: _cython_3_1_4.cython_function_or_method
zgeru: _cython_3_1_4.cython_function_or_method
zgeru_64: _cython_3_1_4.cython_function_or_method
zgetrf_batched: _cython_3_1_4.cython_function_or_method
zgetri_batched: _cython_3_1_4.cython_function_or_method
zgetrs_batched: _cython_3_1_4.cython_function_or_method
zhbmv: _cython_3_1_4.cython_function_or_method
zhbmv_64: _cython_3_1_4.cython_function_or_method
zhemm: _cython_3_1_4.cython_function_or_method
zhemm_64: _cython_3_1_4.cython_function_or_method
zhemm_strided_batched_64: _cython_3_1_4.cython_function_or_method
zhemv: _cython_3_1_4.cython_function_or_method
zhemv_64: _cython_3_1_4.cython_function_or_method
zher: _cython_3_1_4.cython_function_or_method
zher2: _cython_3_1_4.cython_function_or_method
zher2_64: _cython_3_1_4.cython_function_or_method
zher2k: _cython_3_1_4.cython_function_or_method
zher2k_64: _cython_3_1_4.cython_function_or_method
zher_64: _cython_3_1_4.cython_function_or_method
zherk: _cython_3_1_4.cython_function_or_method
zherk_64: _cython_3_1_4.cython_function_or_method
zherkx: _cython_3_1_4.cython_function_or_method
zherkx_64: _cython_3_1_4.cython_function_or_method
zhpmv: _cython_3_1_4.cython_function_or_method
zhpmv_64: _cython_3_1_4.cython_function_or_method
zhpr: _cython_3_1_4.cython_function_or_method
zhpr2: _cython_3_1_4.cython_function_or_method
zhpr2_64: _cython_3_1_4.cython_function_or_method
zhpr_64: _cython_3_1_4.cython_function_or_method
zmatinv_batched: _cython_3_1_4.cython_function_or_method
zrot: _cython_3_1_4.cython_function_or_method
zrot_64: _cython_3_1_4.cython_function_or_method
zrotg: _cython_3_1_4.cython_function_or_method
zscal: _cython_3_1_4.cython_function_or_method
zscal_64: _cython_3_1_4.cython_function_or_method
zswap: _cython_3_1_4.cython_function_or_method
zswap_64: _cython_3_1_4.cython_function_or_method
zsymm: _cython_3_1_4.cython_function_or_method
zsymm_64: _cython_3_1_4.cython_function_or_method
zsymm_strided_batched_64: _cython_3_1_4.cython_function_or_method
zsymv: _cython_3_1_4.cython_function_or_method
zsymv_64: _cython_3_1_4.cython_function_or_method
zsyr: _cython_3_1_4.cython_function_or_method
zsyr2: _cython_3_1_4.cython_function_or_method
zsyr2_64: _cython_3_1_4.cython_function_or_method
zsyr2k: _cython_3_1_4.cython_function_or_method
zsyr2k_64: _cython_3_1_4.cython_function_or_method
zsyr_64: _cython_3_1_4.cython_function_or_method
zsyrk: _cython_3_1_4.cython_function_or_method
zsyrk_64: _cython_3_1_4.cython_function_or_method
zsyrkx: _cython_3_1_4.cython_function_or_method
zsyrkx_64: _cython_3_1_4.cython_function_or_method
ztbmv: _cython_3_1_4.cython_function_or_method
ztbmv_64: _cython_3_1_4.cython_function_or_method
ztbsv: _cython_3_1_4.cython_function_or_method
ztbsv_64: _cython_3_1_4.cython_function_or_method
ztpmv: _cython_3_1_4.cython_function_or_method
ztpmv_64: _cython_3_1_4.cython_function_or_method
ztpsv: _cython_3_1_4.cython_function_or_method
ztpsv_64: _cython_3_1_4.cython_function_or_method
ztpttr: _cython_3_1_4.cython_function_or_method
ztrmm: _cython_3_1_4.cython_function_or_method
ztrmm_64: _cython_3_1_4.cython_function_or_method
ztrmm_strided_batched_64: _cython_3_1_4.cython_function_or_method
ztrmv: _cython_3_1_4.cython_function_or_method
ztrmv_64: _cython_3_1_4.cython_function_or_method
ztrsm: _cython_3_1_4.cython_function_or_method
ztrsm_64: _cython_3_1_4.cython_function_or_method
ztrsm_batched: _cython_3_1_4.cython_function_or_method
ztrsm_batched_64: _cython_3_1_4.cython_function_or_method
ztrsv: _cython_3_1_4.cython_function_or_method
ztrsv_64: _cython_3_1_4.cython_function_or_method
ztrttp: _cython_3_1_4.cython_function_or_method

class AtomicsMode(enum.IntEnum):
    """See `cublasAtomicsMode_t`."""
    __new__: ClassVar[Callable] = ...
    ALLOWED: ClassVar[AtomicsMode] = ...
    NOT_ALLOWED: ClassVar[AtomicsMode] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class ComputeType(enum.IntEnum):
    """See `cublasComputeType_t`."""
    __new__: ClassVar[Callable] = ...
    COMPUTE_16F: ClassVar[ComputeType] = ...
    COMPUTE_16F_PEDANTIC: ClassVar[ComputeType] = ...
    COMPUTE_32F: ClassVar[ComputeType] = ...
    COMPUTE_32F_EMULATED_16BFX9: ClassVar[ComputeType] = ...
    COMPUTE_32F_FAST_16BF: ClassVar[ComputeType] = ...
    COMPUTE_32F_FAST_16F: ClassVar[ComputeType] = ...
    COMPUTE_32F_FAST_TF32: ClassVar[ComputeType] = ...
    COMPUTE_32F_PEDANTIC: ClassVar[ComputeType] = ...
    COMPUTE_32I: ClassVar[ComputeType] = ...
    COMPUTE_32I_PEDANTIC: ClassVar[ComputeType] = ...
    COMPUTE_64F: ClassVar[ComputeType] = ...
    COMPUTE_64F_PEDANTIC: ClassVar[ComputeType] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class DiagType(enum.IntEnum):
    """See `cublasDiagType_t`."""
    __new__: ClassVar[Callable] = ...
    NON_UNIT: ClassVar[DiagType] = ...
    UNIT: ClassVar[DiagType] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class EmulationStrategy(enum.IntEnum):
    """See `cublasEmulationStrategy_t`."""
    __new__: ClassVar[Callable] = ...
    DEFAULT: ClassVar[EmulationStrategy] = ...
    EAGER: ClassVar[EmulationStrategy] = ...
    PERFORMANT: ClassVar[EmulationStrategy] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class FillMode(enum.IntEnum):
    """See `cublasFillMode_t`."""
    __new__: ClassVar[Callable] = ...
    FULL: ClassVar[FillMode] = ...
    LOWER: ClassVar[FillMode] = ...
    UPPER: ClassVar[FillMode] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class GemmAlgo(enum.IntEnum):
    """See `cublasGemmAlgo_t`."""
    __new__: ClassVar[Callable] = ...
    ALGO0: ClassVar[GemmAlgo] = ...
    ALGO0_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO1: ClassVar[GemmAlgo] = ...
    ALGO10: ClassVar[GemmAlgo] = ...
    ALGO10_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO11: ClassVar[GemmAlgo] = ...
    ALGO11_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO12: ClassVar[GemmAlgo] = ...
    ALGO12_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO13: ClassVar[GemmAlgo] = ...
    ALGO13_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO14: ClassVar[GemmAlgo] = ...
    ALGO14_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO15: ClassVar[GemmAlgo] = ...
    ALGO15_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO16: ClassVar[GemmAlgo] = ...
    ALGO17: ClassVar[GemmAlgo] = ...
    ALGO18: ClassVar[GemmAlgo] = ...
    ALGO19: ClassVar[GemmAlgo] = ...
    ALGO1_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO2: ClassVar[GemmAlgo] = ...
    ALGO20: ClassVar[GemmAlgo] = ...
    ALGO21: ClassVar[GemmAlgo] = ...
    ALGO22: ClassVar[GemmAlgo] = ...
    ALGO23: ClassVar[GemmAlgo] = ...
    ALGO2_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO3: ClassVar[GemmAlgo] = ...
    ALGO3_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO4: ClassVar[GemmAlgo] = ...
    ALGO4_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO5: ClassVar[GemmAlgo] = ...
    ALGO5_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO6: ClassVar[GemmAlgo] = ...
    ALGO6_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO7: ClassVar[GemmAlgo] = ...
    ALGO7_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO8: ClassVar[GemmAlgo] = ...
    ALGO8_TENSOR_OP: ClassVar[GemmAlgo] = ...
    ALGO9: ClassVar[GemmAlgo] = ...
    ALGO9_TENSOR_OP: ClassVar[GemmAlgo] = ...
    AUTOTUNE: ClassVar[GemmAlgo] = ...
    DEFAULT: ClassVar[GemmAlgo] = ...
    DEFAULT_TENSOR_OP: ClassVar[GemmAlgo] = ...
    DFALT: ClassVar[GemmAlgo] = ...
    DFALT_TENSOR_OP: ClassVar[GemmAlgo] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class Math(enum.IntEnum):
    """See `cublasMath_t`."""
    __new__: ClassVar[Callable] = ...
    DEFAULT_MATH: ClassVar[Math] = ...
    DISALLOW_REDUCED_PRECISION_REDUCTION: ClassVar[Math] = ...
    FP32_EMULATED_BF16X9_MATH: ClassVar[Math] = ...
    PEDANTIC_MATH: ClassVar[Math] = ...
    TENSOR_OP_MATH: ClassVar[Math] = ...
    TF32_TENSOR_OP_MATH: ClassVar[Math] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class Operation(enum.IntEnum):
    """See `cublasOperation_t`."""
    __new__: ClassVar[Callable] = ...
    C: ClassVar[Operation] = ...
    CONJG: ClassVar[Operation] = ...
    HERMITAN: ClassVar[Operation] = ...
    N: ClassVar[Operation] = ...
    T: ClassVar[Operation] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class PointerMode(enum.IntEnum):
    """See `cublasPointerMode_t`."""
    __new__: ClassVar[Callable] = ...
    DEVICE: ClassVar[PointerMode] = ...
    HOST: ClassVar[PointerMode] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class SideMode(enum.IntEnum):
    """See `cublasSideMode_t`."""
    __new__: ClassVar[Callable] = ...
    LEFT: ClassVar[SideMode] = ...
    RIGHT: ClassVar[SideMode] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class Status(enum.IntEnum):
    """See `cublasStatus_t`."""
    __new__: ClassVar[Callable] = ...
    ALLOC_FAILED: ClassVar[Status] = ...
    ARCH_MISMATCH: ClassVar[Status] = ...
    EXECUTION_FAILED: ClassVar[Status] = ...
    INTERNAL_ERROR: ClassVar[Status] = ...
    INVALID_VALUE: ClassVar[Status] = ...
    LICENSE_ERROR: ClassVar[Status] = ...
    MAPPING_ERROR: ClassVar[Status] = ...
    NOT_INITIALIZED: ClassVar[Status] = ...
    NOT_SUPPORTED: ClassVar[Status] = ...
    SUCCESS: ClassVar[Status] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _value2member_map_: ClassVar[dict] = ...

class cuBLASError(Exception):
    def __init__(self, status) -> Any:
        """cuBLASError.__init__(self, status)"""
    def __reduce__(self) -> Any:
        """cuBLASError.__reduce__(self)"""
