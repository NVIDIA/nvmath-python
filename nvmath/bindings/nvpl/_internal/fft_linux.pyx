# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.3.0. Do not modify it directly.

from libc.stdint cimport intptr_t

from ..._internal.utils import FunctionNotFoundError, NotSupportedError


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

cdef bint __py_nvpl_fft_init = False
cdef str __current_so_name = ""
cdef tuple __lib_so_names = ("libnvpl_fftw.so.0", "libmkl_rt.so.2",)


cdef void* __nvpl_fft_get_version = NULL
cdef void* __fftw_plan_many_dft = NULL
cdef void* __fftw_plan_many_dft_r2c = NULL
cdef void* __fftw_plan_many_dft_c2r = NULL
cdef void* __fftw_execute_dft = NULL
cdef void* __fftw_execute_dft_r2c = NULL
cdef void* __fftw_execute_dft_c2r = NULL
cdef void* __fftwf_plan_many_dft = NULL
cdef void* __fftwf_plan_many_dft_r2c = NULL
cdef void* __fftwf_plan_many_dft_c2r = NULL
cdef void* __fftwf_execute_dft = NULL
cdef void* __fftwf_execute_dft_r2c = NULL
cdef void* __fftwf_execute_dft_c2r = NULL
cdef void* __fftw_init_threads = NULL
cdef void* __fftwf_init_threads = NULL
cdef void* __fftw_plan_with_nthreads = NULL
cdef void* __fftwf_plan_with_nthreads = NULL
cdef void* __fftw_planner_nthreads = NULL
cdef void* __fftwf_planner_nthreads = NULL
cdef void* __fftw_cleanup_threads = NULL
cdef void* __fftwf_cleanup_threads = NULL
cdef void* __fftw_destroy_plan = NULL
cdef void* __fftwf_destroy_plan = NULL


cdef void* load_library() except* with gil:
    cdef void* handle;
    cdef str all_err_msg = ""
    if len(__lib_so_names) == 0:
        raise RuntimeError("Cannot load FFTW-compatible library. No lib names were specified.")
    for so_name in __lib_so_names:
        handle = dlopen(so_name.encode(), RTLD_NOW | RTLD_GLOBAL)
        if handle != NULL:
            global __current_so_name
            __current_so_name = so_name
            break
        else:
            error_msg = dlerror()
            all_err_msg += f"\n{error_msg.decode()}"
    else:
        all_libs = ", ".join(__lib_so_names)
        raise RuntimeError(
            f"Failed to dlopen either of the following libraries: {all_libs}. {all_err_msg}"
        )


cdef int _check_or_init_nvpl_fft() except -1 nogil:
    global __py_nvpl_fft_init
    if __py_nvpl_fft_init:
        return 0

    # Load function
    cdef void* handle = NULL
    global __nvpl_fft_get_version
    __nvpl_fft_get_version = dlsym(RTLD_DEFAULT, 'nvpl_fft_get_version')
    if __nvpl_fft_get_version == NULL:
        if handle == NULL:
            handle = load_library()
        __nvpl_fft_get_version = dlsym(handle, 'nvpl_fft_get_version')

    global __fftw_plan_many_dft
    __fftw_plan_many_dft = dlsym(RTLD_DEFAULT, 'fftw_plan_many_dft')
    if __fftw_plan_many_dft == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_plan_many_dft = dlsym(handle, 'fftw_plan_many_dft')

    global __fftw_plan_many_dft_r2c
    __fftw_plan_many_dft_r2c = dlsym(RTLD_DEFAULT, 'fftw_plan_many_dft_r2c')
    if __fftw_plan_many_dft_r2c == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_plan_many_dft_r2c = dlsym(handle, 'fftw_plan_many_dft_r2c')

    global __fftw_plan_many_dft_c2r
    __fftw_plan_many_dft_c2r = dlsym(RTLD_DEFAULT, 'fftw_plan_many_dft_c2r')
    if __fftw_plan_many_dft_c2r == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_plan_many_dft_c2r = dlsym(handle, 'fftw_plan_many_dft_c2r')

    global __fftw_execute_dft
    __fftw_execute_dft = dlsym(RTLD_DEFAULT, 'fftw_execute_dft')
    if __fftw_execute_dft == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_execute_dft = dlsym(handle, 'fftw_execute_dft')

    global __fftw_execute_dft_r2c
    __fftw_execute_dft_r2c = dlsym(RTLD_DEFAULT, 'fftw_execute_dft_r2c')
    if __fftw_execute_dft_r2c == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_execute_dft_r2c = dlsym(handle, 'fftw_execute_dft_r2c')

    global __fftw_execute_dft_c2r
    __fftw_execute_dft_c2r = dlsym(RTLD_DEFAULT, 'fftw_execute_dft_c2r')
    if __fftw_execute_dft_c2r == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_execute_dft_c2r = dlsym(handle, 'fftw_execute_dft_c2r')

    global __fftwf_plan_many_dft
    __fftwf_plan_many_dft = dlsym(RTLD_DEFAULT, 'fftwf_plan_many_dft')
    if __fftwf_plan_many_dft == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_plan_many_dft = dlsym(handle, 'fftwf_plan_many_dft')

    global __fftwf_plan_many_dft_r2c
    __fftwf_plan_many_dft_r2c = dlsym(RTLD_DEFAULT, 'fftwf_plan_many_dft_r2c')
    if __fftwf_plan_many_dft_r2c == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_plan_many_dft_r2c = dlsym(handle, 'fftwf_plan_many_dft_r2c')

    global __fftwf_plan_many_dft_c2r
    __fftwf_plan_many_dft_c2r = dlsym(RTLD_DEFAULT, 'fftwf_plan_many_dft_c2r')
    if __fftwf_plan_many_dft_c2r == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_plan_many_dft_c2r = dlsym(handle, 'fftwf_plan_many_dft_c2r')

    global __fftwf_execute_dft
    __fftwf_execute_dft = dlsym(RTLD_DEFAULT, 'fftwf_execute_dft')
    if __fftwf_execute_dft == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_execute_dft = dlsym(handle, 'fftwf_execute_dft')

    global __fftwf_execute_dft_r2c
    __fftwf_execute_dft_r2c = dlsym(RTLD_DEFAULT, 'fftwf_execute_dft_r2c')
    if __fftwf_execute_dft_r2c == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_execute_dft_r2c = dlsym(handle, 'fftwf_execute_dft_r2c')

    global __fftwf_execute_dft_c2r
    __fftwf_execute_dft_c2r = dlsym(RTLD_DEFAULT, 'fftwf_execute_dft_c2r')
    if __fftwf_execute_dft_c2r == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_execute_dft_c2r = dlsym(handle, 'fftwf_execute_dft_c2r')

    global __fftw_init_threads
    __fftw_init_threads = dlsym(RTLD_DEFAULT, 'fftw_init_threads')
    if __fftw_init_threads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_init_threads = dlsym(handle, 'fftw_init_threads')

    global __fftwf_init_threads
    __fftwf_init_threads = dlsym(RTLD_DEFAULT, 'fftwf_init_threads')
    if __fftwf_init_threads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_init_threads = dlsym(handle, 'fftwf_init_threads')

    global __fftw_plan_with_nthreads
    __fftw_plan_with_nthreads = dlsym(RTLD_DEFAULT, 'fftw_plan_with_nthreads')
    if __fftw_plan_with_nthreads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_plan_with_nthreads = dlsym(handle, 'fftw_plan_with_nthreads')

    global __fftwf_plan_with_nthreads
    __fftwf_plan_with_nthreads = dlsym(RTLD_DEFAULT, 'fftwf_plan_with_nthreads')
    if __fftwf_plan_with_nthreads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_plan_with_nthreads = dlsym(handle, 'fftwf_plan_with_nthreads')

    global __fftw_planner_nthreads
    __fftw_planner_nthreads = dlsym(RTLD_DEFAULT, 'fftw_planner_nthreads')
    if __fftw_planner_nthreads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_planner_nthreads = dlsym(handle, 'fftw_planner_nthreads')

    global __fftwf_planner_nthreads
    __fftwf_planner_nthreads = dlsym(RTLD_DEFAULT, 'fftwf_planner_nthreads')
    if __fftwf_planner_nthreads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_planner_nthreads = dlsym(handle, 'fftwf_planner_nthreads')

    global __fftw_cleanup_threads
    __fftw_cleanup_threads = dlsym(RTLD_DEFAULT, 'fftw_cleanup_threads')
    if __fftw_cleanup_threads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_cleanup_threads = dlsym(handle, 'fftw_cleanup_threads')

    global __fftwf_cleanup_threads
    __fftwf_cleanup_threads = dlsym(RTLD_DEFAULT, 'fftwf_cleanup_threads')
    if __fftwf_cleanup_threads == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_cleanup_threads = dlsym(handle, 'fftwf_cleanup_threads')

    global __fftw_destroy_plan
    __fftw_destroy_plan = dlsym(RTLD_DEFAULT, 'fftw_destroy_plan')
    if __fftw_destroy_plan == NULL:
        if handle == NULL:
            handle = load_library()
        __fftw_destroy_plan = dlsym(handle, 'fftw_destroy_plan')

    global __fftwf_destroy_plan
    __fftwf_destroy_plan = dlsym(RTLD_DEFAULT, 'fftwf_destroy_plan')
    if __fftwf_destroy_plan == NULL:
        if handle == NULL:
            handle = load_library()
        __fftwf_destroy_plan = dlsym(handle, 'fftwf_destroy_plan')

    __py_nvpl_fft_init = True
    return 0


cdef dict func_ptrs = None


cpdef void _set_lib_so_names(tuple lib_so_names):
    global __lib_so_names
    __lib_so_names = lib_so_names


cpdef tuple _get_lib_so_names():
    global __lib_so_names
    return __lib_so_names


cpdef str _get_current_lib_so_name():
    global __current_so_name
    return __current_so_name


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvpl_fft()
    cdef dict data = {}

    global __nvpl_fft_get_version
    data["__nvpl_fft_get_version"] = <intptr_t>__nvpl_fft_get_version

    global __fftw_plan_many_dft
    data["__fftw_plan_many_dft"] = <intptr_t>__fftw_plan_many_dft

    global __fftw_plan_many_dft_r2c
    data["__fftw_plan_many_dft_r2c"] = <intptr_t>__fftw_plan_many_dft_r2c

    global __fftw_plan_many_dft_c2r
    data["__fftw_plan_many_dft_c2r"] = <intptr_t>__fftw_plan_many_dft_c2r

    global __fftw_execute_dft
    data["__fftw_execute_dft"] = <intptr_t>__fftw_execute_dft

    global __fftw_execute_dft_r2c
    data["__fftw_execute_dft_r2c"] = <intptr_t>__fftw_execute_dft_r2c

    global __fftw_execute_dft_c2r
    data["__fftw_execute_dft_c2r"] = <intptr_t>__fftw_execute_dft_c2r

    global __fftwf_plan_many_dft
    data["__fftwf_plan_many_dft"] = <intptr_t>__fftwf_plan_many_dft

    global __fftwf_plan_many_dft_r2c
    data["__fftwf_plan_many_dft_r2c"] = <intptr_t>__fftwf_plan_many_dft_r2c

    global __fftwf_plan_many_dft_c2r
    data["__fftwf_plan_many_dft_c2r"] = <intptr_t>__fftwf_plan_many_dft_c2r

    global __fftwf_execute_dft
    data["__fftwf_execute_dft"] = <intptr_t>__fftwf_execute_dft

    global __fftwf_execute_dft_r2c
    data["__fftwf_execute_dft_r2c"] = <intptr_t>__fftwf_execute_dft_r2c

    global __fftwf_execute_dft_c2r
    data["__fftwf_execute_dft_c2r"] = <intptr_t>__fftwf_execute_dft_c2r

    global __fftw_init_threads
    data["__fftw_init_threads"] = <intptr_t>__fftw_init_threads

    global __fftwf_init_threads
    data["__fftwf_init_threads"] = <intptr_t>__fftwf_init_threads

    global __fftw_plan_with_nthreads
    data["__fftw_plan_with_nthreads"] = <intptr_t>__fftw_plan_with_nthreads

    global __fftwf_plan_with_nthreads
    data["__fftwf_plan_with_nthreads"] = <intptr_t>__fftwf_plan_with_nthreads

    global __fftw_planner_nthreads
    data["__fftw_planner_nthreads"] = <intptr_t>__fftw_planner_nthreads

    global __fftwf_planner_nthreads
    data["__fftwf_planner_nthreads"] = <intptr_t>__fftwf_planner_nthreads

    global __fftw_cleanup_threads
    data["__fftw_cleanup_threads"] = <intptr_t>__fftw_cleanup_threads

    global __fftwf_cleanup_threads
    data["__fftwf_cleanup_threads"] = <intptr_t>__fftwf_cleanup_threads

    global __fftw_destroy_plan
    data["__fftw_destroy_plan"] = <intptr_t>__fftw_destroy_plan

    global __fftwf_destroy_plan
    data["__fftwf_destroy_plan"] = <intptr_t>__fftwf_destroy_plan

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

cdef int _nvpl_fft_get_version() except* nogil:
    global __nvpl_fft_get_version
    _check_or_init_nvpl_fft()
    if __nvpl_fft_get_version == NULL:
        with gil:
            raise FunctionNotFoundError("function nvpl_fft_get_version is not found")
    return (<int (*)() nogil>__nvpl_fft_get_version)(
        )


cdef fftw_plan _fftw_plan_many_dft(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except* nogil:
    global __fftw_plan_many_dft
    _check_or_init_nvpl_fft()
    if __fftw_plan_many_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_many_dft is not found")
    return (<fftw_plan (*)(int, const int*, int, fftw_complex*, const int*, int, int, fftw_complex*, const int*, int, int, int, unsigned) nogil>__fftw_plan_many_dft)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)


cdef fftw_plan _fftw_plan_many_dft_r2c(int rank, const int* n, int batch, double* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    global __fftw_plan_many_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftw_plan_many_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_many_dft_r2c is not found")
    return (<fftw_plan (*)(int, const int*, int, double*, const int*, int, int, fftw_complex*, const int*, int, int, unsigned) nogil>__fftw_plan_many_dft_r2c)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef fftw_plan _fftw_plan_many_dft_c2r(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, double* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    global __fftw_plan_many_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftw_plan_many_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_many_dft_c2r is not found")
    return (<fftw_plan (*)(int, const int*, int, fftw_complex*, const int*, int, int, double*, const int*, int, int, unsigned) nogil>__fftw_plan_many_dft_c2r)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef void _fftw_execute_dft(const fftw_plan plan, fftw_complex* idata, fftw_complex* odata) except* nogil:
    global __fftw_execute_dft
    _check_or_init_nvpl_fft()
    if __fftw_execute_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_execute_dft is not found")
    (<void (*)(const fftw_plan, fftw_complex*, fftw_complex*) nogil>__fftw_execute_dft)(
        plan, idata, odata)


cdef void _fftw_execute_dft_r2c(const fftw_plan plan, double* idata, fftw_complex* odata) except* nogil:
    global __fftw_execute_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftw_execute_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_execute_dft_r2c is not found")
    (<void (*)(const fftw_plan, double*, fftw_complex*) nogil>__fftw_execute_dft_r2c)(
        plan, idata, odata)


cdef void _fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* idata, double* odata) except* nogil:
    global __fftw_execute_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftw_execute_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_execute_dft_c2r is not found")
    (<void (*)(const fftw_plan, fftw_complex*, double*) nogil>__fftw_execute_dft_c2r)(
        plan, idata, odata)


cdef fftwf_plan _fftwf_plan_many_dft(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except* nogil:
    global __fftwf_plan_many_dft
    _check_or_init_nvpl_fft()
    if __fftwf_plan_many_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_many_dft is not found")
    return (<fftwf_plan (*)(int, const int*, int, fftwf_complex*, const int*, int, int, fftwf_complex*, const int*, int, int, int, unsigned) nogil>__fftwf_plan_many_dft)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)


cdef fftwf_plan _fftwf_plan_many_dft_r2c(int rank, const int* n, int batch, float* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    global __fftwf_plan_many_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftwf_plan_many_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_many_dft_r2c is not found")
    return (<fftwf_plan (*)(int, const int*, int, float*, const int*, int, int, fftwf_complex*, const int*, int, int, unsigned) nogil>__fftwf_plan_many_dft_r2c)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef fftwf_plan _fftwf_plan_many_dft_c2r(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, float* out, const int* onembed, int ostride, int odist, unsigned flags) except* nogil:
    global __fftwf_plan_many_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftwf_plan_many_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_many_dft_c2r is not found")
    return (<fftwf_plan (*)(int, const int*, int, fftwf_complex*, const int*, int, int, float*, const int*, int, int, unsigned) nogil>__fftwf_plan_many_dft_c2r)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef void _fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* idata, fftwf_complex* odata) except* nogil:
    global __fftwf_execute_dft
    _check_or_init_nvpl_fft()
    if __fftwf_execute_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_execute_dft is not found")
    (<void (*)(const fftwf_plan, fftwf_complex*, fftwf_complex*) nogil>__fftwf_execute_dft)(
        plan, idata, odata)


cdef void _fftwf_execute_dft_r2c(const fftwf_plan plan, float* idata, fftwf_complex* odata) except* nogil:
    global __fftwf_execute_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftwf_execute_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_execute_dft_r2c is not found")
    (<void (*)(const fftwf_plan, float*, fftwf_complex*) nogil>__fftwf_execute_dft_r2c)(
        plan, idata, odata)


cdef void _fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* idata, float* odata) except* nogil:
    global __fftwf_execute_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftwf_execute_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_execute_dft_c2r is not found")
    (<void (*)(const fftwf_plan, fftwf_complex*, float*) nogil>__fftwf_execute_dft_c2r)(
        plan, idata, odata)


cdef int _fftw_init_threads() except* nogil:
    global __fftw_init_threads
    _check_or_init_nvpl_fft()
    if __fftw_init_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_init_threads is not found")
    return (<int (*)() nogil>__fftw_init_threads)(
        )


cdef int _fftwf_init_threads() except* nogil:
    global __fftwf_init_threads
    _check_or_init_nvpl_fft()
    if __fftwf_init_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_init_threads is not found")
    return (<int (*)() nogil>__fftwf_init_threads)(
        )


cdef void _fftw_plan_with_nthreads(int nthreads) except* nogil:
    global __fftw_plan_with_nthreads
    _check_or_init_nvpl_fft()
    if __fftw_plan_with_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_with_nthreads is not found")
    (<void (*)(int) nogil>__fftw_plan_with_nthreads)(
        nthreads)


cdef void _fftwf_plan_with_nthreads(int nthreads) except* nogil:
    global __fftwf_plan_with_nthreads
    _check_or_init_nvpl_fft()
    if __fftwf_plan_with_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_with_nthreads is not found")
    (<void (*)(int) nogil>__fftwf_plan_with_nthreads)(
        nthreads)


cdef int _fftw_planner_nthreads() except* nogil:
    global __fftw_planner_nthreads
    _check_or_init_nvpl_fft()
    if __fftw_planner_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_planner_nthreads is not found")
    return (<int (*)() nogil>__fftw_planner_nthreads)(
        )


cdef int _fftwf_planner_nthreads() except* nogil:
    global __fftwf_planner_nthreads
    _check_or_init_nvpl_fft()
    if __fftwf_planner_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_planner_nthreads is not found")
    return (<int (*)() nogil>__fftwf_planner_nthreads)(
        )


cdef void _fftw_cleanup_threads() except* nogil:
    global __fftw_cleanup_threads
    _check_or_init_nvpl_fft()
    if __fftw_cleanup_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_cleanup_threads is not found")
    (<void (*)() nogil>__fftw_cleanup_threads)(
        )


cdef void _fftwf_cleanup_threads() except* nogil:
    global __fftwf_cleanup_threads
    _check_or_init_nvpl_fft()
    if __fftwf_cleanup_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_cleanup_threads is not found")
    (<void (*)() nogil>__fftwf_cleanup_threads)(
        )


cdef void _fftw_destroy_plan(fftw_plan plan) except* nogil:
    global __fftw_destroy_plan
    _check_or_init_nvpl_fft()
    if __fftw_destroy_plan == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_destroy_plan is not found")
    (<void (*)(fftw_plan) nogil>__fftw_destroy_plan)(
        plan)


cdef void _fftwf_destroy_plan(fftwf_plan plan) except* nogil:
    global __fftwf_destroy_plan
    _check_or_init_nvpl_fft()
    if __fftwf_destroy_plan == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_destroy_plan is not found")
    (<void (*)(fftwf_plan) nogil>__fftwf_destroy_plan)(
        plan)
