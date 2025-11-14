# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 0.4.2. Do not modify it directly.

cimport cython
from libc.stdint cimport intptr_t

import os
import site
import threading
import win32api

from ..._internal.utils import FunctionNotFoundError, NotSupportedError

from libc.stddef cimport wchar_t
from libc.stdint cimport uintptr_t
from cpython cimport PyUnicode_AsWideCharString, PyMem_Free

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from "windows.h" nogil:
    ctypedef void* HMODULE
    ctypedef void* HANDLE
    ctypedef void* FARPROC
    ctypedef unsigned long DWORD
    ctypedef const wchar_t *LPCWSTR
    ctypedef const char *LPCSTR

    cdef DWORD LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800
    cdef DWORD LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
    cdef DWORD LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100

    HMODULE _LoadLibraryExW "LoadLibraryExW"(
        LPCWSTR lpLibFileName,
        HANDLE hFile,
        DWORD dwFlags
    )

    FARPROC _GetProcAddress "GetProcAddress"(HMODULE hModule, LPCSTR lpProcName)

cdef inline uintptr_t LoadLibraryExW(str path, HANDLE hFile, DWORD dwFlags):
    cdef uintptr_t result
    cdef wchar_t* wpath = PyUnicode_AsWideCharString(path, NULL)
    with nogil:
        result = <uintptr_t>_LoadLibraryExW(
            wpath,
            hFile,
            dwFlags
        )
    PyMem_Free(wpath)
    return result

cdef inline void *GetProcAddress(uintptr_t hModule, const char* lpProcName) nogil:
    return _GetProcAddress(<HMODULE>hModule, lpProcName)

cdef int get_cuda_version():
    cdef int err, driver_ver = 0

    # Load driver to check version
    handle = LoadLibraryExW("nvcuda.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32)
    if handle == 0:
        raise NotSupportedError('CUDA driver is not found')
    cuDriverGetVersion = GetProcAddress(handle, 'cuDriverGetVersion')
    if cuDriverGetVersion == NULL:
        raise RuntimeError('something went wrong')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError('something went wrong')

    return driver_ver


###############################################################################
# Wrapper init
###############################################################################

cdef object __symbol_lock = threading.Lock()
cdef bint __py_nvpl_fft_init = False
cdef str __current_dll_name = ""
cdef tuple __lib_dll_names = ("mkl_rt.2.dll", )

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


cdef inline list get_site_packages():
    return [site.getusersitepackages()] + site.getsitepackages()


cdef void* load_library() except* with gil:
    handle = 0
    cdef str all_err_msg = ""
    cdef str env_lib_dll_name = os.getenv("NVMATH_FFT_CPU_LIBRARY", "")

    if env_lib_dll_name != "":
        try:
            handle = win32api.GetModuleHandle(env_lib_dll_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to dlopen NVMATH_FFT_CPU_LIBRARY={env_lib_dll_name}. "
                f"Please check that NVMATH_FFT_CPU_LIBRARY is the name of a DLL on the PATH. {e}"
            )
        global __current_dll_name
        __current_dll_name = env_lib_dll_name
        assert handle != 0
        return <void*><intptr_t>handle

    if len(__lib_dll_names) == 0:
        raise RuntimeError("Cannot load a FFT-compatible library. No DLL names were specified.")
    for dll_name in __lib_dll_names:

        # First check if the DLL has been loaded by 3rd parties
        try:
            handle = win32api.GetModuleHandle(dll_name)
        except Exception as e:
            all_err_msg += f"\n{e}"
            pass
        else:
            break  # stop at first successful open

        # Next, check if DLLs are installed via pip
        for sp in get_site_packages():
            mod_path = os.path.join(sp, "..", "..", "Library", "bin")
            if not os.path.isdir(mod_path):
                continue
            os.add_dll_directory(mod_path)
        try:
            handle = win32api.LoadLibraryEx(
                # Note: LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs an abs path...
                os.path.join(mod_path, dll_name),
                0, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR)
        except Exception as e:
            all_err_msg += f"\n{e}"
            pass
        else:
            break  # stop at first successful open

        # Finally, try default search
        try:
            handle = win32api.LoadLibrary(dll_name)
        except Exception as e:
            all_err_msg += f"\n{e}"
            pass
        else:
            break  # stop at first successful open
    else:
        all_libs = ", ".join(__lib_dll_names)
        raise RuntimeError(
            f"Failed to dlopen all of the following libraries: {all_libs}. "
            "Install/add one of these libraries to the PATH or "
            f"use environment variable NVMATH_FFT_CPU_LIBRARY to name a DLL on the PATH. {all_err_msg}"
        )

    global __current_dll_name
    __current_dll_name = dll_name

    assert handle != 0
    return <void*><intptr_t>handle


cdef int _check_or_init_nvpl_fft() except -1 nogil:
    global __py_nvpl_fft_init
    if __py_nvpl_fft_init:
        return 0

    with gil, __symbol_lock:
        # Load library
        handle = <intptr_t>load_library()

        # Load function
        global __nvpl_fft_get_version
        __nvpl_fft_get_version = GetProcAddress(handle, 'nvpl_fft_get_version')

        global __fftw_plan_many_dft
        __fftw_plan_many_dft = GetProcAddress(handle, 'fftw_plan_many_dft')

        global __fftw_plan_many_dft_r2c
        __fftw_plan_many_dft_r2c = GetProcAddress(handle, 'fftw_plan_many_dft_r2c')

        global __fftw_plan_many_dft_c2r
        __fftw_plan_many_dft_c2r = GetProcAddress(handle, 'fftw_plan_many_dft_c2r')

        global __fftw_execute_dft
        __fftw_execute_dft = GetProcAddress(handle, 'fftw_execute_dft')

        global __fftw_execute_dft_r2c
        __fftw_execute_dft_r2c = GetProcAddress(handle, 'fftw_execute_dft_r2c')

        global __fftw_execute_dft_c2r
        __fftw_execute_dft_c2r = GetProcAddress(handle, 'fftw_execute_dft_c2r')

        global __fftwf_plan_many_dft
        __fftwf_plan_many_dft = GetProcAddress(handle, 'fftwf_plan_many_dft')

        global __fftwf_plan_many_dft_r2c
        __fftwf_plan_many_dft_r2c = GetProcAddress(handle, 'fftwf_plan_many_dft_r2c')

        global __fftwf_plan_many_dft_c2r
        __fftwf_plan_many_dft_c2r = GetProcAddress(handle, 'fftwf_plan_many_dft_c2r')

        global __fftwf_execute_dft
        __fftwf_execute_dft = GetProcAddress(handle, 'fftwf_execute_dft')

        global __fftwf_execute_dft_r2c
        __fftwf_execute_dft_r2c = GetProcAddress(handle, 'fftwf_execute_dft_r2c')

        global __fftwf_execute_dft_c2r
        __fftwf_execute_dft_c2r = GetProcAddress(handle, 'fftwf_execute_dft_c2r')

        global __fftw_init_threads
        __fftw_init_threads = GetProcAddress(handle, 'fftw_init_threads')

        global __fftwf_init_threads
        __fftwf_init_threads = GetProcAddress(handle, 'fftwf_init_threads')

        global __fftw_plan_with_nthreads
        __fftw_plan_with_nthreads = GetProcAddress(handle, 'fftw_plan_with_nthreads')

        global __fftwf_plan_with_nthreads
        __fftwf_plan_with_nthreads = GetProcAddress(handle, 'fftwf_plan_with_nthreads')

        global __fftw_planner_nthreads
        __fftw_planner_nthreads = GetProcAddress(handle, 'fftw_planner_nthreads')

        global __fftwf_planner_nthreads
        __fftwf_planner_nthreads = GetProcAddress(handle, 'fftwf_planner_nthreads')

        global __fftw_cleanup_threads
        __fftw_cleanup_threads = GetProcAddress(handle, 'fftw_cleanup_threads')

        global __fftwf_cleanup_threads
        __fftwf_cleanup_threads = GetProcAddress(handle, 'fftwf_cleanup_threads')

        global __fftw_destroy_plan
        __fftw_destroy_plan = GetProcAddress(handle, 'fftw_destroy_plan')

        global __fftwf_destroy_plan
        __fftwf_destroy_plan = GetProcAddress(handle, 'fftwf_destroy_plan')

    __py_nvpl_fft_init = True
    return 0


cdef dict func_ptrs = None


cpdef void _set_lib_so_names(tuple lib_so_names):
    global __lib_dll_names
    __lib_dll_names = lib_so_names


cpdef tuple _get_lib_so_names():
    global __lib_dll_names
    return __lib_dll_names


cpdef str _get_current_lib_so_name():
    global __current_dll_name
    return __current_dll_name


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

cdef int _nvpl_fft_get_version() except?-42 nogil:
    global __nvpl_fft_get_version
    _check_or_init_nvpl_fft()
    if __nvpl_fft_get_version == NULL:
        with gil:
            raise FunctionNotFoundError("function nvpl_fft_get_version is not found")
    return (<int (*)() noexcept nogil>__nvpl_fft_get_version)(
        )


cdef fftw_plan _fftw_plan_many_dft(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except?NULL nogil:
    global __fftw_plan_many_dft
    _check_or_init_nvpl_fft()
    if __fftw_plan_many_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_many_dft is not found")
    return (<fftw_plan (*)(int, const int*, int, fftw_complex*, const int*, int, int, fftw_complex*, const int*, int, int, int, unsigned) noexcept nogil>__fftw_plan_many_dft)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)


cdef fftw_plan _fftw_plan_many_dft_r2c(int rank, const int* n, int batch, double* in_, const int* inembed, int istride, int idist, fftw_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil:
    global __fftw_plan_many_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftw_plan_many_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_many_dft_r2c is not found")
    return (<fftw_plan (*)(int, const int*, int, double*, const int*, int, int, fftw_complex*, const int*, int, int, unsigned) noexcept nogil>__fftw_plan_many_dft_r2c)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef fftw_plan _fftw_plan_many_dft_c2r(int rank, const int* n, int batch, fftw_complex* in_, const int* inembed, int istride, int idist, double* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil:
    global __fftw_plan_many_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftw_plan_many_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_many_dft_c2r is not found")
    return (<fftw_plan (*)(int, const int*, int, fftw_complex*, const int*, int, int, double*, const int*, int, int, unsigned) noexcept nogil>__fftw_plan_many_dft_c2r)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


@cython.show_performance_hints(False)
cdef void _fftw_execute_dft(const fftw_plan plan, fftw_complex* idata, fftw_complex* odata) except* nogil:
    global __fftw_execute_dft
    _check_or_init_nvpl_fft()
    if __fftw_execute_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_execute_dft is not found")
    (<void (*)(const fftw_plan, fftw_complex*, fftw_complex*) noexcept nogil>__fftw_execute_dft)(
        plan, idata, odata)


@cython.show_performance_hints(False)
cdef void _fftw_execute_dft_r2c(const fftw_plan plan, double* idata, fftw_complex* odata) except* nogil:
    global __fftw_execute_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftw_execute_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_execute_dft_r2c is not found")
    (<void (*)(const fftw_plan, double*, fftw_complex*) noexcept nogil>__fftw_execute_dft_r2c)(
        plan, idata, odata)


@cython.show_performance_hints(False)
cdef void _fftw_execute_dft_c2r(const fftw_plan plan, fftw_complex* idata, double* odata) except* nogil:
    global __fftw_execute_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftw_execute_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_execute_dft_c2r is not found")
    (<void (*)(const fftw_plan, fftw_complex*, double*) noexcept nogil>__fftw_execute_dft_c2r)(
        plan, idata, odata)


cdef fftwf_plan _fftwf_plan_many_dft(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, int sign, unsigned flags) except?NULL nogil:
    global __fftwf_plan_many_dft
    _check_or_init_nvpl_fft()
    if __fftwf_plan_many_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_many_dft is not found")
    return (<fftwf_plan (*)(int, const int*, int, fftwf_complex*, const int*, int, int, fftwf_complex*, const int*, int, int, int, unsigned) noexcept nogil>__fftwf_plan_many_dft)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, sign, flags)


cdef fftwf_plan _fftwf_plan_many_dft_r2c(int rank, const int* n, int batch, float* in_, const int* inembed, int istride, int idist, fftwf_complex* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil:
    global __fftwf_plan_many_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftwf_plan_many_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_many_dft_r2c is not found")
    return (<fftwf_plan (*)(int, const int*, int, float*, const int*, int, int, fftwf_complex*, const int*, int, int, unsigned) noexcept nogil>__fftwf_plan_many_dft_r2c)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


cdef fftwf_plan _fftwf_plan_many_dft_c2r(int rank, const int* n, int batch, fftwf_complex* in_, const int* inembed, int istride, int idist, float* out, const int* onembed, int ostride, int odist, unsigned flags) except?NULL nogil:
    global __fftwf_plan_many_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftwf_plan_many_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_many_dft_c2r is not found")
    return (<fftwf_plan (*)(int, const int*, int, fftwf_complex*, const int*, int, int, float*, const int*, int, int, unsigned) noexcept nogil>__fftwf_plan_many_dft_c2r)(
        rank, n, batch, in_, inembed, istride, idist, out, onembed, ostride, odist, flags)


@cython.show_performance_hints(False)
cdef void _fftwf_execute_dft(const fftwf_plan plan, fftwf_complex* idata, fftwf_complex* odata) except* nogil:
    global __fftwf_execute_dft
    _check_or_init_nvpl_fft()
    if __fftwf_execute_dft == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_execute_dft is not found")
    (<void (*)(const fftwf_plan, fftwf_complex*, fftwf_complex*) noexcept nogil>__fftwf_execute_dft)(
        plan, idata, odata)


@cython.show_performance_hints(False)
cdef void _fftwf_execute_dft_r2c(const fftwf_plan plan, float* idata, fftwf_complex* odata) except* nogil:
    global __fftwf_execute_dft_r2c
    _check_or_init_nvpl_fft()
    if __fftwf_execute_dft_r2c == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_execute_dft_r2c is not found")
    (<void (*)(const fftwf_plan, float*, fftwf_complex*) noexcept nogil>__fftwf_execute_dft_r2c)(
        plan, idata, odata)


@cython.show_performance_hints(False)
cdef void _fftwf_execute_dft_c2r(const fftwf_plan plan, fftwf_complex* idata, float* odata) except* nogil:
    global __fftwf_execute_dft_c2r
    _check_or_init_nvpl_fft()
    if __fftwf_execute_dft_c2r == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_execute_dft_c2r is not found")
    (<void (*)(const fftwf_plan, fftwf_complex*, float*) noexcept nogil>__fftwf_execute_dft_c2r)(
        plan, idata, odata)


cdef int _fftw_init_threads() except?-42 nogil:
    global __fftw_init_threads
    _check_or_init_nvpl_fft()
    if __fftw_init_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_init_threads is not found")
    return (<int (*)() noexcept nogil>__fftw_init_threads)(
        )


cdef int _fftwf_init_threads() except?-42 nogil:
    global __fftwf_init_threads
    _check_or_init_nvpl_fft()
    if __fftwf_init_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_init_threads is not found")
    return (<int (*)() noexcept nogil>__fftwf_init_threads)(
        )


@cython.show_performance_hints(False)
cdef void _fftw_plan_with_nthreads(int nthreads) except* nogil:
    global __fftw_plan_with_nthreads
    _check_or_init_nvpl_fft()
    if __fftw_plan_with_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_plan_with_nthreads is not found")
    (<void (*)(int) noexcept nogil>__fftw_plan_with_nthreads)(
        nthreads)


@cython.show_performance_hints(False)
cdef void _fftwf_plan_with_nthreads(int nthreads) except* nogil:
    global __fftwf_plan_with_nthreads
    _check_or_init_nvpl_fft()
    if __fftwf_plan_with_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_plan_with_nthreads is not found")
    (<void (*)(int) noexcept nogil>__fftwf_plan_with_nthreads)(
        nthreads)


cdef int _fftw_planner_nthreads() except?-42 nogil:
    global __fftw_planner_nthreads
    _check_or_init_nvpl_fft()
    if __fftw_planner_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_planner_nthreads is not found")
    return (<int (*)() noexcept nogil>__fftw_planner_nthreads)(
        )


cdef int _fftwf_planner_nthreads() except?-42 nogil:
    global __fftwf_planner_nthreads
    _check_or_init_nvpl_fft()
    if __fftwf_planner_nthreads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_planner_nthreads is not found")
    return (<int (*)() noexcept nogil>__fftwf_planner_nthreads)(
        )


@cython.show_performance_hints(False)
cdef void _fftw_cleanup_threads() except* nogil:
    global __fftw_cleanup_threads
    _check_or_init_nvpl_fft()
    if __fftw_cleanup_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_cleanup_threads is not found")
    (<void (*)() noexcept nogil>__fftw_cleanup_threads)(
        )


@cython.show_performance_hints(False)
cdef void _fftwf_cleanup_threads() except* nogil:
    global __fftwf_cleanup_threads
    _check_or_init_nvpl_fft()
    if __fftwf_cleanup_threads == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_cleanup_threads is not found")
    (<void (*)() noexcept nogil>__fftwf_cleanup_threads)(
        )


@cython.show_performance_hints(False)
cdef void _fftw_destroy_plan(fftw_plan plan) except* nogil:
    global __fftw_destroy_plan
    _check_or_init_nvpl_fft()
    if __fftw_destroy_plan == NULL:
        with gil:
            raise FunctionNotFoundError("function fftw_destroy_plan is not found")
    (<void (*)(fftw_plan) noexcept nogil>__fftw_destroy_plan)(
        plan)


@cython.show_performance_hints(False)
cdef void _fftwf_destroy_plan(fftwf_plan plan) except* nogil:
    global __fftwf_destroy_plan
    _check_or_init_nvpl_fft()
    if __fftwf_destroy_plan == NULL:
        with gil:
            raise FunctionNotFoundError("function fftwf_destroy_plan is not found")
    (<void (*)(fftwf_plan) noexcept nogil>__fftwf_destroy_plan)(
        plan)
