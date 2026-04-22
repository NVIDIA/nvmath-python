# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import threading

# handle cuda.core < 0.5.0
try:
    from cuda.core import Device, ObjectCode, ProgramOptions, Program
except ImportError:
    from cuda.core.experimental import Device, ObjectCode, ProgramOptions, Program

# Kernels are cached based on the kernel_code (and device arch).
# The process-wide cache is a nested dictionary
# arch -> kernel_code -> cuda.core.ObjectCode
# On top of that, each Python thread has its own thread-local cache
# device_id -> kernel_code -> kernel_name -> cuda.core.Kernel
# First, we lookup in the thread-local cache, without explicit locking.
# If there is no hit, we acquire _obj_code_lock to lookup or compile and update
# the process-wide cache.

_obj_code_lock = threading.Lock()
# _obj_code_cache: arch -> kernel_code -> cuda.core.ObjectCode
_obj_code_cache = {}

# thread_local.local_kernel_cache: device_id -> kernel_code -> kernel_name -> cuda.core.Kernel
thread_local = threading.local()


cpdef _invalidate_kernel_cache():
    """
    WARNING: this is internal utility meant for use in unit tests.
    All threads using the cache MUST call this function and wait
    for all other threads to complete the call before using the cache
    again.
    """
    thread_local.local_kernel_cache = {}
    with _obj_code_lock:
        _obj_code_cache.clear()
    return 0


cdef inline str _get_device_arch(int device_id):
    # device_id -> arch
    cdef dict device_archs
    try:
        device_archs = thread_local.device_archs
    except AttributeError:
        thread_local.device_archs = device_archs = {}

    cdef str arch = device_archs.get(device_id)
    if arch is None:
        arch = f"sm_{Device(device_id).arch}"
        device_archs[device_id] = arch
    return arch


cdef inline _compile_kernel(str kernel_code, str arch, include_path, object logger=None):
    options = ProgramOptions(std="c++17", arch=arch, include_path=include_path, device_as_default_execution_space=True)
    prog = Program(kernel_code, code_type="c++", options=options)
    return prog.compile("cubin")


cdef inline _get_or_compile_object_code(str kernel_code, int device_id, include_path, object logger=None):
    """
    Lookup or compile and store object code for the kernel_code and device's arch.
    Note, this function is not thread-safe, it should be called while holding _obj_code_lock.
    """
    cdef str arch = _get_device_arch(device_id)

    # _obj_code_cache : arch -> kernel_code -> cuda.core.ObjectCode
    cdef dict arch_cache = _obj_code_cache.get(arch)
    if arch_cache is None:
        arch_cache = {}
        _obj_code_cache[arch] = arch_cache
    # arch_cache: kernel_code -> cuda.core.ObjectCode

    obj_code = arch_cache.get(kernel_code)
    if obj_code is None:
        obj_code = _compile_kernel(kernel_code, arch, include_path, logger)
        arch_cache[kernel_code] = obj_code
        if logger is not None:
            logger.debug(f"Stored compiled object code ({obj_code}) for device {device_id} in global cache.\n{kernel_code}")
    elif logger is not None:
        logger.debug(f"Loaded object code ({obj_code}) for device {device_id} from global cache.\n{kernel_code}")
    return obj_code


cdef intptr_t get_kernel(str kernel_code, str kernel_name, int device_id, include_path, object logger=None) except? 0:
    """
    Get or compile a kernel and return cuda.core.Kernel object.
    Note, the kernels are cached based on the kernel_code (and implicitly device/device_arch).

    It is user responsibility to provide consistently the same include_path
    for the same kernel_code.
    """

    cdef dict local_kernel_cache
    try:
        local_kernel_cache = thread_local.local_kernel_cache
    except AttributeError:
        thread_local.local_kernel_cache = local_kernel_cache = {}

    # local_kernel_cache: device_id -> kernel_code -> kernel_name -> cuda.core.Kernel

    cdef dict device_kernels = local_kernel_cache.get(device_id)
    if device_kernels is None:
        device_kernels = {}
        local_kernel_cache[device_id] = device_kernels
    # device_kernels: kernel_code -> kernel_name -> cuda.core.Kernel

    cdef dict obj_kernels = device_kernels.get(kernel_code)
    if obj_kernels is None:
        obj_kernels = {}
        device_kernels[kernel_code] = obj_kernels
    # obj_kernels: kernel_name -> cuda.core.Kernel

    cdef kernel = obj_kernels.get(kernel_name)
    if kernel is None:
        with _obj_code_lock:
            obj_code = _get_or_compile_object_code(kernel_code, device_id, include_path, logger)
            kernel = obj_code.get_kernel(kernel_name)
            obj_kernels[kernel_name] = kernel
    elif logger is not None:
        logger.debug(f"Loaded kernel {kernel_name} ({kernel}) for device {device_id} from thread local cache.\n{kernel_code}")
    try:
        return int(kernel.handle)
    except AttributeError:
        # the API has changed in cuda-core 0.6.0
        return int(kernel._handle)
