# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import gc
import importlib
import threading
import typing
import weakref

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nvmath.internal import package_wrapper, tensor_wrapper, utils
from nvmath.internal._device_utils import get_device
from nvmath.internal.utils import cached_get_or_create_stream
from nvmath.memory import _MEMORY_MANAGER

try:
    _device_count = system.get_num_devices()
except AttributeError:
    _device_count = system.num_devices

_cupy_available = False
try:
    import cupy as cp
    from cupy.cuda.runtime import getDevice, setDevice

    _cupy_available = True
except ModuleNotFoundError:
    pass

try:
    import torch
except ModuleNotFoundError:
    pass


class TestDeviceCtx:
    @pytest.mark.skipif(_device_count < 2, reason="2+ GPUs required for this test.")
    @pytest.mark.skipif(not _cupy_available, reason="CuPy required for this test.")
    def test_device_ctx(self):
        assert getDevice() == 0
        with utils.device_ctx(0):
            assert getDevice() == 0
            with utils.device_ctx(1):
                assert getDevice() == 1
                with utils.device_ctx(0):
                    assert getDevice() == 0
                assert getDevice() == 1
            assert getDevice() == 0
        assert getDevice() == 0

        with utils.device_ctx(1):
            assert getDevice() == 1
            setDevice(0)
            with utils.device_ctx(1):
                assert getDevice() == 1
            assert getDevice() == 0
        assert getDevice() == 0

    @pytest.mark.skipif(_device_count < 2, reason="2+ GPUs required for this test.")
    @pytest.mark.skipif(not _cupy_available, reason="CuPy required for this test.")
    def test_thread_safe(self):
        # adopted from https://github.com/cupy/cupy/blob/master/tests/cupy_tests/cuda_tests/test_device.py
        # recall that the CUDA context is maintained per-thread, so when each thread
        # starts it is on the default device (=device 0).
        t0_setup = threading.Event()
        t1_setup = threading.Event()
        t0_first_exit = threading.Event()

        t0_exit_device = []
        t1_exit_device = []

        def t0_seq():
            with utils.device_ctx(0):
                with utils.device_ctx(1):
                    t0_setup.set()
                    t1_setup.wait()
                    t0_exit_device.append(getDevice())
                t0_exit_device.append(getDevice())
                t0_first_exit.set()
            assert getDevice() == 0

        def t1_seq():
            t0_setup.wait()
            with utils.device_ctx(1):
                with utils.device_ctx(0):
                    t1_setup.set()
                    t0_first_exit.wait()
                    t1_exit_device.append(getDevice())
                t1_exit_device.append(getDevice())
            assert getDevice() == 0

        try:
            cp.cuda.runtime.setDevice(1)
            t0 = threading.Thread(target=t0_seq)
            t1 = threading.Thread(target=t1_seq)
            t1.start()
            t0.start()
            t0.join()
            t1.join()
            assert t0_exit_device == [1, 0]
            assert t1_exit_device == [0, 1]
        finally:
            cp.cuda.runtime.setDevice(0)

    def test_one_shot(self):
        dev = utils.device_ctx(0)
        with dev:
            pass
        # CPython raises AttributeError, but we should not care here
        with pytest.raises(Exception):  # noqa: B017, SIM117
            with dev:
                pass


@given(package_name=st.sampled_from(["cupy", "torch", "numpy"]), id0=st.sampled_from(["cpu", 0]))
def test_tensor_empty_device_ctx(package_name: str, id0: int | typing.Literal["cpu"]) -> None:
    try:
        tensor_wrapper.maybe_register_package(package_name)
    except ModuleNotFoundError:
        return
    tensor_type = tensor_wrapper._TENSOR_TYPES[package_name]
    if package_name == "cupy" and isinstance(id0, str):
        return
    if package_name == "numpy" and isinstance(id0, int):
        return
    id1 = _device_count - 1
    stream_holder = (
        None if isinstance(id0, str) else utils.get_or_create_stream(device_id=id0, stream=None, op_package=package_name)
    )
    with utils.device_ctx(id1):
        _ = utils.create_empty_tensor(
            tensor_type,
            device_id=id0,
            extents=(64, 64, 64),
            dtype="float32",
            stream_holder=stream_holder,
            verify_strides=False,
        )


@given(package_name=st.sampled_from(["cupy", "torch"]))
def test_stream_ifc(package_name: str):
    # make sure the package is registered
    try:
        tensor_wrapper.maybe_register_package(package_name)
    except ModuleNotFoundError:
        return
    package = package_wrapper.PACKAGE[package_name]
    core = package_wrapper.PACKAGE["cuda"]

    # test pointers are the same for current stream
    stream = package.get_current_stream(device_id=0)
    ptr = package.to_stream_pointer(stream)
    obj = package.create_stream(stream, 0)
    core_ptr = core.to_stream_pointer(obj)
    assert core_ptr == ptr, f"cuda.core stream pointer ({core_ptr}) not equal to package stream pointer ({ptr})!"

    # test pointers are the same for new stream
    module = importlib.import_module(package_name)
    stream = module.cuda.Stream()
    ptr = package.to_stream_pointer(stream)
    obj = package.create_stream(stream, 0)
    core_ptr = core.to_stream_pointer(obj)
    assert core_ptr == ptr, f"cuda.core stream pointer ({core_ptr}) not equal to package stream pointer ({ptr})!"


@given(package_name=st.sampled_from(["cuda", "cupy", "torch"]))
def test_default_allocator_user_stream_lifetime(package_name: str):
    # make sure the package is registered
    try:
        tensor_wrapper.maybe_register_package(package_name)
    except ModuleNotFoundError:
        return

    memory_manager = _MEMORY_MANAGER["cuda"](device_id=0, logger=None)

    if package_name == "cuda":
        external_stream = get_device(0).create_stream()
    elif package_name == "cupy":
        external_stream = cp.cuda.Stream(non_blocking=True)
    elif package_name == "torch":
        external_stream = torch.cuda.Stream(device="cuda:0")
    stream_holder = utils.get_or_create_stream(device_id=0, stream=external_stream, op_package=package_name)
    alloc = memory_manager.memalloc_async(size=1024, stream=stream_holder.obj)

    external_ref = None if package_name == "cuda" else weakref.ref(external_stream)
    holder_ref = weakref.ref(stream_holder)

    # Get rid of any references to the original stream object
    del stream_holder
    del external_stream
    # simulate stream_holder being evicted from the cache
    cached_get_or_create_stream.cache_clear()
    gc.collect()

    if package_name != "cuda":
        assert external_ref() is not None
    assert holder_ref() is None

    # The cuda.core memory resource remembers the allocation stream
    # to deallocate on it. Here we check if StreamHolder managed to
    # extend that dependency to the foreign stream object, such that:
    # alloc -> stream_holder.obj : cuda.core.Stream -> stream : cupy | torch Stream
    # If not, the following may end up with a dangling pointer.

    alloc.free()
