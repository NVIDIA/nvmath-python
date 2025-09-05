# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import threading
import typing

import cuda.core.experimental as ccx
import pytest

from hypothesis import given, strategies as st
from nvmath.internal import package_wrapper, tensor_wrapper, utils

_device_count = ccx.system.num_devices

_cupy_available = False
try:
    import cupy as cp
    from cupy.cuda.runtime import getDevice, setDevice

    _cupy_available = True
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
        with pytest.raises(Exception):  # noqa: SIM117
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
    id1 = ccx.system.num_devices - 1
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
    obj = package.create_stream(stream)
    core_ptr = core.to_stream_pointer(obj)
    assert core_ptr == ptr, f"cuda.core stream pointer ({core_ptr}) not equal to package stream pointer ({ptr})!"

    # test pointers are the same for new stream
    module = importlib.import_module(package_name)
    stream = module.cuda.Stream()
    ptr = package.to_stream_pointer(stream)
    obj = package.create_stream(stream)
    core_ptr = core.to_stream_pointer(obj)
    assert core_ptr == ptr, f"cuda.core stream pointer ({core_ptr}) not equal to package stream pointer ({ptr})!"
