# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

try:
    from cuda.core import Device, system
except ImportError:
    from cuda.core.experimental import Device, system


import threading

thread_local = threading.local()


def get_device(device_id: int) -> Device:
    """
    NOTE: the function DOES NOT change the current cuda device. If it needs
    to call .set_current(), it immediately restores the previous device
    before exiting. To temporarily change the current device,
    please use utils.device_ctx().

    A counterpart to cuda.core.Device(int), with extra initialization.
    Returns initialized cuda.core.Device(device_id) object, i.e.
    makes sure to call .set_current() on the instance once per thread.
    Calling .set_current() is not just about a valid CUDA context in a thread,
    but specifically cuda.core requirement to initialize the device instance.

    The function keeps a thread-local cache of "initialized"
    device instances. It mirrors cuda.core.Device(int) caching,
    adding the set_current() call on the first access.
    """
    try:
        devices = thread_local.devices
    except AttributeError:
        try:
            num_devices = system.get_num_devices()
        except AttributeError:
            # cuda.core < 0.5.0
            num_devices = system.num_devices
        thread_local.devices = devices = [None] * num_devices

    device = devices[device_id]
    if device is None:
        current_device = Device()
        current_device_id = current_device.device_id
        try:
            device = Device(device_id)
            device.set_current()
            devices[device_id] = device
        finally:
            if current_device_id != device_id:
                current_device.set_current()
    return device
