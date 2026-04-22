# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test nvmath.distributed initialization and parameter validation.
"""

import re

import pytest

import nvmath.distributed

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system


def test_initialize_invalid_device_id_type(process_group):
    """Test that initialize raises TypeError when device_id is not an integer"""

    # Test with string
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The device ID used to initialize the nvmath.distributed module "
            "must be an integer. The provided device ID is invalid_device."
        ),
    ):
        nvmath.distributed.initialize("invalid_device", process_group, backends=["nvshmem"])

    # Test with None
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The device ID used to initialize the nvmath.distributed module must be an integer. The provided device ID is None."
        ),
    ):
        nvmath.distributed.initialize(None, process_group, backends=["nvshmem"])


def test_initialize_invalid_backend(process_group):
    """Test that initialize raises ValueError when an invalid backend is specified"""

    with pytest.raises(ValueError, match=re.escape("backend must be one of ('nvshmem', 'nccl'), got invalid_backend")):
        nvmath.distributed.initialize(0, process_group, backends=["invalid_backend"])

    # Test with empty backends list
    with pytest.raises(
        ValueError, match=re.escape("Need to specify at least one of ('nvshmem', 'nccl') communication backends")
    ):
        nvmath.distributed.initialize(0, process_group, backends=[])


def test_initialize_invalid_communicator_type():
    """Test that initialize raises TypeError when communicator is not a ProcessGroup"""
    # Test with string
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Unrecognized process group type (not_a_communicator). Need nvmath.distributed.ProcessGroup or mpi4py communicator."
        ),
    ):
        nvmath.distributed.initialize(0, "not_a_communicator", backends=["nvshmem"])


def test_initialize_already_initialized(process_group):
    """Test that initialize raises RuntimeError when called twice"""

    try:
        num_devices = system.get_num_devices()
    except AttributeError:
        num_devices = system.num_devices

    device_id = process_group.rank % num_devices

    # First initialization should succeed
    nvmath.distributed.initialize(device_id, process_group, backends=["nvshmem"])

    try:
        # Second initialization should fail
        with pytest.raises(RuntimeError, match=re.escape("nvmath.distributed has already been initialized")):
            nvmath.distributed.initialize(device_id, process_group, backends=["nvshmem"])
    finally:
        # Clean up
        nvmath.distributed.finalize()
