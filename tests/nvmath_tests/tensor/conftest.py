# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
import hashlib
import pytest

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def pytest_collection_modifyitems(config, items):
    """
    Skip all tests in this directory if compute capability <= 7.0
    and generate test seeds
    """
    # starting cutensor 2.3.0, support only compute capability > 7.0
    if Device().compute_capability <= (7, 0):
        skip_marker = pytest.mark.skip(reason="cuTensor 2.3.1+ requires compute capability > 7.0")
        for item in items:
            item.add_marker(skip_marker)

    # Generate a unique seed for each test based on its nodeid
    for item in items:
        nodeid_hash = hashlib.md5(item.nodeid.encode()).hexdigest()
        # Convert hash to seed (take first 8 hex digits and convert to int)
        seed = int(nodeid_hash[:8], 16)
        # Store the seed in the item for later use
        item.random_seed = seed


@pytest.fixture(autouse=False)
def seeder(request):
    """Per-test seed based on test nodeid hash."""
    # Get the test-specific seed from the item (set in pytest_collection_modifyitems above)
    seed = request.node.random_seed

    # Set the seed for numpy
    np.random.seed(seed)

    # Only set cupy seed if cupy is already imported
    # Seed all available devices since cp.random.seed only affects the current device
    # See https://docs.cupy.dev/en/latest/reference/generated/cupy.random.seed.html
    if HAS_CUPY:
        num_devices = cp.cuda.runtime.getDeviceCount()
        for device_id in range(num_devices):
            with cp.cuda.Device(device_id):
                cp.random.seed(seed)

    return seed
