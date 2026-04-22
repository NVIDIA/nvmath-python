# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
import hashlib

import numpy as np
import pytest

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def pytest_collection_modifyitems(config, items):
    """Generate test seeds for reproducibility."""
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
