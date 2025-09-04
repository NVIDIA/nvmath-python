# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys


def test_no_cupy():
    if "cupy" in sys.modules or "torch" in sys.modules:
        raise RuntimeError("Can't test: cupy or torch are already loaded")

    import nvmath.distributed  # noqa: F401

    assert "cupy" not in sys.modules
    assert "torch" not in sys.modules


def test_no_cupy_tensor_wrapper():
    if "cupy" in sys.modules or "torch" in sys.modules:
        raise RuntimeError("Can't test: cupy or torch are already loaded")

    import nvmath.distributed
    import nvmath.distributed._internal.tensor_wrapper

    assert "cupy" not in sys.modules
    assert "torch" not in sys.modules

    import numpy as np

    assert "cupy" not in sys.modules
    assert "torch" not in sys.modules

    a = nvmath.distributed._internal.tensor_wrapper.wrap_operand(np.arange(10))

    assert "cupy" not in sys.modules
    assert "torch" not in sys.modules

    stream = nvmath.internal.utils.get_or_create_stream(0, None, "cuda")
    b = a.to(device_id=0, stream_holder=stream)

    assert "cupy" not in sys.modules
    assert "torch" not in sys.modules

    b.to(device_id="cpu", stream_holder=stream)

    assert "cupy" not in sys.modules
    assert "torch" not in sys.modules
