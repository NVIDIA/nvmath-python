# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of FFT with support on truncation and padding"""

__all__ = ["fft"]

import nvmath


def create(shape, dtype, device_id, package, *, stream=None, creator="zeros"):
    if package == "torch":
        import torch

        if device_id is None:
            return getattr(torch, creator)(*shape, dtype=dtype, device=device_id)
        if stream is None:
            stream = torch.cuda.current_stream()
        elif isinstance(stream, int):
            stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(stream):
            return getattr(torch, creator)(*shape, dtype=dtype, device=device_id)

    if package == "cupy":
        import cupy as cp

        if stream is None:
            stream = cp.cuda.get_current_stream()
        elif isinstance(stream, int):
            stream = cp.cuda.ExternalStream(stream)
        with cp.cuda.Device(device_id), stream:
            return getattr(cp, creator)(shape, dtype=dtype)

    if package == "numpy":
        import numpy as np

        return getattr(np, creator)(shape, dtype=dtype)

    raise AssertionError("Unsupported package.")


def fft(
    a,
    *,
    axes=None,
    extents=None,
    direction=None,
    options=None,
    prolog=None,
    epilog=None,
    stream=None,
    engine=nvmath.fft.fft,
):
    """
    This version supports truncation and padding of the operand, to match the functionality of NumPy FFT.

    Args:
        extents: An array specifying the truncated or padded extents for the FFT axes. If not specified, the extents of the operand dimensions corresponding
            to the FFT axes will be used.
        engine: a callable to execute the FFT operation. The engine can be `fft` from the nvmath.fft package, or `caching.fft`, `fftn1.fftn`, `fftn2.fftn`
            etc. from the examples.
    """
    if extents is None:
        return engine(a, axes=axes, direction=direction, options=options, prolog=prolog, epilog=epilog, stream=stream)

    package = a.__class__.__module__.split(".")[0]

    rank = a.ndim
    num_axes = len(extents)

    if axes is not None and len(extents) != len(axes):
        raise ValueError(f"The FFT axes length ({len(axes)}) must match that of extents ({len(extents)}).")

    if axes is None:
        axes = list(range(rank - num_axes, rank))

    shape = a.shape

    if all(shape[axes[i]] == extents[i] for i in range(num_axes)):
        # No need to pad or truncate if the transform axes extents already match the extents.
        return engine(a, axes=axes, direction=direction, options=options, prolog=prolog, epilog=epilog, stream=stream)

    if all(extents[i] < shape[axes[i]] for i in range(num_axes)):  # All axes truncated.
        creator = "empty"
    else:  # Some axes padded.
        creator = "zeros"

    new_shape = list(shape)
    for i, axis in enumerate(axes):
        new_shape[axis] = extents[i]

    device_id = a.device.id if package == "cupy" else a.device.index if package == "torch" else None
    b = create(new_shape, dtype=a.dtype, device_id=device_id, package=package, stream=stream, creator=creator)

    z = tuple(slice(0, min(s, t)) for s, t in zip(new_shape, shape, strict=True))
    b[z] = a[z]
    return engine(b, axes=axes, direction=direction, options=options, prolog=prolog, epilog=epilog, stream=stream)
