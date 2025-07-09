# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation for N-D FFT as a composition of the 1D, 2D or 3D batched FFTs"""

__all__ = ["fftn"]

import nvmath


def fftn(a, *, axes=None, direction=None, options=None, prolog=None, epilog=None, stream=None, engine=nvmath.fft.fft):
    """
    Perform an N-D FFT as a composition of the 1D, 2D, or 3D batched FFTs supported by
    cuFFT. This is version 1.

    Args:
        engine: a callable to execute the FFT operation. The engine can be `fft` from the
            nvmath.fft package, or `caching.fft` from the examples.
    """

    rank = a.ndim

    axes = [a % rank for a in axes]
    if any(a >= rank for a in axes):
        raise ValueError(f"Invalid axes = {axes}.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate axis IDs are not allowed: axes = {axes}.")
    # Add check to ensure C2C.

    composition = list(range(rank))
    while axes:
        chunk, axes = axes[:3], axes[3:]

        permutation = [d for d in range(rank) if d not in chunk] + chunk

        ipermutation = {v: p for p, v in enumerate(permutation)}
        axes = [ipermutation[a] for a in axes]

        a = a.transpose(*permutation).copy()

        last = list(range(rank - len(chunk), rank))
        a = engine(a, axes=last, direction=direction, options=options, prolog=prolog, epilog=epilog, stream=stream)

        composition = [composition[p] for p in permutation]

    icomposition = {v: c for c, v in enumerate(composition)}
    a = a.transpose(tuple(icomposition[c] for c in range(rank)))
    return a
