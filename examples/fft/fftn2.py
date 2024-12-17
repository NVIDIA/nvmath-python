# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""N-D FFT as a composition of the 1D, 2D or 3D batched FFTs with the number of copies
minimized."""

__all__ = ["fftn"]

import nvmath


def upto_three_contiguous_axes(ordered_axes, ordered_all_axes):
    """
    Given FFT axes and tensor dimensions, both ordered from smallest stride to largest.
    Return (copy flag, chunk slice, remainder slice).
    """
    left, e, f = 0, 0, ordered_all_axes.index(ordered_axes[0])
    if e == f:
        while left < 3 and e < len(ordered_axes) and ordered_axes[e] == ordered_all_axes[f]:
            left, e, f = left + 1, e + 1, f + 1

    right, e, f = 0, -1, ordered_all_axes.index(ordered_axes[-1])
    if e % len(ordered_all_axes) == f:
        while right < 3 and -e <= len(ordered_axes) and ordered_axes[e] == ordered_all_axes[f]:
            right, e, f = right + 1, e - 1, f - 1

    d = max(left, right)
    if d == 0 or (d < 3 and min(left, right) == 0 and len(ordered_axes) > d) or (d < 3 and d + 1 < len(ordered_axes) - d):
        return True, slice(None, -4, -1), slice(-4, None, -1)

    if left > right:
        return False, slice(None, d), slice(d, None)

    return (
        False,
        slice(-1, -d - 1, -1),
        slice(-d - 1, None, -1),
    )


def fftn(a, *, axes=None, direction=None, options=None, prolog=None, epilog=None, stream=None, engine=nvmath.fft.fft):
    """
    Perform an N-D FFT as a composition of the 1D, 2D, or 3D batched FFTs supported by
    cuFFT, minimizing the number of copies needed. This is version 2.

    Args:
        engine: a callable to execute the FFT operation. The engine can be `fft` from the
            nvmath.fft package, or `caching.fft` from the examples.
    """

    rank = a.ndim

    axes = list(a % rank for a in axes)
    if any(a >= rank for a in axes):
        raise ValueError(f"Invalid axes = {axes}.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate axis IDs are not allowed: axes = {axes}.")
    # Add check for C2C.

    composition = list(range(rank))
    while axes:
        _, ordered_all_axes = zip(
            *sorted(zip((a.strides[axis] for axis in range(rank)), range(rank), strict=True), reverse=True), strict=True
        )
        _, axes = zip(*sorted(zip((a.strides[axis] for axis in axes), axes, strict=True), reverse=True), strict=True)
        axes = list(axes)
        copy, c, r = upto_three_contiguous_axes(axes, ordered_all_axes)
        chunk, axes = axes[c], axes[r]

        last = chunk
        if copy:
            permutation = axes + list(d for d in range(rank) if d not in axes + chunk) + chunk
            ipermutation = {v: p for p, v in enumerate(permutation)}

            a = a.transpose(*permutation).copy()

            if axes:
                axes = list(ipermutation[a] for a in axes)

            composition = list(composition[p] for p in permutation)

            last = list(range(rank - len(chunk), rank))

        a = engine(a, axes=last, direction=direction, options=options, prolog=prolog, epilog=epilog, stream=stream)

    icomposition = {v: c for c, v in enumerate(composition)}
    a = a.transpose(tuple(icomposition[c] for c in range(rank)))
    return a
