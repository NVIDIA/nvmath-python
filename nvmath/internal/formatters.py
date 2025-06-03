# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Formatters for printing data.
"""

import numpy as np


class MemoryStr:
    """
    A simple type to pretty-print memory-like values.
    """

    def __init__(self, memory, base_unit="B"):
        self.memory = memory
        self.base_unit = base_unit
        self.base = 1024

    def __str__(self):
        """
        Convert large values to powers of 1024 for readability.
        """

        base, base_unit, memory = self.base, self.base_unit, self.memory

        if memory < base:
            value, unit = memory, base_unit
        elif memory < base**2:
            value, unit = memory / base, f"Ki{base_unit}"
        elif memory < base**3:
            value, unit = memory / base**2, f"Mi{base_unit}"
        else:
            value, unit = memory / base**3, f"Gi{base_unit}"

        return f"{value:0.2f} {unit}"


class FLOPSStr:
    """
    A simple type to pretty-print FLOP count and FLOP/s-like values.
    """

    def __init__(self, flops, base_unit):
        self.flops = flops
        self.base_unit = base_unit
        self.base = 1000

    def __str__(self):
        """
        Convert large values to powers of 1000 for readability.
        """

        base, base_unit, flops = self.base, self.base_unit, self.flops

        if flops < base:
            value, unit = flops, base_unit
        elif flops < base**2:
            value, unit = flops / base, f"K{base_unit}"
        elif flops < base**3:
            value, unit = flops / base**2, f"M{base_unit}"
        elif flops < base**4:
            value, unit = flops / base**3, f"G{base_unit}"
        elif flops < base**5:
            value, unit = flops / base**4, f"T{base_unit}"
        else:
            value, unit = flops / base**5, f"P{base_unit}"

        return f"{value:0.3f} {unit}"


def array2string(array_like):
    """
    String representation of an array-like object with possible truncation of "interior"
    values to limit string size.

    The NumPy function "set_printoptions" can be used to control the display of the array.
    """

    return np.array2string(
        np.asanyarray(array_like, dtype=object),
        separator=", ",
        # NumPy hates empty strings so we print 'None' instead.
        formatter={"object": lambda s: s if s != "" else "None"},
    )
