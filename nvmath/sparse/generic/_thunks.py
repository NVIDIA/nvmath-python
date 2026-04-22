# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "default_prolog",
]


def default_prolog(multiplier, *, is_conjugate):
    if is_conjugate:

        def prolog(a):
            return multiplier * a.conjugate()

        return prolog

    def prolog(a):
        return multiplier * a

    return prolog
