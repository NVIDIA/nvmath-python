# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# Defined in CPython:
# https://github.com/python/cpython/blob/26bc2cc06128890ac89492eca20e83abe0789c1c/Objects/unicodetype_db.h#L6311-L6349

__all__ = ["WHITESPACE_UNICODE"]

_WHITESPACE_UNICODE_INTS = [
    0x0009,
    0x000A,
    0x000B,
    0x000C,
    0x000D,
    0x001C,
    0x001D,
    0x001E,
    0x001F,
    0x0020,
    0x0085,
    0x00A0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
]


WHITESPACE_UNICODE = "".join(chr(s) for s in _WHITESPACE_UNICODE_INTS)
