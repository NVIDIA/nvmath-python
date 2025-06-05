# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Memory specification regular expression.
"""

__all__ = ["MEM_LIMIT_RE_PCT", "MEM_LIMIT_RE_VAL", "MEM_LIMIT_DOC", "check_memory_str"]

import re

MEM_LIMIT_RE_PCT = re.compile(r"(?P<value>[+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*%\s*$")
MEM_LIMIT_RE_VAL = re.compile(
    r"(?P<value>[+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(?P<units>[kmg])?(?P<binary>(?<=[kmg])i)?b\s*$",
    re.IGNORECASE,
)
MEM_LIMIT_DOC = """The {kind} must be specified in one of the following forms:
  (1) A number (int or float). If the number is a float between 0 and 1 inclusive, the
      {kind} is interpreted as a fraction of the total device memory; otherwise it is
      interpreted as the number of bytes of memory, with float value being cast to int.
      Examples: 0.75, 50E6, 50000000, ...
  (2) A string containing a positive value followed by B, kB, MB, or GB for powers of 1000.
      Examples: "0.05 GB", "50 MB", "50000000 B" ...
  (3) A string containing a positive value followed by kiB, MiB, or GiB for powers of 1024.
      Examples:  "0.05 GiB", "51.2 MiB", "53687091 B" ...
  (4) A string with value in the range [0, 100] followed by a % symbol.
      Examples: "26%","82%", ...

  Whitespace between values and units is optional.

The provided {kind} is "{value}".
"""


def check_memory_str(value, kind):
    """
    Check if the memory specification string is valid.

    value = the memory specification string.
    kind  = a string denoting the type of memory being checked, used in error messages.
    """
    if not isinstance(value, int | float):
        m1 = MEM_LIMIT_RE_PCT.match(value)
        if m1:
            factor = float(m1.group("value"))
            if factor < 0 or factor > 100:
                raise ValueError(f"The {kind} percentage must be in the range [0, 100].")
        m2 = MEM_LIMIT_RE_VAL.match(value)
        if not (m1 or m2):
            raise ValueError(MEM_LIMIT_DOC.format(kind=kind, value=value))
