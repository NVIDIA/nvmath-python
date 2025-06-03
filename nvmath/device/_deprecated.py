# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import sys
from warnings import warn


def _deprecated(
    msg,
    category: type[Warning] | None = DeprecationWarning,
    stacklevel: int = 1,
):
    """Dropin replacement for @warnings.deprecated(...)"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warn(msg, category=category, stacklevel=stacklevel + 1)
            return func(*args, **kwargs)

        return wrapper

    return decorator


deprecated = _deprecated

if (sys.version_info.major, sys.version_info.minor) >= (3, 13):
    from warnings import deprecated as _warnings_deprecated  # type: ignore

    deprecated = _warnings_deprecated
