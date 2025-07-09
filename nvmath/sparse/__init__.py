# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath._utils import force_loading_cudss

try:
    force_loading_cudss("12")
except RuntimeError:
    pass

del force_loading_cudss

from . import advanced  # noqa: E402

__all__ = [
    "advanced",
]
