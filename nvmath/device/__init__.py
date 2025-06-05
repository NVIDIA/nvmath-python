# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from .patch import patch_codegen
from nvmath._utils import force_loading_libmathdx

patch_codegen()
force_loading_libmathdx("12")

from .common_cuda import *  # noqa: E402, F403
from .cufftdx import *  # noqa: E402, F403
from .cublasdx import *  # noqa: E402, F403
from .cublasdx_backend import *  # noqa: E402, F403
from .vector_types_numba import *  # noqa: E402, F403
from . import nvrtc  # noqa: E402, F403, F401
from .common import make_tensor  # noqa: E402, F401

# register models in numba
from . import cublasdx_numba  # noqa: E402, F401

del patch_codegen
del force_loading_libmathdx
