# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union, Tuple
import random

import numpy as np
import cupy as cp
try:
    import torch
except:
    torch = None

from .common_axes import Backend, Framework, DType, ShapeKind
from .axes_utils import get_framework_dtype, is_complex


def get_random_input_data(
    framework: Framework,
    shape: Union[int, Tuple[int]],
    dtype: DType,
    backend: Backend,
    seed: int,
    lo: float = -0.5,
    hi: float = 0.5,
    device_id=None,
):
    assert lo < hi
    framework_dtype = get_framework_dtype(framework, dtype)
    if framework in [Framework.numpy, Framework.cupy]:

        def _create_array():
            if framework == Framework.numpy:
                assert backend == Backend.cpu
                rng = np.random.default_rng(seed)
            else:
                assert backend == Backend.gpu
                rng = cp.random.default_rng(seed)
            if not is_complex(dtype):
                a = rng.uniform(lo, hi, size=shape).astype(framework_dtype)
            else:
                real = rng.uniform(lo, hi, size=shape)
                imag = rng.uniform(lo, hi, size=shape)
                a = (real + 1j * imag).astype(framework_dtype)
            assert a.dtype == framework_dtype
            return a

        if backend == Backend.gpu and device_id is not None:
            with cp.cuda.Device(device_id):
                return _create_array()
        else:
            return _create_array()

    elif framework == Framework.torch:
        if backend == Backend.cpu:
            device = "cpu"
        elif device_id is not None:
            device = f"cuda:{device_id}"
        else:
            device = "cuda"
        g = torch.Generator(device=device)
        g = g.manual_seed(seed)
        t = torch.rand(size=shape, generator=g, device=device, dtype=framework_dtype)
        scale = torch.tensor(hi - lo, dtype=framework_dtype)
        if not is_complex(dtype):
            shift = torch.tensor(lo, dtype=framework_dtype)
        else:
            shift = torch.tensor(lo + 1j * lo, dtype=framework_dtype)
        t = t * scale + shift
        assert t.dtype == framework_dtype
        return t
    else:
        raise ValueError(f"Unknown framework {framework}")


def get_1d_shape_cases(
    shape_kinds: List[ShapeKind], rng: random.Random, incl_1: bool = True
):
    """
    Sample concrete shapes to test according to shape_kinds
    """
    assert len(shape_kinds), f"{shape_kinds}"

    def shape_gen():
        for shape_kind in shape_kinds:
            if shape_kind == ShapeKind.pow2:
                if incl_1:
                    yield ShapeKind.pow2, 1
                random_pow_2 = 2 ** rng.randint(1, 12)
                yield ShapeKind.pow2, random_pow_2
            elif shape_kind == ShapeKind.pow2357:
                a, b, c, d = rng.choices([0, 1, 2, 3], k=4)
                random_pow_2357 = (2**a) * (3**b) * (5**c) * (7**d)
                yield ShapeKind.pow2357, random_pow_2357
            elif shape_kind == ShapeKind.prime:
                random_prime = rng.choice([89, 103, 131, 397, 541, 853, 997, 12799])
                yield ShapeKind.prime, random_prime
            else:
                assert shape_kind == ShapeKind.random
                yield ShapeKind.random, rng.randint(1, 4096)

    return list(shape_gen())


def get_random_1d_shape(
    shape_kinds: List[ShapeKind], rng: random.Random, incl_1: bool = False
):
    assert len(shape_kinds), f"{shape_kinds}"
    return rng.choice(get_1d_shape_cases(shape_kinds, rng=rng, incl_1=incl_1))


def get_custom_stream(framework: Framework, device_id=None):
    if framework in [Framework.numpy, Framework.cupy]:
        if device_id is None:
            return cp.cuda.Stream(non_blocking=True)
        else:
            with cp.cuda.Device(device_id):
                return cp.cuda.Stream(non_blocking=True)
    elif framework == Framework.torch:
        return torch.cuda.Stream()
    else:
        raise ValueError(f"Unknown GPU framework {framework}")
