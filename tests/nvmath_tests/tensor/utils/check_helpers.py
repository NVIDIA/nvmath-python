# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import numpy as np
from collections.abc import Sequence

try:
    import cupy as cp

    CP_NDARRAY = cp.ndarray
except ImportError:
    cp = CP_NDARRAY = None

try:
    import torch
except ImportError:
    torch = None


from nvmath.internal import tensor_wrapper
from nvmath.tensor import ComputeDesc, Operator

from .axes_utils import TORCH_TENSOR


def get_contraction_ref(
    eq: str,
    a: np.ndarray | CP_NDARRAY | TORCH_TENSOR,
    b: np.ndarray | CP_NDARRAY | TORCH_TENSOR,
    *,
    c: np.ndarray | CP_NDARRAY | TORCH_TENSOR | None = None,
    d: np.ndarray | CP_NDARRAY | TORCH_TENSOR | None = None,
    alpha: float = 1.0,
    beta: float | None = None,
    qualifiers: Sequence[Operator] = [],
):
    num_inputs = eq.count(",") + 1
    if len(qualifiers) == 0:
        qualifiers = [Operator.OP_IDENTITY] * (num_inputs + 1)
    else:
        assert len(qualifiers) == num_inputs + 1, f"The qualifiers must be a sequence of length {num_inputs + 1}"
    if num_inputs == 2:
        iterator = zip([a, b], qualifiers[:num_inputs], strict=False)
        if c is None and beta is not None:
            raise ValueError("beta can only be set if c is specified in a binary contraction")
        elif c is not None and beta is None:
            raise ValueError("beta must be set when c is specified in a binary contraction")
    else:
        iterator = zip([a, b, c], qualifiers[:num_inputs], strict=False)
        if d is None and beta is not None:
            raise ValueError("beta can only be set if d is specified in a ternary contraction")
        elif d is not None and beta is None:
            raise ValueError("beta must be set when d is specified in a ternary contraction")
    operands = []
    for op, qualifier in iterator:
        if qualifier not in {Operator.OP_IDENTITY, Operator.OP_CONJ}:
            raise ValueError(f"Invalid operator: {qualifier}")
        if op is not None:
            if qualifier == Operator.OP_CONJ:
                op = op.conj()
            operands.append(op)

    offset = None
    match num_inputs:
        case 2:
            offset = c
            assert d is None, "d cannot be set for binary contractions"
        case 3:
            assert c is not None, "c must be set for ternary contractions"
            offset = d
        case _:
            raise ValueError(f"Invalid number of inputs: {num_inputs}")

    # make sure operands are compatible (package, device_id)
    wrapped_operands = tensor_wrapper.wrap_operands(operands)
    package = wrapped_operands[0].name
    module = importlib.import_module(package)
    output = module.einsum(eq, *operands) * alpha
    if offset is not None:
        output = output + offset * beta
    return output


dtype_names = [
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
machine_epsilon_values = [np.finfo(dtype).eps for dtype in dtype_names]

rtol_mapper = dict(zip(dtype_names, [np.sqrt(m_eps) for m_eps in machine_epsilon_values], strict=False))

atol_mapper = dict(zip(dtype_names, [10 * m_eps for m_eps in machine_epsilon_values], strict=False))


def get_contraction_tolerance(dtype_name, compute_type):
    if compute_type == ComputeDesc.COMPUTE_32F() and dtype_name in {"float64", "complex128"}:
        return {"atol": atol_mapper["float32"], "rtol": rtol_mapper["float32"]}
    elif compute_type in {ComputeDesc.COMPUTE_16F(), ComputeDesc.COMPUTE_16BF()}:
        return {"atol": atol_mapper["float16"], "rtol": rtol_mapper["float16"]}
    else:
        tolerance = {"atol": atol_mapper[dtype_name], "rtol": rtol_mapper[dtype_name]}
        if compute_type in {ComputeDesc.COMPUTE_TF32(), ComputeDesc.COMPUTE_3XTF32()}:
            tolerance["rtol"] *= 100
            tolerance["atol"] *= 100
        return tolerance


def assert_all_close(a, b, rtol, atol):
    if isinstance(a, np.ndarray):
        return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif CP_NDARRAY is not None and isinstance(a, CP_NDARRAY):
        return cp.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    elif TORCH_TENSOR is not None and isinstance(a, TORCH_TENSOR):
        return torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"Unknown array type {a}")
