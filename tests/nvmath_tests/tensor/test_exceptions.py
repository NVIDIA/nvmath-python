# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

import nvmath
from nvmath.tensor._internal.cutensor_config_ifc import ContractionPlanPreference
from nvmath.tensor._internal.typemaps import get_supported_compute_types
from nvmath.tensor.contract import (
    ComputeDesc,
    ContractionOptions,
    binary_contraction,
)

try:
    import torch
except Exception:
    torch = None

from types import SimpleNamespace

torch_required = pytest.mark.skipif(
    torch is None,
    reason="PyTorch is required for this test",
)

DEFAULT_SHAPE = (2, 2)
DEFAULT_DTYPE = np.float32
# Shared numpy tensors reused across tests to reduce per-test boilerplate.
a = np.ones(DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)
b = np.ones(DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)
c = np.ones(DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)
d = np.ones(DEFAULT_SHAPE, dtype=DEFAULT_DTYPE)

if torch is not None:
    a_torch_cuda = torch.ones(DEFAULT_SHAPE, device="cuda", dtype=torch.float32)
    b_torch_cuda = torch.ones(DEFAULT_SHAPE, device="cuda", dtype=torch.float32)
    c_torch_cuda = torch.ones(DEFAULT_SHAPE, device="cuda", dtype=torch.float32)
    d_torch_cuda = torch.ones(DEFAULT_SHAPE, device="cuda", dtype=torch.float32)
else:
    a_torch_cuda = b_torch_cuda = c_torch_cuda = d_torch_cuda = None


def test_binary_contraction_beta_requires_c():
    expr = "ij,jk->ik"

    message = "beta can only be set if c is specified in a binary contraction"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, beta=0.5)


def test_binary_contraction_c_requires_beta():
    expr = "ij,jk->ik"

    message = "beta must be set when c is specified in a binary contraction"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, c=c)


def test_ternary_contraction_beta_requires_d():
    expr = "ij,jk,kl->il"

    message = "beta can only be set if d is specified in a ternary contraction"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.ternary_contraction(expr, a, b, c, beta=0.5)


def test_ternary_contraction_d_requires_beta():
    expr = "ij,jk,kl->il"

    message = "beta must be set when d is specified in a ternary contraction"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.ternary_contraction(expr, a, b, c, d=d)


@torch_required
def test_contraction_rejects_mixed_packages():
    expr = "ij,jk->ik"
    c = c_torch_cuda

    message = "operand has package 'torch' but expected 'numpy'"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, c=c, beta=1.0)


def test_contraction_invalid_compute_type():
    # expect a int but give a string
    expr = "ij,jk->ik"
    invalid_compute_type = "255"
    options = nvmath.tensor.ContractionOptions(compute_type=invalid_compute_type)

    message = f"Invalid compute type: {invalid_compute_type}"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, options=options)


def test_binary_contraction_invalid_qualifiers_length():
    expr = "ij,jk->ik"
    qualifiers = np.array([0, 0], dtype=np.int32)

    message = "The qualifiers must be a numpy array of length 3"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, qualifiers=qualifiers)


def test_ternary_contraction_invalid_qualifiers_length():
    expr = "ij,jk,kl->il"
    qualifiers = np.array([0, 0, 0], dtype=np.int32)

    message = "The qualifiers must be a numpy array of length 4"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.ternary_contraction(expr, a, b, c, qualifiers=qualifiers)


def test_binary_contraction_invalid_qualifier_operator():
    expr = "ij,jk->ik"
    qualifiers = np.array(
        [
            255,
            nvmath.tensor.Operator.OP_IDENTITY,
            nvmath.tensor.Operator.OP_IDENTITY,
        ],
        dtype=np.int32,
    )

    message = "Each operator must be a valid cuTensor operator"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, qualifiers=qualifiers)


def test_ternary_contraction_invalid_qualifier_operator():
    expr = "ij,jk,kl->il"
    qualifiers = np.array(
        [
            nvmath.tensor.Operator.OP_IDENTITY,
            255,
            nvmath.tensor.Operator.OP_IDENTITY,
            nvmath.tensor.Operator.OP_IDENTITY,
        ],
        dtype=np.int32,
    )

    message = "Each operator must be a valid cuTensor operator"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.ternary_contraction(expr, a, b, c, qualifiers=qualifiers)


@torch_required
def test_binary_contraction_out_requires_same_package():
    expr = "ij,jk->ik"
    out = d_torch_cuda

    message = "The output operand out must be a numpy tensor"
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, out=out)


@torch_required
def test_binary_contraction_out_requires_same_device():
    expr = "ij,jk->ik"
    a = a_torch_cuda
    b = b_torch_cuda
    out = torch.ones((2, 2), device="cpu", dtype=torch.float32)

    message = "The output operand out must be on the same device as the input operands."
    with pytest.raises(ValueError, match=message):
        nvmath.tensor.binary_contraction(expr, a, b, out=out)


def test_binary_contraction_execute_requires_operands_after_release():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        contraction.plan()
        contraction.release_operands()

        message = "cannot be performed after the operands have been released"
        with pytest.raises(RuntimeError, match=message):
            contraction.execute()


def test_ternary_contraction_execute_requires_operands_after_release():
    expr = "ij,jk,kl->il"

    with nvmath.tensor.TernaryContraction(expr, a, b, c) as contraction:
        contraction.plan()
        contraction.release_operands()

        message = "cannot be performed after the operands have been released"
        with pytest.raises(RuntimeError, match=message):
            contraction.execute()


def test_binary_contraction_reset_operands_requires_all_args_after_reset():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b, out=d) as contraction:
        contraction.plan()
        contraction.release_operands()

        message = "all operands must be provided"
        # only provide a subset but not all of the operands that were provided
        # during initialization, so that the error should always be raised
        # with the same message
        for kwargs in [{"a": a, "b": b}, {"a": a}, {"b": b}]:
            with pytest.raises(ValueError, match=message):
                contraction.reset_operands(**kwargs)


def test_binary_contraction_execute_requires_plan():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        message = re.escape("Execution cannot be performed before plan() has been called.")
        with pytest.raises(RuntimeError, match=message):
            contraction.execute()


def test_ternary_contraction_execute_requires_plan():
    expr = "ij,jk,kl->il"

    with nvmath.tensor.TernaryContraction(expr, a, b, c) as contraction:
        message = re.escape("Execution cannot be performed before plan() has been called.")
        with pytest.raises(RuntimeError, match=message):
            contraction.execute()


def test_binary_contraction_allocator_invalid_alloc_interface():
    class InvalidAllocator:
        def memalloc(self, size, stream):
            pass

    expr = "ijk,jkl->il"
    a = np.ones((8, 8, 8), dtype=np.float32)
    b = np.ones((8, 8, 8), dtype=np.float32)
    options = nvmath.tensor.ContractionOptions(allocator=InvalidAllocator())

    with nvmath.tensor.BinaryContraction(expr, a, b, options=options) as contraction:
        contraction.plan()
        if contraction.workspace_size == 0:
            contraction.workspace_size = 1024

        message = "The method 'memalloc' in the allocator object must conform to the interface"
        with pytest.raises(TypeError, match=message):
            contraction.execute()


def test_binary_contraction_release_operands_clears_provided_out():
    expr = "ij,jk->ik"
    out = d

    with nvmath.tensor.BinaryContraction(expr, a, b, out=out) as contraction:
        contraction.release_operands()
        assert contraction.out_return.tensor is None


def test_binary_contraction_reset_operands_rejects_unspecified_c():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        message = re.escape(
            "operand c was not specified during the initialization "
            "of the ElementaryContraction object and therefore can not be reset "
            "to a concrete tensor."
        )
        with pytest.raises(ValueError, match=message):
            contraction.reset_operands(c=c)


@torch_required
def test_binary_contraction_reset_operands_requires_same_package():
    expr = "ij,jk->ik"
    new_a = torch.ones((2, 2), dtype=torch.float32)

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        message = "The operand a must be a numpy tensor"
        with pytest.raises(ValueError, match=message):
            contraction.reset_operands(a=new_a)


@torch_required
def test_binary_contraction_reset_operands_requires_same_device():
    expr = "ij,jk->ik"
    a = a_torch_cuda
    b = b_torch_cuda
    new_a = torch.ones((2, 2), device="cpu", dtype=torch.float32)

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        message = (
            "The operand a must be on the same device as the operands provided during the initialization of "
            "the ElementaryContraction object."
        )
        with pytest.raises(ValueError, match=message):
            contraction.reset_operands(a=new_a)


def test_binary_contraction_reset_operands_requires_same_dtype():
    expr = "ij,jk->ik"
    new_a = np.ones((2, 2), dtype=np.float64)

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        message = (
            "The operand a must have the same dtype as the one specified during the initialization of the "
            "ElementaryContraction object."
        )
        with pytest.raises(ValueError, match=message):
            contraction.reset_operands(a=new_a)


@torch_required
def test_binary_contraction_reset_operands_requires_compatible_pointer_alignment():
    expr = "ij,jk->ik"

    base_tensor = torch.empty(4, device="cuda", dtype=torch.float32)
    base_storage = base_tensor.untyped_storage()
    aligned = torch.empty((0,), device="cuda", dtype=torch.float32)
    aligned.set_(base_storage, 0, (2, 2), (2, 1))

    misaligned_tensor = torch.empty(5, device="cuda", dtype=torch.float32)
    misaligned_storage = misaligned_tensor.untyped_storage()
    misaligned = torch.empty((0,), device="cuda", dtype=torch.float32)
    misaligned.set_(misaligned_storage, 1, (2, 2), (2, 1))

    b = b_torch_cuda

    with nvmath.tensor.BinaryContraction(expr, aligned, b) as contraction:
        message = (
            "The pointer alignment of the operand a must be the same or a multiple of the corresponding "
            "pointer alignment specified during the initialization of the BinaryContraction object."
        )
        with pytest.raises(ValueError, match=message):
            contraction.reset_operands(a=misaligned)


@torch_required
def test_ternary_contraction_reset_operands_requires_compatible_pointer_alignment():
    expr = "ij,jk,kl->il"

    base_tensor = torch.empty(4, device="cuda", dtype=torch.float32)
    base_storage = base_tensor.untyped_storage()
    aligned = torch.empty((0,), device="cuda", dtype=torch.float32)
    aligned.set_(base_storage, 0, (2, 2), (2, 1))

    misaligned_tensor = torch.empty(5, device="cuda", dtype=torch.float32)
    misaligned_storage = misaligned_tensor.untyped_storage()
    misaligned = torch.empty((0,), device="cuda", dtype=torch.float32)
    misaligned.set_(misaligned_storage, 1, (2, 2), (2, 1))

    b = b_torch_cuda
    c = c_torch_cuda

    with nvmath.tensor.TernaryContraction(expr, aligned, b, c) as contraction:
        message = (
            "The pointer alignment of the operand a must be the same or a multiple of the corresponding "
            "pointer alignment specified during the initialization of the TernaryContraction object."
        )
        with pytest.raises(ValueError, match=message):
            contraction.reset_operands(a=misaligned)


@torch_required
def test_binary_contraction_reset_operands_reuses_internal_out_on_device_mismatch():
    expr = "ij,jk->ik"
    a = torch.ones((2, 2), device="cpu", dtype=torch.float32)
    b = torch.ones((2, 2), device="cpu", dtype=torch.float32)
    out = torch.ones((2, 2), device="cpu", dtype=torch.float32)
    execution = nvmath.tensor.ExecutionCUDA(device_id=torch.cuda.current_device())

    with nvmath.tensor.BinaryContraction(expr, a, b, out=out, execution=execution) as contraction:
        original_internal_out = contraction.out
        new_out = torch.zeros((2, 2), device="cpu", dtype=torch.float32)

        old_out_return = contraction.out_return

        contraction.reset_operands(a=a, b=b, out=new_out)

        assert contraction.out is original_internal_out
        assert contraction.out_return is not old_out_return


def test_binary_contraction_reset_operands_rejects_d():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        message = re.escape("Internal Error: For pairwise contractions, d can not be set.")
        with pytest.raises(RuntimeError, match=message):
            super(nvmath.tensor.BinaryContraction, contraction).reset_operands(d=d)


def test_binary_contraction_execute_requires_beta_when_c_specified():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b, c=c) as contraction:
        contraction.plan()
        message = "beta must be set when c is specified in a binary contraction"
        with pytest.raises(ValueError, match=message):
            contraction.execute()


def test_binary_contraction_execute_rejects_beta_without_c():
    expr = "ij,jk->ik"

    with nvmath.tensor.BinaryContraction(expr, a, b) as contraction:
        contraction.plan()
        message = "For binary contraction, beta can only be set if c is specified"
        with pytest.raises(ValueError, match=message):
            contraction.execute(beta=0.5)


def test_ternary_contraction_execute_requires_beta_when_d_specified():
    expr = "ij,jk,kl->il"

    with nvmath.tensor.TernaryContraction(expr, a, b, c, d=d) as contraction:
        contraction.plan()
        message = "beta must be set when d is specified in a ternary contraction"
        with pytest.raises(ValueError, match=message):
            contraction.execute()


def test_ternary_contraction_execute_rejects_beta_without_d():
    expr = "ij,jk,kl->il"

    with nvmath.tensor.TernaryContraction(expr, a, b, c) as contraction:
        contraction.plan()
        message = "For ternary contraction, beta can only be set if d is specified"
        with pytest.raises(ValueError, match=message):
            contraction.execute(beta=0.5)


def test_binary_contraction_rejects_numpy_int_dtype():
    a = np.ones((2, 2), dtype=np.int32)
    b = np.ones_like(a)

    with pytest.raises(ValueError, match="Invalid data type"):
        binary_contraction("ij,jk->ik", a, b)


def test_binary_contraction_rejects_incompatible_compute_type():
    a = np.ones((2, 2), dtype=np.float64)
    b = np.ones((2, 2), dtype=np.float64)
    options = ContractionOptions(compute_type=ComputeDesc.COMPUTE_3XTF32())

    with pytest.raises(ValueError, match="Invalid compute type"):
        binary_contraction("ij,jk->ik", a, b, options=options)


def test_typemap_invalid_dtype_errors():
    invalid_dtype = "float256"

    with pytest.raises(ValueError, match="Invalid data type"):
        get_supported_compute_types(invalid_dtype)


def test_plan_preference_rejects_invalid_contraction():
    plan_pref = ContractionPlanPreference(SimpleNamespace(valid_state=False, handle=None))

    error_message = "The ContractionPlanPreference object cannot be used after its contraction object is free'd."

    with pytest.raises(RuntimeError, match=re.escape(error_message)):
        _ = plan_pref.autotune_mode
