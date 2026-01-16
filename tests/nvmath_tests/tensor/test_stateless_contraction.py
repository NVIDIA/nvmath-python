# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import itertools
import logging

try:
    from cuda.core import system
except ImportError:
    from cuda.core.experimental import system

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import torch
except ImportError:
    torch = None

from contextlib import nullcontext

from nvmath.bindings import cutensor
from nvmath.internal import tensor_wrapper
from nvmath.memory import _MEMORY_MANAGER
from nvmath.tensor import binary_contraction, ExecutionCUDA, Operator, tensor_qualifiers_dtype

from nvmath_tests.helpers import use_stream, order_streams
from .utils.common_axes import Framework, ComputeType, BlockingOption, MemBackend
from .utils.check_helpers import get_contraction_tolerance, assert_all_close
from .utils.data import contraction_test_cases
from .utils.support_matrix import framework_backend_support, framework_type_support
from .utils.base_testers import run_stateless_impl, parse_operands, run_coefficients_test_impl
from .utils.support_matrix import compute_type_support
from .utils.axes_utils import is_complex
from .utils.input_fixtures import get_custom_stream

try:
    num_devices = system.get_num_devices()
except AttributeError:
    num_devices = system.num_devices


@pytest.mark.parametrize(
    (
        "test_case",
        "framework",
        "mem_backend",
        "dtype",
    ),
    [
        (
            test_case,
            framework,
            mem_backend,
            dtype,
        )
        for test_case in contraction_test_cases
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
        for dtype in framework_type_support[framework]
    ],
)
class TestStatelessContraction:
    @pytest.mark.parametrize("alpha", [False, 0, 0.3, 0.2 + 0.3j])
    @pytest.mark.parametrize("beta", [False, 0, 0.5, 0.4 + 0.5j])
    @pytest.mark.parametrize("use_offset", [False, True])
    def test_coefficients(self, seeder, alpha, beta, use_offset, test_case, framework, mem_backend, dtype):
        run_coefficients_test_impl(test_case, framework, mem_backend, dtype, "stateless", alpha, beta, use_offset)

    @pytest.mark.parametrize("offset_format", ["out", "new", False])
    def test_inplace_output(self, seeder, offset_format, test_case, framework, mem_backend, dtype):
        kwargs = {
            "alpha": 0.3,
            "beta": 0.5,
        }
        out = test_case.gen_random_output(framework, dtype, mem_backend)

        # Determine offset based on offset_format
        if offset_format == "out":
            # Offset is the same as output
            if test_case.num_inputs == 2:
                a, b, c, d, _ = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False, c=out, out=out)
            else:  # num_inputs == 3
                a, b, c, d, _ = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False, d=out, out=out)
        elif offset_format == "new":
            # Offset is a new tensor (different from output)
            offset = test_case.gen_random_output(framework, dtype, mem_backend)
            if test_case.num_inputs == 2:
                a, b, c, d, _ = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False, c=offset, out=out)
            else:  # num_inputs == 3
                a, b, c, d, _ = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False, d=offset, out=out)
        elif offset_format is False:
            # No offset
            a, b, c, d, _ = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False, out=out)
            kwargs["beta"] = None
        else:
            raise ValueError(f"Invalid offset_format: {offset_format}")

        run_stateless_impl(test_case, framework, mem_backend, dtype, a, b, c=c, d=d, out=out, **kwargs)

    def test_qualifiers(self, seeder, test_case, framework, mem_backend, dtype):
        for ops in itertools.product([Operator.OP_IDENTITY, Operator.OP_CONJ], repeat=test_case.num_inputs + 1):
            qualifiers = np.asarray(ops, dtype=tensor_qualifiers_dtype)
            if not is_complex(dtype) and any(op == Operator.OP_CONJ for op in qualifiers):
                context = pytest.raises(ValueError)
            elif qualifiers[test_case.num_inputs] != Operator.OP_IDENTITY:
                context = pytest.raises(ValueError)  # output operand must be the identity operator
            else:
                context = nullcontext()

            a, b, c, d, out = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False)
            run_stateless_impl(
                test_case, framework, mem_backend, dtype, a, b, c=c, d=d, out=out, context=context, qualifiers=qualifiers
            )

    def test_stream(self, seeder, test_case, framework, mem_backend, dtype):
        # for input operands generation
        s0 = get_custom_stream(framework, is_numpy_stream_oriented=True)
        # for execution on GPU (both reference and result)
        s1 = get_custom_stream(framework, is_numpy_stream_oriented=True)

        with use_stream(s0):
            a, b, c, d, out = parse_operands(test_case, framework, mem_backend, dtype, use_offset=True)

        order_streams(s0, s1)
        run_stateless_impl(test_case, framework, mem_backend, dtype, a, b, c=c, d=d, out=out, beta=0.6, stream=s1)

    @pytest.mark.parametrize("compute_type", ComputeType)
    def test_compute_type(self, seeder, compute_type, test_case, framework, mem_backend, dtype):
        if compute_type in compute_type_support[dtype]:
            context = nullcontext()
        else:
            context = pytest.raises(ValueError)
        compute_type = compute_type.value

        a, b, c, d, out = parse_operands(test_case, framework, mem_backend, dtype, use_offset=False)
        run_stateless_impl(
            test_case,
            framework,
            mem_backend,
            dtype,
            a,
            b,
            c=c,
            d=d,
            out=out,
            context=context,
            options={"compute_type": compute_type},
        )


@pytest.mark.parametrize(
    (
        "framework",
        "mem_backend",
    ),
    [
        (
            framework,
            mem_backend,
        )
        for framework in Framework.enabled()
        for mem_backend in framework_backend_support[framework]
    ],
)
class TestMiscellaneous:
    def _run_test(self, framework, mem_backend, *, execution=None, options=None):
        if isinstance(execution, ExecutionCUDA):
            device_id = execution.device_id
        else:
            device_id = execution.get("device_id", 0) if execution is not None else 0
        if framework == Framework.numpy:
            a = np.random.rand(10, 10)
        elif framework == Framework.cupy:
            with cp.cuda.Device(device_id):
                a = cp.random.rand(10, 10)
        elif framework == Framework.torch:
            if mem_backend == MemBackend.cuda:
                a = torch.rand(10, 10, device=f"cuda:{device_id}")
            else:
                a = torch.rand(10, 10, device="cpu")
        result = binary_contraction("ij,jk->ik", a, a, execution=execution, options=options)
        reference = a @ a
        tolerance = get_contraction_tolerance("float32", None)
        assert_all_close(result, reference, **tolerance)

    @pytest.mark.parametrize("device_id", range(num_devices))
    def test_execution_device_id(self, seeder, framework, mem_backend, device_id):
        self._run_test(framework, mem_backend, execution={"name": "cuda", "device_id": device_id})

    @pytest.mark.parametrize("memory_limit", [1024**2, "1GB", "60%"])
    def test_memory_limit(self, seeder, framework, mem_backend, memory_limit):
        self._run_test(framework, mem_backend, options={"memory_limit": memory_limit})

    def test_handle(self, seeder, framework, mem_backend):
        try:
            handle = cutensor.create()
            self._run_test(framework, mem_backend, options={"handle": handle})
        finally:
            cutensor.destroy(handle)

    @pytest.mark.parametrize("blocking", BlockingOption)
    def test_blocking(self, seeder, blocking, framework, mem_backend):
        self._run_test(framework, mem_backend, options={"blocking": blocking.value})

    def test_allocator(self, seeder, framework, mem_backend):
        framework_name = {
            Framework.cupy: "cupy",
            Framework.torch: "torch",
            Framework.numpy: "cuda",
        }[framework]
        tensor_wrapper.maybe_register_package(framework_name)
        BaseAllocatorClass = _MEMORY_MANAGER[framework_name]

        class MockAllocator(BaseAllocatorClass):
            def __init__(self, device_id, logger):
                super().__init__(device_id, logger)
                self.counter = 0

            def memalloc(self, size, *args, **kwargs):
                self.counter += 1
                return super().memalloc(size, *args, **kwargs)

        for cls in [BaseAllocatorClass, MockAllocator]:
            allocator = cls(0, logging.getLogger())
            self._run_test(framework, mem_backend, options={"allocator": allocator})
