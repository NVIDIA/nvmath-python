# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import itertools
import logging

import cuda.core.experimental as ccx
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    import torch
except ImportError:
    torch = None

from nvmath.bindings import cutensor
from nvmath.internal import tensor_wrapper
from nvmath.memory import _MEMORY_MANAGER
from nvmath.tensor import binary_contraction, ExecutionCUDA, Operator, tensor_qualifiers_dtype

from .utils.common_axes import Framework, ComputeType, BlockingOption, MemBackend
from .utils.check_helpers import get_contraction_tolerance, assert_all_close
from .utils.data import contraction_test_cases
from .utils.support_matrix import framework_backend_support, framework_type_support
from .utils.base_testers import BaseStatelessTester


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
class TestStatelessContraction(BaseStatelessTester):
    @pytest.mark.parametrize("alpha", [False, 0, 0.3, 0.2 + 0.3j])
    @pytest.mark.parametrize("beta", [False, 0, 0.5, 0.4 + 0.5j])
    @pytest.mark.parametrize("use_offset", [False, True])
    def test_coefficients(self, alpha, beta, use_offset, test_case, framework, mem_backend, dtype):
        self._test_coefficients(alpha, beta, use_offset, test_case, framework, mem_backend, dtype)

    @pytest.mark.parametrize("offset_format", ["out", "new", False])
    def test_inplace_output(self, offset_format, test_case, framework, mem_backend, dtype):
        self._test_inplace_output(offset_format, test_case, framework, mem_backend, dtype)

    def test_qualifiers(self, test_case, framework, mem_backend, dtype):
        for ops in itertools.product([Operator.OP_IDENTITY, Operator.OP_CONJ], repeat=test_case.num_inputs + 1):
            qualifiers = np.asarray(ops, dtype=tensor_qualifiers_dtype)
            self._test_qualifiers(qualifiers, test_case, framework, mem_backend, dtype)

    @pytest.mark.parametrize("stream", [None, True])
    def test_stream(self, stream, test_case, framework, mem_backend, dtype):
        self.run_test(test_case, framework, mem_backend, dtype, 13, use_offset=True, beta=0.6, stream=stream)

    @pytest.mark.parametrize("compute_type", ComputeType)
    def test_compute_type(self, compute_type, test_case, framework, mem_backend, dtype):
        self._test_compute_type(compute_type, test_case, framework, mem_backend, dtype)


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

    @pytest.mark.parametrize("device_id", range(ccx.system.num_devices))
    def test_execution_device_id(self, framework, mem_backend, device_id):
        self._run_test(framework, mem_backend, execution={"name": "cuda", "device_id": device_id})

    @pytest.mark.parametrize("memory_limit", [1024**2, "1GB", "60%"])
    def test_memory_limit(self, framework, mem_backend, memory_limit):
        self._run_test(framework, mem_backend, options={"memory_limit": memory_limit})

    def test_handle(self, framework, mem_backend):
        try:
            handle = cutensor.create()
            self._run_test(framework, mem_backend, options={"handle": handle})
        finally:
            cutensor.destroy(handle)

    @pytest.mark.parametrize("blocking", BlockingOption)
    def test_blocking(self, blocking, framework, mem_backend):
        self._run_test(framework, mem_backend, options={"blocking": blocking.value})

    def test_allocator(self, framework, mem_backend):
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
