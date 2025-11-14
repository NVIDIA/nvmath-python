# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface classes to encapsulate low-level calls to get or set configuration information.
"""

__all__ = ["ContractionPlanPreference"]


import numpy as np

from nvmath.bindings import cutensor
from nvmath.internal import utils


class ContractionPlanPreference:
    """
    An interface to configure :meth:`nvmath.tensor.BinaryContraction.plan` and
    :meth:`nvmath.tensor.TernaryContraction.plan`. The
    current configuration can also be queried.
    """

    def __init__(self, contraction):
        """
        ctor for internal use only.
        """
        self._contraction = contraction
        self._handle = self._contraction.handle

        get_dtype = cutensor.get_plan_preference_attribute_dtype

        self._autotune_mode = np.zeros((1,), dtype=get_dtype(cutensor.PlanPreferenceAttribute.AUTOTUNE_MODE))
        self._cache_mode = np.zeros((1,), dtype=get_dtype(cutensor.PlanPreferenceAttribute.CACHE_MODE))
        self._incremental_count = np.zeros((1,), dtype=get_dtype(cutensor.PlanPreferenceAttribute.INCREMENTAL_COUNT))
        self._algo = np.zeros((1,), dtype=get_dtype(cutensor.PlanPreferenceAttribute.ALGO))
        self._kernel_rank = np.zeros((1,), dtype=get_dtype(cutensor.PlanPreferenceAttribute.KERNEL_RANK))
        self._jit = np.zeros((1,), dtype=get_dtype(cutensor.PlanPreferenceAttribute.JIT))

    def _check_valid_contraction_wrapper(self, *args, **kwargs):
        if not self._contraction.valid_state:
            raise RuntimeError("The ContractionPlanPreference object cannot be used after its contraction object is free'd.")

    @staticmethod
    def _get_scalar_attribute(contraction, name, attribute):
        """
        name      = cutensor PlanPreference enum for the attribute
        attribute = numpy ndarray object into which the value is stored by cutensornet
        """
        raise AttributeError("cuTensor does not support a getter for plan preference attributes.")

    @staticmethod
    def _set_scalar_attribute(contraction, name, attribute, value):
        """
        name      = cutensor PlanPreference enum for the attribute
        attribute = numpy ndarray object into which the value is stored
        value     = the value to set the the attribute to
        """
        assert contraction.plan_preference_ptr is not None, "Internal error"
        attribute[0] = value
        cutensor.plan_preference_set_attribute(
            contraction.handle, contraction.plan_preference_ptr, name, attribute.ctypes.data, attribute.dtype.itemsize
        )

    @property
    @utils.precondition(_check_valid_contraction_wrapper)
    def autotune_mode(self):
        """
        Query the autotune mode. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorautotunemode-t>`__
        for more information.
        """
        ContractionPlanPreference._get_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.AUTOTUNE_MODE, self._autotune_mode
        )
        return self._autotune_mode.item()

    @autotune_mode.setter
    @utils.precondition(_check_valid_contraction_wrapper)
    def autotune_mode(self, autotune_mode):
        """
        Set the autotune mode. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorautotunemode-t>`__
        for more information.

        Args:
            autotune_mode: (nvmath.tensor.ContractionAutotuneMode) The autotune mode.

        """
        ContractionPlanPreference._set_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.AUTOTUNE_MODE, self._autotune_mode, autotune_mode
        )

    @property
    @utils.precondition(_check_valid_contraction_wrapper)
    def cache_mode(self):
        """
        Query the cache mode. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorcachemode-t>`__
        for more information.
        """
        ContractionPlanPreference._get_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.CACHE_MODE, self._cache_mode
        )
        return self._cache_mode.item()

    @cache_mode.setter
    @utils.precondition(_check_valid_contraction_wrapper)
    def cache_mode(self, cache_mode):
        """
        Set the cache mode. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorcachemode-t>`__
        for more information.

        Args:
            cache_mode: (nvmath.tensor.ContractionCacheMode) The cache mode.

        """
        ContractionPlanPreference._set_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.CACHE_MODE, self._cache_mode, cache_mode
        )

    @property
    @utils.precondition(_check_valid_contraction_wrapper)
    def incremental_count(self):
        """
        Query the incremental count. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorincrementalcount-t>`__
        for more information.
        """
        ContractionPlanPreference._get_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.INCREMENTAL_COUNT, self._incremental_count
        )
        return self._incremental_count.item()

    @incremental_count.setter
    @utils.precondition(_check_valid_contraction_wrapper)
    def incremental_count(self, incremental_count):
        """
        Set the incremental count. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorincrementalcount-t>`__
        for more information.

        Args:
            incremental_count: The incremental count.

        """
        ContractionPlanPreference._set_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.INCREMENTAL_COUNT, self._incremental_count, incremental_count
        )

    @property
    @utils.precondition(_check_valid_contraction_wrapper)
    def algo(self):
        """
        Query the algo. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensoralgo-t>`__
        for more information.
        """
        ContractionPlanPreference._get_scalar_attribute(self._contraction, cutensor.PlanPreferenceAttribute.ALGO, self._algo)
        return self._algo.item()

    @algo.setter
    @utils.precondition(_check_valid_contraction_wrapper)
    def algo(self, algo):
        """
        Set the algo. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensoralgo-t>`__
        for more information.

        Args:
            algo: (nvmath.tensor.ContractionAlgo) The contraction algorithm.

        """
        ContractionPlanPreference._set_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.ALGO, self._algo, algo
        )

    @property
    @utils.precondition(_check_valid_contraction_wrapper)
    def kernel_rank(self):
        """
        Query the kernel rank. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorkernelrank-t>`__
        for more information.
        """
        ContractionPlanPreference._get_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.KERNEL_RANK, self._kernel_rank
        )
        return self._kernel_rank.item()

    @kernel_rank.setter
    @utils.precondition(_check_valid_contraction_wrapper)
    def kernel_rank(self, kernel_rank):
        """
        Set the kernel rank. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorkernelrank-t>`__
        for more information.

        Args:
            kernel_rank: The kernel rank.

        """
        ContractionPlanPreference._set_scalar_attribute(
            self._contraction, cutensor.PlanPreferenceAttribute.KERNEL_RANK, self._kernel_rank, kernel_rank
        )

    @property
    @utils.precondition(_check_valid_contraction_wrapper)
    def jit(self):
        """
        Query the jit compilation mode. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorjit-t>`__
        for more information.
        """
        ContractionPlanPreference._get_scalar_attribute(self._contraction, cutensor.PlanPreferenceAttribute.JIT, self._jit)
        return self._jit.item()

    @jit.setter
    @utils.precondition(_check_valid_contraction_wrapper)
    def jit(self, jit):
        """
        Set the jit compilation mode. See the
        `cuTensor documentation
        <https://docs.nvidia.com/cuda/cutensor/latest/api/types.html#cutensorjit-t>`__
        for more information.

        Args:
            jit: (nvmath.tensor.ContractionJitMode) The JIT compilation mode.

        """
        ContractionPlanPreference._set_scalar_attribute(self._contraction, cutensor.PlanPreferenceAttribute.JIT, self._jit, jit)
