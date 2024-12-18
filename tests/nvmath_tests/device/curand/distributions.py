# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import scipy.stats as stats
import nvmath.device.random as R
from collections.abc import Callable
from .generators import *


class Distribution:
    def cdf(self, x):
        """
        Cumulative distribution function.
        """
        raise NotImplementedError

    def ppf(self):
        """
        Inverse CDF (Percent Point Function)
        """
        raise NotImplementedError

    def curand_variants(self) -> dict[tuple[str, int, Generator], Callable]:
        """
        A dictionary mapping (dtype, group size, generator) to curand distribution function.
        """
        raise NotImplementedError

    def _get_curand_function(self, dtype_name: str, group_size: int) -> Callable:
        """
        Finds curand distribution function for specific dtype and group size.
        """
        for variant, f in self.curand_variants().items():
            variant_dtype, variant_group_size, _ = variant
            if variant_dtype == dtype_name and variant_group_size == group_size:
                return f
        raise NotImplementedError

    def curand(self, dtype_name: str, group_size: int) -> tuple[Callable, tuple]:
        """
        Returns curand distribution function for specific dtype and group size, together
        with its extra arguments.
        """
        raise NotImplementedError

    def pretty(self):
        """
        Pretty-prints the distribution for pytest.
        """
        raise NotImplementedError


class ContinuousDistribution(Distribution):
    def is_discrete(self):
        return False


class DiscreteDistribution(Distribution):
    def is_discrete(self):
        return True


class Uniform(ContinuousDistribution):
    """
    Uniform(0, 1) distribution.
    """

    def cdf(self, x):
        return stats.uniform.cdf(x)

    def curand_variants(self):
        return {
            ("float", 1, XorwowGenerator()): R.uniform,
            ("float", 1, MrgGenerator()): R.uniform,
            ("float", 1, PhiloxGenerator()): R.uniform,
            ("float", 1, Sobol32Generator()): R.uniform,
            ("float", 1, ScrambledSobol32Generator()): R.uniform,
            ("float", 1, Sobol64Generator()): R.uniform,
            ("float", 1, ScrambledSobol64Generator()): R.uniform,
            ("double", 1, XorwowGenerator()): R.uniform_double,
            ("double", 1, MrgGenerator()): R.uniform_double,
            ("double", 1, PhiloxGenerator()): R.uniform_double,
            ("double", 1, Sobol32Generator()): R.uniform_double,
            ("double", 1, ScrambledSobol32Generator()): R.uniform_double,
            ("double", 1, Sobol64Generator()): R.uniform_double,
            ("double", 1, ScrambledSobol64Generator()): R.uniform_double,
            ("double", 2, PhiloxGenerator()): R.uniform2_double,
            ("float", 4, PhiloxGenerator()): R.uniform4,
        }

    def curand(self, dtype_name, group_size):
        return self._get_curand_function(dtype_name, group_size), ()

    def pretty(self):
        return "uniform(0,1)"


class Normal(ContinuousDistribution):
    """
    Normal(0, 1) distribution.
    """

    def cdf(self, x):
        return stats.norm.cdf(x)

    def curand_variants(self):
        return {
            ("float", 1, XorwowGenerator()): R.normal,
            ("float", 1, MrgGenerator()): R.normal,
            ("float", 1, PhiloxGenerator()): R.normal,
            ("float", 1, Sobol32Generator()): R.normal,
            ("float", 1, ScrambledSobol32Generator()): R.normal,
            ("float", 1, Sobol64Generator()): R.normal,
            ("float", 1, ScrambledSobol64Generator()): R.normal,
            ("double", 1, XorwowGenerator()): R.normal_double,
            ("double", 1, MrgGenerator()): R.normal_double,
            ("double", 1, PhiloxGenerator()): R.normal_double,
            ("double", 1, Sobol32Generator()): R.normal_double,
            ("double", 1, ScrambledSobol32Generator()): R.normal_double,
            ("double", 1, Sobol64Generator()): R.normal_double,
            ("double", 1, ScrambledSobol64Generator()): R.normal_double,
            ("float", 2, XorwowGenerator()): R.normal2,
            ("float", 2, MrgGenerator()): R.normal2,
            ("float", 2, PhiloxGenerator()): R.normal2,
            ("double", 2, XorwowGenerator()): R.normal2_double,
            ("double", 2, MrgGenerator()): R.normal2_double,
            ("double", 2, PhiloxGenerator()): R.normal2_double,
            ("float", 4, PhiloxGenerator()): R.normal4,
        }

    def curand(self, dtype_name, group_size):
        return self._get_curand_function(dtype_name, group_size), ()

    def pretty(self):
        return "normal(0,1)"


class LogNormal(ContinuousDistribution):
    """
    Log-normal distribution with configurable mean and stddev.
    """

    def __init__(self, mean=0, stddev=1):
        self.mean, self.stddev = mean, stddev

    def cdf(self, x):
        return stats.lognorm.cdf(x, s=self.stddev, scale=np.exp(self.mean))

    def curand_variants(self):
        return {
            ("float", 1, XorwowGenerator()): R.log_normal,
            ("float", 1, MrgGenerator()): R.log_normal,
            ("float", 1, PhiloxGenerator()): R.log_normal,
            ("float", 1, Sobol32Generator()): R.log_normal,
            ("float", 1, ScrambledSobol32Generator()): R.log_normal,
            ("float", 1, Sobol64Generator()): R.log_normal,
            ("float", 1, ScrambledSobol64Generator()): R.log_normal,
            ("double", 1, XorwowGenerator()): R.log_normal_double,
            ("double", 1, MrgGenerator()): R.log_normal_double,
            ("double", 1, PhiloxGenerator()): R.log_normal_double,
            ("double", 1, Sobol32Generator()): R.log_normal_double,
            ("double", 1, ScrambledSobol32Generator()): R.log_normal_double,
            ("double", 1, Sobol64Generator()): R.log_normal_double,
            ("double", 1, ScrambledSobol64Generator()): R.log_normal_double,
            ("float", 2, XorwowGenerator()): R.log_normal2,
            ("float", 2, MrgGenerator()): R.log_normal2,
            ("float", 2, PhiloxGenerator()): R.log_normal2,
            ("double", 2, XorwowGenerator()): R.log_normal2_double,
            ("double", 2, MrgGenerator()): R.log_normal2_double,
            ("double", 2, PhiloxGenerator()): R.log_normal2_double,
            ("float", 4, PhiloxGenerator()): R.log_normal4,
        }

    def curand(self, dtype_name, group_size):
        return self._get_curand_function(dtype_name, group_size), (
            self.mean,
            self.stddev,
        )

    def pretty(self):
        return f"lognormal({self.mean}, {self.stddev})"


class Poisson(DiscreteDistribution):
    """
    Poisson distribution.
    """

    def __init__(self, l):
        self.l = l

    def cdf(self, x):
        return stats.poisson.cdf(x, self.l)

    def ppf(self, x):
        return stats.poisson.ppf(x, self.l)

    def curand_variants(self):
        return {
            ("uint32", 1, XorwowGenerator()): R.poisson,
            ("uint32", 1, MrgGenerator()): R.poisson,
            ("uint32", 1, PhiloxGenerator()): R.poisson,
            ("uint32", 1, Sobol32Generator()): R.poisson,
            ("uint32", 1, ScrambledSobol32Generator()): R.poisson,
            ("uint32", 1, Sobol64Generator()): R.poisson,
            ("uint32", 1, ScrambledSobol64Generator()): R.poisson,
            ("uint32", 4, PhiloxGenerator()): R.poisson4,
        }

    def curand(self, dtype_name, group_size):
        return self._get_curand_function(dtype_name, group_size), (self.l,)

    def pretty(self):
        return f"poisson({self.l})"
