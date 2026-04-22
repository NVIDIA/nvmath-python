# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import pytest

try:
    import cupy as cp
    import cupyx.scipy.sparse as sp

    HAS_CUPY_SPARSE = True
except ImportError:
    HAS_CUPY_SPARSE = False

from nvmath.bindings import cudss
from nvmath.sparse.advanced import (
    DirectSolverFactorizationConfig,
    DirectSolverFactorizationPreferences,
    DirectSolverPlanConfig,
    DirectSolverPlanPreferences,
    DirectSolverSolutionConfig,
    DirectSolverSolutionPreferences,
    direct_solver,
)


@pytest.mark.parametrize(
    ("preferences_class", "config_class"),
    [
        (DirectSolverPlanPreferences, DirectSolverPlanConfig),
        (DirectSolverFactorizationPreferences, DirectSolverFactorizationConfig),
        (DirectSolverSolutionPreferences, DirectSolverSolutionConfig),
    ],
)
def test_preferences_attributes_match_config(preferences_class, config_class):
    # Verify that the public attributes of the preferences and config classes match.
    fields_preferences = [field.name for field in fields(preferences_class)]
    fields_config = []
    for field in vars(config_class):
        if not field.startswith("_"):
            fields_config.append(field)
    assert set(fields_preferences) == set(fields_config)


@pytest.mark.skipif(HAS_CUPY_SPARSE is False, reason="cupy.sparse is not available")
class TestPreferences:
    # NOTE: cuDSS contains lots of invalid configuration combos,
    # therefore here we only test certain input configurations and formats
    # are functional.

    def setup_class(self):
        n = 16
        # Prepare sample input data.
        # Create a diagonally-dominant random CSR matrix.
        self.a = sp.random(n, n, density=0.5, format="csr", dtype="float64")
        self.a += sp.diags([2.0] * n, format="csr", dtype="float64")

        # Create the RHS, which can be a matrix or vector in column-major layout.
        self.b = cp.ones((n, 2), order="F")
        # Compute the solution with default settings.
        self.x_default = direct_solver(self.a, self.b)

    def _verify_with_preferences(self, **kwargs):
        # Verify that the solution is all close to the default solution.
        # Ideally the reference solution should be computed using the stateful API
        # and setting the corresponding preferences. However it seems cuDSS does not
        # guarantee the same output for the same input and preferences.
        # Therefore we only verify that the solution is all close to the default solution.
        out = direct_solver(self.a, self.b, **kwargs)
        cp.testing.assert_allclose(self.x_default, out)

    def test_plan_preference(self):
        plan_preferences_dict = {
            "use_matching": True,
            "matching_algorithm": cudss.AlgType.ALG_1,
        }
        plan_preferences_object = DirectSolverPlanPreferences(**plan_preferences_dict)

        self._verify_with_preferences(plan_preferences=plan_preferences_dict)
        self._verify_with_preferences(plan_preferences=plan_preferences_object)

    def test_factorization_preference(self):
        factorization_preferences_dict = {
            "factorization_algorithm": cudss.AlgType.ALG_1,
        }
        factorization_preferences_object = DirectSolverFactorizationPreferences(**factorization_preferences_dict)

        self._verify_with_preferences(factorization_preferences=factorization_preferences_dict)
        self._verify_with_preferences(factorization_preferences=factorization_preferences_object)

    def test_solution_preference(self):
        solution_preferences_dict = {
            "ir_num_steps": 10,
        }
        solution_preferences_object = DirectSolverSolutionPreferences(**solution_preferences_dict)

        self._verify_with_preferences(solution_preferences=solution_preferences_dict)
        self._verify_with_preferences(solution_preferences=solution_preferences_object)

    @pytest.mark.parametrize("preference_type", ["plan_preferences", "factorization_preferences", "solution_preferences"])
    def test_invalid_preferences(self, preference_type):
        kwargs = {preference_type: {"invalid_arg": 100}}

        with pytest.raises(TypeError, match=r".* got an unexpected keyword argument 'invalid_arg'"):
            direct_solver(self.a, self.b, **kwargs)
