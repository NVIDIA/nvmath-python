# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# The following configs are needed to deselect/ignore collected tests for
# various reasons, see pytest-dev/pytest#3730. In particular, this strategy
# is borrowed from https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496.

from collections.abc import Iterable
import datetime
import os

import hypothesis
import pytest

ci_phases = [
    hypothesis.Phase.explicit,
    # Skip reuse phase on CI because runners do not cache previous runs
    hypothesis.Phase.generate,
    hypothesis.Phase.target,
    hypothesis.Phase.shrink,
]
hypothesis.settings.register_profile(
    "nightly",
    deadline=datetime.timedelta(seconds=10),
    derandomize=False,
    max_examples=10_000,
    print_blob=True,
    verbosity=hypothesis.Verbosity.normal,
    phases=ci_phases,
)
hypothesis.settings.register_profile(
    "merge",
    deadline=datetime.timedelta(seconds=10),
    derandomize=False,
    max_examples=1_000,
    print_blob=True,
    verbosity=hypothesis.Verbosity.normal,
    phases=ci_phases,
)
hypothesis.settings.register_profile(
    "local",
    deadline=datetime.timedelta(seconds=10),
    derandomize=False,
    max_examples=100,
    phases=[
        hypothesis.Phase.explicit,
        hypothesis.Phase.reuse,
        hypothesis.Phase.generate,
        hypothesis.Phase.target,
        hypothesis.Phase.shrink,
        # https://github.com/HypothesisWorks/hypothesis/issues/4339
        # hypothesis.Phase.explain,
    ],
    print_blob=True,
    verbosity=hypothesis.Verbosity.normal,
)
if os.environ.get("NVMATH_NIGHTLY", default="").lower() == "true":
    hypothesis.settings.load_profile("nightly")
elif os.environ.get("CI", default="").lower() == "true":
    hypothesis.settings.load_profile("merge")
else:
    hypothesis.settings.load_profile("local")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "uncollect_if(*, func): function to unselect tests from parametrization",
    )


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        is_removed = False
        m = item.get_closest_marker("uncollect_if")
        if m:
            funcs = m.kwargs["func"]
            if not isinstance(funcs, Iterable):
                funcs = (funcs,)
            # loops over all deselect requirements
            for func in funcs:
                if func(**item.callspec.params):
                    removed.append(item)
                    is_removed = True
                    break
        if not is_removed:
            kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


try:
    import cupy

    _mempool = cupy.get_default_memory_pool()

    @pytest.fixture(autouse=True)
    def free_cupy_mempool():
        """Force the cupy mempool to release all memory after each test."""
        global _mempool
        yield
        _mempool.free_all_blocks()

except ModuleNotFoundError:
    # If cupy is not installed, then we don't need to do anything
    pass
