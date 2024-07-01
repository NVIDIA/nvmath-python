# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

# The following configs are needed to deselect/ignore collected tests for
# various reasons, see pytest-dev/pytest#3730. In particular, this strategy
# is borrowed from https://github.com/pytest-dev/pytest/issues/3730#issuecomment-567142496.

from collections.abc import Iterable
import os


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "uncollect_if(*, func): function to unselect tests from parametrization"
    )


def pytest_collection_modifyitems(config, items):
    removed = []
    kept = []
    for item in items:
        is_removed = False
        m = item.get_closest_marker('uncollect_if')
        if m:
            funcs = m.kwargs['func']
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
