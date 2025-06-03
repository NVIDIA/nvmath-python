# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import sphinx.cmd.build


@contextlib.contextmanager
def os_cd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def test_docstrings():
    with os_cd("docs/sphinx"):
        ret = sphinx.cmd.build.main(
            ["-M", "doctest", ".", os.path.join("../..", "docs/_build/doctest"), "--tag", "exclude-nvmath-distributed"]
        )
        assert ret == 0


if __name__ == "__main__":
    test_docstrings()
