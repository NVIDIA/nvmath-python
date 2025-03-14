# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


class Workspace:
    def __init__(self, handle, workspace):
        self._handle = handle
        self._workspace = workspace

    def __str__(self):
        return f"Workspace(handle={self.handle:#x}, workspace={self.workspace:#x})"

    @property
    def handle(self):
        return self._handle

    @property
    def workspace(self):
        return self._workspace
