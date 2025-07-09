# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Package-agnostic interface for key sparse tensor attributes.
"""

__all__ = ["PACKAGE_HELPER_MAP", "CupySparseFormatHelper", "ScipySparseFormatHelper", "TorchSparseFormatHelper"]

from abc import ABC, abstractmethod

import re


class SparseFormatHelper(ABC):
    @staticmethod
    @abstractmethod
    def sparse_format_name(tensor):
        raise NotImplementedError

    # Attribute name map: canonical name <> package name.
    # One for each named sparse format type.
    @staticmethod
    @abstractmethod
    def csr_attribute_names():
        raise NotImplementedError

    # A ctor for each named sparse format type.
    @staticmethod
    @abstractmethod
    def create_csr(shape, crow_indices, col_indices, values):
        raise NotImplementedError


class TorchSparseFormatHelper(SparseFormatHelper):
    @staticmethod
    def sparse_format_name(tensor):
        name = re.sub(r"^torch\.sparse_", "", str(tensor.layout)).upper()
        return name

    @staticmethod
    def csr_attribute_names():
        return {
            "crow_indices": lambda a: a.crow_indices(),
            "col_indices": lambda a: a.col_indices(),
            "values": lambda a: a.values(),
        }

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        # import torch
        # return torch.sparse_csr_tensor(crow_indices.tensor, col_indices.tensor,
        #                                values.tensor, shape, device=values.device_id)
        raise NotImplementedError


class CupySparseFormatHelper(SparseFormatHelper):
    @staticmethod
    def sparse_format_name(tensor):
        name = re.sub(r"_(array|matrix)$", "", type(tensor).__name__).upper()
        return name

    @staticmethod
    def csr_attribute_names():
        return {"crow_indices": lambda a: a.indptr, "col_indices": lambda a: a.indices, "values": lambda a: a.data}

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        raise NotImplementedError


class ScipySparseFormatHelper(SparseFormatHelper):
    @staticmethod
    def sparse_format_name(tensor):
        name = re.sub(r"_(array|matrix)$", "", type(tensor).__name__).upper()
        return name

    @staticmethod
    def csr_attribute_names():
        return {"crow_indices": lambda a: a.indptr, "col_indices": lambda a: a.indices, "values": lambda a: a.data}

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        raise NotImplementedError


PACKAGE_HELPER_MAP = {"cupyx": CupySparseFormatHelper, "scipy": ScipySparseFormatHelper, "torch": TorchSparseFormatHelper}
