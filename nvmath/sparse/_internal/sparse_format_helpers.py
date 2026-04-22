# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Package-agnostic interface for key sparse tensor attributes.
"""

__all__ = ["PACKAGE_HELPER_MAP", "CupySparseFormatHelper", "ScipySparseFormatHelper", "TorchSparseFormatHelper"]

import re
from abc import ABC, abstractmethod


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

    @staticmethod
    @abstractmethod
    def bsr_attribute_names():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def csc_attribute_names():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def coo_attribute_names():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def dia_attribute_names():
        raise NotImplementedError

    # A ctor for each named sparse format type.
    @staticmethod
    @abstractmethod
    def create_bsr(shape, block_size, crow_indices, col_indices, values):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_bsc(shape, block_size, ccol_indices, row_indices, values):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_csr(shape, crow_indices, col_indices, values):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_csc(shape, ccol_indices, row_indices, values):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_coo(shape, indices, values):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_dia(shape, offsets, values):
        raise NotImplementedError


class TorchSparseFormatHelper(SparseFormatHelper):
    @staticmethod
    def sparse_format_name(tensor):
        name = re.sub(r"^torch\.sparse_", "", str(tensor.layout)).upper()
        return name

    @staticmethod
    def csr_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": TorchSparseFormatHelper,
            "num_sparse_dim": lambda a: a.sparse_dim(),
            "num_dense_dim": lambda a: a.dense_dim(),
            # Format-specific attributes.
            "crow_indices": lambda a: a.crow_indices(),
            "col_indices": lambda a: a.col_indices(),
            "values": lambda a: a.values(),
            # Checker attributes.
            "has_sorted_indices": lambda a: True,
            "is_coalesced": lambda a: True,
        }

    @staticmethod
    def bsr_attribute_names():
        # TODO: improve the logic for block order/layout. Can the smallest block stride
        # be greater than 1?
        return {
            # General sparse attributes.
            "sparse_format_helper": TorchSparseFormatHelper,
            "num_sparse_dim": lambda a: a.sparse_dim(),
            "num_dense_dim": lambda a: a.dense_dim(),
            # Format-specific attributes.
            "block_size": lambda a: a.values().shape[-2:],
            "block_order": lambda a: "right" if a.values().stride()[-1] == 1 else "left",
            "crow_indices": lambda a: a.crow_indices(),
            "col_indices": lambda a: a.col_indices(),
            "values": lambda a: a.values(),
            # Checker attributes.
            "has_sorted_indices": lambda a: True,
            "is_coalesced": lambda a: True,
        }

    @staticmethod
    def bsc_attribute_names():
        # TODO: improve the logic for block order/layout. Can the smallest block stride
        # be greater than 1?
        return {
            # General sparse attributes.
            "sparse_format_helper": TorchSparseFormatHelper,
            "num_sparse_dim": lambda a: a.sparse_dim(),
            "num_dense_dim": lambda a: a.dense_dim(),
            # Format-specific attributes.
            "block_size": lambda a: a.values().shape[-2:],
            "block_order": lambda a: "right" if a.values().stride()[-1] == 1 else "left",
            "ccol_indices": lambda a: a.ccol_indices(),
            "row_indices": lambda a: a.row_indices(),
            "values": lambda a: a.values(),
            # Checker attributes.
            "has_sorted_indices": lambda a: True,
            "is_coalesced": lambda a: True,
        }

    @staticmethod
    def csc_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": TorchSparseFormatHelper,
            "num_sparse_dim": lambda a: a.sparse_dim(),
            "num_dense_dim": lambda a: a.dense_dim(),
            # Format-specific attributes.
            "ccol_indices": lambda a: a.ccol_indices(),
            "row_indices": lambda a: a.row_indices(),
            "values": lambda a: a.values(),
            # Checker attributes.
            "has_sorted_indices": lambda a: True,
            "is_coalesced": lambda a: True,
        }

    @staticmethod
    def coo_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": TorchSparseFormatHelper,
            "num_sparse_dim": lambda a: a.sparse_dim(),
            "num_dense_dim": lambda a: a.dense_dim(),
            # Format-specific attributes.
            "indices": lambda a: [a.indices()[i] for i in range(len(a.shape) - a.dense_dim())],
            "values": lambda a: a.values(),
            # Checker attributes.
            "is_coalesced": lambda a: a.is_coalesced(),
        }

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        import torch

        csr = torch.sparse_csr_tensor(crow_indices.tensor, col_indices.tensor, values.tensor, shape, device=values.device_id)

        return csr

    @staticmethod
    def create_bsr(shape, block_size, crow_indices, col_indices, values):
        import torch

        bsr = torch.sparse_bsr_tensor(crow_indices.tensor, col_indices.tensor, values.tensor, shape)

        return bsr

    @staticmethod
    def create_bsc(shape, block_size, ccol_indices, row_indices, values):
        import torch

        bsr = torch.sparse_bsc_tensor(ccol_indices.tensor, row_indices.tensor, values.tensor, shape)

        return bsr

    @staticmethod
    def create_csc(shape, ccol_indices, row_indices, values):
        import torch

        csc = torch.sparse_csc_tensor(ccol_indices.tensor, row_indices.tensor, values.tensor, shape, device=values.device_id)

        return csc

    @staticmethod
    def create_coo(shape, indices, values):
        import torch

        base = indices[0].tensor._base
        # TODO: we should assert base exists and raise an error instead of stacking?
        # if base is None:
        #    num_dimensions = len(shape)
        #    base = torch.stack([indices[d].tensor for d in range(num_dimensions))], axis=0)

        coo = torch.sparse_coo_tensor(torch.as_tensor(base), torch.as_tensor(values.tensor), shape, is_coalesced=True)

        return coo

    @staticmethod
    def create_dia(shape, offsets, values):
        raise NotImplementedError


class CupySparseFormatHelper(SparseFormatHelper):
    @staticmethod
    def sparse_format_name(tensor):
        name = re.sub(r"_(array|matrix)$", "", type(tensor).__name__).upper()
        return name

    @staticmethod
    def csr_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": CupySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "crow_indices": lambda a: a.indptr,
            "col_indices": lambda a: a.indices,
            "values": lambda a: a.data,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.has_sorted_indices,
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def csc_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": CupySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "ccol_indices": lambda a: a.indptr,
            "row_indices": lambda a: a.indices,
            "values": lambda a: a.data,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.has_sorted_indices,
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def coo_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": CupySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "indices": lambda a: [a.row, a.col],
            "values": lambda a: a.data,
            # Checker attributes.
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def dia_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": CupySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "offsets": lambda a: a.offsets,
            "values": lambda a: a.data,
        }

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        assert len(shape) == 2, "Internal error."

        import cupyx.scipy.sparse as sp

        csr = sp.csr_matrix((values.tensor, col_indices.tensor, crow_indices.tensor), shape=shape, copy=False)

        return csr

    @staticmethod
    def create_csc(shape, ccol_indices, row_indices, values):
        assert len(shape) == 2, "Internal error."

        import cupyx.scipy.sparse as sp

        csc = sp.csc_matrix((values.tensor, row_indices.tensor, ccol_indices.tensor), shape=shape, copy=False)

        return csc

    @staticmethod
    def create_coo(shape, indices, values):
        assert len(shape) == 2, "Internal error."

        import cupyx.scipy.sparse as sp

        coo = sp.coo_matrix((values.tensor, (indices[0].tensor, indices[1].tensor)), shape=shape, copy=False)

        return coo

    @staticmethod
    def create_dia(shape, offsets, values):
        assert len(shape) == 2, "Internal error."

        import cupyx.scipy.sparse as sp

        dia = sp.dia_matrix((values.tensor, offsets.tensor), shape=shape, copy=False)

        return dia


class ScipySparseFormatHelper(SparseFormatHelper):
    @staticmethod
    def sparse_format_name(tensor):
        name = re.sub(r"_(array|matrix)$", "", type(tensor).__name__).upper()
        return name

    @staticmethod
    def bsr_attribute_names():
        # TODO: improve the logic for block order/layout. Can the smallest block stride
        # be greater than 1?
        return {
            # General sparse attributes.
            "sparse_format_helper": ScipySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "block_size": lambda a: a.blocksize,
            "block_order": lambda a: "right" if a.data.strides[-1] == a.data.itemsize else "left",
            "crow_indices": lambda a: a.indptr,
            "col_indices": lambda a: a.indices,
            "values": lambda a: a.data,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.has_sorted_indices,
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def csr_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": ScipySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "crow_indices": lambda a: a.indptr,
            "col_indices": lambda a: a.indices,
            "values": lambda a: a.data,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.has_sorted_indices,
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def csc_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": ScipySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "ccol_indices": lambda a: a.indptr,
            "row_indices": lambda a: a.indices,
            "values": lambda a: a.data,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.has_sorted_indices,
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def coo_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": ScipySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "indices": lambda a: [a.row, a.col],
            "values": lambda a: a.data,
            # Checker attributes.
            "is_coalesced": lambda a: a.has_canonical_format,
        }

    @staticmethod
    def dia_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": ScipySparseFormatHelper,
            "num_sparse_dim": lambda a: len(a.shape),
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "offsets": lambda a: a.offsets,
            "values": lambda a: a.data,
        }

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        assert len(shape) == 2, "Internal error."

        import scipy.sparse as sp

        csr = sp.csr_array((values.tensor, col_indices.tensor, crow_indices.tensor), shape=shape, copy=False)

        return csr

    @staticmethod
    def create_bsr(shape, block_size, crow_indices, col_indices, values):
        assert len(shape) == 2, "Internal error."

        import scipy.sparse as sp

        bsr = sp.bsr_array((values.tensor, col_indices.tensor, crow_indices.tensor), shape=shape, copy=False)

        return bsr

    @staticmethod
    def create_csc(shape, ccol_indices, row_indices, values):
        assert len(shape) == 2, "Internal error."

        import scipy.sparse as sp

        csc = sp.csc_array((values.tensor, row_indices.tensor, ccol_indices.tensor), shape=shape, copy=False)

        return csc

    @staticmethod
    def create_coo(shape, indices, values):
        assert len(shape) == 2, "Internal error."

        import scipy.sparse as sp

        coo = sp.coo_matrix((values.tensor, (indices[0].tensor, indices[1].tensor)), shape=shape, copy=False)

        return coo

    @staticmethod
    def create_dia(shape, offsets, values):
        assert len(shape) == 2, "Internal error."

        import scipy.sparse as sp

        dia = sp.dia_array((values.tensor, offsets.tensor), shape=shape, copy=False)

        return dia


class USTFormatHelper(SparseFormatHelper):
    """
    CSR is really UST 3D BatchedCSR.
    """

    # TODO: add more formats for canonical names.
    NAME_EXPRESSIONS = {
        r"BSR(Left|Right)(d)?\d+x\d+": "BSR",
        r"BSC(Left|Right)(d)?\d+x\d+": "BSC",
        r"COO(\d+)?": "COO",
        r"(Batched)?CSR(\d+)?": "CSR",
        r"CSC(\d+)?": "CSC",
    }

    @staticmethod
    def sparse_format_name(tensor):
        name = tensor.tensor_format.name
        # We need to get the canonical name for certain formats.
        # TODO: Can we handle this in TensorFormat, or is there a better/more robust way?
        for expr in USTFormatHelper.NAME_EXPRESSIONS:
            if re.match(expr, name):
                name = USTFormatHelper.NAME_EXPRESSIONS[expr]
        return name

    # TODO: check generalization for batched tensors.
    @staticmethod
    def csr_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": USTFormatHelper,
            "num_sparse_dim": lambda a: 2,
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "crow_indices": lambda a: a.pos(a.tensor_format.num_dimensions - 1),
            "col_indices": lambda a: a.crd(a.tensor_format.num_dimensions - 1),
            "values": lambda a: a.val,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.tensor_format.is_ordered,
            "is_coalesced": lambda a: a.tensor_format.is_ordered and a.tensor_format.is_unique,
        }

    @staticmethod
    def bsr_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": USTFormatHelper,
            "num_sparse_dim": lambda a: 2,
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "block_size": lambda a: a.levels[-2:],
            "block_order": lambda a: "right" if "Right" in a.tensor_format.name else "left",
            "crow_indices": lambda a: a.pos(a.tensor_format.num_dimensions - 1),
            "col_indices": lambda a: a.crd(a.tensor_format.num_dimensions - 1),
            "values": lambda a: a.val,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.tensor_format.is_ordered,
            "is_coalesced": lambda a: a.tensor_format.is_ordered and a.tensor_format.is_unique,
        }

    @staticmethod
    def csc_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": USTFormatHelper,
            "num_sparse_dim": lambda a: 2,
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "ccol_indices": lambda a: a.pos(a.tensor_format.num_dimensions - 1),
            "row_indices": lambda a: a.crd(a.tensor_format.num_dimensions - 1),
            "values": lambda a: a.val,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.tensor_format.is_ordered,
            "is_coalesced": lambda a: a.tensor_format.is_ordered and a.tensor_format.is_unique,
        }

    @staticmethod
    def bsc_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": USTFormatHelper,
            "num_sparse_dim": lambda a: 2,
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "block_size": lambda a: a.levels[-2:],
            "block_order": lambda a: "right" if "Right" in a.tensor_format.name else "left",
            "ccol_indices": lambda a: a.pos(a.tensor_format.num_dimensions - 1),
            "row_indices": lambda a: a.crd(a.tensor_format.num_dimensions - 1),
            "values": lambda a: a.val,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.tensor_format.is_ordered,
            "is_coalesced": lambda a: a.tensor_format.is_ordered and a.tensor_format.is_unique,
        }

    @staticmethod
    def coo_attribute_names():
        return {
            # General sparse attributes.
            "sparse_format_helper": USTFormatHelper,
            "num_sparse_dim": lambda a: 2,
            "num_dense_dim": lambda a: 0,
            # Format-specific attributes.
            "indices": lambda a: [a.crd(i) for i in range(a.tensor_format.num_dimensions)],
            "values": lambda a: a.val,
            # Checker attributes.
            "has_sorted_indices": lambda a: a.tensor_format.is_ordered,
            "is_coalesced": lambda a: a.tensor_format.is_ordered and a.tensor_format.is_unique,
        }

    @staticmethod
    def create_csr(shape, crow_indices, col_indices, values):
        raise NotImplementedError

    @staticmethod
    def create_csc(shape, ccol_indices, row_indices, values):
        raise NotImplementedError

    @staticmethod
    def create_bsr(shape, block_size, crow_indices, col_indices, values):
        raise NotImplementedError

    @staticmethod
    def create_bsc(shape, block_size, ccol_indices, row_indices, values):
        raise NotImplementedError

    @staticmethod
    def create_coo(shape, indices, values):
        raise NotImplementedError

    @staticmethod
    def create_dia(shape, offsets, values):
        raise NotImplementedError


PACKAGE_HELPER_MAP = {
    "cupyx": CupySparseFormatHelper,
    "scipy": ScipySparseFormatHelper,
    "torch": TorchSparseFormatHelper,
    "nvmath": USTFormatHelper,
}
