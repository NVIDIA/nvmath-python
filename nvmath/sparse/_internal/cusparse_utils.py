# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from nvmath.bindings import cusparse
from nvmath.bindings._internal.utils import FunctionNotFoundError
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE

COMPATIBLE_SP_DN_PACKAGES = {("scipy", "numpy"), ("cupyx", "cupy"), ("torch", "torch"), ("nvmath", "nvmath")}

SUPPORTED_NAMED_FORMATS = {"BSR", "CSR", "CSC", "COO"}

INDEX_TYPE_MAP = {
    "int32": cusparse.IndexType.INDEX_32I,
    "int64": cusparse.IndexType.INDEX_64I,
    "uint16": cusparse.IndexType.INDEX_16U,
}

ORDER_TYPE_MAP = {"left": cusparse.Order.COL, "right": cusparse.Order.ROW}

# The index base is always 0.
BASE = 0


class SparseIfc(ABC):
    """
    Abstract base to create and update sparse matrix descriptors.
    """

    @abstractmethod
    def create(self):
        """
        Create a cuSPARSE matrix descriptor.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, new_operand):
        """
        Update the pointers in the sparse matrix descriptor.
        """
        raise NotImplementedError

    # Destructor is not dependent on sparse or dense, so no abstraction is needed.


class COOIfc(SparseIfc):
    """
    Create and update sparse COO matrix descriptor.
    """

    def __init__(self, operand, layout_traits):
        """
        The operand must be a wrapped sparse tensor.
        """
        if layout_traits.batch_count != 1 or len(operand.indices) != 2:
            raise NotImplementedError("Non-uniform batched COO matrices are not supported by cuSPARSE.")
        self.values_size = operand.values.size
        self.index_type = operand.index_type
        self.dtype = operand.dtype
        self.layout_traits = layout_traits
        self.descriptor = None
        # Updatable attributes
        self.row_indices_data_ptr = operand.indices[0].data_ptr
        self.col_indices_data_ptr = operand.indices[1].data_ptr
        self.values_data_ptr = operand.values.data_ptr

    def create(self):
        """
        Create a cuSPARSE COO matrix descriptor.
        """
        t = self.layout_traits
        shape, nnz = t.mm_shape, self.values_size

        if self.index_type not in INDEX_TYPE_MAP:
            raise TypeError(f"The index type {self.index_type} is not in the supported types: {INDEX_TYPE_MAP.keys()}.")

        index_type, value_type = INDEX_TYPE_MAP[self.index_type], NAME_TO_DATA_TYPE[self.dtype]

        self.descriptor = cusparse.create_coo(
            *shape,
            nnz,
            self.row_indices_data_ptr,
            self.col_indices_data_ptr,
            self.values_data_ptr,
            index_type,
            BASE,
            value_type,
        )

        return self.descriptor

    def update(self, new_operand):
        """
        Update the pointers in the sparse matrix descriptor.
        """
        self.row_indices_data_ptr = row_ptr = new_operand.indices[0].data_ptr
        self.col_indices_data_ptr = col_ptr = new_operand.indices[1].data_ptr
        self.values_data_ptr = values_data_ptr = new_operand.values.data_ptr
        cusparse.coo_set_pointers(self.descriptor, row_ptr, col_ptr, values_data_ptr)


class CSRIfc(SparseIfc):
    """
    Create and update sparse CSR matrix descriptor.
    """

    # TODO: take logger to log debug messages.
    def __init__(self, operand, layout_traits):
        """
        The operand must be a wrapped sparse tensor.
        """
        self.values_size = operand.values.size
        self.crow_indices_size = operand.crow_indices.size
        self.index_type = operand.index_type
        self.dtype = operand.dtype
        self.layout_traits = layout_traits
        self.descriptor = None
        # Updatable attributes
        self.crow_indices_data_ptr = operand.crow_indices.data_ptr
        self.col_indices_data_ptr = operand.col_indices.data_ptr
        self.values_data_ptr = operand.values.data_ptr

    def create(self):
        """
        Create a cuSPARSE CSR matrix descriptor.
        """
        t = self.layout_traits

        if t.batch_broadcast:
            shape, nnz = t.mm_shape, self.values_size
        else:
            assert self.values_size % t.batch_count == 0, "Internal error."
            shape, nnz = t.mm_shape, self.values_size // t.batch_count

        if self.index_type not in INDEX_TYPE_MAP:
            raise TypeError(f"The index type {self.index_type} is not in the supported types: {INDEX_TYPE_MAP.keys()}.")

        index_type, value_type = INDEX_TYPE_MAP[self.index_type], NAME_TO_DATA_TYPE[self.dtype]

        self.descriptor = cusparse.create_csr(
            *shape,
            nnz,
            self.crow_indices_data_ptr,
            self.col_indices_data_ptr,
            self.values_data_ptr,
            index_type,
            index_type,
            BASE,
            value_type,
        )

        # Handle batching.
        if t.batch_count > 1:
            if t.batch_broadcast:
                crow_indices_batch_offset = 0
                values_batch_offset = 0
            else:
                assert self.crow_indices_size % t.batch_count == 0, "Internal error."
                crow_indices_batch_offset = self.crow_indices_size // t.batch_count

                assert self.values_size % t.batch_count == 0, "Internal error."
                values_batch_offset = self.values_size // t.batch_count

            cusparse.csr_set_strided_batch(self.descriptor, t.batch_count, crow_indices_batch_offset, values_batch_offset)

        return self.descriptor

    def update(self, new_operand):
        """
        Update the pointers in the sparse matrix descriptor.
        """
        self.crow_indices_data_ptr = new_operand.crow_indices.data_ptr
        self.col_indices_data_ptr = new_operand.col_indices.data_ptr
        self.values_data_ptr = new_operand.values.data_ptr
        cusparse.csr_set_pointers(self.descriptor, self.crow_indices_data_ptr, self.col_indices_data_ptr, self.values_data_ptr)


class BSRIfc(SparseIfc):
    """
    Create and update sparse BSR matrix descriptor.
    """

    # TODO: take logger to log debug messages.
    def __init__(self, operand, layout_traits):
        """
        The operand must be a wrapped sparse tensor.
        """
        self.block_size = operand.block_size
        self.values_size = operand.values.size
        self.crow_indices_size = operand.crow_indices.size
        self.col_indices_size = operand.col_indices.size
        self.index_type = operand.index_type
        self.dtype = operand.dtype
        self.block_order = ORDER_TYPE_MAP[operand.block_order]
        self.layout_traits = layout_traits
        self.descriptor = None
        # Updatable attributes
        self.crow_indices_data_ptr = operand.crow_indices.data_ptr
        self.col_indices_data_ptr = operand.col_indices.data_ptr
        self.values_data_ptr = operand.values.data_ptr

    def create(self):
        """
        Create a cuSPARSE BSR matrix descriptor.
        """
        t = self.layout_traits
        b0, b1 = self.block_size

        # TODO: more elegant way of getting the number of non-zero blocks.
        if t.batch_broadcast:
            shape, bnnz = t.mm_shape, self.values_size // (b0 * b1)
        else:
            assert self.values_size % t.batch_count == 0, "Internal error."
            shape, bnnz = t.mm_shape, self.values_size // (t.batch_count * b0 * b1)

        assert shape[0] % b0 == 0 and shape[1] % b1 == 0, "Internal error."
        shape_in_blocks = shape[0] // b0, shape[1] // b1

        if self.index_type not in INDEX_TYPE_MAP:
            raise TypeError(f"The index type {self.index_type} is not in the supported types: {INDEX_TYPE_MAP.keys()}.")

        index_type, value_type = INDEX_TYPE_MAP[self.index_type], NAME_TO_DATA_TYPE[self.dtype]

        try:
            self.descriptor = cusparse.create_bsr(
                *shape_in_blocks,
                bnnz,
                b0,
                b1,
                self.crow_indices_data_ptr,
                self.col_indices_data_ptr,
                self.values_data_ptr,
                index_type,
                index_type,
                BASE,
                value_type,
                self.block_order,
            )
        except FunctionNotFoundError as e:
            raise RuntimeError("BSR is not supported by the installed cuSPARSE library.") from e

        # Handle batching.
        if t.batch_count > 1:
            if t.batch_broadcast:
                crow_indices_batch_offset = 0
                col_indices_batch_offset = 0
                values_batch_offset = 0
            else:
                assert self.crow_indices_size % t.batch_count == 0, "Internal error."
                crow_indices_batch_offset = self.crow_indices_size // t.batch_count

                assert self.col_indices_size % t.batch_count == 0, "Internal error."
                col_indices_batch_offset = self.col_indices_size // t.batch_count

                assert self.values_size % t.batch_count == 0, "Internal error."
                values_batch_offset = self.values_size // t.batch_count

            cusparse.bsr_set_strided_batch(
                self.descriptor, t.batch_count, crow_indices_batch_offset, col_indices_batch_offset, values_batch_offset
            )

        return self.descriptor

    def update(self, new_operand):
        """
        Update the pointers in the sparse matrix descriptor.
        """
        self.crow_indices_data_ptr = new_operand.crow_indices.data_ptr
        self.col_indices_data_ptr = new_operand.col_indices.data_ptr
        self.values_data_ptr = new_operand.values.data_ptr

        # There is no API currently to just update the pointers so we've to create a new
        # descriptor.
        if self.descriptor is not None:
            cusparse.destroy_sp_mat(self.descriptor)
        self.create()


class CSCIfc(SparseIfc):
    """
    Create and update sparse CSC matrix descriptor.
    """

    # TODO: take logger to log debug messages.
    def __init__(self, operand, layout_traits):
        """
        The operand must be a wrapped sparse tensor.
        """
        self.values_size = operand.values.size
        self.index_type = operand.index_type
        self.dtype = operand.dtype
        self.layout_traits = layout_traits
        self.descriptor = None

        # Updatable attributes
        self.ccol_indices_data_ptr = operand.ccol_indices.data_ptr
        self.row_indices_data_ptr = operand.row_indices.data_ptr
        self.values_data_ptr = operand.values.data_ptr

    def create(self):
        """
        Create a cuSPARSE CSC matrix descriptor.
        """
        t = self.layout_traits

        if t.batch_broadcast:
            shape, nnz = t.mm_shape, self.values_size
        else:
            assert self.values_size % t.batch_count == 0, "Internal error."
            shape, nnz = t.mm_shape, self.values_size // t.batch_count

        if self.index_type not in INDEX_TYPE_MAP:
            raise TypeError(f"The index type {self.index_type} is not in the supported types: {INDEX_TYPE_MAP.keys()}.")

        index_type, value_type = INDEX_TYPE_MAP[self.index_type], NAME_TO_DATA_TYPE[self.dtype]

        self.descriptor = cusparse.create_csc(
            *shape,
            nnz,
            self.ccol_indices_data_ptr,
            self.row_indices_data_ptr,
            self.values_data_ptr,
            index_type,
            index_type,
            BASE,
            value_type,
        )

        # Handle batching.
        if t.batch_count > 1:
            raise NotImplementedError("The cuSPARSE library doesn't currently support batched CSC matrices.")

        return self.descriptor

    def update(self, new_operand):
        """
        Update the pointers in the sparse matrix descriptor.
        """
        self.ccol_indices_data_ptr = ccol_indices_data_ptr = new_operand.ccol_indices.data_ptr
        self.row_indices_data_ptr = row_indices_data_ptr = new_operand.row_indices.data_ptr
        self.values_data_ptr = values_data_ptr = new_operand.values.data_ptr
        cusparse.csc_set_pointers(self.descriptor, ccol_indices_data_ptr, row_indices_data_ptr, values_data_ptr)


class DenseMatrixIfc:
    """
    Create and update dense matrix descriptors.
    """

    def __init__(self, operand, layout_traits):
        """
        The operand must be a wrapped dense tensor.
        """
        self.dtype = operand.dtype
        self.layout_traits = layout_traits
        self.descriptor = None
        # Updatable attributes
        self.data_ptr = operand.data_ptr

    def create(self):
        """
        Create a cuSPARSE dense matrix descriptor.
        """

        t = self.layout_traits

        self.descriptor = cusparse.create_dn_mat(*t.mm_shape, t.ld, self.data_ptr, NAME_TO_DATA_TYPE[self.dtype], t.order)

        if t.batch_count > 1:
            cusparse.dn_mat_set_strided_batch(self.descriptor, t.batch_count, t.batch_offset)

        return self.descriptor

    def update(self, new_operand):
        """
        Update the pointers in the dense matrix descriptor.
        """
        self.data_ptr = data_ptr = new_operand.data_ptr
        cusparse.dn_mat_set_values(self.descriptor, data_ptr)
