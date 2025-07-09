# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import numpy as np

from nvmath.bindings import cudss
from nvmath.internal.tensor_wrapper import wrap_operand

COMPATIBLE_LHS_RHS_PACKAGES = {("scipy", "numpy"), ("cupyx", "cupy"), ("torch", "torch")}


# TODO: Add debug log messages.
def wrap_dense_tensors(tensor_type, native_operands):
    # Unchecked faster version of tensor_wrapper.wrap_operands().
    return tuple(tensor_type(o) for o in native_operands)


def wrap_sparse_tensors(tensor_type, attr_name_map, native_operands):
    # A more direct version of nvmath.sparse._internal.common_utils.wrap_sparse_operands().
    return tuple(tensor_type.create_from_tensor(o, attr_name_map=attr_name_map) for o in native_operands)


def create_cudss_dense_matrix(cuda_value_type, rhs):
    # The rhs should be a wrapped (TensorHolder object) matrix or vector in a memory space
    # consistent with the execution mode.
    shape = rhs.shape
    assert len(shape) <= 2, "Internal error."

    m = shape[0]
    n = shape[1] if len(shape) == 2 else 1

    # Only column-major is supported.
    ld = m

    rhs_ptr = cudss.matrix_create_dn(m, n, ld, rhs.data_ptr, cuda_value_type, cudss.Layout.COL_MAJOR)
    references = []
    return references, rhs_ptr


def create_cudss_dense_explicit_batch(cuda_index_type, cuda_value_type, index_type, rhs, stream_holder):
    # The rhs should be a sequence of wrapped (TensorHolder object) matrix or vector in a
    # memory space consistent with the execution mode.
    assert isinstance(rhs, Sequence), "Internal error."

    batch_count = len(rhs)

    m = wrap_operand(np.array([r.shape[0] for r in rhs], dtype=index_type))
    n = wrap_operand(np.array([r.shape[1] if len(r.shape) == 2 else 1 for r in rhs], dtype=index_type))

    # Take device ID of the first matrix, since it's the same for all.
    device_id = rhs[0].device_id

    # The pointers must be a device array. We'll first create it on the host, wrap and
    # copy it to the device.
    p = wrap_operand(np.array([r.data_ptr for r in rhs], dtype=np.uint64))
    p = p.to(device_id=device_id, stream_holder=stream_holder)

    # LD is the same as M, since only column-major is supported.
    rhs_ptr = cudss.matrix_create_batch_dn(
        batch_count, m.data_ptr, n.data_ptr, m.data_ptr, p.data_ptr, cuda_index_type, cuda_value_type, cudss.Layout.COL_MAJOR
    )

    references = [m, n, p]
    return references, rhs_ptr


def create_cudss_dense_implicit_batch(cuda_index_type, cuda_value_type, index_type, batch_indices, rhs, stream_holder):
    # The rhs should be a wrapped (TensorHolder object) NDarray 3D or greater dimension.
    assert len(rhs.shape) > 2, "Internal error."

    # Create a sequence of samples in the batch.
    # Explicitly unpack indices to avoid tensor[*b, ...] (support 3.10).
    unpacked_indices = [tuple(b) + (...,) for b in batch_indices]
    rhs_sequence = [rhs.tensor[u] for u in unpacked_indices]

    # Wrap the samples using the lightweight interface, since we've already checked.
    rhs_sequence = wrap_dense_tensors(rhs.__class__, rhs_sequence)

    return create_cudss_dense_explicit_batch(cuda_index_type, cuda_value_type, index_type, rhs_sequence, stream_holder)


def create_cudss_dense_wrapper(cuda_index_type, cuda_value_type, index_type, batch_indices, rhs, stream_holder):
    # A convenience function to forward to the right implementation.

    if isinstance(rhs, Sequence):
        return create_cudss_dense_explicit_batch(cuda_index_type, cuda_value_type, index_type, rhs, stream_holder)

    if len(rhs.shape) > 2:
        return create_cudss_dense_implicit_batch(
            cuda_index_type, cuda_value_type, index_type, batch_indices, rhs, stream_holder
        )

    return create_cudss_dense_matrix(cuda_value_type, rhs)


def update_cudss_dense_matrix_ptr(rhs_ptr, new_rhs):
    # The new rhs should be a wrapped (TensorHolder object) matrix or vector in a memory
    #   space consistent with the execution mode.
    # rhs_ptr is a cuDSS matrix ptr.
    assert len(new_rhs.shape) <= 2, "Internal error."
    cudss.matrix_set_values(rhs_ptr, new_rhs.data_ptr)

    references = []
    return references


def update_cudss_dense_explicit_batch_ptr(rhs_ptr, new_rhs, stream_holder):
    # The new rhs should be a sequence of wrapped (TensorHolder object) matrix or vector
    # in a memory space consistent with the execution mode.
    assert isinstance(new_rhs, Sequence), "Internal error."

    # Take device ID of the first matrix, since it's the same for all.
    device_id = new_rhs[0].device_id

    # The pointers must be a device array. We'll first create it on the host, wrap and
    # copy it to the device.
    p = wrap_operand(np.array([r.data_ptr for r in new_rhs], dtype=np.uint64))
    p = p.to(device_id=device_id, stream_holder=stream_holder)

    cudss.matrix_set_batch_values(rhs_ptr, p.data_ptr)

    references = [p]
    return references


def update_cudss_dense_implicit_batch_ptr(rhs_ptr, batch_indices, new_rhs, stream_holder):
    # The new rhs should be a wrapped (TensorHolder object) NDarray 3D or greater dimension.
    assert len(new_rhs.shape) > 2, "Internal error."

    # Create a sequence of samples in the batch.
    # Explicitly unpack indices to avoid tensor[*b, ...] (support 3.10).
    unpacked_indices = [tuple(b) + (...,) for b in batch_indices]
    rhs_sequence = [new_rhs.tensor[u] for u in unpacked_indices]

    # Wrap the samples using the lightweight interface, since we've already checked.
    rhs_sequence = wrap_dense_tensors(new_rhs.__class__, rhs_sequence)

    return update_cudss_dense_explicit_batch_ptr(rhs_ptr, rhs_sequence, stream_holder)


def update_cudss_dense_ptr_wrapper(rhs_ptr, *, batch_indices=None, new_rhs=None, stream_holder=None):
    # A convenience function to forward to the right implementation.
    assert new_rhs is not None, "Internal error."

    if isinstance(new_rhs, Sequence):
        assert stream_holder is not None, "Internal error."
        return update_cudss_dense_explicit_batch_ptr(rhs_ptr, new_rhs, stream_holder)

    if len(new_rhs.shape) > 2:
        assert batch_indices is not None and stream_holder is not None, "Internal error."
        return update_cudss_dense_implicit_batch_ptr(rhs_ptr, batch_indices, new_rhs, stream_holder)

    return update_cudss_dense_matrix_ptr(rhs_ptr, new_rhs)


def create_cudss_csr_matrix(cuda_index_type, cuda_value_type, matrix_type, matrix_view_type, lhs):
    # The lhs should be a wrapped (CSRTensorHolder object) matrix in a memory space
    # consistent with the execution mode.
    assert lhs.format_name == "CSR", "Internal error."

    shape = lhs.shape
    assert len(shape) == 2, "Internal error."

    m, n = shape
    nnz = lhs.values.size

    lhs_ptr = cudss.matrix_create_csr(
        m,
        n,
        nnz,
        lhs.crow_indices.data_ptr,
        0,
        lhs.col_indices.data_ptr,
        lhs.values.data_ptr,
        cuda_index_type,
        cuda_value_type,
        matrix_type,
        matrix_view_type,
        cudss.IndexBase.ZERO,
    )
    references = []
    return references, lhs_ptr


def create_cudss_csr_explicit_batch(
    cuda_index_type, cuda_value_type, index_type, matrix_type, matrix_view_type, lhs, stream_holder
):
    # The lhs should be a sequence of wrapped (CSRTensorHolder object) matrix in a memory
    # space consistent with the execution mode.
    assert isinstance(lhs, Sequence), "Internal error."

    batch_count = len(lhs)

    m = wrap_operand(np.array([o.shape[0] for o in lhs], dtype=index_type))
    n = wrap_operand(np.array([o.shape[1] for o in lhs], dtype=index_type))
    nnz = wrap_operand(np.array([o.values.size for o in lhs], dtype=index_type))

    # Take device ID of the first operator, since it's the same for all.
    device_id = lhs[0].device_id

    # The pointer arrays must be on device. We'll first create it on the host, wrap and
    # copy it to the device.
    crow_indices = wrap_operand(np.array([o.crow_indices.data_ptr for o in lhs], dtype=np.uint64))
    crow_indices = crow_indices.to(device_id=device_id, stream_holder=stream_holder)

    col_indices = wrap_operand(np.array([o.col_indices.data_ptr for o in lhs], dtype=np.uint64))
    col_indices = col_indices.to(device_id=device_id, stream_holder=stream_holder)

    values = wrap_operand(np.array([o.values.data_ptr for o in lhs], dtype=np.uint64))
    values = values.to(device_id=device_id, stream_holder=stream_holder)

    lhs_ptr = cudss.matrix_create_batch_csr(
        batch_count,
        m.data_ptr,
        n.data_ptr,
        nnz.data_ptr,
        crow_indices.data_ptr,
        0,
        col_indices.data_ptr,
        values.data_ptr,
        cuda_index_type,
        cuda_value_type,
        matrix_type,
        matrix_view_type,
        cudss.IndexBase.ZERO,
    )
    references = [m, n, nnz, crow_indices, col_indices, values]
    return references, lhs_ptr


def create_cudss_csr_implicit_batch(
    cuda_index_type, cuda_value_type, index_type, matrix_type, matrix_view_type, batch_indices, lhs, stream_holder
):
    # The lhs should be a wrapped CSRTensorHolder object in a memory space consistent with
    # the execution mode.
    assert lhs.num_dimensions > 2, "Internal error."

    # We can't use an elegant implementation like for the dense implicit batch, where we
    # index the samples in the tensor to form an explicit sequence and forward to the dense
    # explicit batch implementation. This is because we can't rely on having the native
    # LHS tensor after copying across memory spaces.

    # For this function, we assume that the batch_indices are consistent with the
    # specified lhs.
    batch_count = len(batch_indices)

    # The shape is constant for each sample.
    shape = lhs.shape[-2], lhs.shape[-1]
    m = wrap_operand(np.full(batch_count, shape[0], dtype=index_type))
    n = wrap_operand(np.full(batch_count, shape[1], dtype=index_type))

    # Explicitly unpack indices to avoid tensor[*b, ...] (support 3.10).
    unpacked_indices = [tuple(b) + (...,) for b in batch_indices]

    # Create a sequence of wrapped samples in the batch for each of the constituent dense
    # arrays.
    crow_indices_sequence = [lhs.crow_indices.__class__(lhs.crow_indices.tensor[u]) for u in unpacked_indices]
    col_indices_sequence = [lhs.col_indices.__class__(lhs.col_indices.tensor[u]) for u in unpacked_indices]
    values_sequence = [lhs.values.__class__(lhs.values.tensor[u]) for u in unpacked_indices]

    # The pointer arrays must be on device. We'll first create it on the host, wrap and
    # copy it to the device.
    crow_indices = wrap_operand(np.array([o.data_ptr for o in crow_indices_sequence], dtype=np.uint64))
    crow_indices = crow_indices.to(device_id=lhs.device_id, stream_holder=stream_holder)

    col_indices = wrap_operand(np.array([o.data_ptr for o in col_indices_sequence], dtype=np.uint64))
    col_indices = col_indices.to(device_id=lhs.device_id, stream_holder=stream_holder)

    values = wrap_operand(np.array([o.data_ptr for o in values_sequence], dtype=np.uint64))
    values = values.to(device_id=lhs.device_id, stream_holder=stream_holder)

    # The nnz should be the same for each sample, but we'll iterate anyway.
    nnz = wrap_operand(np.array([o.size for o in values_sequence], dtype=index_type))

    lhs_ptr = cudss.matrix_create_batch_csr(
        batch_count,
        m.data_ptr,
        n.data_ptr,
        nnz.data_ptr,
        crow_indices.data_ptr,
        0,
        col_indices.data_ptr,
        values.data_ptr,
        cuda_index_type,
        cuda_value_type,
        matrix_type,
        matrix_view_type,
        cudss.IndexBase.ZERO,
    )
    references = [m, n, nnz, crow_indices, col_indices, values]
    return references, lhs_ptr


def create_cudss_csr_wrapper(
    cuda_index_type, cuda_value_type, index_type, matrix_type, matrix_view_type, batch_indices, lhs, stream_holder
):
    if isinstance(lhs, Sequence):
        return create_cudss_csr_explicit_batch(
            cuda_index_type, cuda_value_type, index_type, matrix_type, matrix_view_type, lhs, stream_holder
        )

    if len(lhs.shape) > 2:
        return create_cudss_csr_implicit_batch(
            cuda_index_type, cuda_value_type, index_type, matrix_type, matrix_view_type, batch_indices, lhs, stream_holder
        )

    return create_cudss_csr_matrix(cuda_index_type, cuda_value_type, matrix_type, matrix_view_type, lhs)


def update_cudss_csr_matrix_ptr(lhs_ptr, new_lhs):
    # The new lhs should be a wrapped (CSRTensorHolder object) matrix in a memory space
    # consistent with the execution mode.
    # lhs_ptr is a cuDSS CSR matrix ptr.
    assert len(new_lhs.shape) == 2, "Internal error."
    cudss.matrix_set_csr_pointers(
        lhs_ptr, new_lhs.crow_indices.data_ptr, 0, new_lhs.col_indices.data_ptr, new_lhs.values.data_ptr
    )
    references = []
    return references


def update_cudss_csr_explicit_batch_ptr(lhs_ptr, new_lhs, stream_holder):
    # The new lhs should be a sequence of wrapped (CSRTensorHolder object) matrix in a
    # memory space consistent with the execution mode.
    assert isinstance(new_lhs, Sequence), "Internal error."

    # Take device ID of the first matrix, since it's the same for all.
    device_id = new_lhs[0].device_id

    # The pointers must be a device array. We'll first create it on the host, wrap and
    # copy it to the device.
    crow_indices = wrap_operand(np.array([o.crow_indices.data_ptr for o in new_lhs], dtype=np.uint64))
    crow_indices = crow_indices.to(device_id=device_id, stream_holder=stream_holder)

    col_indices = wrap_operand(np.array([o.col_indices.data_ptr for o in new_lhs], dtype=np.uint64))
    col_indices = col_indices.to(device_id=device_id, stream_holder=stream_holder)

    values = wrap_operand(np.array([o.values.data_ptr for o in new_lhs], dtype=np.uint64))
    values = values.to(device_id=device_id, stream_holder=stream_holder)

    cudss.matrix_set_batch_csr_pointers(lhs_ptr, crow_indices.data_ptr, 0, col_indices.data_ptr, values.data_ptr)

    references = [crow_indices, col_indices, values]
    return references


def update_cudss_csr_implicit_batch_ptr(lhs_ptr, batch_indices, new_lhs, stream_holder):
    # The new lhs should be a wrapped (CSRTensorHolder object) tensor 3D or greater
    # dimension in a memory space consistent with the execution mode.
    assert new_lhs.num_dimensions > 2, "Internal error."

    # We can't use an elegant implementation like for the dense implicit batch, where we
    # index the samples in the tensor to form an explicit sequence and forward to the dense
    # explicit batch implementation. This is because we can't rely on having the native
    # LHS tensor after copying across memory spaces.

    # Explicitly unpack indices to avoid tensor[*b, ...] (support 3.10).
    unpacked_indices = [tuple(b) + (...,) for b in batch_indices]

    # Create a sequence of wrapped samples in the batch for each of the constituent dense
    # arrays.
    crow_indices_sequence = [new_lhs.crow_indices.__class__(new_lhs.crow_indices.tensor[u]) for u in unpacked_indices]
    col_indices_sequence = [new_lhs.col_indices.__class__(new_lhs.col_indices.tensor[u]) for u in unpacked_indices]
    values_sequence = [new_lhs.values.__class__(new_lhs.values.tensor[u]) for u in unpacked_indices]

    # The pointer arrays must be on device. We'll first create it on the host, wrap and
    # copy it to the device.
    crow_indices = wrap_operand(np.array([o.data_ptr for o in crow_indices_sequence], dtype=np.uint64))
    crow_indices = crow_indices.to(device_id=new_lhs.device_id, stream_holder=stream_holder)

    col_indices = wrap_operand(np.array([o.data_ptr for o in col_indices_sequence], dtype=np.uint64))
    col_indices = col_indices.to(device_id=new_lhs.device_id, stream_holder=stream_holder)

    values = wrap_operand(np.array([o.data_ptr for o in values_sequence], dtype=np.uint64))
    values = values.to(device_id=new_lhs.device_id, stream_holder=stream_holder)

    cudss.matrix_set_batch_csr_pointers(lhs_ptr, crow_indices.data_ptr, 0, col_indices.data_ptr, values.data_ptr)

    references = [crow_indices, col_indices, values]
    return references


def update_cudss_csr_ptr_wrapper(lhs_ptr, *, batch_indices=None, new_lhs=None, stream_holder=None):
    # A convenience function to forward to the right implementation.
    assert new_lhs is not None, "Internal error."

    if isinstance(new_lhs, Sequence):
        assert stream_holder is not None, "Internal error."
        return update_cudss_csr_explicit_batch_ptr(lhs_ptr, new_lhs, stream_holder)

    if len(new_lhs.shape) > 2:
        assert batch_indices is not None and stream_holder is not None, "Internal error."
        return update_cudss_csr_implicit_batch_ptr(lhs_ptr, batch_indices, new_lhs, stream_holder)

    return update_cudss_csr_matrix_ptr(lhs_ptr, new_lhs)
