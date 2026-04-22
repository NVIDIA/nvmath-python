# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Torch tensor objects.
"""

__all__ = ["TorchTensor"]

import contextlib
from collections.abc import Sequence

import torch

from .package_ifc import StreamHolder
from .tensor_ifc import TensorHolder


class TorchTensor(TensorHolder[torch.Tensor]):
    """
    TensorHolder for Torch Tensors.
    """

    name = "torch"
    module = torch
    name_to_dtype = TensorHolder.create_name_dtype_map(
        conversion_function=lambda name: getattr(torch, name), exception_type=AttributeError
    )

    def __init__(self, tensor):
        super().__init__(tensor)

    @property
    def data_ptr(self):
        return self.tensor.data_ptr()

    @property
    def device(self):
        return str(self.tensor.device).split(":")[0]

    @property
    def device_id(self):
        index = self.tensor.device.index
        return "cpu" if index is None else index

    @property
    def dtype(self):
        """Name of the data type"""
        return str(self.tensor.dtype).split(".")[-1]

    @property
    def itemsize(self):
        return self.tensor.itemsize

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.nelement()

    @property
    def strides(self):
        return self.tensor.stride()

    @classmethod
    def empty(
        cls, shape, device_id="cpu", *, dtype="float32", strides=None, stream_holder: StreamHolder | None = None, **context
    ):
        """
        Create an empty tensor of the specified shape and data type on the specified device
        (None, 'cpu', or device id).

        Note, that the strides, if specified, should correspond to a dense
        (possibly permuted) tensor and MUST NOT overlap.
        Otherwise, the behaviour is not defined.
        """
        dtype = TorchTensor.name_to_dtype[dtype]
        stream_ctx = contextlib.nullcontext() if stream_holder is None else stream_holder.ctx
        with stream_ctx:
            if strides:
                # note: torch strides is not scaled by bytes
                tensor = torch.empty_strided(shape, strides, dtype=dtype, device=device_id)
            else:
                tensor = torch.empty(shape, dtype=dtype, device=device_id)

        return cls(tensor)

    def asndbuffer(self):
        raise RuntimeError("Converting torch tensor to ndbuffer is not supported")

    def to(self, device_id, stream_holder):
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        # For h2d the data on the CPU is available and we block on `to()`, so a stream
        # is not needed. It is currently ignored if provided.

        # For d2h and d2d, we require a stream to ensure that the source data is ready
        # before the copy operation is launched.
        if self.device_id != "cpu" and stream_holder is None:
            raise AssertionError("Internal error: a stream holder should be provided for d2h or d2d copies.")

        # Always block for h2d and d2h copies to ensure that the copied data is ready
        # for consumption when `to()` return.

        # For d2d, the `torch.to()` operation is launched on the source device on the
        # specified stream, so this stream must be on the source device. Their
        # implementation ensures correct ordering between the current streams on the
        # source and target devices.

        blocking = self.device_id == "cpu" or device_id == "cpu"

        stream_ctx = contextlib.nullcontext() if stream_holder is None else stream_holder.ctx
        with stream_ctx:
            tensor_device = self.tensor.to(device=device_id, non_blocking=not blocking)

        return TorchTensor(tensor_device)

    def copy_(self, src, stream_holder):
        """
        Inplace copy of src (copy the data from src into self).
        """

        # For d2h and d2d, we require a stream to ensure that the source data is ready
        # before the copy operation is launched.
        if src.device_id != "cpu" and stream_holder is None:
            raise AssertionError("Internal error: a stream holder should be provided for d2h or d2d copies.")

        stream_ctx = contextlib.nullcontext() if stream_holder is None else stream_holder.ctx
        with stream_ctx:
            self.tensor.copy_(src.tensor)

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, torch.Tensor)

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None):
        if copy:
            raise NotImplementedError("reshape with copy=True not implemented")
        if copy is False:
            return self.__class__(self.tensor.view(shape))
        return self.__class__(self.tensor.reshape(shape))

    def memory_buffer(self):
        """Creates a view of the memory buffer as a 1D tensor."""
        storage = self.tensor.untyped_storage()
        size = self.size
        v = self.tensor.view(self.shape)
        # TODO: ensure tensor is dense for now, and later support linear
        # memory with constant stride.
        v.set_(storage, self.tensor.storage_offset(), (size,), (1,))
        return TorchTensor(v)

    def memory_buffer_to_tensor(self, shape, strides):
        """
        Creates a N-D tensor view of the memory buffer according to the specified
        shape and strides.
        """
        assert len(self.shape) == 1, "Internal error."
        storage = self.tensor.untyped_storage()
        t = self.tensor.view(self.shape)
        t.set_(storage, self.tensor.storage_offset(), shape, strides)
        return TorchTensor(t)

    def _broadcast_to(self, shape):
        reshaped_tensor = torch.broadcast_to(self.tensor, shape)
        return self.__class__(reshaped_tensor)

    @property
    def is_conjugate(self) -> bool:
        return self.tensor.is_conj()
