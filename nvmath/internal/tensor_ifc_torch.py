# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Torch tensor objects.
"""

__all__ = ["TorchTensor"]

import contextlib
from collections.abc import Sequence

import torch

from .tensor_ifc import TensorHolder
from .package_ifc import StreamHolder


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

    def to(self, device_id, stream_holder):
        if not (device_id == "cpu" or isinstance(device_id, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")

        assert stream_holder is not None, "Internal Error: moving TorchTensor requires a stream."
        with stream_holder.ctx:
            tensor_device = self.tensor.to(device=device_id, non_blocking=(device_id != "cpu"))

        return TorchTensor(tensor_device)

    def copy_(self, src, stream_holder):
        """
        Inplace copy of src (copy the data from src into self).
        """
        with (stream_holder and stream_holder.ctx) or contextlib.nullcontext():
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
