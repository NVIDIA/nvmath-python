# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Interface to seamlessly use Torch tensor objects.
"""

__all__ = ["TorchTensor"]

import contextlib

import torch

from .package_ifc import StreamHolder
from .tensor_ifc import Tensor


class TorchTensor(Tensor):
    """
    Tensor wrapper for Torch Tensors.
    """

    name = "torch"
    module = torch
    name_to_dtype = Tensor.create_name_dtype_map(
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
        return self.tensor.device.index

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

    def numpy(self, stream_holder=StreamHolder()):
        # We currently do not use this.
        raise NotImplementedError

    @classmethod
    def empty(cls, shape, *, dtype="float32", device_id=None, strides=None):
        """
        Create an empty tensor of the specified shape and data type on the specified device
        (None, 'cpu', or device id).

        Note, that the strides, if specified, should correspond to a dense
        (possibly permuted) tensor and MUST NOT overlap.
        Otherwise, the behaviour is not defined.
        """
        dtype = TorchTensor.name_to_dtype[dtype]
        if strides:
            # note: torch strides is not scaled by bytes
            tensor = torch.empty_strided(shape, strides, dtype=dtype, device=device_id)
        else:
            tensor = torch.empty(shape, dtype=dtype, device=device_id)

        return cls(tensor)

    def to(self, device="cpu", stream_holder=StreamHolder()):
        """
        Create a copy of the tensor on the specified device (integer or
          'cpu'). Copy to  Numpy ndarray if CPU, otherwise return Cupy type.
        """
        if not (device == "cpu" or isinstance(device, int)):
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device}'.")

        non_blocking = device != "cpu"

        with stream_holder.ctx:
            tensor_device = self.tensor.to(device=device, non_blocking=non_blocking)

        return tensor_device

    def copy_(self, src, stream_holder=StreamHolder()):
        """
        Inplace copy of src (copy the data from src into self).
        """
        with (stream_holder and stream_holder.ctx) or contextlib.nullcontext():
            self.tensor.copy_(src)

        return self.tensor

    def istensor(self):
        """
        Check if the object is ndarray-like.
        """
        return isinstance(self.tensor, torch.Tensor)
