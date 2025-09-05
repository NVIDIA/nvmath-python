# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Mixins for handling host <-> device symmetry for distributed tensors
for packages that support only host or device tensors.

Most of the symmetric-memory logic is not package-specific, but handling
regular allocations or local tensor operations is, and can be delegated
to the non-distributed tensor implementation.
For this reason, the functionality is implemented as mixins, so that
we can delegate the `super()` calls without specifying the parent class.
"""

from abc import abstractmethod, ABC
from collections.abc import Sequence

__all__ = ["HostDistributedTensorMixIn", "CudaDistributedTensorMixIn"]

import nvmath.distributed
from nvmath.internal.tensor_ifc import TensorHolder
from nvmath.internal.package_ifc import StreamHolder
from nvmath.internal.utils import device_ctx
from nvmath.distributed._internal.nvshmem import NvshmemNDBufferAllocator
from nvmath.internal.typemaps import NAME_TO_ITEM_SIZE
from nvmath.internal.ndbuffer import ndbuffer


class HostDistributedTensorMixIn(ABC):  # noqa: B024
    """
    Host counterpart for distributed tensor wrapping package that
    does not support host memory space (e.g. cupy). The class is marked
    as abstract, because the mixin is not meant to be instantiated directly.
    """

    def to(self, device_id, stream_holder, symmetric_memory: bool = False):
        """
        In addition to the base class semantics:
          - If symmetric_memory=True, target device must be the one used to initialize
            NVSHMEM on this process.
          - Strides must be dense non-overlapping.
          - Memory layout is preserved (if strides are dense non-overlapping,
            the base class guarantees that)
        """
        if not symmetric_memory or device_id == "cpu":
            tensor = super().to(device_id, stream_holder)  # type: ignore
        elif isinstance(device_id, int):
            device_cls = self.device_tensor_class  # type: ignore
            tensor = device_cls.create_from_host(self, device_id, stream_holder, symmetric_memory)
        else:
            raise ValueError(f"The device must be specified as an integer or 'cpu', not '{device_id}'.")
        assert tensor.is_symmetric_memory == symmetric_memory
        return tensor


class CudaDistributedTensorMixIn(ABC):
    @classmethod
    def empty(
        cls,
        shape,
        device_id="cpu",
        *,
        dtype="float32",
        strides=None,
        stream_holder: StreamHolder | None = None,
        **context,
    ):
        """
        Create an empty tensor of the specified shape and data type.

        Note, that the strides, if specified, MUST correspond to a dense (possibly permuted)
        tensor, otherwise the created tensor may be corrupted.
        """
        symmetric_memory = context.get("symmetric_memory", False)
        make_symmetric = context.get("make_symmetric", False)
        skip_symmetric_check = context.get("skip_symmetric_check", False)

        if not symmetric_memory:
            if make_symmetric or skip_symmetric_check:
                raise ValueError("Use of symmetric memory option with symmetric_memory=False")
            return super().empty(  # type: ignore
                shape,
                device_id=device_id,
                dtype=dtype,
                strides=strides,
                stream_holder=stream_holder,
                **context,
            )

        logger = context.get("logger")
        with device_ctx(device_id):
            ctx = nvmath.distributed.get_context()
            assert ctx is not None, "nvmath.distributed has not been initialized"
            allocator = NvshmemNDBufferAllocator(
                device_id, ctx, make_symmetric=make_symmetric, skip_symmetric_check=skip_symmetric_check
            )
            nd_dst = ndbuffer.empty(
                shape,
                device_id=device_id,
                dtype_name=dtype,
                itemsize=NAME_TO_ITEM_SIZE[dtype],
                strides=strides,
                stream=stream_holder,
                device_memory_pool=allocator,
                logger=logger,
            )
            if nd_dst.cf_order() == "K":
                raise ValueError("CudaDistributedTensor only supports 'C' or 'F' order")
            return cls.wrap_ndbuffer(nd_dst)

    @classmethod
    def create_from_host(
        cls,
        tensor: TensorHolder,
        device_id: int,
        stream_holder: StreamHolder,
        symmetric_memory: bool = False,
    ):
        if not symmetric_memory:
            return super().create_from_host(tensor, device_id, stream_holder)  # type: ignore
        with device_ctx(device_id):
            ctx = nvmath.distributed.get_context()
            assert ctx is not None, "nvmath.distributed has not been initialized"
            allocator = NvshmemNDBufferAllocator(device_id, ctx, make_symmetric=True, skip_symmetric_check=False)
            src_ndbuffer = tensor.asndbuffer()
            dst_ndbuffer = ndbuffer.empty_like(
                src_ndbuffer,
                device_id=device_id,
                stream=stream_holder,
                device_memory_pool=allocator,
            )
            if dst_ndbuffer.cf_order() == "K":
                raise ValueError("CudaDistributedTensor only supports 'C' or 'F' order")
            ndbuffer.copy_into(dst_ndbuffer, src_ndbuffer, stream_holder)
            return cls.wrap_ndbuffer(dst_ndbuffer)

    @classmethod
    @abstractmethod
    def wrap_ndbuffer(cls, ndbuffer: ndbuffer.NDBuffer):
        """
        Defines how to wrap NDBuffer into a distributed tensor instance.
        The exact implementation depends on `self.tensor` type
        expected by the TensorHolder implementation to be mixed in.
        E.g. for NDBufferTensor, it suffices to call `cls(ndbuffer)`,
        while for CupyTensor, we need to wrap the NDBuffer into a cupy.ndarray first.
        """
        raise NotImplementedError

    def to(self, device_id, stream_holder, symmetric_memory: bool = False):
        """
        In addition to the base class semantics:
          - Target device must be the one used to initialize NVSHMEM on this process.
          - Strides must be dense non-overlapping.
          - Memory layout is preserved (if strides are dense non-overlapping,
            the base class guarantees that)
        """
        tensor = super().to(device_id, stream_holder)  # type: ignore
        assert tensor.is_symmetric_memory == symmetric_memory
        return tensor

    def reshape(self, shape: Sequence[int], *, copy: bool | None = None):
        if copy:
            raise NotImplementedError("reshape with copy=True is not supported for CUDA distributed tensor")
        return super().reshape(shape, copy=copy)  # type: ignore
