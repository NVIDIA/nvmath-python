# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
from typing import Literal, TypeAlias, TypeVar

import numpy as np

__all__ = [
    "ReductionOp",
    "ProcessGroup",
    "MPIProcessGroup",
    "TorchProcessGroup",
]


class ReductionOp(IntEnum):
    SUM = 0
    MIN = 1
    MAX = 2


T = TypeVar("T")
Reducer: TypeAlias = Callable[[T, T], T]


class ProcessGroup(ABC):
    """A ProcessGroup represents a set of processes collectively running
    ``nvmath.distributed`` operations. A ProcessGroup implements a small set of
    communication collectives used by nvmath-python to initialize the distributed
    runtime and set up distributed math operations. It is not used for compute
    (each math library uses its chosen communication backend(s), like NCCL or NVSHMEM).
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        "Index of this process in the group."
        raise NotImplementedError

    @property
    @abstractmethod
    def nranks(self) -> int:
        """Number of processes in the group."""
        raise NotImplementedError

    @abstractmethod
    def broadcast_buffer(self, array: np.ndarray, *, root: int = 0) -> None:
        """Broadcast an array from one process to every process.

        Args:
            array: input (on root) and output of the collective.

            root: rank of sending process.
        """
        raise NotImplementedError

    @abstractmethod
    def allreduce_buffer(self, array: np.ndarray, *, op: ReductionOp) -> None:
        """Allreduce an array.

        Args:
            array: Input and output of the collective. The function operates in-place.

            op: One of the values from :class:`ReduceOp` enum. Specifies an operation for
                element-wise reductions.
        """
        raise NotImplementedError

    @abstractmethod
    def allreduce_object(self, obj: T, *, op: Reducer[T]) -> T:
        """Reduces all Python objects contributed by members of the group. The result
        is a single reduced object which is returned on every process.

        Args:
            obj: object contributed by this process.

            op: A Python function that takes two objects and returns a single (reduced)
                object.
        """
        raise NotImplementedError


class MPIProcessGroup(ProcessGroup):
    """ProcessGroup implemented on mpi4py.

    Args:
        mpi_comm: mpi4py communicator.
    """

    # Map from mpi4py reduction ops to this module's ReductionOp.
    _reduction_op_map = {}  # type: ignore

    def __init__(self, mpi_comm):
        if not MPIProcessGroup._reduction_op_map:
            from mpi4py import MPI

            MPIProcessGroup._reduction_op_map = {
                ReductionOp.MIN: MPI.MIN,
                ReductionOp.MAX: MPI.MAX,
                ReductionOp.SUM: MPI.SUM,
            }
        self._mpi_comm = mpi_comm

    @property
    def rank(self) -> int:
        return self._mpi_comm.Get_rank()

    @property
    def nranks(self) -> int:
        return self._mpi_comm.Get_size()

    def broadcast_buffer(self, array: np.ndarray, *, root: int = 0) -> None:
        self._mpi_comm.Bcast(array, root=root)

    def allreduce_buffer(self, array: np.ndarray, *, op: ReductionOp) -> None:
        mpi_reduction_op = MPIProcessGroup._reduction_op_map[op]
        from mpi4py import MPI

        self._mpi_comm.Allreduce(MPI.IN_PLACE, array, mpi_reduction_op)

    def allreduce_object(self, obj: T, *, op: Reducer[T]) -> T:
        return self._mpi_comm.allreduce(obj, op=op)


def _init_binomial_tree(rank: int, nranks: int) -> tuple[int | None, list[int]]:
    """Return parent and children of process with given rank in a binomial
    tree of nranks processes."""
    max_height = math.ceil(math.log2(nranks))
    max_size = 1 << max_height  # max number of nodes in tree (2^max_height)

    # Compute order/height of this subtree (in a full binomial tree of max_size nodes).
    # Relabel this node for the computation (so 0 is deepest leaf instead of root).
    p = label = max_size - 1 - rank
    order = 0
    while p > 0:
        if p % 2 == 0:
            break
        p //= 2
        order += 1

    if rank == 0:
        parent = None
    else:
        parent = label + (1 << order)  # parent = label + 2^order
        parent = max_size - 1 - parent  # convert back to original numbering scheme

    children = []
    # Max number of children is the order of this subtree.
    for i in range(order):
        child = label - (1 << i)  # child = label - 2^i
        child = max_size - 1 - child  # convert back to original numbering scheme
        if child <= nranks - 1:
            children.append(child)

    return parent, children


class MsgSizeError(Exception):
    def __init__(self, size: int):
        self.size = size


class TorchProcessGroup(ProcessGroup):
    """ProcessGroup implemented on ``torch.distributed``.

    Args:
        device_id: Device used by the ``torch.distributed`` process group backend.

        torch_process_group: ``torch.distributed`` process group handle (e.g. returned
            by ``torch.distributed.new_group()``), or None to use the default torch
            process group.
    """

    # Map from torch.distributed reduction ops to this module's ReductionOp.
    _reduction_op_map = {}  # type: ignore
    # The minimum buffer size (in bytes) for allreduce_object()
    MIN_ALL_REDUCE_OBJ_BUFFER_SIZE = 128

    def __init__(
        self,
        *,
        device_id: int | Literal["cpu"],
        torch_process_group=None,
        logger=None,
    ):
        import torch
        import torch.distributed as dist

        if not isinstance(device_id, int) and device_id != "cpu":
            raise ValueError("device_id must be int or 'cpu'")

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        # Get communication backend used by this torch process group.
        backend = dist.get_backend(torch_process_group)
        if backend in ("nccl", "xccl") and device_id == "cpu":
            raise ValueError(f"This process group is using a GPU-only backend ({backend}), but the provided device_id is 'cpu'")
        if backend == "mpi":
            logger.warning(
                "You're using torch.distributed with MPI backend. In this case it's "
                "recommended to initialize nvmath.distributed with "
                "nvmath.distributed.process_group.MPIProcessGroup instead of "
                "TorchProcessGroup."
            )

        self._device_id = device_id
        self._device = torch.device(f"cuda:{device_id}") if device_id != "cpu" else torch.device("cpu")
        self._torch_process_group = torch_process_group

        if not TorchProcessGroup._reduction_op_map:
            TorchProcessGroup._reduction_op_map = {
                ReductionOp.MIN: dist.ReduceOp.MIN,
                ReductionOp.MAX: dist.ReduceOp.MAX,
                ReductionOp.SUM: dist.ReduceOp.SUM,
            }

        if backend == "nccl":
            # Ensure a NCCL communicator is formed for the whole group. Otherwise if a
            # parent NCCL communicator is not yet formed when batched P2P operations are
            # called on a subset of ranks (see allreduce_object() below), they could fail.
            with torch.cuda.device(self._device):
                a = torch.ones(1, device=self._device)
                dist.all_reduce(a, op=dist.ReduceOp.SUM, group=self._torch_process_group)
                if a[0] != self.nranks:
                    raise RuntimeError("torch.distributed incorrect all_reduce output")

        # A serialized MsgSizeError must fit in MIN_ALL_REDUCE_OBJ_BUFFER_SIZE
        assert len(pickle.dumps(MsgSizeError(1 << 30))) < self.MIN_ALL_REDUCE_OBJ_BUFFER_SIZE, "Internal error."
        # Initial buffer size for allreduce_object (in bytes)
        self.allreduce_obj_buffer_size = 2048

        # Determine my parent and children for allreduce_object() implementation.
        # A binomial tree is good for small message sizes.
        self._parent, self._children = _init_binomial_tree(self.rank, self.nranks)

    @property
    def device_id(self) -> int | Literal["cpu"]:
        """Device used by the communication backend of this torch process group."""
        return self._device_id

    @property
    def rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank(group=self._torch_process_group)

    @property
    def nranks(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(group=self._torch_process_group)

    def broadcast_buffer(self, array: np.ndarray, *, root: int = 0) -> None:
        import torch

        if self._device_id == "cpu":
            torch.distributed.broadcast(torch.from_numpy(array), group=self._torch_process_group, group_src=root)
        else:
            tensor_gpu = torch.from_numpy(array).to(self._device)
            torch.distributed.broadcast(tensor_gpu, group=self._torch_process_group, group_src=root)
            array[:] = tensor_gpu.cpu().numpy()

    def allreduce_buffer(self, array: np.ndarray, *, op: ReductionOp) -> None:
        import torch

        torch_reduction_op = TorchProcessGroup._reduction_op_map[op]
        if self._device_id == "cpu":
            torch.distributed.all_reduce(torch.from_numpy(array), op=torch_reduction_op, group=self._torch_process_group)
        else:
            tensor_gpu = torch.from_numpy(array).to(self._device)
            torch.distributed.all_reduce(tensor_gpu, op=torch_reduction_op, group=self._torch_process_group)
            array[:] = tensor_gpu.cpu().numpy()

    @property
    def allreduce_obj_buffer_size(self):
        """Current buffer size for allreduce_object() (in bytes)"""
        return self._allreduce_obj_buffer_size

    @allreduce_obj_buffer_size.setter
    def allreduce_obj_buffer_size(self, value):
        """**NOTE: Same size needs to be set on every process**"""
        if not isinstance(value, int) or value < self.MIN_ALL_REDUCE_OBJ_BUFFER_SIZE:
            raise ValueError(f"Buffer size must be >= {self.MIN_ALL_REDUCE_OBJ_BUFFER_SIZE}")
        self._allreduce_obj_buffer_size = value

    def allreduce_object(self, obj: T, *, op: Reducer[T]) -> T:
        import torch

        NUM_TRIES = 5
        for _ in range(NUM_TRIES):
            try:
                if self._device_id != "cpu":
                    # It's recommended to set torch cuda device for batch_isend_irecv
                    with torch.cuda.device(self._device):
                        return self._allreduce_object(obj, op)
                else:
                    return self._allreduce_object(obj, op)
            except MsgSizeError as e:
                new_size = int(e.size * 1.2)
                self.logger.warning(
                    "allreduce_obj_buffer_size is too small for this allreduce: "
                    f"current size is {self.allreduce_obj_buffer_size} and the required "
                    f"size is {e.size}. Increasing to {new_size}. "
                    "You can set the size manually with 'allreduce_obj_buffer_size' property "
                    " to avoid dynamic resizing."
                )
                self.allreduce_obj_buffer_size = new_size
        raise RuntimeError(
            f"Could not find buffer size for allreduce_object() after {NUM_TRIES} tries. "
            "Please set the limit manually with 'allreduce_obj_buffer_size' property."
        )

    def _allreduce_object(self, obj: T, reducer: Reducer[T]) -> T:
        import torch
        import torch.distributed as dist

        def to_tensor(obj, out_tensor):
            """Serializes Python object into out_tensor buffer.
            If the tensor buffer is not large enough to accommodate the serialized object,
            insert a serialized MsgSizeError (with required size) instead."""
            pickled_obj = pickle.dumps(obj)
            pickled_obj_arr = np.frombuffer(pickled_obj, dtype=np.uint8).copy()
            if pickled_obj_arr.nbytes > out_tensor.nbytes:
                pickled_obj = pickle.dumps(MsgSizeError(pickled_obj_arr.nbytes))
                pickled_obj_arr = np.frombuffer(pickled_obj, dtype=np.uint8).copy()
            out_tensor[: pickled_obj_arr.nbytes] = torch.from_numpy(pickled_obj_arr)
            return out_tensor

        def from_tensor(tensor):
            """Deserializes Python object from tensor buffer."""
            return pickle.loads(tensor.cpu().numpy().tobytes())

        num_children = len(self._children)

        if num_children > 0:
            # 1) Get partially reduced objects from children.

            # Post a recv for each child.
            # Use batch_isend_irecv because, when using the NCCL backend, individual
            # (unbatched) point-to-point (P2P) send or receive operations can trigger
            # the creation of new, separate NCCL communicators for each pair of
            # communicating processes, which can have significant overhead or use
            # large amounts of GPU memory. Instead, we want the p2p operations to use
            # the parent NCCL communicator whenever possible.
            data = [
                torch.empty(self._allreduce_obj_buffer_size, dtype=torch.uint8, device=self._device)
                for _ in range(num_children)
            ]
            recv_ops = []
            for i, child in enumerate(self._children):
                # The tag is to match the recv against the remote send (in case the sender
                # is sending multiple different messages).
                recv_op = dist.P2POp(dist.irecv, data[i], group=self._torch_process_group, tag=2355, group_peer=child)
                recv_ops.append(recv_op)

            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()

            # 2) Apply the reducer on my object and objects received from children.
            reduced_obj = obj
            errors = []
            for i in range(num_children):
                child_obj = from_tensor(data[i])
                if isinstance(child_obj, MsgSizeError):
                    errors.append(child_obj)
                elif not errors:
                    reduced_obj = reducer(reduced_obj, child_obj)
            if errors:
                # Max reduction of all received errors to get the maximum required buffer
                # size for this subtree.
                reduced_obj = MsgSizeError(max(e.size for e in errors))  # type: ignore
        else:
            reduced_obj = obj

        data = [torch.empty(self._allreduce_obj_buffer_size, dtype=torch.uint8, device=self._device)]
        if self._parent is not None:
            # Send reduced object to parent.
            send_op = dist.P2POp(
                dist.isend, to_tensor(reduced_obj, data[0]), group=self._torch_process_group, tag=2355, group_peer=self._parent
            )
            dist.batch_isend_irecv([send_op])[0].wait()

            # Wait for broadcast of reduction from root.
            dist.broadcast(data[0], group=self._torch_process_group, group_src=0)
            result = from_tensor(data[0])
        else:
            # This is the root: broadcast the result.
            assert self.rank == 0
            dist.broadcast(to_tensor(reduced_obj, data[0]), group=self._torch_process_group, group_src=0)
            result = reduced_obj

        if isinstance(result, MsgSizeError):
            raise result

        return result
