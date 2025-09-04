# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import re

import nvmath.distributed
from nvmath.internal.utils import device_ctx, get_or_create_stream
from nvmath.distributed import free_symmetric_memory
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand, maybe_register_package

from .helpers import calculate_strides, gather_array, generate_random_data, is_close, to_host
from .helpers_fft import calc_slab_shape

import cuda.core.experimental

package_name_to_package = {"numpy": np}


@pytest.fixture(scope="module")
def nvmath_distributed():
    """Pytest fixture that initializes nvmath.distributed and finalizes it on exit"""
    from mpi4py import MPI

    maybe_register_package("cupy")
    try:
        import torch

        maybe_register_package("torch")
        package_name_to_package["torch"] = torch
    except ImportError:
        pass

    device_id = MPI.COMM_WORLD.Get_rank() % cuda.core.experimental.system.num_devices
    nvmath.distributed.initialize(device_id, MPI.COMM_WORLD)

    yield

    nvmath.distributed.finalize()


def _calculate_local_box(global_shape, partition_dim, rank, nranks):
    """Given a global shape of data that is partitioned across ranks along the
    `partition_dim` dimension, return the local box of this rank (as a lower and
    upper coordinate in the global shape).
    """
    lower = [0 for _ in range(len(global_shape))]
    for i in range(rank):
        shape = calc_slab_shape(global_shape, partition_dim, i, nranks)
        lower[partition_dim] += shape[partition_dim]
    shape = calc_slab_shape(global_shape, partition_dim, rank, nranks)
    upper = list(shape)
    upper[partition_dim] += lower[partition_dim]
    return lower, upper


@pytest.mark.parametrize("dtype", [np.int8, np.int16])
def test_unsupported_itemsize(dtype, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    global_shape = (16, 16)
    shape = calc_slab_shape(global_shape, 0, rank, nranks)
    box = _calculate_local_box(global_shape, 0, rank, nranks)

    itemsize = dtype().itemsize
    with pytest.raises(
        ValueError,
        match=re.escape(f"Reshape only supports element sizes in (4, 8, 16) bytes. The operand's element size is {itemsize}"),
    ):
        data = np.ones(shape, dtype=dtype)
        nvmath.distributed.reshape.reshape(data, input_box=box, output_box=box)


def test_wrong_boxes1(nvmath_distributed, check_symmetric_memory_leaks):
    """In this test, the input and output box of one process overlaps with those
    of another process."""
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires nranks > 1")

    with pytest.raises(
        ValueError, match=re.escape("The global number of elements is incompatible with the inferred global shape (2, 2)")
    ):
        data = np.array([0, 1, 2, 3], dtype=np.int32).reshape((2, 2))
        nvmath.distributed.reshape.reshape(data, input_box=[(0, 0), (2, 2)], output_box=[(0, 0), (2, 2)])


def test_wrong_boxes2(nvmath_distributed, check_symmetric_memory_leaks):
    """In this test each rank has 2x2=4 elements, but the box arguments
    used imply a global shape of (6,6), which has more elements than the
    actual number of global elements.
    """
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks < 2 or nranks > 8:
        pytest.skip("This test requires nranks in [2,8]")

    dtype = np.int32
    with pytest.raises(
        ValueError, match=re.escape("The global number of elements is incompatible with the inferred global shape (6, 6)")
    ):
        if rank % 2 == 0:
            data = np.array([0, 1, 2, 3], dtype=dtype).reshape((2, 2))
            nvmath.distributed.reshape.reshape(data, input_box=[(0, 0), (2, 2)], output_box=[(0, 0), (2, 2)])
        else:
            data = np.array([4, 5, 6, 7], dtype=dtype).reshape((2, 2))
            nvmath.distributed.reshape.reshape(data, input_box=[(4, 4), (6, 6)], output_box=[(4, 4), (6, 6)])


def test_wrong_boxes3(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires nranks > 1")

    with pytest.raises(
        ValueError,
        match=re.escape("The upper coordinates must be larger than the lower coordinates, but got lower=(2, 2) upper=(0, 0)"),
    ):
        data = np.array([0, 1, 2, 3], dtype=np.int32).reshape((2, 2))
        nvmath.distributed.reshape.reshape(data, input_box=[(2, 2), (0, 0)], output_box=[(2, 2), (0, 0)])


def test_inconsistent_layout(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires nranks > 1")

    global_shape = (16, 16)
    shape = calc_slab_shape(global_shape, 0, rank, nranks)
    box = _calculate_local_box(global_shape, 0, rank, nranks)

    with pytest.raises(
        ValueError,
        match=re.escape("The input memory layout is not C or Fortran, or is inconsistent across processes"),
    ):
        data = np.ones(shape, dtype=np.float64)
        if rank % 2 == 1:
            data = np.asfortranarray(data)
        nvmath.distributed.reshape.reshape(data, input_box=box, output_box=box)


@pytest.mark.parametrize("memory_order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reshape_matrix_2_processes(memory_order, dtype, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks != 2:
        pytest.skip("This test requires 2 ranks")

    # Distributed Reshape matrix of global shape (2, 3) in this fashion:
    #
    # |0  1  |2 |           |0 |1  2 |
    # |3  4  |5 |     =>    |3 |4  5 |
    #
    #
    # The drawing shows the elements/values of the matrix, with each block
    # owned by one process.

    def F(a):
        if memory_order == "F":
            return np.asfortranarray(a)
        return a

    if rank == 0:
        data = F(np.array([0, 1, 3, 4], dtype=dtype).reshape((2, 2)))
        result = nvmath.distributed.reshape.reshape(data, input_box=[(0, 0), (2, 2)], output_box=[(0, 0), (2, 1)])
        expected = np.array([0, 3], dtype=dtype).reshape((2, 1))
    else:
        data = F(np.array([2, 5], dtype=dtype).reshape((2, 1)))
        result = nvmath.distributed.reshape.reshape(data, input_box=[(0, 2), (2, 3)], output_box=[(0, 1), (2, 3)])
        expected = np.array([1, 2, 4, 5], dtype=dtype).reshape((2, 2))
    np.testing.assert_equal(result, expected)


@pytest.mark.need_4_procs
@pytest.mark.parametrize("memory_order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_reshape_matrix_4_processes(memory_order, dtype, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks != 4:
        pytest.skip("This test requires 4 ranks")

    # Distributed Reshape matrix of global shape (4, 4) in this fashion:
    #
    # |0  1  |2  3 |          |0 |1 |2 |3 |
    # |4  5  |6  7 |          |4 |5 |6 |7 |
    # --------------    =>    |8 |9 |10|11|
    # |8  9  |10 11|          |12|13|14|15|
    # |12 13 |14 15|
    #
    # The drawing shows the elements/values of the matrix, with each block
    # owned by one process.

    def F(a):
        if memory_order == "F":
            return np.asfortranarray(a)
        return a

    if rank == 0:
        data = F(np.array([(0, 1), (4, 5)], dtype=dtype))
        result = nvmath.distributed.reshape.reshape(data, input_box=[(0, 0), (2, 2)], output_box=[(0, 0), (4, 1)])
        expected = np.array([0, 4, 8, 12], dtype=dtype).reshape((4, 1))
    elif rank == 1:
        data = F(np.array([(2, 3), (6, 7)], dtype=dtype))
        result = nvmath.distributed.reshape.reshape(data, input_box=[(0, 2), (2, 4)], output_box=[(0, 1), (4, 2)])
        expected = np.array([1, 5, 9, 13], dtype=dtype).reshape((4, 1))
    elif rank == 2:
        data = F(np.array([(8, 9), (12, 13)], dtype=dtype))
        result = nvmath.distributed.reshape.reshape(data, input_box=[(2, 0), (4, 2)], output_box=[(0, 2), (4, 3)])
        expected = np.array([2, 6, 10, 14], dtype=dtype).reshape((4, 1))
    else:
        data = F(np.array([(10, 11), (14, 15)], dtype=dtype))
        result = nvmath.distributed.reshape.reshape(data, input_box=[(2, 2), (4, 4)], output_box=[(0, 3), (4, 4)])
        expected = np.array([3, 7, 11, 15], dtype=dtype).reshape((4, 1))
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
def test_reset_operand_none(input_memory_space, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    device_id = distributed_ctx.device_id
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    global_shape = (16, 16)
    shape = calc_slab_shape(global_shape, 0, rank, nranks)
    box = _calculate_local_box(global_shape, 0, rank, nranks)
    dtype = np.complex64

    stream = None
    if input_memory_space == "gpu":
        stream = get_or_create_stream(device_id, stream=None, op_package="cupy")

    _, data_in = generate_random_data(np, input_memory_space, shape, dtype, stream)

    with nvmath.distributed.reshape.Reshape(data_in.tensor, box, box) as reshape:
        reshape.plan()
        result1 = reshape.execute()
        result2 = reshape.execute()
        if input_memory_space == "gpu":
            free_symmetric_memory(result1)
            free_symmetric_memory(result2)
        reshape.reset_operand(None)
        with pytest.raises(RuntimeError, match="Execution cannot be performed if the input operand has been set to None"):
            reshape.execute()
        reshape.reset_operand(data_in.tensor)
        result3 = reshape.execute()
        if input_memory_space == "gpu":
            free_symmetric_memory(result3)

    if input_memory_space == "gpu":
        free_symmetric_memory(data_in.tensor)


@pytest.mark.parametrize("package", ["numpy", "torch"])  # numpy uses cupy for GPU
@pytest.mark.parametrize("global_shape", [(128, 32), (128, 32, 64), (32, 32, 32)])
@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
@pytest.mark.parametrize("memory_order", ["C", "F"])
@pytest.mark.parametrize("provide_out", [False])
@pytest.mark.parametrize("reset_inplace", [True, False])
# For blocking we just test that it runs without error.
@pytest.mark.parametrize("blocking", [True, "auto"])
def test_distributed_reshape(
    package,
    global_shape,
    input_memory_space,
    memory_order,
    provide_out,
    reset_inplace,
    blocking,
    nvmath_distributed,
    check_symmetric_memory_leaks,
):
    """This test generates random data of the given global shape, partitioned across
    ranks according to the X-slab distribution used by cuFFTMp, and reshapes it to Y-slab
    distribution using the distributed Reshape operation. To verify correctness, at the
    end of the test:
      - The original input is gathered on the X dimension
      - The result is gathered on the Y-dimension
      - The gathered arrays must be equal

    The operands are of the type given by the specified package:
      - numpy if package is "numpy" and input_memory_space == "cpu"
      - cupy if package is "numpy" and input_memory_space == "gpu"
      - torch if package is "torch"
    """
    if input_memory_space == "cpu" and blocking == "auto":
        # CPU is always blocking, already captured by blocking=True.
        return

    if input_memory_space == "cpu" and reset_inplace:
        # reset_inplace tests resetting operand's data without calling reset_operand, and is
        # only for GPU operands.
        return

    try:
        pkg = package_name_to_package[package]
    except KeyError:
        pytest.skip(f"{package} is not available")

    package = pkg

    distributed_ctx = nvmath.distributed.get_context()
    device_id = distributed_ctx.device_id
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    stream = None
    if input_memory_space == "gpu":
        stream_package = "cupy" if package is np else package.__name__
        stream = get_or_create_stream(device_id, stream=None, op_package=stream_package)

    # With the generic NVSHMEM allocation helpers used in this test, we can only support
    # X and Y dimensions exactly divisible by # ranks.
    assert global_shape[0] % nranks == 0
    assert global_shape[1] % nranks == 0

    # Get X-slab shape for this rank.
    shape = calc_slab_shape(global_shape, 0, rank, nranks)
    dtype = np.complex64
    data_in_cpu, data_in = generate_random_data(package, input_memory_space, shape, dtype, stream, memory_layout=memory_order)

    # Reshape from X-slab to Y-slab distribution.
    input_box = _calculate_local_box(global_shape, 0, rank, nranks)
    output_box = _calculate_local_box(global_shape, 1, rank, nranks)

    if memory_order == "C":
        axis_order = tuple(reversed(range(len(global_shape))))
    else:
        assert memory_order == "F"
        axis_order = tuple(range(len(global_shape)))

    out = None
    if provide_out:
        out_shape = calc_slab_shape(global_shape, 1, rank, nranks)
        strides = calculate_strides(out_shape, axis_order)
        out = data_in.__class__.empty(
            out_shape, data_in.device_id, dtype=data_in.dtype, strides=strides, symmetric_memory=(input_memory_space == "gpu")
        )

    options = {"blocking": blocking}
    with nvmath.distributed.reshape.Reshape(data_in.tensor, input_box, output_box, options=options) as reshape:
        reshape.plan()
        # We'll do two distributed reshapes per test case (to test reset_operand).
        reshape_count = 0
        while True:
            # Copy input data just to check that the operation doesn't change the input in
            # any way.
            original_data_in = data_in.__class__.empty(
                data_in.shape,
                data_in.device_id,
                dtype=data_in.dtype,
                strides=data_in.strides,
                symmetric_memory=(input_memory_space == "gpu"),
            )
            with device_ctx(device_id):
                original_data_in.copy_(data_in, stream)

            # Run distributed reshape.
            result = reshape.execute()
            result = dist_wrap_operand(result)
            # Check that the result has the same memory layout as the input.
            assert result.strides == tuple(calculate_strides(result.shape, axis_order))
            reshape_count += 1
            assert data_in.module is result.module

            # Assert that the operation didn't change the input data.
            assert is_close(original_data_in, data_in), "Input changed by Reshape"
            if nranks > 1:
                assert not is_close(original_data_in, result)

            if original_data_in.device == "cuda":
                free_symmetric_memory(original_data_in.tensor)

            if provide_out:
                assert result.tensor is out.tensor

            if input_memory_space == "gpu":
                assert result.device == "cuda"
                result_cpu = to_host(result, device_id, stream)
            else:
                assert result.device == "cpu"
                result_cpu = result

            # If reset_inplace and provide_out, we can't free the result memory
            # yet, since it's going to be reused.
            if not (reset_inplace and provide_out) and input_memory_space == "gpu":
                free_symmetric_memory(result.tensor)

            # Check that the result has the expected shape.
            assert result_cpu.shape == calc_slab_shape(global_shape, 1, rank, nranks)

            # Gathering the input and output on their respective partition dimension, the
            # gathered arrays must be equal.
            data_in_cpu_global = gather_array(data_in_cpu, 0, comm, rank)
            del data_in_cpu

            result_cpu_global = gather_array(result_cpu, 1, comm, rank)
            del result_cpu

            if rank == 0:
                try:
                    assert is_close(result_cpu_global, data_in_cpu_global), "Gathered arrays don't match"
                    comm.bcast(None)
                except Exception as e:
                    # Broadcast the exception to avoid deadlock.
                    comm.bcast(e)
                    raise
            else:
                # If rank 0 raises an exception, every process has to do the same to avoid
                # deadlock.
                e = comm.bcast(None)
                if e is not None:
                    raise e
            del data_in_cpu_global, result_cpu_global

            if reshape_count < 2:
                data_in_cpu, data_in_new = generate_random_data(
                    package, input_memory_space, shape, dtype, stream, memory_layout=memory_order
                )
                if input_memory_space == "gpu" and reset_inplace:
                    with device_ctx(device_id):
                        data_in.copy_(data_in_new, stream)
                    free_symmetric_memory(data_in_new.tensor)
                else:
                    if input_memory_space == "gpu":
                        free_symmetric_memory(data_in.tensor)
                    data_in = data_in_new
                    if provide_out:
                        out = out.__class__.empty(
                            out_shape,
                            out.device_id,
                            dtype=out.dtype,
                            strides=out.strides,
                            symmetric_memory=(input_memory_space == "gpu"),
                        )
                        reshape.reset_operand(data_in.tensor, out=out.tensor)
                    else:
                        # assert reshape.out is None
                        reshape.reset_operand(data_in.tensor)
            else:
                if input_memory_space == "gpu":
                    free_symmetric_memory(data_in.tensor)
                    if reset_inplace and provide_out and input_memory_space == "gpu":
                        free_symmetric_memory(result.tensor)
                break


# This test only uses CPU operand.
@pytest.mark.parametrize("package", ["numpy", "torch"])
def test_distributed_reshape_1D(package, nvmath_distributed, check_symmetric_memory_leaks):
    """This test reshapes a 1D array that is evenly partitioned across ranks
    to one where the first 80 elements are on rank 0 and the remaining elements
    are evenly divided across the other ranks."""

    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test needs to be run with 2 or 4 PEs")

    assert nranks in (2, 4)

    try:
        pkg = package_name_to_package[package]
    except KeyError:
        pytest.skip(f"{package} is not available")

    package = pkg

    stream = None
    global_shape = (128,)
    shape = calc_slab_shape(global_shape, 0, rank, nranks)
    dtype = np.complex64
    data_in, _ = generate_random_data(package, "cpu", shape, dtype, stream)

    # Input box.
    input_box = _calculate_local_box(global_shape, 0, rank, nranks)
    # Calculate output box.
    nelems_per_other_rank = (global_shape[0] - 80) // (nranks - 1)
    if rank == 0:
        output_box = ([0], [80])
    else:
        lower = 80
        for i in range(1, rank):
            lower += nelems_per_other_rank
        output_box = [lower], [lower + nelems_per_other_rank]

    # Run distributed reshape.
    result = nvmath.distributed.reshape.reshape(data_in.tensor, input_box, output_box)
    result = dist_wrap_operand(result)

    assert data_in.module is result.module

    if rank == 0:
        assert result.shape == (80,)
    else:
        assert result.shape == (nelems_per_other_rank,)

    data_in_global = gather_array(data_in, 0, comm, rank)

    if rank == 0:
        result_global = result.__class__.empty(global_shape, result.device_id, dtype=result.dtype, symmetric_memory=False)
        sendcounts = [80] + [nelems_per_other_rank for i in range(nranks - 1)]
        comm.Gatherv(sendbuf=result.tensor, recvbuf=(result_global.tensor, sendcounts))
        try:
            assert is_close(result_global, data_in_global), "Gathered arrays don't match"
            comm.bcast(None)
        except Exception as e:
            # Broadcast the exception to avoid deadlock.
            comm.bcast(e)
            raise
    else:
        comm.Gatherv(result.tensor, [])
        # If rank 0 raises an exception, every process has to do the same to avoid deadlock.
        e = comm.bcast(None)
        if e is not None:
            raise e
