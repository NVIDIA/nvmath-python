# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import nvmath.distributed
from nvmath.internal.utils import device_ctx, get_or_create_stream
from nvmath.distributed import free_symmetric_memory
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand, maybe_register_package
from nvmath.distributed.fft._configuration import Slab

from .helpers import gather_array, generate_random_data, is_close, to_host
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


def test_unsupported_rank(nvmath_distributed, check_symmetric_memory_leaks):
    data = np.ones(10, dtype="complex64")
    with pytest.raises(
        ValueError,
        match="Distributed FFT is currently supported only for 2-D and 3-D tensors. "
        "The number of dimensions of the operand is 1.",
    ):
        nvmath.distributed.fft.fft(data, distribution=Slab.X)


@pytest.mark.parametrize("distribution", [Slab.X, Slab.Y])
def test_inconsistent_shape(distribution, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires multiple processes")

    if rank == 0:
        shape = (32, 20) if distribution == Slab.X else (20, 32)
    else:
        shape = (32, 32)

    data = np.ones(shape, dtype=np.complex64)
    with pytest.raises(ValueError, match="problem size is inconsistent"):
        nvmath.distributed.fft.fft(data, distribution=distribution)


@pytest.mark.parametrize("distribution", [Slab.X, Slab.Y])
def test_wrong_slab_shape(distribution, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks not in (2, 4):
        pytest.skip("This test requires 2 or 4 ranks")

    if nranks == 2:
        if rank == 0:
            shape = (10, 64) if distribution == Slab.X else (64, 10)
        else:
            shape = (20, 64) if distribution == Slab.X else (64, 20)
    else:
        if rank in (0, 1):
            shape = (20, 64) if distribution == Slab.X else (64, 20)
        elif rank == 2:
            shape = (15, 64) if distribution == Slab.X else (64, 15)
        elif rank == 3:
            shape = (25, 64) if distribution == Slab.X else (64, 25)

    data = np.ones(shape, dtype=np.complex64)
    with pytest.raises(ValueError, match=(r"The operand shape is \(\d+, \d+\), but the expected slab shape is \(\d+, \d+\)")):
        nvmath.distributed.fft.fft(data, distribution=distribution)


def test_inconsistent_rank(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires multiple processes")

    shape = (8, 8) if rank == 0 else (8, 8, 2)

    data = np.ones(shape, dtype=np.complex64)
    with pytest.raises(ValueError, match="The number of dimensions of the input operand is inconsistent across processes"):
        nvmath.distributed.fft.fft(data, distribution=Slab.Y)


def test_inconsistent_dtype(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires multiple processes")

    dtype = np.complex64 if rank == 0 else np.complex128
    data = np.ones((8, 8, 2), dtype=dtype)
    with pytest.raises(ValueError, match="The operand dtype is inconsistent across processes"):
        nvmath.distributed.fft.fft(data, distribution=Slab.X)


def test_inconsistent_options(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires multiple processes")

    options = {"reshape": True} if rank == 0 else {"reshape": False}
    data = np.ones((4, 4), dtype=np.complex64)
    with pytest.raises(ValueError, match="options are inconsistent across processes"):
        nvmath.distributed.fft.fft(data, distribution=Slab.Y, options=options)


def test_inconsistent_package(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires multiple processes")

    if "torch" not in package_name_to_package:
        pytest.skip("torch is not available")

    shape = (8, 8, 8)
    memory_space = "cpu"
    dtype = np.complex64
    if rank == 0:
        data, _ = generate_random_data(np, memory_space, shape, dtype, stream=None)
    else:
        import torch

        data, _ = generate_random_data(torch, memory_space, shape, dtype, stream=None)
    with pytest.raises(ValueError, match="operand doesn't belong to the same package on all processes"):
        nvmath.distributed.fft.fft(data.tensor, distribution=Slab.X)


def test_inconsistent_memory_space(nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    device_id = distributed_ctx.device_id
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if nranks == 1:
        pytest.skip("This test requires multiple processes")

    if "torch" not in package_name_to_package:
        pytest.skip("torch is not available")

    import torch

    shape = (8, 8, 8)
    dtype = np.complex64

    stream = get_or_create_stream(device_id, stream=None, op_package="cupy")
    _, gpu_data = generate_random_data(torch, "gpu", shape, dtype, stream=stream)

    if rank == 0:
        _, data = generate_random_data(torch, "cpu", shape, dtype, stream=None)
    else:
        data = gpu_data

    with pytest.raises(ValueError, match="operand is not on the same memory space"):
        nvmath.distributed.fft.fft(data.tensor, distribution=Slab.Y)

    free_symmetric_memory(gpu_data.tensor)


@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
def test_reset_operand_none(input_memory_space, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    device_id = distributed_ctx.device_id
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    global_shape = (16, 16)
    shape = calc_slab_shape(global_shape, 0, rank, nranks)
    dtype = np.complex64

    stream = None
    if input_memory_space == "gpu":
        stream = get_or_create_stream(device_id, stream=None, op_package="cupy")

    _, data_in = generate_random_data(np, input_memory_space, shape, dtype, stream)

    with nvmath.distributed.fft.FFT(data_in.tensor, distribution=Slab.X) as fft:
        fft.plan()
        fft.execute()
        fft.reset_operand(None)
        with pytest.raises(RuntimeError, match="Execution cannot be performed if the input operand has been set to None"):
            fft.execute()
        fft.reset_operand(data_in.tensor, distribution=Slab.X)
        fft.execute()

    if input_memory_space == "gpu":
        free_symmetric_memory(data_in.tensor)


def generate_data_with_padding(
    global_shape, distribution, package, device_id, input_memory_space, in_dtype, fft_type, stream, rank, nranks
):
    if isinstance(distribution, Slab):
        partition_dim = 0 if distribution == Slab.X else 1
        shape = calc_slab_shape(global_shape, partition_dim, rank, nranks)
    else:
        lower, upper = distribution[0]
        shape = tuple(upper[i] - lower[i] for i in range(len(global_shape)))

    # First generate random data without padding, then allocate padded arrays to copy into.
    data_in_cpu0, data_in0 = generate_random_data(package, input_memory_space, shape, in_dtype, stream)

    # Allocate padded CPU array.
    data_in_cpu = nvmath.distributed.fft.allocate_operand(
        shape, package, distribution=distribution, input_dtype=data_in_cpu0.tensor.dtype, memory_space="cpu", fft_type=fft_type
    )

    # Allocate padded GPU array.
    if input_memory_space == "gpu" and package is np:
        import cupy as package

    data_in = nvmath.distributed.fft.allocate_operand(
        shape,
        package,
        distribution=distribution,
        input_dtype=data_in0.tensor.dtype,
        memory_space="cuda" if input_memory_space == "gpu" else "cpu",
        fft_type=fft_type,
    )

    # Copy data to padded arrays and free the non-padded ones.
    data_in_cpu[:] = data_in_cpu0.tensor[:]
    with device_ctx(device_id):
        data_in[:] = data_in0.tensor[:]
    if input_memory_space == "gpu":
        free_symmetric_memory(data_in0.tensor)

    return dist_wrap_operand(data_in_cpu), dist_wrap_operand(data_in)


@pytest.mark.parametrize("package", ["numpy", "torch"])  # "numpy" value uses cupy for GPU
@pytest.mark.parametrize(
    "global_shape", [(8, 9), (8, 8), (9, 11, 8), (11, 9, 8), (31, 31, 31), (128, 32), (128, 32, 64), (32, 32, 32)]
)
@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
@pytest.mark.parametrize("fft_type", ["R2C", ("C2R", "even"), ("C2R", "odd"), "C2C"])
@pytest.mark.parametrize("reshape", [True, False, "use_box"])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
@pytest.mark.parametrize("reset_inplace", [True, False])
# For blocking we just test that it runs without error.
@pytest.mark.parametrize("blocking", [True, "auto"])
def test_distributed_fft(
    package,
    global_shape,
    input_memory_space,
    fft_type,
    reshape,
    direction,
    reset_inplace,
    blocking,
    nvmath_distributed,
    check_symmetric_memory_leaks,
):
    """This test runs distributed FFT with various combinations of options, and checks
    correctness by gathering the distributed result and comparing to cuFFT (single-GPU
    library).

    Test parameters:
      - package: Package that the input operand belongs to: numpy/cupy or torch.
      - global_shape: Global shape of the input for distributed FFT.
      - input_memory_space: Whether the input operand is in CPU or GPU.
      - reshape:
            - If bool, this indicates whether to redistribute the result back to the
              original slab distribution or not, using the cuFFTMp reshape API.
            - With reshape="use_box" we run the FFT using the custom slab/pencil
              distribution of cuFFTMp (by using the `box` option), and we have the
              output be the complementary slab distribution.
      - direction: initial FFT direction.
      - reset_inplace: Whether to reset operand by changing the contents of the current
        operand inplace or by calling `reset_operand(new_operand)`.
      - blocking: Operation is blocking or not.
    """

    last_axis_parity = "even"
    if isinstance(fft_type, tuple):
        fft_type, last_axis_parity = fft_type
        assert last_axis_parity in ("even", "odd")

    assert fft_type in ("C2C", "R2C", "C2R")

    if input_memory_space == "cpu" and blocking == "auto":
        # CPU is always blocking, already captured by blocking=True.
        pytest.skip("redundant test: input_memory_space='cpu' and blocking='auto'")

    if input_memory_space == "cpu" and reset_inplace:
        # reset_inplace tests resetting operand's data without calling reset_operand, and is
        # only for GPU operands.
        pytest.skip("reset_inplace doesn't apply to CPU operands")

    if fft_type == "R2C" and direction == "inverse":
        pytest.skip("invalid test parameter combination: R2C and direction='inverse'")
    if fft_type == "C2R" and direction == "forward":
        pytest.skip("invalid test parameter combination: C2R and direction='forward'")

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

    # To test both slab distributions, we use Slab.X distribution when starting with
    # FORWARD direction and Slab.Y when starting with INVERSE (note that Slab.X and Slab.Y
    # is a requirement for 2D R2C and C2R, respectively).
    distribution = Slab.X if direction == "forward" else Slab.Y
    partition_dim = 0 if distribution == Slab.X else 1
    complementary_partition_dim = 1 - partition_dim

    global_output_shape = list(global_shape)
    if fft_type == "C2C":
        in_dtype = np.complex64
    elif fft_type == "R2C":
        in_dtype = np.float32
        global_output_shape[-1] = global_output_shape[-1] // 2 + 1
    elif fft_type == "C2R":
        in_dtype = np.complex64
        global_output_shape[-1] = (global_output_shape[-1] - 1) * 2
        if last_axis_parity == "odd":
            global_output_shape[-1] += 1

    if reshape == "use_box":
        # Use the FFT box distribution option to get the complementary slab distribution
        # as output.
        in_shapes = [calc_slab_shape(global_shape, partition_dim, i, nranks) for i in range(nranks)]
        out_shapes = [calc_slab_shape(global_output_shape, complementary_partition_dim, i, nranks) for i in range(nranks)]
        input_box = calculate_box(partition_dim, None, in_shapes, global_shape, rank)
        output_box = calculate_box(complementary_partition_dim, None, out_shapes, global_output_shape, rank)
        distribution = [input_box, output_box]

    data_in_cpu, data_in = generate_data_with_padding(
        global_shape, distribution, package, device_id, input_memory_space, in_dtype, fft_type, stream, rank, nranks
    )

    options = {
        "reshape": reshape is True,
        "blocking": blocking,
        "fft_type": fft_type,
        "last_axis_parity": last_axis_parity,
    }
    with nvmath.distributed.fft.FFT(
        data_in.tensor,
        distribution=distribution,
        options=options,
    ) as fft:
        assert tuple(fft.global_extents) == tuple(global_shape)
        fft.plan()
        # We do a sequence of distributed FFTs per test case, to test reset_operand
        # and different combinations of changing the direction and distribution.
        fft_count = 0
        FFT_LIMIT = 2 if fft_type in ("R2C", "C2R") else 3 if reshape is True else 4
        while True:
            # Run distributed FFT.
            result = fft.execute(direction=direction, release_workspace=(fft_count == 0))
            result = dist_wrap_operand(result)
            fft_count += 1
            assert data_in.module is result.module

            if fft_type in ("C2C", "R2C"):
                assert result.dtype == "complex64"
            else:
                assert result.dtype == "float32"

            if data_in.shape == result.shape:
                assert data_in.tensor is result.tensor
            assert data_in.data_ptr == result.data_ptr

            if input_memory_space == "gpu":
                assert result.device == "cuda"
                if fft_type == "C2R":
                    # C2R result is strided, causing TensorHolder.to() to fail because of
                    # mismatch between shape and strides, so we just make it contiguous to
                    # avoid the issue.
                    with device_ctx(device_id):
                        tensor_contiguous = result.tensor.copy() if package is np else result.tensor.contiguous()
                    result_cpu = to_host(dist_wrap_operand(tensor_contiguous), device_id, stream)
                else:
                    result_cpu = to_host(result, device_id, stream)
            else:
                assert result.device == "cpu"
                result_cpu = result

            # Compare the result with single-GPU FFT
            data_in_cpu_global = gather_array(data_in_cpu, partition_dim, comm, rank)
            del data_in_cpu
            if reshape is True:
                # With reshape, result must have the original distribution.
                assert result_cpu.shape == calc_slab_shape(global_output_shape, partition_dim, rank, nranks)
                result_cpu_global = gather_array(result_cpu, partition_dim, comm, rank)
            else:
                # Without reshape, the result shape must have the complementary
                # slab distribution.
                complementary_partition_dim = 1 if partition_dim == 0 else 0
                assert result_cpu.shape == calc_slab_shape(global_output_shape, complementary_partition_dim, rank, nranks)
                result_cpu_global = gather_array(result_cpu, complementary_partition_dim, comm, rank)
            if rank == 0:
                with nvmath.fft.FFT(
                    data_in_cpu_global.tensor,
                    options={
                        "inplace": False,
                        "result_layout": "natural",
                        "fft_type": fft_type,
                        "last_axis_parity": last_axis_parity,
                    },
                    execution="cuda",
                ) as single_gpu_fft:
                    single_gpu_fft.plan(direction=direction)
                    result_single_gpu = single_gpu_fft.execute(direction=direction)
                result_single_gpu = nvmath.internal.tensor_wrapper.wrap_operand(result_single_gpu)
                try:
                    assert is_close(result_cpu_global, result_single_gpu, rtol=3e-02, atol=1e-05), (
                        "Gathered result doesn't match single-GPU FFT"
                    )
                    comm.bcast(None)
                except Exception as e:
                    # Broadcast the exception to avoid deadlock.
                    comm.bcast(e)
                    raise
                del result_single_gpu
            else:
                # If rank 0 raises an exception, every process has to do the same to avoid
                # deadlock.
                e = comm.bcast(None)
                if e is not None:
                    raise e
            del data_in_cpu_global, result_cpu_global

            if fft_count == FFT_LIMIT:
                if input_memory_space == "gpu":
                    free_symmetric_memory(data_in.tensor)
                break

            call_reset_operand = True

            def swap_distribution():
                assert reshape != True  # noqa: E712
                assert fft_type == "C2C"
                if reshape == "use_box":
                    dist = (distribution[1], distribution[0])
                else:
                    dist = Slab.X if distribution == Slab.Y else Slab.Y
                p_dim = 1 if partition_dim == 0 else 0
                shape = calc_slab_shape(global_shape, p_dim, rank, nranks)
                return dist, p_dim, shape

            if fft_count == 1 and reset_inplace:
                call_reset_operand = False
            elif fft_count == 2 and reshape is True:
                direction = "inverse" if direction == "forward" else "forward"  # change direction
            elif fft_count == 2 and reshape in (False, "use_box"):
                distribution, partition_dim, shape = swap_distribution()  # change distribution
            elif fft_count == 3 and reshape in (False, "use_box"):
                direction = "inverse" if direction == "forward" else "forward"  # change both
                distribution, partition_dim, shape = swap_distribution()

            data_in_cpu, data_in_new = generate_data_with_padding(
                global_shape, distribution, package, device_id, input_memory_space, in_dtype, fft_type, stream, rank, nranks
            )
            if not call_reset_operand:
                assert reset_inplace and input_memory_space == "gpu"
                with device_ctx(device_id):
                    data_in.copy_(data_in_new, stream)
                free_symmetric_memory(data_in_new.tensor)
            else:
                if input_memory_space == "gpu":
                    free_symmetric_memory(data_in.tensor)
                data_in = data_in_new
                fft.reset_operand(data_in.tensor, distribution=distribution)


def calculate_box(dim0, dim1, shapes, global_shape, rank):
    # Calculate box of this rank within specified global shape,
    # given the local shapes on each rank.
    lower = [0 for i in range(len(global_shape))]
    for i in range(rank):
        if dim1 is not None:
            lower[dim1] = (lower[dim1] + shapes[i][dim1]) % global_shape[dim1]
            if lower[dim1] == 0:
                lower[dim0] += shapes[i][dim0]
        else:
            lower[dim0] += shapes[i][dim0]
    upper = list(lower)
    for i in range(len(upper)):
        upper[i] += shapes[rank][i]
    return (lower, upper)


def gather_pencils(x, dim0, dim1, shape, global_shape, comm, rank, nranks):
    # First we use Reshape to convert pencil distribution to X-slab, then
    # we gather the array on rank 0.
    input_box = calculate_box(dim0, dim1, [shape] * nranks, global_shape, rank)
    slab_shape = calc_slab_shape(global_shape, 0, rank, nranks)
    output_box = calculate_box(0, None, [slab_shape] * nranks, global_shape, rank)
    x = nvmath.distributed.reshape.reshape(x.tensor, input_box, output_box)
    x = dist_wrap_operand(x)
    return gather_array(x, 0, comm, rank)


@pytest.mark.need_4_procs
@pytest.mark.parametrize("package", ["numpy"])  # numpy uses cupy for GPU
@pytest.mark.parametrize("global_shape", [(32, 32, 32)])
@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_distributed_fft_pencils(
    package,
    global_shape,
    input_memory_space,
    direction,
    nvmath_distributed,
    check_symmetric_memory_leaks,
):
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

    if nranks != 4:
        pytest.skip("This test requires 4 ranks")

    assert global_shape[0] % nranks == 0
    assert global_shape[1] % nranks == 0
    assert global_shape[2] % nranks == 0

    stream = None
    if input_memory_space == "gpu":
        stream_package = "cupy" if package is np else package.__name__
        stream = get_or_create_stream(device_id, stream=None, op_package=stream_package)

    X, Y, Z = global_shape
    # Every process has the same pencil shape.
    # Input will have pencils partitioned on X and Y.
    # Output will have pencils partitioned on Y and Z.
    input_pencil_shape = X // 2, Y // 2, Z
    output_pencil_shape = X, Y // 2, Z // 2

    input_box = calculate_box(0, 1, [input_pencil_shape] * nranks, global_shape, rank)
    output_box = calculate_box(1, 2, [output_pencil_shape] * nranks, global_shape, rank)
    distribution = [input_box, output_box]

    dtype = np.complex128
    data_in_cpu, data_in = generate_random_data(package, input_memory_space, input_pencil_shape, dtype, stream)

    with nvmath.distributed.fft.FFT(
        data_in.tensor,
        distribution=distribution,
    ) as fft:
        assert tuple(fft.global_extents) == tuple(global_shape)
        fft.plan()

        # Run distributed FFT.
        result = fft.execute(direction=direction)
        result = dist_wrap_operand(result)
        assert data_in.module is result.module

        assert data_in.shape != result.shape
        assert data_in.data_ptr == result.data_ptr

        if input_memory_space == "gpu":
            assert result.device == "cuda"
            result_cpu = to_host(result, device_id, stream)
        else:
            assert result.device == "cpu"
            result_cpu = result

        # Compare the result with single-GPU FFT
        data_in_cpu_global = gather_pencils(data_in_cpu, 0, 1, input_pencil_shape, global_shape, comm, rank, nranks)
        del data_in_cpu

        result_cpu_global = gather_pencils(result_cpu, 1, 2, output_pencil_shape, global_shape, comm, rank, nranks)
        del result_cpu

        if rank == 0:
            result_single_gpu = nvmath.fft.fft(
                data_in_cpu_global.tensor,
                direction=direction,
                options={"inplace": False, "result_layout": "natural"},
                execution="cuda",
            )
            result_single_gpu = dist_wrap_operand(result_single_gpu)
            try:
                assert is_close(result_cpu_global, result_single_gpu, rtol=3e-02, atol=1e-05), (
                    "Gathered result doesn't match single-GPU FFT"
                )
                comm.bcast(None)
            except Exception as e:
                # Broadcast the exception to avoid deadlock.
                comm.bcast(e)
                raise
            del result_single_gpu
        else:
            # If rank 0 raises an exception, every process has to do the same to avoid
            # deadlock.
            e = comm.bcast(None)
            if e is not None:
                raise e
        del data_in_cpu_global, result_cpu_global

        if input_memory_space == "gpu":
            free_symmetric_memory(data_in.tensor)
