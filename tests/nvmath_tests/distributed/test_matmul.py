# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np
import pytest
import random
import re
import cuda.core.experimental as ccx
from collections.abc import Sequence

from pathlib import Path
import tempfile
import os

import nvmath.distributed
from nvmath.internal.utils import device_ctx, get_or_create_stream
from nvmath.distributed import free_symmetric_memory
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand, maybe_register_package
from nvmath.internal.tensor_wrapper import wrap_operand

from .helpers import gather_array, generate_random_data, is_close, to_host

from nvmath.internal.typemaps import NAME_TO_DATA_TYPE, NAME_TO_DATA_WIDTH

from nvmath.distributed.linalg.advanced import matrix_qualifiers_dtype, MatmulEpilog, MatmulComputeType

from nvmath.distributed.distribution import ProcessGrid, BlockNonCyclic, BlockCyclic, Slab, Box

from nvmath.bindings import cublasMp

import cuda.core.experimental

package_name_to_package = {"numpy": np}


@pytest.fixture(scope="module")
def nvmath_distributed():
    """Pytest fixture that initializes nvmath.distributed and finalizes it on exit"""
    from mpi4py import MPI

    try:
        import cupy

        maybe_register_package("cupy")
        package_name_to_package["cupy"] = cupy
    except ImportError:
        pass

    try:
        import torch

        maybe_register_package("torch")
        package_name_to_package["torch"] = torch
    except ImportError:
        pass

    device_id = MPI.COMM_WORLD.Get_rank() % cuda.core.experimental.system.num_devices
    nvmath.distributed.initialize(device_id, MPI.COMM_WORLD, backends=["nvshmem", "nccl"])

    yield

    nvmath.distributed.finalize()


@pytest.fixture(scope="module", autouse=True)
def cublasmp_logfile():
    # We're not using the cuBLASMp logging runtime APIs for now, which are considered
    # experimental. When setting the log file through env vars, the log file gets fixed
    # when the library is initialized and there is no way to change it per matmul
    # operation. So we need to select the file at the module scope.
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / f"cublasmp_{rank}.log"
        prev_log_level = os.environ.get("CUBLASMP_LOG_LEVEL", "")
        prev_log_file = os.environ.get("CUBLASMP_LOG_FILE", "")
        os.environ["CUBLASMP_LOG_LEVEL"] = "5"
        os.environ["CUBLASMP_LOG_FILE"] = str(temp_file_path)

        yield temp_file_path

        os.environ["CUBLASMP_LOG_LEVEL"] = prev_log_level
        os.environ["CUBLASMP_LOG_FILE"] = prev_log_file


@pytest.fixture(scope="function")
def cublasmp_logfile_with_cleanup(cublasmp_logfile):
    """Clear the log file after every test."""

    def truncate_log():
        try:
            # Can't delete the file because libcublasmp won't reopen without restarting
            # the application, so we truncate it instead to clear its contents and reuse
            # it in the same session.
            os.truncate(cublasmp_logfile, 0)
        except FileNotFoundError:
            pass

    truncate_log()  # in case the previous test hasn't used this fixture

    yield cublasmp_logfile

    truncate_log()


def test_wrong_distribution(nvmath_distributed):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    valid_nranks = (2, 4, 8)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    process_grid = ProcessGrid(shape=(1, nranks), layout=ProcessGrid.Layout.ROW_MAJOR)

    global_shape = (64, 64)
    assert global_shape[1] % nranks == 0

    # ERROR: all ranks must have the same dim 0 length
    nrows = 60 if rank == 0 else global_shape[0]
    ncols = global_shape[1] // nranks  # partition on dim 1
    a = np.zeros((nrows, ncols), dtype=np.float32)
    a = np.asfortranarray(a)

    distributions = [BlockNonCyclic(process_grid)] * 3
    with pytest.raises(ValueError, match="The problem size is inconsistent across processes"):
        _ = nvmath.distributed.linalg.advanced.Matmul(a, a, distributions=distributions)


@pytest.mark.parametrize("symmetric_memory", [False, True])
def test_symmetric_memory(symmetric_memory, nvmath_distributed, check_symmetric_memory_leaks):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    nranks = comm.Get_size()
    device_id = distributed_ctx.device_id

    m, n, k = 64, 32, 48
    a_shape = (k // nranks, m)
    b_shape = (k // nranks, n)

    import cupy as cp

    if symmetric_memory:
        # allocate a and b on symmetric memory
        a = nvmath.distributed.allocate_symmetric_memory(a_shape, cp, dtype=cp.float32, axis_order="F")
        b = nvmath.distributed.allocate_symmetric_memory(b_shape, cp, dtype=cp.float32, axis_order="F")
    else:
        with device_ctx(device_id):
            a = cp.asfortranarray(cp.zeros(a_shape))
            b = cp.asfortranarray(cp.zeros(b_shape))

    with device_ctx(device_id):
        a[:] = cp.random.rand(*a_shape)
        b[:] = cp.random.rand(*b_shape)
        stream = cp.cuda.Stream()

    distributions = [Slab.X, Slab.X, Slab.Y]
    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = True
    d = nvmath.distributed.linalg.advanced.matmul(a, b, distributions=distributions, qualifiers=qualifiers, stream=stream)

    stream.synchronize()

    d = dist_wrap_operand(d)
    assert d.device_id == device_id
    assert d.is_symmetric_memory == symmetric_memory

    if symmetric_memory:
        nvmath.distributed.free_symmetric_memory(a, b, d.tensor)


@pytest.mark.parametrize("global_size", [32, 64, 48])
def test_matmul_execute_sequence(global_size, nvmath_distributed, check_symmetric_memory_leaks):
    """Calculate A^4 where A is a square matrix, by creating and planning three separate
    matmuls, which then execute in sequence."""

    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    device_id = distributed_ctx.device_id

    valid_nranks = (1, 2, 4, 8)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    assert global_size % nranks == 0
    matrix_shape = (global_size // nranks, global_size)

    import cupy as cp

    stream = get_or_create_stream(device_id, stream=None, op_package="cupy")
    with device_ctx(device_id):
        a = cp.random.rand(*matrix_shape).astype(cp.float32)
        a = cp.asfortranarray(a)
        a_ = a.copy(order="F")

    distributions = [Slab.X] * 3
    mm1 = nvmath.distributed.linalg.advanced.Matmul(a, a_, distributions=distributions)
    mm2 = nvmath.distributed.linalg.advanced.Matmul(a, a_, distributions=distributions)
    mm3 = nvmath.distributed.linalg.advanced.Matmul(a, a_, distributions=distributions)

    mm1.plan()
    mm2.plan()
    mm3.plan()

    with device_ctx(device_id):
        a[:] = mm1.execute()
        a[:] = mm2.execute()
        d = mm3.execute()

    for mm in (mm1, mm2, mm3):
        mm.free()

    a_global = gather_array(to_host(dist_wrap_operand(a_), device_id, stream), 0, comm, rank)
    result_global = gather_array(to_host(dist_wrap_operand(d), device_id, stream), 0, comm, rank)
    if rank == 0:
        a = a_global.tensor
        expected = a @ a @ a @ a
        assert is_close(result_global, wrap_operand(expected), rtol=1e-5, atol=1e-5), (
            "Gathered result doesn't match single-GPU matmul"
        )


def generate_process_grids(only_2d=False):
    """Generate all possible process grids for the current number of MPI processes."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nranks = comm.Get_size()
    # Return process grids as tuples of process grid shape and layout. We can't create
    # ProcessGrid objects here because nvmath.distributed has not been initialized yet.
    process_grids = []
    for i in range(1, nranks + 1):
        for j in range(1, nranks + 1):
            if only_2d and (i == 1 or j == 1):
                continue
            if i * j == nranks:
                process_grids.append(((i, j), ProcessGrid.Layout.COL_MAJOR))
                process_grids.append(((i, j), ProcessGrid.Layout.ROW_MAJOR))
    return process_grids


def read_algo_from_log(logfile_path) -> int | None:
    with open(logfile_path) as logfile:
        # NOTE: need to call mm.execute() to see this printed to the logfile.
        regexAlgo = re.compile(r"\[cublasMpMatmul\] using matmul algo (\d+)$")
        for line in logfile:
            m = regexAlgo.search(line)
            if m:
                return int(m.group(1))
    return None


def skip_test_uniform_1d_distributions(
    package,
    input_memory_space,
    M_N_K,
    transA,
    transB,
    A_distribution,
    B_distribution,
    C_distribution,
    input_C,
    epilog_AR,
):
    if epilog_AR:
        if not (transA and not transB):
            # GEMM+AR algo only supported for TN
            return True
        if not (A_distribution == "R" and B_distribution == "R" and C_distribution == "C"):
            # GEMM+AR algo requires A and B row-wise and C col-wise
            return True

    if package == "numpy" and input_memory_space != "cpu":
        return True  # numpy only supports CPU memory space
    if package == "cupy" and input_memory_space != "gpu":
        return True  # cupy only supports GPU memory space


@pytest.mark.uncollect_if(func=skip_test_uniform_1d_distributions)
@pytest.mark.parametrize("package", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
@pytest.mark.parametrize("M_N_K", [(64, 64, 64), (128, 96, 64), (64, 128, 64)])
@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("transB", [True, False])
@pytest.mark.parametrize("A_distribution", ["R", "C"])
@pytest.mark.parametrize("B_distribution", ["R", "C"])
@pytest.mark.parametrize("C_distribution", ["R", "C"])  # same distribution applies to D
@pytest.mark.parametrize("input_C", [False, True])
@pytest.mark.parametrize("epilog_AR", [False, True])
def test_uniform_1d_distributions(
    package,
    input_memory_space,
    M_N_K,
    transA,
    transB,
    A_distribution,
    B_distribution,
    C_distribution,
    input_C,
    epilog_AR,
    nvmath_distributed,
    cublasmp_logfile_with_cleanup,
    check_symmetric_memory_leaks,
):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    device_id = distributed_ctx.device_id

    valid_nranks = (1, 2, 4, 8)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    try:
        pkg = package_name_to_package[package]
    except KeyError:
        pytest.skip(f"{package} is not available")

    M, N, K = M_N_K
    assert M % nranks == 0
    assert N % nranks == 0
    assert K % nranks == 0

    assert all(d in ("C", "R") for d in (A_distribution, B_distribution, C_distribution))

    if nranks == 1:
        # With nranks=1 cuBLASMp always does a local GEMM.
        expected_algo = 5  # local GEMM
    else:
        expected_algo = 0  # naive
        if epilog_AR:
            expected_algo = 4  # GEMM+AR
        elif (
            C_distribution == "R" and A_distribution == ("C" if transA else "R") and B_distribution == ("R" if transB else "C")
        ):
            expected_algo = 3  # AG+GEMM
        elif (
            C_distribution == "C" and A_distribution == ("R" if transA else "C") and B_distribution == ("C" if transB else "R")
        ):
            expected_algo = 2  # GEMM+RS

    # Generate some random numbers and broadcast them because every process
    # must use the same.
    r = np.random.rand(3)
    r[2] = random.randint(0, nranks - 1)
    comm.Bcast(r)
    if r[0] < 0.5:
        RowWiseDist = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)))
        assert RowWiseDist._is_row_wise()
    else:
        RowWiseDist = Slab.X

    if r[1] < 0.5:
        ColWiseDist = BlockNonCyclic(ProcessGrid(shape=(1, nranks)))
        assert ColWiseDist._is_col_wise()
    else:
        ColWiseDist = Slab.Y

    if A_distribution == "R":
        if expected_algo == 0:
            # Currently (rsrc, csrc) != (0, 0) only works for algo 0 (naive algorithm).
            first_process = (int(r[2]), 0)
            distribution_A = BlockNonCyclic(ProcessGrid(shape=(nranks, 1)), first_process=first_process)
        else:
            distribution_A = RowWiseDist
        A_shape = (K // nranks, M) if transA else (M // nranks, K)
    else:
        distribution_A = ColWiseDist
        A_shape = (K, M // nranks) if transA else (M, K // nranks)

    if B_distribution == "R":
        distribution_B = RowWiseDist
        B_shape = (N // nranks, K) if transB else (K // nranks, N)
    else:
        distribution_B = ColWiseDist
        B_shape = (N, K // nranks) if transB else (K, N // nranks)

    if C_distribution == "R":
        distribution_C = RowWiseDist
        C_shape = (M // nranks, N) if not epilog_AR else (M, N)
    else:
        distribution_C = ColWiseDist
        C_shape = (M, N // nranks) if not epilog_AR else (M, N)

    stream = None
    if input_memory_space == "gpu":
        stream = get_or_create_stream(device_id, stream=None, op_package=package)

    dtype = np.float32

    def generate_random_matrix(shape, dtype, symmetric_memory):
        return generate_random_data(
            np if package != "torch" else pkg,
            input_memory_space,
            shape,
            dtype,
            stream,
            memory_layout="F",
            symmetric_memory=symmetric_memory,
        )

    a_cpu, a = generate_random_matrix(A_shape, dtype, False)
    b_cpu, b = generate_random_matrix(B_shape, dtype, True)
    if input_C:
        beta = 0.8
        c_cpu, c = generate_random_matrix(C_shape, dtype, False)
        if epilog_AR:
            # For epilog_AR cuBLASMp has each process contribute its C to the result.
            # To get the same result as single-GPU MM, we have to set the values
            # to zero on every rank except one.
            c_cpu.tensor[:] = 7.0 if rank == 0 else 0.0
            with device_ctx(device_id):
                c.tensor[:] = 7.0 if rank == 0 else 0.0
    else:
        beta = c_cpu = c = None

    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = transA
    qualifiers[1]["is_transpose"] = transB
    distributions = [distribution_A, distribution_B, distribution_C]
    # For 1D uniform distribution we don't have to pass blocking sizes and can
    # let Matmul infer them.
    with nvmath.distributed.linalg.advanced.Matmul(
        a.tensor,
        b.tensor,
        c=c.tensor if input_C else None,
        beta=beta,
        distributions=distributions,
        qualifiers=qualifiers,
    ) as mm:
        assert M == mm.mm_traits.M
        assert N == mm.mm_traits.N
        assert K == mm.mm_traits.K
        mm.plan(epilog=MatmulEpilog.ALLREDUCE if epilog_AR else None)
        assert expected_algo == mm._expected_algo

        mm_count = 0
        MM_LIMIT = 2
        while True:
            d = mm.execute()
            mm_count += 1
            d = dist_wrap_operand(d)

            assert d.module is a.module
            assert d.dtype == "float32"
            if input_memory_space == "gpu":
                assert d.device == "cuda"
                d_cpu = to_host(d, device_id, stream)
                # matmul was called with some operands on symmetric memory and others not,
                # so the result won't be on symmetric memory.
                assert not d.is_symmetric_memory
            else:
                assert d.device == "cpu"
                d_cpu = d

            if b.is_symmetric_memory:
                free_symmetric_memory(b.tensor)

            if isinstance(distributions[0], BlockCyclic) and distributions[0].first_process != (0, 0):
                # Reshape A to a BlockNonCyclic distribution with first_process=(0,0) before
                # gathering.
                assert distributions[0].process_grid.shape == (nranks, 1)
                assert distributions[0].first_process[0] > 0
                rank_adjusted = (rank - distributions[0].first_process[0]) % nranks
                mb, nb = A_shape

                lower = (mb * rank_adjusted, 0)
                upper = (mb * rank_adjusted + mb, nb)
                input_box = Box(lower, upper)

                lower = (mb * rank, 0)
                upper = (mb * rank + mb, nb)
                output_box = Box(lower, upper)

                a_cpu = nvmath.distributed.reshape.reshape(a_cpu.tensor, input_box, output_box)
                a_cpu = dist_wrap_operand(a_cpu)

            a_global = gather_array(a_cpu, 0 if A_distribution == "R" else 1, comm, rank)
            b_global = gather_array(b_cpu, 0 if B_distribution == "R" else 1, comm, rank)
            if epilog_AR:
                # C/D is not actually distributed (it's replicated on all processes).
                if input_C:
                    c_global = c_cpu
                d_global = d_cpu
            else:
                if input_C:
                    c_global = gather_array(c_cpu, 0 if C_distribution == "R" else 1, comm, rank)
                d_global = gather_array(d_cpu, 0 if C_distribution == "R" else 1, comm, rank)
            if rank == 0:
                if input_C:
                    assert c_global.shape == (M, N)
                assert d_global.shape == (M, N)
                single_gpu_result = nvmath.linalg.advanced.matmul(
                    a_global.tensor.T if transA else a_global.tensor,
                    b_global.tensor.T if transB else b_global.tensor,
                    c=c_global.tensor if input_C else None,
                    beta=beta,
                )
                single_gpu_result = wrap_operand(single_gpu_result)
                try:
                    assert is_close(d_global, single_gpu_result, rtol=1e-5, atol=1e-5), (
                        "Gathered result doesn't match single-GPU matmul"
                    )

                    algo = read_algo_from_log(cublasmp_logfile_with_cleanup)
                    assert algo is not None, "Couldn't determine the distributed matmul algorithm used"
                    assert algo == expected_algo, (
                        f"cuBLASMp didn't run the expected distributed algorithm: algo is {algo}, "
                        f"expected algo is {expected_algo}"
                    )

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

            if mm_count == MM_LIMIT:
                break

            # Reset operands.
            a_cpu, a = generate_random_matrix(A_shape, dtype, False)
            b_cpu, b = generate_random_matrix(B_shape, dtype, True)
            if input_C:
                beta = 0.5
                c_cpu, c = generate_random_matrix(C_shape, dtype, False)
                if epilog_AR:
                    # For epilog_AR cuBLASMp has each process contribute its C to
                    # the result. To get the same result as single-GPU MM, we have
                    # to set the values to zero on every rank except one.
                    c_cpu.tensor[:] = 10.0 if rank == 0 else 0.0
                    with device_ctx(device_id):
                        c.tensor[:] = 10.0 if rank == 0 else 0.0
            else:
                beta = c_cpu = c = None
            mm.reset_operands(a.tensor, b.tensor, c.tensor if c is not None else None, beta=beta)


@pytest.mark.parametrize("M_N_K", [(64, 64, 64), (128, 96, 64), (64, 128, 64)])
@pytest.mark.parametrize("transA", [True, False])
@pytest.mark.parametrize("transB", [True, False])
@pytest.mark.parametrize("input_C", [False, True])
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("process_grid", generate_process_grids(only_2d=True))
def test_2d_block(
    M_N_K,
    transA,
    transB,
    input_C,
    cyclic,
    process_grid,
    nvmath_distributed,
    cublasmp_logfile_with_cleanup,
    check_symmetric_memory_leaks,
):
    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    valid_nranks = (1, 2, 4)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    M, N, K = M_N_K
    assert M % nranks == 0
    assert N % nranks == 0
    assert K % nranks == 0

    # Use the same process grid for A, B and C/D.
    process_grid = ProcessGrid(shape=process_grid[0], layout=process_grid[1])
    if cyclic:
        distribution = BlockCyclic(process_grid, (4, 4))
    else:
        distribution = BlockNonCyclic(process_grid)

    A_shape = distribution.shape(rank, (K, M) if transA else (M, K))
    B_shape = distribution.shape(rank, (N, K) if transB else (K, N))
    C_shape = distribution.shape(rank, (M, N))

    a = np.asfortranarray(np.random.rand(*A_shape).astype(np.float32))
    b = np.asfortranarray(np.random.rand(*B_shape).astype(np.float32))
    if input_C:
        beta = 0.8
        c = np.asfortranarray(np.random.rand(*C_shape).astype(np.float32))
    else:
        beta = c = None

    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = transA
    qualifiers[1]["is_transpose"] = transB
    with nvmath.distributed.linalg.advanced.Matmul(
        a,
        b,
        c=c,
        beta=beta,
        distributions=[distribution] * 3,
        qualifiers=qualifiers,
    ) as mm:
        # Check that the global matrix sizes were inferred correctly.
        assert M == mm.mm_traits.M
        assert N == mm.mm_traits.N
        assert K == mm.mm_traits.K
        mm.plan()
        assert mm._expected_algo in (-1, 0, 5)  # unknown or naive or local
        d = mm.execute()

    nprow, npcol = process_grid.shape
    myprow = rank % nprow if process_grid.layout == ProcessGrid.Layout.COL_MAJOR else rank // npcol
    mypcol = rank // nprow if process_grid.layout == ProcessGrid.Layout.COL_MAJOR else rank % npcol

    def gather_matrix(matrix, mb, nb, global_shape):
        # Reshape matrix to 1D column-wise (partitioning on Y) to be able to gather it.
        lower = (myprow * mb, mypcol * nb)
        upper = (lower[0] + mb, lower[1] + nb)
        input_box = Box(lower, upper)
        output_box = Box((0, global_shape[1] // nranks * rank), (global_shape[0], global_shape[1] // nranks * (rank + 1)))
        matrix = nvmath.distributed.reshape.reshape(matrix, input_box, output_box)
        # Gather matrix
        return gather_array(dist_wrap_operand(matrix), 1, comm, rank)

    # For gather we ignore the cyclic property, it doesn't affect correctness testing
    # since cyclic determines a global permutation of values but the values themselves
    # don't change.
    a_global = gather_matrix(a, A_shape[0], A_shape[1], (K, M) if transA else (M, K))
    b_global = gather_matrix(b, B_shape[0], B_shape[1], (N, K) if transB else (K, N))
    if input_C:
        c_global = gather_matrix(c, C_shape[0], C_shape[1], (M, N))
    d_global = gather_matrix(d, C_shape[0], C_shape[1], (M, N))

    if rank == 0:
        single_gpu_result = nvmath.linalg.advanced.matmul(
            a_global.tensor.T if transA else a_global.tensor,
            b_global.tensor.T if transB else b_global.tensor,
            c=c_global.tensor if input_C else None,
            beta=beta,
        )
        single_gpu_result = wrap_operand(single_gpu_result)
        try:
            assert is_close(d_global, single_gpu_result, rtol=1e-5, atol=1e-5), (
                "Gathered result doesn't match single-GPU matmul"
            )

            algo = read_algo_from_log(cublasmp_logfile_with_cleanup)
            assert algo is not None, "Couldn't determine the distributed matmul algorithm used"
            expected_algo = 1 if cyclic else 0  # 0 is naive, 1 is SUMMA.
            assert algo == expected_algo, (
                f"cuBLASMp didn't run the expected distributed algorithm: algo is {algo}, expected algo is {expected_algo}"
            )

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


def skip_test_global_shape_inference(global_shape, process_grid, blocking_sizes):
    nprow, npcol = process_grid[0]
    if blocking_sizes != "non-cyclic":
        mb, nb = blocking_sizes
        if nprow != 1 and npcol != 1 and (mb == "all" or nb == "all"):
            return True  # 'all' block size not used for 2D block distributions
        if mb == "all" and nprow != 1:
            return True  # "mb='all' row block size requires (1, N) process grid
        if nb == "all" and npcol != 1:
            return True  # nb='all' col block size requires (N, 1) process grid
    return False


@pytest.mark.uncollect_if(func=skip_test_global_shape_inference)
# This test only uses square matrices.
@pytest.mark.parametrize("global_shape", [64])
@pytest.mark.parametrize("process_grid", generate_process_grids())
@pytest.mark.parametrize(
    "blocking_sizes",
    [
        (1, "all"),
        (2, "all"),
        (3, "all"),
        ("all", 1),
        ("all", 4),
        ("all", 7),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (4, 3),
        (4, 4),
        (5, 3),
        "non-cyclic",
    ],
)
def test_global_shape_inference(global_shape, process_grid, blocking_sizes, nvmath_distributed):
    """This tests that global shape inference works under a wide range of BlockCyclic
    distributions, generated from all possible 1D and 2D process grids (given the
    number of processes running the test) and various block sizes (cyclic and non-cyclic).
    It doesn't run matmul end-to-end because the block sizes don't match across
    A, B, C/D for matching dimensions."""

    # test parameter with blocking size "all" means that the block size in that dimension
    # is the full length of the global matrix in that dimension. Only used with 1D
    # distributions for the dimension that is not partitioned.

    # Note that for 1D distributions, for the dimension that is not partitioned, a
    # blocking size < 'all' is also valid (it simply means that the process has a number of
    # contiguous blocks), and is in fact used to specify some algorithms in cuBLASMp.

    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()

    process_grid = ProcessGrid(shape=process_grid[0], layout=process_grid[1])
    nprow, npcol = process_grid.shape

    if blocking_sizes == "non-cyclic":
        mb = math.ceil(global_shape / nprow)
        nb = math.ceil(global_shape / npcol)
    else:
        mb, nb = blocking_sizes
        if mb == "all":
            mb = global_shape
        if nb == "all":
            nb = global_shape

    myprow = rank % nprow if process_grid.layout == ProcessGrid.Layout.COL_MAJOR else rank // npcol
    mypcol = rank // nprow if process_grid.layout == ProcessGrid.Layout.COL_MAJOR else rank % npcol

    local_nrows = cublasMp.numroc(global_shape, mb, myprow, 0, nprow)
    local_ncols = cublasMp.numroc(global_shape, nb, mypcol, 0, npcol)

    from mpi4py import MPI

    total_elements = np.array([local_nrows * local_ncols], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, total_elements, op=MPI.SUM)
    assert total_elements == global_shape * global_shape

    a = np.zeros((local_nrows, local_ncols))
    a = np.asfortranarray(a)

    distributions = [BlockCyclic(process_grid, (mb, nb))] * 3
    mm = nvmath.distributed.linalg.advanced.Matmul(a, a, distributions=distributions)
    # Check that the global matrix sizes were inferred correctly.
    assert mm.mm_traits.M == mm.mm_traits.N == mm.mm_traits.K == global_shape
    mm.free()


def valid_matrix_dtypes():
    SUPPORTED_TYPES = nvmath.linalg._internal.typemaps.SUPPORTED_TYPES
    # cupy doesn't support complex32
    return [dt for dt in SUPPORTED_TYPES if dt != "complex32"]


def is_invalid_compute_and_dtype_combination(compute_type, a_dtype, b_dtype, c_dtype, d_dtype, M_N_K):
    assert all(dtype is not None for dtype in (a_dtype, b_dtype, d_dtype))

    # TODO: c_type != None

    if "complex" in a_dtype or "complex" in d_dtype:
        if not (a_dtype == b_dtype == d_dtype):
            return True
        if a_dtype == "complex64" and not compute_type.startswith("COMPUTE_32F"):
            return True
        if a_dtype == "complex128" and not compute_type.startswith("COMPUTE_64F"):
            return True

    if compute_type in ("COMPUTE_32F", "COMPUTE_32F_PEDANTIC") and NAME_TO_DATA_WIDTH[d_dtype] == 16 and a_dtype == "float32":
        return True

    if compute_type in ("COMPUTE_32F", "COMPUTE_32F_PEDANTIC") and NAME_TO_DATA_WIDTH[d_dtype] == 64 and a_dtype == "float32":
        return True

    if compute_type in ("COMPUTE_32I", "COMPUTE_32I_PEDANTIC"):
        return True

    if compute_type in ("COMPUTE_16F", "COMPUTE_16F_PEDANTIC") and a_dtype != "float16":
        return True

    if (compute_type in ("COMPUTE_64F", "COMPUTE_64F_PEDANTIC")) ^ (a_dtype in ("float64", "complex128")):
        return True

    if compute_type in ("COMPUTE_32F_FAST_16F", "COMPUTE_32F_FAST_16BF", "COMPUTE_32F_FAST_TF32"):
        if a_dtype not in ("float32", "complex64"):
            return True
        if not (a_dtype == b_dtype == d_dtype):
            # NOTE: cuBLASLt and cuBLASMp don't throw an error for this case
            # (e.g. a_dtype=float32, b_dtype=float32, d_dtype=float64)
            # but according to docs cuBLASMp doesn't support this, and the result doesn't
            # match cuBLASLt.
            return True

    if NAME_TO_DATA_WIDTH[a_dtype] != 8 and NAME_TO_DATA_WIDTH[b_dtype] != 8:
        if a_dtype != b_dtype:
            return True
    else:
        # FP8
        if d_dtype == "float64" or d_dtype.startswith("complex"):
            return True
        if d_dtype == "float8_e5m2" and a_dtype != "float8_e5m2" and b_dtype != "float8_e5m2":
            return True
        if compute_type != "COMPUTE_32F":
            return True
        if NAME_TO_DATA_WIDTH[a_dtype] != NAME_TO_DATA_WIDTH[b_dtype]:
            return True
        if a_dtype == "float8_e5m2" and b_dtype == "float8_e5m2":
            return True

    if NAME_TO_DATA_WIDTH[d_dtype] == 8 and NAME_TO_DATA_WIDTH[a_dtype] != 8:
        return True

    if NAME_TO_DATA_WIDTH[a_dtype] == 64 and (NAME_TO_DATA_WIDTH[b_dtype] != 64 or NAME_TO_DATA_WIDTH[d_dtype] != 64):
        return True

    if a_dtype == b_dtype and NAME_TO_DATA_WIDTH[a_dtype] == 16 and d_dtype != a_dtype:
        if compute_type in ("COMPUTE_16F", "COMPUTE_16F_PEDANTIC"):
            return True
        if d_dtype != "float32":
            return True

    if (a_dtype == b_dtype == "bfloat16") and (d_dtype not in ("bfloat16", "float32")):
        return True


# Skip invalid compute_type and matrix dtype combinations. It might be better to check
# that Matmul correctly throws an error for invalid combinations, but the number of
# invalid combinations is very large, and destroying distributed Matmul objects currently
# takes too long.
@pytest.mark.uncollect_if(func=is_invalid_compute_and_dtype_combination)
# Use the compute_type name instead of the enum value in order to see the name
# in the pytest output instead of an integer code.
@pytest.mark.parametrize("compute_type", [compute_type.name for compute_type in MatmulComputeType])
@pytest.mark.parametrize("a_dtype", valid_matrix_dtypes())
@pytest.mark.parametrize("b_dtype", valid_matrix_dtypes())
@pytest.mark.parametrize("c_dtype", [None])
@pytest.mark.parametrize("d_dtype", valid_matrix_dtypes())
@pytest.mark.parametrize("M_N_K", [(64, 64, 64), (128, 96, 64), (64, 128, 64)])
def test_dtypes(
    compute_type,
    a_dtype,
    b_dtype,
    c_dtype,
    d_dtype,
    M_N_K,
    nvmath_distributed,
    check_symmetric_memory_leaks,
):
    """Test various combinations of compute_type and matrix dtypes (including mixed and
    narrow-precision) and check that the result matches single-GPU matmul."""

    distributed_ctx = nvmath.distributed.get_context()
    comm = distributed_ctx.communicator
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    device_id = distributed_ctx.device_id

    # TODO: c_dtype != None

    compute_type = MatmulComputeType[compute_type]
    dtypes = (a_dtype, b_dtype, c_dtype, d_dtype)

    torch_required = set(dtypes) & {"float8_e4m3fn", "float8_e5m2", "bfloat16"}
    if torch_required:
        pytest.skip("FP8 not supported")
    if torch_required and "torch" not in package_name_to_package:
        pytest.skip(f"torch is required for one of {torch_required} but is not installed")

    m, n, k = M_N_K
    assert k % nranks == 0

    # Use TN: (k, m) * (k, n) = (m, n)
    # (note that we use x.T on the created matrices so that cuBLASMp sees
    # Fortran memory order)
    a_shape = (m, k // nranks)
    b_shape = (n, k // nranks)

    beta = None if c_dtype is None else 1.0

    scales = None
    if torch_required:
        # Allocate all operands with PyTorch.
        import torch

        stream = get_or_create_stream(device_id, stream=None, op_package="torch")
        name_to_dtype = nvmath.internal.tensor_ifc_torch.TorchTensor.name_to_dtype
        # transpose to get Fortran order
        a = (torch.rand(*a_shape, device=f"cuda:{device_id}") * 10).type(name_to_dtype[a_dtype]).T
        b = (torch.rand(*b_shape, device=f"cuda:{device_id}") * 10).type(name_to_dtype[b_dtype]).T
        c = None
        if c_dtype is not None:
            raise NotImplementedError
        if NAME_TO_DATA_WIDTH[a_dtype] == 8:
            scales = {"a": 0.8, "b": 0.9}
            if NAME_TO_DATA_WIDTH[d_dtype] == 8:
                scales["d"] = 0.1
    else:
        # Allocate all operands with CuPy.
        import cupy as cp

        stream = get_or_create_stream(device_id, stream=None, op_package="cupy")
        name_to_dtype = nvmath.internal.tensor_ifc_numpy.NumpyTensor.name_to_dtype
        with device_ctx(device_id):
            # transpose to get Fortran order
            if "complex" in a_dtype:
                assert a_dtype != "complex32"
                float_dtype = cp.float32 if a_dtype == "complex64" else cp.float64
                a = (cp.random.rand(*a_shape, dtype=float_dtype) + 1j * cp.random.rand(*a_shape, dtype=float_dtype)).T
                b = (cp.random.rand(*b_shape, dtype=float_dtype) + 1j * cp.random.rand(*b_shape, dtype=float_dtype)).T
            else:
                a = (cp.random.rand(*a_shape) * 10).astype(name_to_dtype[a_dtype]).T
                b = (cp.random.rand(*b_shape) * 10).astype(name_to_dtype[b_dtype]).T
            c = None
        if c_dtype is not None:
            raise NotImplementedError

    cc = ccx.Device(device_id).compute_capability
    if any(NAME_TO_DATA_WIDTH[dt] <= 8 for dt in dtypes if dt is not None) and cc < (8, 9):
        pytest.skip("FP8 requires compute capability >= 8.9")

    options = {"compute_type": compute_type, "result_type": NAME_TO_DATA_TYPE[d_dtype]}
    if NAME_TO_DATA_WIDTH[d_dtype] <= 8:
        options["result_amax"] = True
    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    if "complex" in a_dtype:
        qualifiers[0]["is_conjugate"] = qualifiers[0]["is_transpose"] = True
    else:
        qualifiers[0]["is_transpose"] = True
    d = nvmath.distributed.linalg.advanced.matmul(
        a,
        b,
        c=c,
        distributions=[Slab.X] * 3,
        beta=beta,
        qualifiers=qualifiers,
        # quantization_scales=scales,
        options=options,
    )
    if isinstance(d, Sequence) and len(d) == 2:
        d, aux = d
        if "result_amax" in aux:
            from mpi4py import MPI

            aux_global = comm.allreduce(aux["result_amax"].item(), op=MPI.MAX)

    a = dist_wrap_operand(a)
    b = dist_wrap_operand(b)
    d = dist_wrap_operand(d)
    assert d.shape == (m // nranks, n)
    assert d.dtype == d_dtype
    assert d.module is a.module
    assert d.device == "cuda" and d.device_id == device_id

    if "complex" in a_dtype:
        a_global = gather_array(to_host(dist_wrap_operand(a.tensor), device_id, stream), 0, comm, rank)
    else:
        a_global = gather_array(to_host(dist_wrap_operand(a.tensor.T), device_id, stream), 1, comm, rank)
    b_global = gather_array(to_host(dist_wrap_operand(b.tensor.T), device_id, stream), 1, comm, rank)
    d_global = gather_array(to_host(d, device_id, stream), 0, comm, rank)
    if rank == 0:
        qualifiers = None
        c = None
        if "complex" in a_dtype:
            qualifiers = np.zeros((3,), dtype=nvmath.linalg.advanced.matrix_qualifiers_dtype)
            qualifiers[0]["is_conjugate"] = True
            # cuBLASLt fails to query heuristics for conjugate transpose unless
            # C is provided.
            beta = 1.0
            c = np.zeros((m, n), dtype=name_to_dtype[d_dtype])
        single_gpu_result = nvmath.linalg.advanced.matmul(
            a_global.tensor if "complex" not in a_dtype else a_global.tensor.T,
            b_global.tensor.T,
            c=c,
            beta=beta,
            qualifiers=qualifiers,
            quantization_scales=scales,
            options=options,
        )
        single_gpu_aux = {}
        if isinstance(single_gpu_result, Sequence) and len(single_gpu_result) == 2:
            single_gpu_result, single_gpu_aux = single_gpu_result

        single_gpu_result = wrap_operand(single_gpu_result)
        try:
            if "result_amax" in single_gpu_aux:
                assert math.isclose(aux_global, single_gpu_aux["result_amax"].item(), rel_tol=1e-3, abs_tol=1e-3)
            if NAME_TO_DATA_WIDTH[a_global.dtype] <= 16:
                rtol, atol = 1e-1, 1
            elif compute_type in (MatmulComputeType.COMPUTE_32F_FAST_TF32, MatmulComputeType.COMPUTE_32F_FAST_16F):
                rtol, atol = 1e-2, 1e-1
            else:
                rtol, atol = 1e-5, 1e-5
            assert is_close(d_global, single_gpu_result, rtol, atol), "Gathered result doesn't match single-GPU matmul"
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
