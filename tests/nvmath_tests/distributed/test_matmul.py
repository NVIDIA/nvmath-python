# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import random
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import nvmath.distributed
from nvmath.bindings import cublasLt, cublasMp
from nvmath.distributed import free_symmetric_memory
from nvmath.distributed._internal.tensor_wrapper import maybe_register_package
from nvmath.distributed._internal.tensor_wrapper import wrap_operand as dist_wrap_operand
from nvmath.distributed.distribution import BlockCyclic, BlockNonCyclic, Box, ProcessGrid, Slab
from nvmath.distributed.linalg._internal.epilog_protocol import gelu_aux_mm_shape, relu_aux_mm_shape
from nvmath.distributed.linalg.advanced import MatmulComputeType, MatmulEpilog, matrix_qualifiers_dtype
from nvmath.distributed.process_group import MPIProcessGroup, ReductionOp
from nvmath.internal.tensor_wrapper import wrap_operand
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE, NAME_TO_DATA_WIDTH
from nvmath.internal.utils import device_ctx, get_or_create_stream
from nvmath.linalg.advanced.helpers.matmul import apply_mxfp8_scale

from ..helpers import check_freed_after
from .helpers import assert_close, gather_array, generate_random_data, process_group_broadcast, to_host

try:
    from cuda.core import Device, system
except ImportError:
    from cuda.core.experimental import Device, system

package_name_to_package = {"numpy": np}


@pytest.fixture(scope="module")
def nvmath_distributed(process_group):
    """Pytest fixture that initializes nvmath.distributed and finalizes it on exit"""

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

    try:
        num_devices = system.get_num_devices()
    except AttributeError:
        num_devices = system.num_devices
    device_id = process_group.rank % num_devices
    # nvshmem is needed for distributed reshape operation (used by some tests).
    backends = ["nvshmem", "nccl"] if process_group.nranks > 1 else ["nccl"]
    nvmath.distributed.initialize(device_id, process_group, backends=backends)

    yield

    nvmath.distributed.finalize()


@pytest.fixture(scope="module", autouse=True)
def cublasmp_logfile():
    # We're not using the cuBLASMp logging runtime APIs for now, which are considered
    # experimental. When setting the log file through env vars, the log file gets fixed
    # when the library is initialized and there is no way to change it per matmul
    # operation. So we need to select the file at the module scope.

    if "TORCHELASTIC_RUN_ID" in os.environ:
        rank = int(os.environ["RANK"])
    else:
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
    process_group = distributed_ctx.process_group
    rank = process_group.rank
    nranks = process_group.nranks

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
    process_group = distributed_ctx.process_group
    nranks = process_group.nranks
    device_id = distributed_ctx.device_id

    if not nvmath.distributed._internal.nvshmem.is_initialized():
        pytest.skip("NVSHMEM is not initialized")

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
    process_group = distributed_ctx.process_group
    rank = process_group.rank
    nranks = process_group.nranks
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

    a_global = gather_array(to_host(dist_wrap_operand(a_), device_id, stream), 0, process_group, rank)
    result_global = gather_array(to_host(dist_wrap_operand(d), device_id, stream), 0, process_group, rank)
    if rank == 0:
        a = a_global.tensor
        expected = a @ a @ a @ a
        assert_close(result_global, wrap_operand(expected), rtol=1e-5, atol=1e-5)


def generate_process_grids(only_2d=False):
    """Generate all possible process grids for the current number of MPI processes."""

    if "TORCHELASTIC_RUN_ID" in os.environ:
        nranks = int(os.environ["WORLD_SIZE"])
    else:
        from mpi4py import MPI

        nranks = MPI.COMM_WORLD.Get_size()

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


algo_str_to_enum = {
    "fallback Matmul": 0,
    "generic Matmul": 1,  # SUMMA algo
    "Matmul + ReduceScatter": 2,
    "AllGather + Matmul": 3,
    "Matmul + AllReduce": 4,
    "local Matmul": 5,
}


def read_algo_from_log(logfile_path, cublasMp_version) -> int | None:
    with open(logfile_path) as logfile:
        # NOTE: need to call mm.execute() to see this printed to the logfile.
        if cublasMp_version >= 700:
            regexAlgo = re.compile(r"\[cublasMpMatmul\] Using (.*)$")
        else:
            regexAlgo = re.compile(r"\[cublasMpMatmul\] using matmul algo (\d+)$")
        for line in logfile:
            m = regexAlgo.search(line)
            if m:
                if cublasMp_version >= 700:
                    algo = m.group(1)
                    if algo not in algo_str_to_enum:
                        # The regex pattern for cuBLASMp 0.7+ is very generic, so
                        # continue searching just in case.
                        continue
                    return algo_str_to_enum[algo]
                else:
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
    inplace,
    epilog_AR,
):
    if inplace and not input_C:
        return True

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
@pytest.mark.parametrize("inplace", [False, True])
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
    inplace,
    epilog_AR,
    nvmath_distributed,
    cublasmp_logfile_with_cleanup,
    check_symmetric_memory_leaks,
):
    distributed_ctx = nvmath.distributed.get_context()
    process_group = distributed_ctx.process_group
    rank = process_group.rank
    nranks = process_group.nranks
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

    cublasMp_version = cublasMp.get_version()

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
    process_group.broadcast_buffer(r)
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

    def generate_random_matrix(shape, dtype, symmetric_memory=False):
        return generate_random_data(
            np if package != "torch" else pkg,
            input_memory_space,
            shape,
            dtype,
            stream,
            memory_layout="F",
            symmetric_memory=symmetric_memory,
        )

    a_cpu, a = generate_random_matrix(A_shape, dtype)
    b_cpu, b = generate_random_matrix(B_shape, dtype)
    if input_C:
        beta = 0.8
        c_cpu, c = generate_random_matrix(C_shape, dtype)
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
    options = {"inplace": inplace}
    with nvmath.distributed.linalg.advanced.Matmul(
        a.tensor,
        b.tensor,
        c=c.tensor if input_C else None,
        beta=beta,
        distributions=distributions,
        qualifiers=qualifiers,
        options=options,
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
            if inplace:
                assert d is c.tensor
            elif c is not None:
                assert d is not c.tensor
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

            a_global = gather_array(a_cpu, 0 if A_distribution == "R" else 1, process_group, rank)
            b_global = gather_array(b_cpu, 0 if B_distribution == "R" else 1, process_group, rank)
            if epilog_AR:
                # C/D is not actually distributed (it's replicated on all processes).
                if input_C:
                    c_global = c_cpu
                d_global = d_cpu
            else:
                if input_C:
                    c_global = gather_array(c_cpu, 0 if C_distribution == "R" else 1, process_group, rank)
                d_global = gather_array(d_cpu, 0 if C_distribution == "R" else 1, process_group, rank)
            if rank == 0:
                if input_C:
                    assert c_global.shape == (M, N)
                assert d_global.shape == (M, N)
                single_gpu_result = nvmath.linalg.advanced.matmul(
                    a_global.tensor.T if transA else a_global.tensor,
                    b_global.tensor.T if transB else b_global.tensor,
                    c=c_global.tensor if input_C else None,
                    beta=beta,
                    options=options,
                )
                single_gpu_result = wrap_operand(single_gpu_result)
                try:
                    if inplace:
                        assert single_gpu_result.tensor is c_global.tensor

                    assert_close(d_global, single_gpu_result, rtol=1e-5, atol=1e-5)

                    if os.environ.get("CUBLASMP_ALGO_CHECK") == "1":
                        algo = read_algo_from_log(cublasmp_logfile_with_cleanup, cublasMp_version)
                        assert algo is not None, "Couldn't determine the distributed matmul algorithm used"
                        assert algo == expected_algo, (
                            f"cuBLASMp didn't run the expected distributed algorithm: algo is {algo}, "
                            f"expected algo is {expected_algo}"
                        )

                    process_group_broadcast(process_group, None)

                except Exception as e:
                    # Broadcast the exception to avoid deadlock.
                    process_group_broadcast(process_group, e)
                    raise
            else:
                # If rank 0 raises an exception, every process has to do the same to avoid
                # deadlock.
                e = process_group_broadcast(process_group, None)
                if e is not None:
                    raise e

            if mm_count == MM_LIMIT:
                break

            # Reset operands.
            a_cpu, a = generate_random_matrix(A_shape, dtype)
            b_cpu, b = generate_random_matrix(B_shape, dtype)
            if input_C:
                beta = 0.5
                c_cpu, c = generate_random_matrix(C_shape, dtype)
                if epilog_AR:
                    # For epilog_AR cuBLASMp has each process contribute its C to
                    # the result. To get the same result as single-GPU MM, we have
                    # to set the values to zero on every rank except one.
                    c_cpu.tensor[:] = 10.0 if rank == 0 else 0.0
                    with device_ctx(device_id):
                        c.tensor[:] = 10.0 if rank == 0 else 0.0
            else:
                beta = c_cpu = c = None
            mm.reset_operands(a=a.tensor, b=b.tensor, c=c.tensor if c is not None else None, beta=beta)


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
    process_group = distributed_ctx.process_group
    rank = process_group.rank
    nranks = process_group.nranks

    valid_nranks = (1, 2, 4)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    M, N, K = M_N_K
    assert M % nranks == 0
    assert N % nranks == 0
    assert K % nranks == 0

    cublasMp_version = cublasMp.get_version()

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
        return gather_array(dist_wrap_operand(matrix), 1, process_group, rank)

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
            assert_close(d_global, single_gpu_result, rtol=1e-5, atol=1e-5)

            if os.environ.get("CUBLASMP_ALGO_CHECK") == "1":
                algo = read_algo_from_log(cublasmp_logfile_with_cleanup, cublasMp_version)
                assert algo is not None, "Couldn't determine the distributed matmul algorithm used"
                expected_algo = 1 if cyclic else 0  # 0 is naive, 1 is SUMMA.
                assert algo == expected_algo, (
                    f"cuBLASMp didn't run the expected distributed algorithm: algo is {algo}, expected algo is {expected_algo}"
                )

            process_group_broadcast(process_group, None)

        except Exception as e:
            # Broadcast the exception to avoid deadlock.
            process_group_broadcast(process_group, e)
            raise
    else:
        # If rank 0 raises an exception, every process has to do the same to avoid
        # deadlock.
        e = process_group_broadcast(process_group, None)
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
    process_group = distributed_ctx.process_group
    rank = process_group.rank

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

    total_elements = np.array([local_nrows * local_ncols], dtype=np.int64)
    process_group.allreduce_buffer(total_elements, op=ReductionOp.SUM)
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


def is_invalid_compute_and_dtype_combination(compute_type, a_dtype, b_dtype, c_dtype, d_dtype, M_N_K, algo, inplace, mxfp8):
    assert all(dtype is not None for dtype in (a_dtype, b_dtype, d_dtype))

    if mxfp8 and NAME_TO_DATA_WIDTH[a_dtype] != 8:
        return True

    if mxfp8 and all(x < 512 for x in M_N_K):
        # MXFP8 requires matrix sizes for local GEMMs that are divisible by 128. Given the
        # matrix sizes that test_dtypes tests for, the simplest way to guarantee this is
        # is to only test MXFP8 with the larger sizes.
        return True

    d_dtype_bitwidth = NAME_TO_DATA_WIDTH[d_dtype]

    if c_dtype is not None:
        if d_dtype_bitwidth == 8:
            # if d_dtype is FP8 c_dtype must be FP16
            if NAME_TO_DATA_WIDTH[c_dtype] != 16:
                return True
        elif c_dtype != d_dtype:
            return True

    if inplace and (c_dtype is None or c_dtype != d_dtype):
        return True

    if "complex" in a_dtype or "complex" in d_dtype:
        if not (a_dtype == b_dtype == d_dtype):
            return True
        if a_dtype == "complex64" and not compute_type.startswith("COMPUTE_32F"):
            return True
        if a_dtype == "complex128" and not compute_type.startswith("COMPUTE_64F"):
            return True

    if compute_type in ("COMPUTE_16F", "COMPUTE_16F_PEDANTIC") and a_dtype != "float16":
        return True

    if (compute_type in ("COMPUTE_64F", "COMPUTE_64F_PEDANTIC", "COMPUTE_64F_EMULATED_FIXEDPOINT")) ^ (
        a_dtype in ("float64", "complex128")
    ):
        return True

    if compute_type in (
        "COMPUTE_32F_FAST_16F",
        "COMPUTE_32F_FAST_16BF",
        "COMPUTE_32F_FAST_TF32",
        "COMPUTE_32F_EMULATED_16BFX9",
    ):
        if a_dtype not in ("float32", "complex64"):
            return True
        if not (a_dtype == b_dtype == d_dtype):
            # NOTE: cuBLASLt and cuBLASMp don't throw an error for this case
            # (e.g. a_dtype=float32, b_dtype=float32, d_dtype=float64)
            # but according to docs cuBLASMp doesn't support this, and the result doesn't
            # match cuBLASLt.
            return True

    if compute_type in ("COMPUTE_32F", "COMPUTE_32F_PEDANTIC") and d_dtype_bitwidth == 16 and a_dtype == "float32":
        return True

    if compute_type in ("COMPUTE_32F", "COMPUTE_32F_PEDANTIC") and d_dtype_bitwidth == 64 and a_dtype == "float32":
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

    if d_dtype_bitwidth == 8 and NAME_TO_DATA_WIDTH[a_dtype] != 8:
        return True

    if NAME_TO_DATA_WIDTH[a_dtype] == 64 and (NAME_TO_DATA_WIDTH[b_dtype] != 64 or d_dtype_bitwidth != 64):
        return True

    if a_dtype == b_dtype and NAME_TO_DATA_WIDTH[a_dtype] == 16 and d_dtype != a_dtype:
        if compute_type in ("COMPUTE_16F", "COMPUTE_16F_PEDANTIC"):
            return True
        if d_dtype != "float32":
            return True

    if (a_dtype == b_dtype == "bfloat16") and (d_dtype not in ("bfloat16", "float32")):
        return True


# Use the compute_type name instead of the enum value in order to see the name
# in the pytest output instead of an integer code.
@pytest.mark.parametrize(
    "compute_type", [compute_type.name for compute_type in MatmulComputeType if not compute_type.name.startswith("COMPUTE_32I")]
)
@pytest.mark.parametrize("d_dtype", valid_matrix_dtypes())
@pytest.mark.parametrize("M_N_K", [(64, 64, 64), (128, 96, 64), (64, 128, 64), (512, 512, 512)])
@pytest.mark.parametrize("algo", ["AG+GEMM", "GEMM+RS"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("mxfp8", [False, True])
def test_dtypes(
    compute_type,
    d_dtype,
    M_N_K,
    algo,
    inplace,
    mxfp8,
    nvmath_distributed,
    subtests,
):
    """Test various combinations of compute_type and matrix dtypes (including mixed and
    narrow-precision) and check that the result matches single-GPU matmul."""

    distributed_ctx = nvmath.distributed.get_context()
    process_group = distributed_ctx.process_group
    nranks = process_group.nranks
    device_id = distributed_ctx.device_id

    if nranks > 1 and mxfp8 and algo == "GEMM+RS" and cublasMp.get_version() == 800:
        pytest.xfail("MXFP8 and GEMM+RS expected to fail with cuBLASMp 0.8 (fixed in >=0.8.1)")

    version = nvmath.bindings.cublasLt.get_version()
    if version < 120900 and compute_type == "COMPUTE_32F_EMULATED_16BFX9":
        pytest.skip("COMPUTE_32F_EMULATED_16BFX9 requires CTK >= 12.9.0 (cuBLASLt >= 12.9.0).")
    if cublasMp.get_version() < 700 and compute_type == "COMPUTE_32F_EMULATED_16BFX9":
        pytest.skip("COMPUTE_32F_EMULATED_16BFX9 requires cuBLASMp >= 0.7.")

    if version < 130200 and compute_type == "COMPUTE_64F_EMULATED_FIXEDPOINT":
        pytest.skip("COMPUTE_64F_EMULATED_FIXEDPOINT requires CTK >= 13.1.0 (cuBLASLt >= 13.2.0).")
    if cublasMp.get_version() < 900 and compute_type == "COMPUTE_64F_EMULATED_FIXEDPOINT":
        pytest.skip("COMPUTE_64F_EMULATED_FIXEDPOINT is not supported in this version of cuBLASMp.")

    if d_dtype == "float4_e2m1fn_x2":
        pytest.skip("FP4 is not supported in distributed matmul")
    torch_required = d_dtype in {"float8_e4m3fn", "float8_e5m2", "bfloat16"}
    if torch_required and "torch" not in package_name_to_package:
        pytest.skip(f"torch is required for {d_dtype} but is not installed")

    cc = Device(device_id).compute_capability
    if NAME_TO_DATA_WIDTH[d_dtype] <= 8 and cc < (8, 9):
        pytest.skip("FP8 requires compute capability >= 8.9")
    if mxfp8 and cc < (10, 0):
        pytest.skip("MXFP8 requires compute capability >= 10.0")

    for a_dtype in valid_matrix_dtypes():
        for b_dtype in valid_matrix_dtypes():
            for c_dtype in [None] + valid_matrix_dtypes():
                if is_invalid_compute_and_dtype_combination(
                    compute_type, a_dtype, b_dtype, c_dtype, d_dtype, M_N_K, algo, inplace, mxfp8
                ):
                    continue

                with subtests.test(msg=f"a_dtype={a_dtype} b_dtype={b_dtype} c_dtype={c_dtype}", i=(a_dtype, b_dtype, c_dtype)):
                    run_test_dtypes(compute_type, a_dtype, b_dtype, c_dtype, d_dtype, M_N_K, algo, inplace, mxfp8)


def run_test_dtypes(
    compute_type,
    a_dtype,
    b_dtype,
    c_dtype,
    d_dtype,
    M_N_K,
    algo,
    inplace,
    mxfp8,
):
    distributed_ctx = nvmath.distributed.get_context()
    process_group = distributed_ctx.process_group
    rank = process_group.rank
    nranks = process_group.nranks
    device_id = distributed_ctx.device_id

    compute_type = MatmulComputeType[compute_type]
    dtypes = (a_dtype, b_dtype, c_dtype, d_dtype)

    if set(dtypes) & {"float4_e2m1fn_x2"}:
        pytest.skip("FP4 is not supported in distributed matmul")
    torch_required = set(dtypes) & {"float8_e4m3fn", "float8_e5m2", "bfloat16"}
    if torch_required and "torch" not in package_name_to_package:
        pytest.skip(f"torch is required for one of {torch_required} but is not installed")

    cc = Device(device_id).compute_capability
    if any(NAME_TO_DATA_WIDTH[dt] <= 8 for dt in dtypes if dt is not None) and cc < (8, 9):
        pytest.skip("FP8 requires compute capability >= 8.9")

    m, n, k = M_N_K
    assert k % nranks == 0

    # Create distributions that we're going to pass to distributed matmul.
    if algo == "AG+GEMM":
        distributions = [Slab.Y, Slab.Y, Slab.X]  # For TN
    elif algo == "GEMM+RS":
        distributions = [Slab.X, Slab.X, Slab.Y]  # For TN
    else:
        raise ValueError(f"test_dtypes doesn't support algo {algo}")

    def transpose_slab(slab):
        if slab.partition_dim == 0:
            return Slab.Y
        return Slab.X

    # Use TN: (k, m) * (k, n) = (m, n)
    # (note that we use x.T on the created matrices so that cuBLASMp sees
    # Fortran memory order)
    # We need to transpose the distribution to get the desired distribution after
    # transposing the matrices.
    a_shape = transpose_slab(distributions[0]).shape(rank, (m, k))
    b_shape = transpose_slab(distributions[1]).shape(rank, (n, k))

    beta = None if c_dtype is None else 1.0 if inplace else 0.6
    if c_dtype is not None:
        c_shape = transpose_slab(distributions[2]).shape(rank, (n, m))

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
            c = (torch.rand(*c_shape, device=f"cuda:{device_id}") * 10).type(name_to_dtype[c_dtype]).T
        if NAME_TO_DATA_WIDTH[a_dtype] == 8:
            if mxfp8:
                scales = {
                    "a": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a, -1),  # 2^-1 = 0.5
                    "b": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b, -1),  # 2^-1 = 0.5
                }
            else:
                scales = {"a": 0.8, "b": 0.9}
                if NAME_TO_DATA_WIDTH[d_dtype] == 8:
                    scales["d"] = 0.1
        c_orig = c
        if inplace:
            # Need to make a copy to compare with cuBLASLt.
            c_orig = c.clone()
    else:
        # Allocate all operands with CuPy.
        import cupy as cp

        stream = get_or_create_stream(device_id, stream=None, op_package="cupy")
        name_to_dtype = nvmath.internal.tensor_ifc_numpy.NumpyTensor.name_to_dtype
        with device_ctx(device_id):
            # transpose to get Fortran order
            c = None
            if "complex" in a_dtype:
                assert a_dtype != "complex32"
                float_dtype = cp.float32 if a_dtype == "complex64" else cp.float64
                a = (cp.random.rand(*a_shape, dtype=float_dtype) + 1j * cp.random.rand(*a_shape, dtype=float_dtype)).T
                b = (cp.random.rand(*b_shape, dtype=float_dtype) + 1j * cp.random.rand(*b_shape, dtype=float_dtype)).T
                if c_dtype is not None:
                    c = (cp.random.rand(*c_shape, dtype=float_dtype) + 1j * cp.random.rand(*c_shape, dtype=float_dtype)).T
            else:
                a = (cp.random.rand(*a_shape) * 10).astype(name_to_dtype[a_dtype]).T
                b = (cp.random.rand(*b_shape) * 10).astype(name_to_dtype[b_dtype]).T
                if c_dtype is not None:
                    c = (cp.random.rand(*c_shape) * 10).astype(name_to_dtype[c_dtype]).T
        c_orig = c
        if inplace:
            # Need to make a copy to compare with cuBLASLt.
            c_orig = c.copy()

    options = {
        "compute_type": compute_type,
        "result_type": NAME_TO_DATA_TYPE[d_dtype],
        "inplace": inplace,
        "block_scaling": mxfp8,
    }
    if NAME_TO_DATA_WIDTH[d_dtype] <= 8 and not mxfp8:
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
        distributions=distributions,
        beta=beta,
        qualifiers=qualifiers,
        quantization_scales=scales,
        options=options,
    )
    if isinstance(d, Sequence) and len(d) == 2:
        d, aux = d
        aux_global = {}
        if "result_amax" in aux:
            result_amax = np.array([aux["result_amax"].item()])
            process_group.allreduce_buffer(result_amax, op=ReductionOp.MAX)
            aux_global["result_amax"] = float(result_amax[0])

    if mxfp8 and NAME_TO_DATA_WIDTH[d_dtype] <= 8:
        # Apply the d_out scales.
        d = apply_mxfp8_scale(d, aux["d_out_scale"], output_dtype=torch.float32)

    if inplace:
        assert d is c
    else:
        assert d is not c
    a = dist_wrap_operand(a)
    b = dist_wrap_operand(b)
    d = dist_wrap_operand(d)
    assert d.shape == distributions[2].shape(rank, (m, n))
    if mxfp8 and NAME_TO_DATA_WIDTH[d_dtype] <= 8:
        assert d.dtype == "float32"  # scales were applied above to convert to FP32
    else:
        assert d.dtype == d_dtype
    assert d.module is a.module
    assert d.device == "cuda" and d.device_id == device_id

    c_global = None
    if "complex" in a_dtype:
        a_global = gather_array(
            to_host(dist_wrap_operand(a.tensor), device_id, stream), distributions[0].partition_dim, process_group, rank
        )
        if c is not None:
            c_global = gather_array(
                to_host(dist_wrap_operand(c_orig), device_id, stream), distributions[2].partition_dim, process_group, rank
            )
    else:
        a_global = gather_array(
            to_host(dist_wrap_operand(a.tensor.T), device_id, stream), 1 - distributions[0].partition_dim, process_group, rank
        )
        if c is not None:
            c_global = gather_array(
                to_host(dist_wrap_operand(c_orig.T), device_id, stream), 1 - distributions[2].partition_dim, process_group, rank
            )
    b_global = gather_array(
        to_host(dist_wrap_operand(b.tensor.T), device_id, stream), 1 - distributions[1].partition_dim, process_group, rank
    )
    d_global = gather_array(to_host(d, device_id, stream), distributions[2].partition_dim, process_group, rank)
    if rank == 0:
        if mxfp8:
            scales_global = {
                "a": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(a_global.tensor, -1),  # 2^-1 = 0.5
                "b": nvmath.linalg.advanced.helpers.matmul.create_mxfp8_scale(b_global.tensor, -1),  # 2^-1 = 0.5
            }
        else:
            scales_global = scales
        qualifiers = None
        c = None
        if c_global is not None:
            c = c_global.tensor.T if "complex" not in a_dtype else c_global.tensor
        if "complex" in a_dtype:
            qualifiers = np.zeros((3,), dtype=nvmath.linalg.advanced.matrix_qualifiers_dtype)
            qualifiers[0]["is_conjugate"] = True
            if c_global is None:
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
            quantization_scales=scales_global,
            options=options,
        )
        single_gpu_aux = {}
        if isinstance(single_gpu_result, Sequence) and len(single_gpu_result) == 2:
            single_gpu_result, single_gpu_aux = single_gpu_result

        if mxfp8 and NAME_TO_DATA_WIDTH[d_dtype] <= 8:
            # Apply the d_out scales.
            single_gpu_result = apply_mxfp8_scale(single_gpu_result, single_gpu_aux["d_out_scale"], output_dtype=torch.float32)

        if inplace:
            assert single_gpu_result is c
        else:
            assert single_gpu_result is not c

        single_gpu_result = wrap_operand(single_gpu_result)
        try:
            if "result_amax" in single_gpu_aux:
                assert math.isclose(
                    aux_global["result_amax"], single_gpu_aux["result_amax"].item(), rel_tol=1e-3, abs_tol=1e-3
                ), "amax doesn't match cuBLASLt"
            if NAME_TO_DATA_WIDTH[a_global.dtype] <= 16:
                rtol, atol = 1e-1, 1
                if algo == "GEMM+RS" and nranks > 1:
                    # We compare distributed results with single-GPU cuBLASLt results, and
                    # for GEMM+RS a large portion of the computation is done in cuBLASMp
                    # (not delegated to cuBLASLt), and so for FP8 the output difference wrt
                    # cuBLASLt (and thus the rtol) ends up varying depending on kernels,
                    # algorithms and hardware used. This is a conservative rtol to allow
                    # tests to pass on different hardware.
                    rtol = 0.25
            elif compute_type in (
                MatmulComputeType.COMPUTE_32F_FAST_TF32,
                MatmulComputeType.COMPUTE_32F_FAST_16F,
                MatmulComputeType.COMPUTE_32F_FAST_16BF,
            ):
                rtol, atol = 1e-2, 1e-1
            else:
                rtol, atol = 1e-5, 1e-5
            assert_close(d_global, single_gpu_result, rtol, atol)
            process_group_broadcast(process_group, None)
        except Exception as e:
            # Broadcast the exception to avoid deadlock.
            process_group_broadcast(process_group, e)
            raise
    else:
        # If rank 0 raises an exception, every process has to do the same to avoid
        # deadlock.
        e = process_group_broadcast(process_group, None)
        if e is not None:
            raise e


def left_shift_buffer(buf, shift_amount):
    assert shift_amount > 0 and shift_amount < 8
    carry = 0
    # Iterate from the end for left shift
    for i in range(len(buf) - 1, -1, -1):
        next_carry = buf[i] << (8 - shift_amount)  # Bits that will carry to the preceding byte
        buf[i] = buf[i] >> shift_amount  # shift this byte by the required amount of bits
        buf[i] |= carry  # Add the carry from the following byte
        carry = next_carry


def gather_relu_mask(mask, m, n, comm, rank, nranks):
    # Gather the bitmask in such a way that it matches the one returned by cuBLASLt
    # if doing matmul with the global matrices on a single process.
    # This function assumes that the output matrix is partitioned on m
    assert m % nranks == 0  # m must divide evenly for the cases we're considering
    global_relu_aux_shape = relu_aux_mm_shape(m, n)
    my_starting_index = (m // nranks) * rank
    pad_up = math.ceil(my_starting_index / 8)
    pad_down = global_relu_aux_shape[0] - pad_up - mask.shape[0]
    if pad_down < 0:
        mask = np.pad(mask, ((pad_up, 0), (0, 0)), mode="constant", constant_values=0)
        mask = mask[:pad_down, :].copy(order="K")
    else:
        mask = np.pad(mask, ((pad_up, pad_down), (0, 0)), mode="constant", constant_values=0)
    if my_starting_index % 8 != 0:
        for i in range(n):
            left_shift_buffer(mask[:, i], 8 - (my_starting_index % 8))
    assert mask.flags["F_CONTIGUOUS"]
    # Now do a bitwise OR reduce
    from mpi4py import MPI

    if rank == 0:
        comm.Reduce(MPI.IN_PLACE, mask, op=MPI.BOR)
    else:
        comm.Reduce(mask, None, op=MPI.BOR)
    return mask


def remove_mask_padding(mask, m):
    mask = mask[: math.ceil(m / 8), :]
    if m % 8 != 0:
        bit_filter = 255 >> (8 - (m % 8))
        # Set unused bits in the last byte of each column to 0.
        for i in range(mask.shape[1]):
            mask[-1, i] &= bit_filter
    return mask


def skip_test_epilogues(M_N_K, transA, transB, algo, epilogue, fp8):
    if fp8 and M_N_K == (64, 32, 48):
        return True
    if fp8 and any(x % 16 != 0 for x in M_N_K):
        return True
    if fp8 and (transA, transB) != (True, False):
        # FP8 matrix multiplications only support TN
        return True
    # Skip DGELU for now because cuBLAS doesn't support if the inferred ctype is float16
    if epilogue in ("GELU", "GELU_BIAS", "DGELU") and fp8:
        # Not supported by cuBLAS
        return True
    if "BGRAD" in epilogue and fp8:
        # Not supported by cuBLAS
        return True
    if epilogue == "DEFAULT":
        return True
    if epilogue == "ALLREDUCE":
        return True  # ALLREDUCE is tested in test_uniform_1d_distributions
    if epilogue in ("DRELU", "DRELU_BGRAD"):
        return True  # cuBLASMp currently doesn't support this.
    if "BGRADA" in epilogue and transA:
        # CUBLASLT_EPILOGUE_BGRADA only works with non-transposed A.
        return True
    # CUBLASLT_EPILOGUE_BGRADB only works with transposed B
    return "BGRADB" in epilogue and not transB


@pytest.mark.uncollect_if(func=skip_test_epilogues)
@pytest.mark.parametrize("M_N_K", [(64, 32, 48), (128, 96, 64), (64, 128, 64), (84, 32, 48)])
@pytest.mark.parametrize("transA", [False, True])
@pytest.mark.parametrize("transB", [False, True])
@pytest.mark.parametrize("algo", ["AG+GEMM", "GEMM+RS"])
# We use the compute_type name instead of the enum value in order to see the name
# in the pytest output instead of an integer code.
@pytest.mark.parametrize("epilogue", [e.name for e in cublasMp.MatmulEpilogue])
@pytest.mark.parametrize("fp8", [False, True])
def test_epilogues(
    M_N_K,
    transA,
    transB,
    algo,
    epilogue,
    fp8,
    nvmath_distributed,
    check_symmetric_memory_leaks,
):
    distributed_ctx = nvmath.distributed.get_context()
    process_group = distributed_ctx.process_group
    rank = process_group.rank
    nranks = process_group.nranks
    device_id = distributed_ctx.device_id

    epilogue = cublasMp.MatmulEpilogue[epilogue]

    if nranks > 1 and fp8 and algo == "GEMM+RS" and cublasMp.get_version() == 800:
        pytest.xfail("Most epilogues with FP8 and GEMM+RS expected to fail with cuBLASMp 0.8 (fixed in >=0.8.1)")

    if "RELU_AUX" in epilogue.name and not isinstance(process_group, MPIProcessGroup):
        pytest.skip("RELU_AUX tests require MPI")

    m, n, k = M_N_K
    assert m % nranks == 0
    assert n % nranks == 0
    assert k % nranks == 0

    if fp8 and "torch" not in package_name_to_package:
        pytest.skip("torch is required for FP8 but is not installed")

    if fp8 and Device(device_id).compute_capability < (8, 9):
        pytest.skip("FP8 requires compute capability >= 8.9")

    if algo == "AG+GEMM":
        distributions = [
            Slab.Y if transA else Slab.X,
            Slab.X if transB else Slab.Y,
            Slab.X,
        ]
    elif algo == "GEMM+RS":
        distributions = [
            Slab.X if transA else Slab.Y,
            Slab.Y if transB else Slab.X,
            Slab.Y,
        ]
    else:
        raise ValueError(f"test_epilogues doesn't support algo {algo}")

    a_shape = distributions[0].shape(rank, (m, k) if not transA else (k, m))
    b_shape = distributions[1].shape(rank, (k, n) if not transB else (n, k))

    import cupy as cp

    if fp8:
        # Allocate all operands with PyTorch.
        import torch

        dtype = torch.float8_e4m3fn
        stream = get_or_create_stream(device_id, stream=None, op_package="torch")
        a = (torch.randn(*a_shape[::-1], device=f"cuda:{device_id}") * 10).type(dtype).T
        b = (torch.randn(*b_shape[::-1], device=f"cuda:{device_id}") * 10).type(dtype).T
        scales = {"a": 0.8, "b": 0.9, "d": 0.1}
    else:
        # Allocate all operands with CuPy.
        dtype = cp.float32
        stream = get_or_create_stream(device_id, stream=None, op_package="cupy")
        with device_ctx(device_id):
            a = cp.random.randn(*a_shape).astype(cp.float32)
            b = cp.random.randn(*b_shape).astype(cp.float32)
            a = cp.asfortranarray(a)
            b = cp.asfortranarray(b)
        scales = None

    epilog_inputs = None
    if "BIAS" in epilogue.name:
        # Bias is a vector of length M that is applied to D.
        if distributions[2].partition_dim == 0:
            # D is partitioned on M, so bias input is partitioned too.
            with device_ctx(device_id):
                bias = cp.random.rand(m // nranks, 1).astype(cp.float32)
        else:
            # Bias vector is not partitioned, therefore it's replicated. Generate it
            # on one rank and broadcast to others.
            bias = np.random.rand(m, 1).astype(cp.float32)
            process_group.broadcast_buffer(bias)
            with device_ctx(device_id):
                bias = cp.asarray(bias)
        if fp8:
            # cuBLAS requires FP16 bias with FP8 d_dtype (and doesn't return
            # an error if it isn't)
            bias = torch.as_tensor(bias, dtype=torch.float16)
        epilog_inputs = {"bias": bias}
    elif "DGELU" in epilogue.name:
        # GELU gradient has same shape as D and follows its distribution.
        dummy_dgelu_input_shape = gelu_aux_mm_shape(*distributions[2].shape(rank, (m, n)))
        with device_ctx(device_id):
            dummy_dgelu_input = cp.random.randn(*dummy_dgelu_input_shape).astype(cp.float32)
            dummy_dgelu_input = cp.asfortranarray(dummy_dgelu_input)
        if fp8:
            dummy_dgelu_input = (torch.as_tensor(dummy_dgelu_input) * 10).type(dtype)
        epilog_inputs = {"gelu_aux": dummy_dgelu_input}

    qualifiers = np.zeros((3,), dtype=matrix_qualifiers_dtype)
    qualifiers[0]["is_transpose"] = transA
    qualifiers[1]["is_transpose"] = transB
    d = nvmath.distributed.linalg.advanced.matmul(
        a,
        b,
        distributions=distributions,
        qualifiers=qualifiers,
        quantization_scales=scales,
        epilog=None if epilogue.name == "DEFAULT" else epilogue,
        epilog_inputs=epilog_inputs,
    )

    aux_out = {}
    if isinstance(d, Sequence) and len(d) == 2:
        d, aux_out = d

    a = dist_wrap_operand(a)
    b = dist_wrap_operand(b)
    d = dist_wrap_operand(d)
    assert d.shape == distributions[2].shape(rank, (m, n))
    assert d.module is a.module
    assert d.device == "cuda" and d.device_id == device_id

    # Gather distributed inputs and outputs for comparison with cuBLASLt.

    a_global = gather_array(to_host(a, device_id, stream), distributions[0].partition_dim, process_group, rank)
    b_global = gather_array(to_host(b, device_id, stream), distributions[1].partition_dim, process_group, rank)
    d_global = gather_array(to_host(d, device_id, stream), distributions[2].partition_dim, process_group, rank)

    if "BIAS" in epilogue.name:
        bias = to_host(wrap_operand(epilog_inputs["bias"]), device_id, stream)
        if distributions[2].partition_dim == 0:
            bias_global = gather_array(dist_wrap_operand(bias.tensor), 0, process_group, rank)
        else:
            bias_global = bias
        if rank == 0:
            epilog_inputs["bias"] = bias_global.tensor

    if "DGELU" in epilogue.name:
        dgelu_input = to_host(wrap_operand(epilog_inputs["gelu_aux"]), device_id, stream)
        # remove padding from dgelu_input if any
        local_m = distributions[2].shape(rank, (m, n))[0]
        dgelu_input = dgelu_input.tensor[:local_m, :]
        dgelu_input_global = gather_array(dist_wrap_operand(dgelu_input), distributions[2].partition_dim, process_group, rank)
        if rank == 0:
            epilog_inputs["gelu_aux"] = np.asfortranarray(dgelu_input_global.tensor)
            epilog_inputs["gelu_aux"] = np.pad(
                epilog_inputs["gelu_aux"], ((0, gelu_aux_mm_shape(m, n)[0] - m), (0, 0)), mode="constant", constant_values=0
            )

    if "RELU_AUX" in epilogue.name:
        local_bitmask = dist_wrap_operand(aux_out["relu_aux"])
        local_bitmask = to_host(local_bitmask, device_id, stream)
        if distributions[2].partition_dim == 1:
            # If partitioned on N, we can directly gather the result.
            aux_out["relu_aux"] = gather_array(local_bitmask, 1, process_group, rank)
        else:
            # When D is partitioned on M we can't do a simple gather because
            # depending on the size of M and number of ranks, there could be
            # bytes in the global bitmask that refer to elements in multiple ranks.
            global_bitmask = gather_relu_mask(local_bitmask.tensor, m, n, process_group._mpi_comm, rank, nranks)
            if rank == 0:
                aux_out["relu_aux"] = wrap_operand(global_bitmask)
        if rank == 0:
            # remove the extra padding (it doesn't contain useful data and may
            # remain uninitialized)
            aux_out["relu_aux"] = wrap_operand(remove_mask_padding(aux_out["relu_aux"].tensor, m))
            if fp8:
                # gather_relu_mask converts to numpy, so convert back to torch
                aux_out["relu_aux"] = wrap_operand(torch.as_tensor(aux_out["relu_aux"].tensor))

    if "GELU_AUX" in epilogue.name:
        gelu_aux = dist_wrap_operand(aux_out["gelu_aux"])
        gelu_aux = to_host(gelu_aux, device_id, stream)
        # remove padding from gelu_aux if any
        local_m = distributions[2].shape(rank, (m, n))[0]
        gelu_aux = gelu_aux.tensor[:local_m, :]
        gelu_aux_global = gather_array(dist_wrap_operand(gelu_aux), distributions[2].partition_dim, process_group, rank)
        aux_out["gelu_aux"] = gelu_aux_global

    if "BGRAD" in epilogue.name:
        bgrad_out = dist_wrap_operand(aux_out[epilogue.name.lower()])
        bgrad_out = to_host(bgrad_out, device_id, stream)
        # For BGRAD and BGRADA, epilogue output distribution follows distribution
        # of inputs (M dimension is always partitioned with AG+GEMM and never
        # partitioned with GEMM+RS).
        # For BGRADB, epilogue output is always replicated.
        do_gather = algo == "AG+GEMM" and "BGRADB" not in epilogue.name
        if do_gather:
            bgrad_out_global = gather_array(bgrad_out, distributions[2].partition_dim, process_group, rank)
        else:
            bgrad_out_global = bgrad_out
        aux_out[epilogue.name.lower()] = bgrad_out_global

    if rank == 0:
        if "BGRAD" in epilogue.name:
            # Some epilogues require matrices in column-order (and the gather helpers always
            # return row-major order).
            a_global = wrap_operand(np.asfortranarray(a_global.tensor))
            b_global = wrap_operand(np.asfortranarray(b_global.tensor))
        if fp8:

            def to_col_major(t):
                new_t = torch.empty(t.shape[::-1], dtype=t.dtype, device=t.device)
                new_t.T[:] = t
                return new_t.T

            a_global = wrap_operand(to_col_major(a_global.tensor))
            b_global = wrap_operand(to_col_major(b_global.tensor))

        single_gpu_result = nvmath.linalg.advanced.matmul(
            a_global.tensor.T if transA else a_global.tensor,
            b_global.tensor.T if transB else b_global.tensor,
            quantization_scales=scales,
            epilog=None if epilogue.name == "DEFAULT" else cublasLt.Epilogue[epilogue.name],
            epilog_inputs=epilog_inputs,
        )
        single_gpu_aux_out = {}
        if isinstance(single_gpu_result, Sequence) and len(single_gpu_result) == 2:
            single_gpu_result, single_gpu_aux_out = single_gpu_result
            if epilogue.name.startswith("RELU"):
                # remove the extra padding (it doesn't contain useful data and may
                # remain uninitialized)
                single_gpu_aux_out["relu_aux"] = remove_mask_padding(single_gpu_aux_out["relu_aux"], m)
            elif epilogue.name.startswith("GELU"):
                single_gpu_aux_out["gelu_aux"] = single_gpu_aux_out["gelu_aux"][:m, :]  # remove padding if any
        single_gpu_result = wrap_operand(single_gpu_result)
        try:
            if fp8:
                rtol, atol = 1e-1, 1
                if algo == "GEMM+RS":
                    rtol = 0.15
            else:
                rtol, atol = 1e-5, 1e-5
            assert_close(d_global, single_gpu_result, rtol=rtol, atol=atol)
            for out_name in aux_out:
                arr1 = aux_out[out_name]
                arr2 = wrap_operand(single_gpu_aux_out[out_name])
                if "RELU_AUX" in epilogue.name:
                    # RELU_AUX output is a bitmask. Check that the number of differing bits
                    # between cuBLASMp and cuBLASLt bitmasks is below a threshold.
                    if fp8:
                        diff_bits = arr1.tensor.numpy() ^ arr2.tensor.numpy()
                    else:
                        diff_bits = arr1.tensor ^ arr2.tensor
                    error = np.sum(np.bitwise_count(diff_bits)) / (arr1.tensor.nbytes * 8)
                    assert error < 1e-3, f"Fraction of different bits is {error}"
                else:
                    if fp8:
                        rtol, atol = 1e-1, 1
                    elif "BGRAD" in epilogue.name:
                        rtol, atol = 1e-3, 1e-3
                    else:
                        rtol, atol = 1e-5, 1e-5
                    assert_close(arr1, arr2, rtol=rtol, atol=atol)
            process_group_broadcast(process_group, None)
        except Exception as e:
            # Broadcast the exception to avoid deadlock.
            process_group_broadcast(process_group, e)
            raise
    else:
        # If rank 0 raises an exception, every process has to do the same to avoid
        # deadlock.
        e = process_group_broadcast(process_group, None)
        if e is not None:
            raise e


@pytest.mark.parametrize("input_memory_space", ["cpu", "gpu"])
@pytest.mark.parametrize("with_c", [False, True], ids=["no_c", "with_c"])
def test_release_operands_refcount(input_memory_space, with_c, nvmath_distributed):
    """
    Test that after release_operands(), the refcounts of user-provided
    main operands (a, b, c) return to their initial values.
    """
    if input_memory_space == "gpu":
        cp = pytest.importorskip("cupy")

    distributed_ctx = nvmath.distributed.get_context()
    nranks = distributed_ctx.process_group.nranks
    device_id = distributed_ctx.device_id

    valid_nranks = (1, 4)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    n = 128
    local_shape = (n // nranks, n)

    # Starting from Python 3.14, sys.getrefcount() behavior changes due to
    # LOAD_FAST_BORROW, a bytecode optimization that skips the refcount
    # increment when the compiler's lifetime analysis can prove the local
    # variable outlives the stack reference.  Whether the optimization
    # applies can vary between load sites depending on code structure.
    # Pre-assigning a, b, c here ensures consistent behavior across all
    # sys.getrefcount() calls.
    # See https://github.com/python/cpython/issues/130704 for example.
    a = None
    b = None
    c = None

    if input_memory_space == "gpu":
        with device_ctx(device_id):
            a = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)
            b = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)
            c = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32) if with_c else None
    else:
        a = np.asfortranarray(np.random.rand(*local_shape), dtype=np.float32)
        b = np.asfortranarray(np.random.rand(*local_shape), dtype=np.float32)
        c = np.asfortranarray(np.random.rand(*local_shape), dtype=np.float32) if with_c else None

    initial_refs = {"a": sys.getrefcount(a), "b": sys.getrefcount(b)}
    if c is not None:
        initial_refs["c"] = sys.getrefcount(c)

    mm = nvmath.distributed.linalg.advanced.Matmul(
        a,
        b,
        c=c,
        beta=1.0 if with_c else None,
        distributions=[Slab.X] * 3,
    )
    mm.plan()
    result = mm.execute()
    with check_freed_after(result, "The caller should hold the only reference to the result buffer"):
        del result

    mm.release_operands()

    assert sys.getrefcount(a) == initial_refs["a"]
    assert sys.getrefcount(b) == initial_refs["b"]
    if c is not None:
        assert sys.getrefcount(c) == initial_refs["c"]

    mm.free()


def test_release_operands_cpu_inplace(nvmath_distributed):
    """
    Test that release_operands() releases cpu_c_ref for CPU inplace case.
    """
    nranks = nvmath.distributed.get_context().process_group.nranks

    valid_nranks = (1, 4)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    n = 128
    local_shape = (n // nranks, n)

    a = np.asfortranarray(np.random.rand(*local_shape), dtype=np.float32)
    b = np.asfortranarray(np.random.rand(*local_shape), dtype=np.float32)
    c = np.asfortranarray(np.random.rand(*local_shape), dtype=np.float32)

    initial_refcount_c = sys.getrefcount(c)
    mm = nvmath.distributed.linalg.advanced.Matmul(
        a,
        b,
        c=c,
        beta=1.0,
        distributions=[Slab.X] * 3,
        options={"inplace": True},
    )
    mm.plan()
    result = mm.execute()
    assert sys.getrefcount(c) > initial_refcount_c, (
        f"c refcount after execute: {sys.getrefcount(c)}, expected > {initial_refcount_c}. "
        f"cpu_c_ref should hold a reference to c for inplace."
    )
    del result
    mm.release_operands()

    assert sys.getrefcount(c) == initial_refcount_c
    mm.free()


def test_release_operands_then_execute_fails(nvmath_distributed):
    """
    Test that execute() raises after release_operands().
    """
    cp = pytest.importorskip("cupy")

    distributed_ctx = nvmath.distributed.get_context()
    nranks = distributed_ctx.process_group.nranks
    device_id = distributed_ctx.device_id

    valid_nranks = (1, 4)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    n = 128
    local_shape = (n // nranks, n)

    with device_ctx(device_id):
        a = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)
        b = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)

    mm = nvmath.distributed.linalg.advanced.Matmul(a, b, distributions=[Slab.X] * 3)
    mm.plan()
    _ = mm.execute()
    mm.release_operands()
    with pytest.raises(RuntimeError, match="cannot be performed after the operands have been released"):
        mm.execute()

    mm.free()


def test_release_operands_then_reset_works(nvmath_distributed):
    """
    Test that reset_operands() restores functionality after release_operands().
    """
    cp = pytest.importorskip("cupy")

    distributed_ctx = nvmath.distributed.get_context()
    nranks = distributed_ctx.process_group.nranks
    device_id = distributed_ctx.device_id

    valid_nranks = (1, 4)
    if nranks not in valid_nranks:
        pytest.skip(f"This test needs nranks in {valid_nranks}")

    n = 128
    local_shape = (n // nranks, n)

    with device_ctx(device_id):
        a = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)
        b = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)
        a_new = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)
        b_new = cp.asfortranarray(cp.random.rand(*local_shape), dtype=cp.float32)

    mm = nvmath.distributed.linalg.advanced.Matmul(a, b, distributions=[Slab.X] * 3)
    mm.plan()
    _ = mm.execute()
    mm.release_operands()
    mm.reset_operands(a=a_new, b=b_new)
    result = mm.execute()
    assert result is not None

    mm.free()
