# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

#
# C++ CUDA implementation reference:
# https://github.com/NVIDIA/CUDALibrarySamples/tree/master/MathDx/cuBLASDx/16_dgemm_emulation
#

#
# Based on the: Hiroyuki Ootomo, Katsuhisa Ozaki, and Rio Yokota. 2018. DGEMM on
#   Integer Matrix Multiplication Unit.
#

#
# Basic idea:
# 1. Split full-precision matrices A,B into multiple lower-precision splits.
# 2. Multiply each low-precision split pair using IMMA instructions (int8 * int8 -> int32).
# 3. Accumulate split products in high precision to recover full-precision accuracy.
#

from numba import cuda
import numpy as np
from numba import int32, int8, int16, float64, int64, types
from numba.types import Tuple
import cuda.cccl.cooperative.experimental as cudax

from common import mm_perf_GFlops, random_real
from common_numba import time_numba
from nvmath.device import matmul
from nvmath.device.cublasdx import MAX_ALIGNMENT, BlasOptionsComplete, SharedStorageCalc
from nvmath.device.common import (
    clear,
    copy,
    copy_fragment,
    copy_wait,
    make_tensor,
)
from nvmath.device.common_cuda import Dim3

MANTISSA_FRACTION_BITS = 52
MANTISSA_FRACTION_MASK = (1 << MANTISSA_FRACTION_BITS) - 1

EXPONENT_BITS = 11
EXPONENT_MASK = (1 << EXPONENT_BITS) - 1
MAX_EXPONENT = (1 << (EXPONENT_BITS - 1)) - 1

SIGN_BITS = 1
SIGN_MASK = (1 << SIGN_BITS) - 1

BYTE_BITS = 8
BYTE_MASK = (1 << BYTE_BITS) - 1

INT32_BYTES = 4
INT32_BITS = INT32_BYTES * BYTE_BITS


@cuda.jit(Tuple((int16, int64))(float64), device=True, forceinline=True, cache=True)
def decompose_float64(x):
    """Extract the mantissa, exponent, and sign from the floating point representation"""
    # Reinterpret bits as uint64
    bits = x.view(np.uint64)

    # Extract sign
    sign = (bits >> np.uint64(MANTISSA_FRACTION_BITS + EXPONENT_BITS)) & np.uint64(SIGN_MASK)

    # Extract raw exponent and compute unbiased exponent
    raw_exp = (bits >> np.uint64(MANTISSA_FRACTION_BITS)) & np.uint64(EXPONENT_MASK)
    exponent = np.int16(raw_exp) - np.int16(MAX_EXPONENT)
    exponent -= np.int16(MANTISSA_FRACTION_BITS)  # Adjust exponent to account for the implicit leading 1

    # Extract mantissa
    mantissa = bits & np.uint64(MANTISSA_FRACTION_MASK)

    # Add implicit leading 1 for normalized numbers
    if raw_exp != 0:
        mantissa = mantissa | np.uint64(1 << MANTISSA_FRACTION_BITS)

    # Improve precsision at zero cost
    # We are using 7 int8 chunks to store mantissa. It is 7*8 = 56 bits.
    # Mantissa is 52 bits + 1 bit for implicit leading 1. Plus we have 1 bit
    # for sign and 1 bit for overflow. 52 + 1 + 1 + 1 = 55 bits. We have
    # 56-55 = 1 bit of free space.
    mantissa <<= np.uint64(1)
    exponent -= 1

    mantissa = int64(mantissa)
    if sign != 0:
        mantissa = -mantissa

    return exponent, mantissa


def build_split_mantissa(splits: int):
    """
    Build a kernel to split the mantissa into int8 chunks.
    """

    @cuda.jit(types.void(int64, int8[:]), device=True, forceinline=True, cache=True)
    def split_mantissa(mantissa, splits_array):
        # Represent mantissa in the signed base of 256 (use -128..127 range
        # instead of 0..255).
        # Here is how it works for 9928:
        # 1. Represent in 256 base: 9928 =  38 * 256 + 200
        # 2. If the value is bigger than 127 reduce it (for each chunk), by
        # shifting 256 to the next chunk:
        #    9928 = 38 * 256 + 256 - 256 + 200 = 39 * 256 - 56
        # WARNING: we can't flip the sign for -128. However we can follow same
        #   algorithm for negative mantissa to achieve negative representation.

        for i in range(splits):
            reg_pack = mantissa & np.uint64(BYTE_MASK)
            split = np.int8(reg_pack)
            mantissa = (mantissa >> np.uint64(BYTE_BITS)) + ((reg_pack >> np.uint64(BYTE_BITS - 1)) & np.uint64(1))
            splits_array[i] = split

        return

    return split_mantissa


def build_split_kernel(k, threads, splits=7, order="C"):
    """
    Build a kernel to split float64 matrix into multiple int8 matrixes with
    common exponent in rows.
    """
    assert k >= threads

    split_mantissa = build_split_mantissa(splits)

    def op_max(a, b):
        return a if a > b else b

    block_reduce = cudax.block.reduce(int16, threads, op_max)

    items_per_thread = (k + threads - 1) // threads

    min_int16 = np.int16(np.iinfo(np.int16).min)

    @cuda.jit(types.void(float64[:, :], int8[:, :, :], int16[:]), link=block_reduce.files)
    def split_kernel(a, splits, column_exponents):
        exponents = cuda.local.array(shape=items_per_thread, dtype=int16)
        mantissas = cuda.local.array(shape=items_per_thread, dtype=int64)
        max_exp_shared = cuda.shared.array(shape=1, dtype=int16)

        max_exp = np.int16(min_int16)

        threadId = cuda.threadIdx.x
        y = cuda.blockIdx.x

        for i in range(items_per_thread):
            x = threadId + i * threads
            if x >= k:
                break

            exp, mnt = decompose_float64(a[x, y] if order == "F" else a[y, x])
            max_exp = max(max_exp, exp)

            exponents[i] = exp
            mantissas[i] = mnt

        max_exp = block_reduce(max_exp)

        if threadId == 0:
            max_exp_shared[0] = max_exp
            column_exponents[y] = max_exp

        cuda.syncthreads()
        max_exp = max_exp_shared[0]

        for i in range(items_per_thread):
            x = threadId + i * threads

            if x >= k:
                break

            exp, mnt = exponents[i], mantissas[i]

            # Align mantissa
            mnt >>= max_exp - exp

            split_mantissa(
                mnt,
                splits[x, y, :] if order == "F" else splits[:, y, x],
            )

    return split_kernel


def matmul_specification(tile_m, tile_n, tile_k, block_size, alignment) -> BlasOptionsComplete:
    return matmul(
        size=(tile_m, tile_n, tile_k),
        precision=(np.int8, np.int8, np.int32),
        data_type="real",
        arrangement=("row_major", "col_major", "col_major"),
        execution="Block",
        block_size=block_size,
        alignment=alignment,
        global_memory_alignment=alignment,
        static_block_dim=True,
        compiler="numba",
        execute_api="tensors",
        tensor_types=("suggested_smem_a", "suggested_smem_b", "suggested_rmem_c"),
    )


def build_single_matmul(m: int, n: int, k: int, MM: BlasOptionsComplete):
    tile_m, tile_n, tile_k = MM.size

    assert m % tile_m == 0
    assert n % tile_n == 0
    assert k % tile_k == 0

    grid_dim = Dim3(m // tile_m, n // tile_n, 1)

    @cuda.jit(link=MM.files, device=True, forceinline=True)
    def matmul_func(a, b, smem_a, smem_b, smem_a_n, smem_b_n, rmem_c):
        block_m = cuda.blockIdx.x
        block_n = cuda.blockIdx.y

        # 1. PREPARE GLOBAL MEMORY TENSORS
        a_slice = a[block_m * tile_m : (block_m + 1) * tile_m, :]
        b_slice = b[:, block_n * tile_n : (block_n + 1) * tile_n]

        # 2. PREPARE 2-STAGE MEMORY PIPELINE
        stages = k // tile_k

        a_tile = a_slice[:, 0:tile_k]
        b_tile = b_slice[0:tile_k, :]

        gmem_a = make_tensor(a_tile, MM.get_layout_gmem_a(k))
        gmem_b = make_tensor(b_tile, MM.get_layout_gmem_b(k))

        copy(gmem_a, smem_a)
        copy(gmem_b, smem_b)

        # 3. EXECUTE GEMM WITH ACCUMULATION IN REGISTERS
        for stage in range(1, stages):
            # Wait for previous stage
            copy_wait()

            # Copy tile for the next iteration
            a_tile = a_slice[:, stage * tile_k : (stage + 1) * tile_k]
            b_tile = b_slice[stage * tile_k : (stage + 1) * tile_k, :]

            gmem_a = make_tensor(a_tile, MM.get_layout_gmem_a(k))
            gmem_b = make_tensor(b_tile, MM.get_layout_gmem_b(k))

            copy(gmem_a, smem_a_n)
            copy(gmem_b, smem_b_n)

            # Accumulate results from this stage
            MM.execute(smem_a, smem_b, rmem_c)

            # Swap for the next iteration
            smem_a_n, smem_a = smem_a, smem_a_n
            smem_b_n, smem_b = smem_b, smem_b_n

        copy_wait()
        MM.execute(smem_a, smem_b, rmem_c)

    return matmul_func, grid_dim


def build_looped_matmul(
    m,
    n,
    k,
    tile_m,
    tile_n,
    tile_k,
    block_size,
    splits,
    alignment,
    device=False,
):
    MM = matmul_specification(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_size=block_size,
        alignment=alignment,
    )

    matmul_func, grid_dim = build_single_matmul(m, n, k, MM)

    a_size = MM.suggest_layout_smem_a().cosize
    b_size = MM.suggest_layout_smem_b().cosize
    c_size = MM.suggest_layout_rmem_c().cosize

    @cuda.jit(link=MM.files, device=device, forceinline=device)
    def matmul_kernel(a_split, b_split, output):
        block_m = cuda.blockIdx.x
        block_n = cuda.blockIdx.y

        ## 1. Prepare shared memory and local memory

        # We have same precision for all input tensors
        smem = cuda.shared.array(shape=(0,), dtype=np.int8, alignment=16)

        # Shared memory
        smem_a_buff, smem = smem[0:a_size], smem[a_size:]
        smem_b_buff, smem = smem[0:b_size], smem[b_size:]
        smem_a_n_buff, smem = smem[0:a_size], smem[a_size:]
        smem_b_n_buff, smem = smem[0:b_size], smem[b_size:]

        smem_a = make_tensor(smem_a_buff, MM.suggest_layout_smem_a())
        smem_b = make_tensor(smem_b_buff, MM.suggest_layout_smem_b())
        smem_a_n = make_tensor(smem_a_n_buff, MM.suggest_layout_smem_a())
        smem_b_n = make_tensor(smem_b_n_buff, MM.suggest_layout_smem_b())

        # Register accumulator memory
        rmem_c_out_buff1 = cuda.local.array(shape=(c_size,), dtype=MM.c_value_type)
        rmem_c_out1 = make_tensor(rmem_c_out_buff1, MM.suggest_layout_rmem_c())

        rmem_c_out_buff2 = cuda.local.array(shape=(c_size,), dtype=MM.c_value_type)
        rmem_c_out2 = make_tensor(rmem_c_out_buff2, MM.suggest_layout_rmem_c())
        rmem_c_out_buff2 = rmem_c_out_buff2.view(np.uint32)

        clear(rmem_c_out1)
        clear(rmem_c_out2)

        ## 2. Matmul with accumulation
        for i in range(splits):
            # Shift result:
            for j in range(c_size):
                rmem_c_out_buff2[j] >>= np.uint32(BYTE_BITS)
                rmem_c_out_buff2[j] |= np.uint32(rmem_c_out_buff1[j] & BYTE_MASK) << ((INT32_BYTES - 1) * BYTE_BITS)
                rmem_c_out_buff1[j] >>= np.uint32(BYTE_BITS)

            for j in range(splits - i):
                a = a_split[j + i, :, :]
                b = b_split[:, :, splits - 1 - j]

                matmul_func(
                    a,
                    b,
                    smem_a,
                    smem_b,
                    smem_a_n,
                    smem_b_n,
                    rmem_c_out1,
                )

        ## 3. Store result to global memory
        output_tile = output[
            block_m * tile_m : (block_m + 1) * tile_m,
            block_n * tile_n : (block_n + 1) * tile_n,
            0,
        ]
        gmem_output = make_tensor(output_tile, MM.get_layout_gmem_c(m))
        copy_fragment(rmem_c_out1, gmem_output)

        output_tile = output[
            block_m * tile_m : (block_m + 1) * tile_m,
            block_n * tile_n : (block_n + 1) * tile_n,
            1,
        ]
        gmem_output = make_tensor(output_tile, MM.get_layout_gmem_c(m))
        copy_fragment(rmem_c_out2, gmem_output)

    smem_calc = SharedStorageCalc()
    itemsize = np.dtype(np.int8).itemsize
    smem_calc.add(MM.alignment.a, itemsize, MM.suggest_layout_smem_a())
    smem_calc.add(MM.alignment.b, itemsize, MM.suggest_layout_smem_b())
    smem_calc.add(MM.alignment.a, itemsize, MM.suggest_layout_smem_a())
    smem_calc.add(MM.alignment.b, itemsize, MM.suggest_layout_smem_b())
    shared_memory_size = smem_calc.get()

    return matmul_kernel, grid_dim, MM.block_dim, shared_memory_size, MM.files


@cuda.jit(float64(int16, int64), device=True, forceinline=True, cache=True)
def compose_float64(exponent, mantissa):
    # same as exponent = np.float64(2) ** exponent
    exponent += 1023
    exponent = np.uint64(exponent) << np.uint64(52)
    exponent = exponent.view(np.float64)

    return np.float64(mantissa) * exponent


def build_compose_kernel(tile_size, threads, exp_shift, device=False):
    tile_m, tile_n = tile_size[0], tile_size[1]

    assert threads % tile_m == 0 or tile_m % threads == 0

    lines = (threads + tile_m - 1) // tile_m
    repeats_per_line = (tile_m + threads - 1) // threads

    exp_shift = int16(exp_shift)

    @cuda.jit(
        types.void(float64, int16[:], int16[:], int32[:, :, :], float64, float64[:, :], float64[:, :]),
        device=device,
        forceinline=device,
        cache=True,
    )
    def compose_kernel(alpha, exponent_a, exponent_b, mantissa, beta, c, out):
        block_m = cuda.blockIdx.x
        block_n = cuda.blockIdx.y

        start_m, start_n = block_m * tile_m, block_n * tile_n
        end_m, end_n = start_m + tile_m, start_n + tile_n

        m1 = mantissa[start_m:end_m, start_n:end_n, 0]
        m2 = mantissa[start_m:end_m, start_n:end_n, 1].view(np.uint32)
        output_tile = out[start_m:end_m, start_n:end_n]
        c_tile = c[start_m:end_m, start_n:end_n]

        thread = cuda.threadIdx.x
        i0 = thread % tile_m

        for r in range(repeats_per_line):
            i = i0 + r * threads
            for l in range(lines):
                exp_shift_a = exp_shift + exponent_a[start_m + i]

                for j_it in range(0, tile_n, lines):
                    j = j_it + l
                    if j >= tile_n:
                        continue

                    exp = exponent_b[j + start_n] + exp_shift_a
                    m = (np.int64(m1[i, j]) << INT32_BITS) | np.int64(m2[i, j])
                    x, y = compose_float64(exp, m), c_tile[i, j]
                    output_tile[i, j] = alpha * x + beta * y

    return compose_kernel


def main(m, n, k, tile_m, tile_n, tile_k, block_size, run_perf=True):
    ncycles = 100
    splits = 7

    alignment = MAX_ALIGNMENT
    split_block_size = min(256, k)

    alpha, beta = 1.1, 1.2
    a = random_real((m, k), np.float64, order="C")
    b = random_real((k, n), np.float64, order="F")
    c = random_real((m, n), np.float64, order="F")

    o = np.empty_like(c)

    a_split = np.empty((splits, m, k), dtype=np.int8, order="C")
    b_split = np.empty((k, n, splits), dtype=np.int8, order="F")

    a_d = cuda.to_device(a)
    b_d = cuda.to_device(b)
    c_d = cuda.to_device(c)
    a_split_d = cuda.to_device(a_split)
    b_split_d = cuda.to_device(b_split)

    max_e_a_d = cuda.to_device(np.empty(m, dtype=np.int16))
    max_e_b_d = cuda.to_device(np.empty(n, dtype=np.int16))

    o_d = cuda.to_device(o)
    m_o_d = cuda.to_device(np.empty((o.shape[0], o.shape[1], 2), dtype=np.int32, order="F"))

    split_a_kernel = build_split_kernel(k, split_block_size, splits=splits, order="C")
    split_b_kernel = build_split_kernel(k, split_block_size, splits=splits, order="F")

    cumulative_matmul, grid_dim, block_dim, shared_memory_size, files = build_looped_matmul(
        m,
        n,
        k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_size=block_size,
        splits=splits,
        alignment=alignment,
        device=True,
    )
    exp_shift = int(BYTE_BITS * (splits + 1))
    assert block_dim[1] == 1 and block_dim[2] == 1
    compose_kernel = build_compose_kernel((tile_m, tile_n), block_dim[0], exp_shift, device=True)

    @cuda.jit(link=files)
    def fused_kernel(alpha, a_split_d, b_split_d, m_o_d, max_e_a_d, max_e_b_d, beta, c, o_d):
        cumulative_matmul(a_split_d, b_split_d, m_o_d)
        cuda.syncthreads()
        compose_kernel(alpha, max_e_a_d, max_e_b_d, m_o_d, beta, c, o_d)

    print(f"Problem size: {m}, {n}, {k}")
    print(f"Tile size: {tile_m}x{tile_n}x{tile_k}")
    print(f"Block size: {block_size}")
    print(f"Alignment: {alignment}")
    print(f"Split block size: {split_block_size}")

    kernel_args = (alpha, a_split_d, b_split_d, m_o_d, max_e_a_d, max_e_b_d, beta, c_d, o_d)
    if run_perf:
        split_a_time_mm_ms = time_numba(split_a_kernel, m, split_block_size, 0, ncycles, a_d, a_split_d, max_e_a_d)
        split_b_time_mm_ms = time_numba(split_b_kernel, n, split_block_size, 0, ncycles, b_d, b_split_d, max_e_b_d)
        time_mm_ms = time_numba(fused_kernel, grid_dim, block_dim, shared_memory_size, ncycles, *kernel_args)

        perf = mm_perf_GFlops((m, n, k), 1, time_mm_ms) / 1000
        e2e_time_ms = split_a_time_mm_ms + split_b_time_mm_ms + time_mm_ms
        e2e_perf = mm_perf_GFlops((m, n, k), 1, e2e_time_ms) / 1000

        print(f"Split A time [ms]: {split_a_time_mm_ms}")
        print(f"Split B time [ms]: {split_b_time_mm_ms}")
        print(f"Matmul time [ms]: {time_mm_ms}")
        print(f"Performance [TFLOPS]: {perf}")
        print(f"Avg E2E time [ms]: {e2e_time_ms}")
        print(f"E2E performance [TFLOPS]: {e2e_perf}")
        print(f"Split performance impact: {(perf - e2e_perf) / e2e_perf * 100:.2f} %")
    else:
        split_a_kernel[m, split_block_size](a_d, a_split_d, max_e_a_d)
        split_b_kernel[n, split_block_size](b_d, b_split_d, max_e_b_d)
        cuda.synchronize()
        fused_kernel[grid_dim, block_dim, 0, shared_memory_size](*kernel_args)
        cuda.synchronize()

    data_test = o_d.copy_to_host()

    data_ref = alpha * (a @ b) + beta * c

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"Error: {error}")
    assert error < 1e-5, "Error is too high"

    return perf if run_perf else None


def blackwell_perf():
    for size, tile_m, tile_n, tile_k, block_size in [
        (256, 64, 64, 128, 128),
        (512, 64, 64, 128, 128),
        (1024, 128, 64, 128, 128),
        (2048, 128, 64, 128, 128),
        (4096, 128, 64, 128, 128),
        (8192, 128, 128, 64, 256),
        (16384, 64, 256, 64, 256),
    ]:
        main(size, size, size, tile_m, tile_n, tile_k, block_size)
        print()


def ada_perf():
    for size, tile_m, tile_n, tile_k, block_size in [
        (256, 64, 64, 128, 256),
        (512, 64, 64, 128, 256),
        (1024, 128, 64, 128, 128),
        (2048, 256, 128, 64, 256),
        (4096, 256, 128, 64, 256),
        (8192, 256, 128, 64, 256),
        (16384, 128, 256, 64, 256),
    ]:
        main(size, size, size, tile_m, tile_n, tile_k, block_size)
        print()


if __name__ == "__main__":
    m, n, k = 2048, 2048, 2048
    tile_m, tile_n, tile_k, block_size = 128, 64, 128, 128
    perf = main(m, n, k, tile_m, tile_n, tile_k, block_size)
