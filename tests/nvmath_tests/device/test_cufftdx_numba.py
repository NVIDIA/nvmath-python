# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda
import pytest

from nvmath.device import FFTOptions
from nvmath.device import current_device_lto, fft, float16x4, float16x2, float64x2_type, float32x2_type, float16x4_type
from nvmath.device.cufftdx import FFTCompiled, FFTNumba
from .helpers import _TOLERANCE, random_complex, random_real, show_FFT_traits, complex64_to_fp16x2, fp16x2_to_complex64

np.random.seed(314 + 271)


def make_numba_input_outputs(fft_type, precision, direction, batch, fft_size, real_fft_options):
    if fft_type == "c2c":
        input_size = fft_size
        output_size = fft_size
        input_h = random_complex((batch, fft_size), precision)
        if direction == "forward":
            output_ref = np.fft.fft(input_h, axis=-1, norm="backward")
        else:
            output_ref = np.fft.ifft(input_h, axis=-1, norm="forward")

    elif fft_type == "r2c":
        assert direction == "forward"
        input_size = fft_size
        input_h = random_real((batch, fft_size), precision)
        if real_fft_options["complex_layout"] == "natural":
            output_ref = np.fft.rfft(input_h, axis=-1, norm="backward")
            output_size = fft_size // 2 + 1
        elif real_fft_options["complex_layout"] == "full":
            output_ref = np.fft.fft(input_h, axis=-1, norm="backward")
            output_size = fft_size
        elif real_fft_options["complex_layout"] == "packed":
            assert fft_size % 2 == 0
            output_ref = np.fft.rfft(input_h, axis=-1, norm="backward")
            output_ref[:, 0].imag = output_ref[:, fft_size // 2].real
            output_ref = output_ref[:, : fft_size // 2]
            output_size = fft_size // 2

    elif fft_type == "c2r":
        assert direction == "inverse"
        output_size = fft_size
        if real_fft_options["complex_layout"] == "natural":
            output_ref = random_real((batch, fft_size), precision)
            input_h = np.fft.rfft(output_ref, norm="forward")  # Necessary for symmetry
            input_size = fft_size // 2 + 1
        elif real_fft_options["complex_layout"] == "full":
            output_ref = random_real((batch, fft_size), precision)
            input_h = np.fft.fft(output_ref, norm="forward")  # Necessary for symmetry
            input_size = fft_size
        elif real_fft_options["complex_layout"] == "packed":
            assert fft_size % 2 == 0
            output_ref = random_real((batch, fft_size), precision)
            input_h = np.fft.fft(output_ref, norm="forward")  # Necessary for symmetry
            input_h[:, 0].imag = input_h[:, fft_size // 2].real
            input_h = input_h[:, : fft_size // 2]
            input_size = fft_size // 2

    assert input_h.shape == (batch, input_size)
    assert output_ref.shape == (batch, output_size)

    if precision == np.float16:
        if fft_type == "r2c":
            input_d = cuda.to_device(input_h)
        else:
            input_d = cuda.to_device(complex64_to_fp16x2(input_h))
        if fft_type == "c2r":
            output_d = cuda.to_device(np.zeros_like(output_ref))
        else:
            output_d = cuda.to_device(complex64_to_fp16x2(np.zeros_like(output_ref)))
    else:
        input_d = cuda.to_device(input_h)
        output_d = cuda.to_device(np.zeros_like(output_ref))

    return input_h, output_ref, input_d, output_d


def convert_output(fft_type, precision, output_d):
    if precision == np.float16:
        if fft_type == "c2r":
            output_test = output_d.copy_to_host()
        else:
            output_test = fp16x2_to_complex64(output_d.copy_to_host())
    else:
        output_test = output_d.copy_to_host()
    return output_test


COMPLEX_TYPE_MAP = {np.float16: float16x4_type, np.float32: float32x2_type, np.float64: float64x2_type}

IMPLICIT_BATCHING_MAP = {
    np.float16: 2,
    np.float32: 1,
    np.float64: 1,
}

DEFAULT_REAL_FFT_OPTIONS = {"complex_layout": "natural", "real_mode": "normal"}

SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 20]
SIZES += [2**l for l in range(1, 8)]
SIZES += [3**l for l in range(1, 3)]
SIZES += [5**l for l in range(1, 3)]
SIZES += [6**l for l in range(1, 3)]
SIZES += [7**l for l in range(1, 3)]
SIZES += [10**l for l in range(1, 3)]
SIZES += [11**l for l in range(1, 2)]
SIZES += [12**l for l in range(1, 2)]
SIZES = sorted(list(set(SIZES)))

TEST_CASES = []

for i, s in enumerate(SIZES):
    # This is done to limit the number of tests
    if i % 3 == 0:
        tt = "c2c"
        ff = "forward" if i % 2 == 0 else "inverse"
        pp = np.float16
    elif i % 3 == 1:
        tt = "r2c"
        ff = "forward"
        pp = np.float32
    elif i % 3 == 2:
        tt = "c2r"
        ff = "inverse"
        pp = np.float64
    if i % 2 == 0:
        api = "thread"
    else:
        api = "smem"
    TEST_CASES.append((tt, s, pp, ff, api, None, None))

TEST_CASES.append(("c2c", 32, np.float16, "inverse", "thread", 2, None))
TEST_CASES.append(("c2r", 32, np.float32, "inverse", "thread", 8, None))
TEST_CASES.append(("r2c", 32, np.float64, "forward", "thread", 16, None))
TEST_CASES.append(("c2c", 64, np.float16, "forward", "thread", 16, None))
TEST_CASES.append(("c2r", 64, np.float32, "inverse", "thread", 32, None))
TEST_CASES.append(("r2c", 32, np.float64, "forward", "thread", 16, None))

# real_mode Normal
TEST_CASES.append(("r2c", 4, np.float16, "forward", "thread", None, {"complex_layout": "packed", "real_mode": "normal"}))
TEST_CASES.append(("r2c", 8, np.float16, "forward", "thread", None, {"complex_layout": "packed", "real_mode": "normal"}))
TEST_CASES.append(("r2c", 16, np.float16, "forward", "smem", None, {"complex_layout": "packed", "real_mode": "normal"}))
TEST_CASES.append(("r2c", 32, np.float16, "forward", "smem", None, {"complex_layout": "packed", "real_mode": "normal"}))

TEST_CASES.append(("r2c", 5, np.float32, "forward", "thread", None, {"complex_layout": "full", "real_mode": "normal"}))
TEST_CASES.append(("r2c", 9, np.float32, "forward", "thread", None, {"complex_layout": "full", "real_mode": "normal"}))
TEST_CASES.append(("r2c", 17, np.float32, "forward", "smem", None, {"complex_layout": "full", "real_mode": "normal"}))
TEST_CASES.append(("r2c", 33, np.float32, "forward", "smem", None, {"complex_layout": "full", "real_mode": "normal"}))

TEST_CASES.append(("c2r", 64, np.float16, "inverse", "thread", None, {"complex_layout": "packed", "real_mode": "normal"}))
TEST_CASES.append(("c2r", 128, np.float16, "inverse", "thread", None, {"complex_layout": "packed", "real_mode": "normal"}))
TEST_CASES.append(("c2r", 256, np.float16, "inverse", "smem", None, {"complex_layout": "packed", "real_mode": "normal"}))
TEST_CASES.append(("c2r", 512, np.float16, "inverse", "smem", None, {"complex_layout": "packed", "real_mode": "normal"}))

TEST_CASES.append(("c2r", 9, np.float64, "inverse", "thread", None, {"complex_layout": "full", "real_mode": "normal"}))
TEST_CASES.append(("c2r", 13, np.float64, "inverse", "thread", None, {"complex_layout": "full", "real_mode": "normal"}))
TEST_CASES.append(("c2r", 17, np.float64, "inverse", "smem", None, {"complex_layout": "full", "real_mode": "normal"}))
TEST_CASES.append(("c2r", 13, np.float64, "inverse", "smem", None, {"complex_layout": "full", "real_mode": "normal"}))

# real_mode Folded
TEST_CASES.append(("c2r", 4, np.float32, "inverse", "thread", None, {"complex_layout": "full", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 8, np.float64, "inverse", "smem", None, {"complex_layout": "natural", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 16, np.float16, "inverse", "smem", None, {"complex_layout": "packed", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 32, np.float32, "inverse", "thread", None, {"complex_layout": "full", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 64, np.float64, "inverse", "thread", None, {"complex_layout": "natural", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 128, np.float16, "inverse", "smem", None, {"complex_layout": "packed", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 256, np.float32, "inverse", "smem", None, {"complex_layout": "full", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 512, np.float64, "inverse", "thread", None, {"complex_layout": "natural", "real_mode": "folded"}))
TEST_CASES.append(("c2r", 1024, np.float16, "inverse", "thread", None, {"complex_layout": "full", "real_mode": "folded"}))

TEST_CASES.append(("r2c", 4, np.float32, "forward", "thread", None, {"complex_layout": "full", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 8, np.float64, "forward", "smem", None, {"complex_layout": "natural", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 16, np.float16, "forward", "smem", None, {"complex_layout": "packed", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 32, np.float32, "forward", "thread", None, {"complex_layout": "full", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 64, np.float64, "forward", "thread", None, {"complex_layout": "natural", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 128, np.float16, "forward", "smem", None, {"complex_layout": "packed", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 256, np.float32, "forward", "smem", None, {"complex_layout": "full", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 512, np.float64, "forward", "thread", None, {"complex_layout": "natural", "real_mode": "folded"}))
TEST_CASES.append(("r2c", 1024, np.float16, "forward", "thread", None, {"complex_layout": "full", "real_mode": "folded"}))


# Supports: Block APIs, C2R/R2C/C2C, all precision, all real_fft_options
@pytest.mark.parametrize("fft_type,size,precision,direction,api_kind,ept,real_fft_options", TEST_CASES)
def test_block(fft_type, size, precision, direction, api_kind, ept, real_fft_options):
    ffts_per_block = 4 if precision == np.float16 else 2
    num_batches = 3 * ffts_per_block

    FFT = fft(
        fft_type=fft_type,
        ffts_per_block=ffts_per_block,
        elements_per_thread=ept,
        size=size,
        precision=precision,
        direction=direction,
        real_fft_options=real_fft_options,
        execution="Block",
        compiler="numba",
    )

    complex_type = FFT.value_type
    storage_size = FFT.storage_size
    shared_memory_size = FFT.shared_memory_size
    files = FFT.files
    stride = FFT.stride
    elements_per_thread = FFT.elements_per_thread
    block_dim = FFT.block_dim
    implicit_type_batching = FFT.implicit_type_batching
    input_type = FFT.input_type
    output_type = FFT.output_type

    # cuFFTDx default ffts_per_block is 1 for FP32/FP64 and 2 for FP16
    assert api_kind in ["thread", "smem"]
    assert FFT.ffts_per_block == ffts_per_block
    assert complex_type == COMPLEX_TYPE_MAP[precision]
    assert all([code.endswith(".ltoir") for code in files])
    assert FFT.size == size
    assert implicit_type_batching == IMPLICIT_BATCHING_MAP[precision]
    assert not FFT.requires_workspace
    if ept is not None:
        assert elements_per_thread == ept
    if real_fft_options is None:
        real_fft_options = DEFAULT_REAL_FFT_OPTIONS

    IS_FP16 = np.float16 == precision
    IS_SMEM = api_kind == "smem"

    IS_INPUT_REAL = fft_type == "r2c"
    IS_OUTPUT_REAL = fft_type == "c2r"

    if real_fft_options["complex_layout"] == "natural":
        complex_size = size // 2 + 1
    elif real_fft_options["complex_layout"] == "packed":
        complex_size = size // 2
    elif real_fft_options["complex_layout"] == "full":
        complex_size = size

    INPUT_SIZE = complex_size if fft_type == "c2r" else size
    OUTPUT_SIZE = complex_size if fft_type == "r2c" else size

    INPUT_FOLDED = real_fft_options["real_mode"] == "folded" and fft_type == "r2c"
    OUTPUT_FOLDED = real_fft_options["real_mode"] == "folded" and fft_type == "c2r"

    make_complex_type = complex_type.make
    if INPUT_FOLDED:
        INPUT_SIZE = INPUT_SIZE // 2
    if OUTPUT_FOLDED:
        OUTPUT_SIZE = OUTPUT_SIZE // 2

    @cuda.jit(link=FFT.files)
    def f(input, output):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=complex_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=complex_type)

        thread_input = thread_data.view(input_type)
        thread_output = thread_data.view(output_type)

        shared_mem_input = shared_mem.view(input_type)
        shared_mem_output = shared_mem.view(output_type)

        local_fft_id = implicit_type_batching * cuda.threadIdx.y
        global_fft_id = (cuda.blockIdx.x * ffts_per_block) + local_fft_id

        for i in range(elements_per_thread):
            idx = i * stride + cuda.threadIdx.x
            smem_idx = cuda.threadIdx.y * INPUT_SIZE + idx
            if idx < INPUT_SIZE:
                if IS_FP16:
                    if IS_INPUT_REAL and not INPUT_FOLDED:
                        # R0, R1
                        r0, r1 = input[global_fft_id, idx], input[global_fft_id + 1, idx]
                        # R0 R1
                        if IS_SMEM:
                            shared_mem_input[smem_idx] = float16x2(r0, r1)
                        else:
                            thread_input[i] = float16x2(r0, r1)
                    else:
                        # R0, R1
                        r0, r1 = input[global_fft_id, 2 * idx + 0], input[global_fft_id + 1, 2 * idx + 0]
                        # I0, I1
                        i0, i1 = input[global_fft_id, 2 * idx + 1], input[global_fft_id + 1, 2 * idx + 1]
                        # R0 R1 I0 I1
                        if IS_SMEM:
                            shared_mem_input[smem_idx] = float16x4(r0, r1, i0, i1)
                        else:
                            thread_input[i] = float16x4(r0, r1, i0, i1)
                else:
                    if INPUT_FOLDED:
                        value = make_complex_type(input[global_fft_id, 2 * idx + 0], input[global_fft_id, 2 * idx + 1])
                    else:
                        value = input[global_fft_id, idx]
                    if IS_SMEM:
                        shared_mem_input[smem_idx] = value
                    else:
                        thread_input[i] = value

        # Execute FFT
        if IS_SMEM:
            cuda.syncthreads()
            FFT(shared_mem)
            cuda.syncthreads()
        else:
            FFT(thread_data, shared_mem)

        # Save results
        for i in range(elements_per_thread):
            idx = i * stride + cuda.threadIdx.x
            smem_idx = cuda.threadIdx.y * OUTPUT_SIZE + idx
            if idx < OUTPUT_SIZE:
                if IS_FP16:
                    if IS_OUTPUT_REAL and not OUTPUT_FOLDED:
                        if IS_SMEM:
                            r0, r1 = shared_mem_output[smem_idx].x, shared_mem_output[smem_idx].y
                        else:
                            r0, r1 = thread_output[i].x, thread_output[i].y
                        output[global_fft_id + 0, idx] = r0
                        output[global_fft_id + 1, idx] = r1
                    else:
                        if IS_SMEM:
                            r0, r1 = shared_mem_output[smem_idx].x, shared_mem_output[smem_idx].y
                            i0, i1 = shared_mem_output[smem_idx].z, shared_mem_output[smem_idx].w
                        else:
                            # R0 R1 I0 I1
                            r0, r1 = thread_output[i].x, thread_output[i].y
                            i0, i1 = thread_output[i].z, thread_output[i].w
                        # R0, R1
                        output[global_fft_id + 0, 2 * idx + 0] = r0
                        output[global_fft_id + 1, 2 * idx + 0] = r1
                        # I0, I1
                        output[global_fft_id + 0, 2 * idx + 1] = i0
                        output[global_fft_id + 1, 2 * idx + 1] = i1
                else:
                    if IS_SMEM:
                        if OUTPUT_FOLDED:
                            output[global_fft_id, 2 * idx + 0], output[global_fft_id, 2 * idx + 1] = (
                                shared_mem_output[smem_idx].x,
                                shared_mem_output[smem_idx].y,
                            )
                        else:
                            output[global_fft_id, idx] = shared_mem_output[smem_idx]
                    else:
                        if OUTPUT_FOLDED:
                            output[global_fft_id, 2 * idx + 0], output[global_fft_id, 2 * idx + 1] = (
                                thread_output[i].x,
                                thread_output[i].y,
                            )
                        else:
                            output[global_fft_id, idx] = thread_output[i]

    input_h, output_ref, input_d, output_d = make_numba_input_outputs(
        fft_type, precision, direction, num_batches, size, real_fft_options
    )

    grid_dim = (num_batches + ffts_per_block - 1) // ffts_per_block

    print(f"grid_dim = {grid_dim}, block_dim = {block_dim}, shared_memory_size = {shared_memory_size}")

    if IS_SMEM:
        smem = max(shared_memory_size, ffts_per_block * size * precision(1.0).itemsize * 2)
        f[grid_dim, block_dim, 0, smem](input_d, output_d)
    else:
        f[grid_dim, block_dim, 0, shared_memory_size](input_d, output_d)
    cuda.synchronize()

    output_test = convert_output(fft_type, precision, output_d)

    print("Input: ", input_h)
    print("Output test: ", output_test)
    print("Output ref: ", output_ref)

    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    assert error < _TOLERANCE[precision]


# Supports: Thread APIs, R2C/C2R/C2C, all precision, real_fft_options except folder
@pytest.mark.parametrize(
    "fft_type,size,precision,direction,real_fft_options",
    [
        ("c2r", 2, np.float16, "inverse", None),
        ("r2c", 4, np.float16, "forward", None),
        ("c2c", 8, np.float16, "forward", None),
        ("c2c", 9, np.float16, "forward", None),
        ("r2c", 16, np.float16, "forward", None),
        ("r2c", 18, np.float16, "forward", {"complex_layout": "natural", "real_mode": "normal"}),
        ("c2r", 32, np.float16, "inverse", {"complex_layout": "packed", "real_mode": "normal"}),
        ("c2r", 33, np.float16, "inverse", None),
        ("c2c", 2, np.float32, "forward", None),
        ("c2c", 3, np.float32, "forward", None),
        ("c2r", 4, np.float32, "inverse", None),
        ("c2r", 6, np.float32, "inverse", {"complex_layout": "packed", "real_mode": "normal"}),
        ("r2c", 8, np.float32, "forward", {"complex_layout": "natural", "real_mode": "normal"}),
        ("c2c", 11, np.float32, "inverse", None),
        ("c2c", 16, np.float32, "inverse", None),
        ("c2c", 15, np.float32, "inverse", None),
        ("c2c", 16, np.float32, "inverse", None),
        ("r2c", 2, np.float64, "forward", {"complex_layout": "natural", "real_mode": "normal"}),
        ("c2c", 4, np.float64, "forward", None),
        ("c2r", 8, np.float64, "inverse", {"complex_layout": "packed", "real_mode": "normal"}),
        ("c2c", 16, np.float64, "forward", None),
        # R2C/C2R folded tests
        ("r2c", 8, np.float64, "forward", {"complex_layout": "natural", "real_mode": "folded"}),
        ("c2r", 8, np.float64, "inverse", {"complex_layout": "packed", "real_mode": "folded"}),
        ("r2c", 16, np.float32, "forward", {"complex_layout": "full", "real_mode": "folded"}),
        ("c2r", 16, np.float32, "inverse", {"complex_layout": "full", "real_mode": "folded"}),
        ("r2c", 32, np.float16, "forward", {"complex_layout": "packed", "real_mode": "folded"}),
        ("c2r", 32, np.float16, "inverse", {"complex_layout": "natural", "real_mode": "folded"}),
    ],
)
def test_thread(fft_type, size, precision, direction, real_fft_options):
    FFT = fft(
        fft_type=fft_type,
        size=size,
        precision=precision,
        direction=direction,
        real_fft_options=real_fft_options,
        execution="Thread",
        compiler="numba",
    )
    show_FFT_traits(FFT)

    complex_type = FFT.value_type
    storage_size = FFT.storage_size
    implicit_type_batching = FFT.implicit_type_batching
    input_type = FFT.input_type
    output_type = FFT.output_type

    assert complex_type == COMPLEX_TYPE_MAP[precision]
    assert all([code.endswith(".ltoir") for code in FFT.files])
    assert FFT.size == size
    assert implicit_type_batching == IMPLICIT_BATCHING_MAP[precision]
    assert not FFT.requires_workspace
    if real_fft_options is None:
        real_fft_options = DEFAULT_REAL_FFT_OPTIONS

    IS_FP16 = precision == np.float16
    IS_INPUT_REAL = fft_type == "r2c"
    IS_OUTPUT_REAL = fft_type == "c2r"

    if real_fft_options["complex_layout"] == "natural":
        complex_size = size // 2 + 1
    elif real_fft_options["complex_layout"] == "packed":
        complex_size = size // 2
    elif real_fft_options["complex_layout"] == "full":
        complex_size = size

    INPUT_SIZE = complex_size if fft_type == "c2r" else size
    OUTPUT_SIZE = complex_size if fft_type == "r2c" else size

    INPUT_FOLDED = real_fft_options["real_mode"] == "folded" and fft_type == "r2c"
    OUTPUT_FOLDED = real_fft_options["real_mode"] == "folded" and fft_type == "c2r"

    make_complex_type = complex_type.make
    if INPUT_FOLDED:
        INPUT_SIZE = INPUT_SIZE // 2
    if OUTPUT_FOLDED:
        OUTPUT_SIZE = OUTPUT_SIZE // 2

    @cuda.jit(link=FFT.files)
    def f(input, output):
        thread_data = cuda.local.array(shape=(storage_size,), dtype=complex_type)
        thread_input = thread_data.view(input_type)
        thread_output = thread_data.view(output_type)

        for i in range(INPUT_SIZE):
            if IS_FP16:
                if IS_INPUT_REAL and not INPUT_FOLDED:
                    r0, r1 = input[0, i], input[1, i]
                    thread_input[i] = float16x2(r0, r1)
                else:
                    # R0 R1 I0 I1
                    r0, r1 = input[0, 2 * i + 0], input[1, 2 * i + 0]
                    i0, i1 = input[0, 2 * i + 1], input[1, 2 * i + 1]
                    thread_input[i] = float16x4(r0, r1, i0, i1)
            else:
                if INPUT_FOLDED:
                    thread_input[i] = make_complex_type(input[0, 2 * i + 0], input[0, 2 * i + 1])
                else:
                    thread_input[i] = input[0, i]

        # Execute FFT
        FFT(thread_data)

        # Save results
        for i in range(OUTPUT_SIZE):
            if IS_FP16:
                if IS_OUTPUT_REAL and not OUTPUT_FOLDED:
                    output[0, i] = thread_output[i].x
                    output[1, i] = thread_output[i].y
                else:
                    # R0 R1 I0 I1
                    output[0, 2 * i + 0], output[1, 2 * i + 0] = thread_output[i].x, thread_output[i].y
                    output[0, 2 * i + 1], output[1, 2 * i + 1] = thread_output[i].z, thread_output[i].w
            else:
                if OUTPUT_FOLDED:
                    output[0, 2 * i + 0], output[0, 2 * i + 1] = thread_output[i].x, thread_output[i].y
                else:
                    output[0, i] = thread_output[i]

    input_h, output_ref, input_d, output_d = make_numba_input_outputs(
        fft_type, precision, direction, implicit_type_batching, size, real_fft_options
    )

    f[1, 1](input_d, output_d)
    cuda.synchronize()

    output_test = convert_output(fft_type, precision, output_d)

    print("Input: ", input_h)
    print("Output test: ", output_test)
    print("Output ref: ", output_ref)

    error = np.linalg.norm(output_test - output_ref) / np.linalg.norm(output_ref)
    assert error < _TOLERANCE[precision]


def test_valid():
    base_FFT = FFTOptions(
        fft_type="c2c",
        size=2,
        precision=np.float32,
        direction="forward",
        execution="Block",
        code_type=current_device_lto(),
    )

    count = 0
    for ept, fpb in base_FFT.valid("elements_per_thread", "ffts_per_block"):
        FFT0 = base_FFT.create(elements_per_thread=ept, ffts_per_block=fpb, compiler="numba")
        assert isinstance(FFT0, FFTNumba)
        FFT1 = base_FFT.create(elements_per_thread=ept, ffts_per_block=fpb)
        assert isinstance(FFT1, FFTCompiled)
        count += 1

    assert count > 0
