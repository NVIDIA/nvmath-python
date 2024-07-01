# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import cuda, types

from nvmath.device import float16x4, float16x2, float32x2, float64x2, float64x2_type, float32x2_type, float16x2_type, float16x4_type
import pytest

# Tests we can use the types to build arrays and read to/from global arrays
# built with Numba basic complex dtypes
@pytest.mark.parametrize("numpy_type,numba_type,numba_fe_type", [
    (np.complex128, float64x2_type, float64x2),
    (np.complex128, float32x2_type, float32x2),
    (np.complex64,  float64x2_type, float64x2),
    (np.complex64,  float32x2_type, float32x2),
])
def test_complex_numpy_numba_interop(numpy_type, numba_type, numba_fe_type):

    print(f"Test for {numpy_type} <-> {numba_type}")

    @cuda.jit
    def f(input, output1, output2):

        lmem = cuda.local.array(shape=(4,), dtype=numba_type)
        smem = cuda.shared.array(shape=(4,), dtype=numba_type)

        for i in range(4):
            lmem[i] = input[i]
            smem[i] = lmem[i]
            output1[i] = smem[i]
            output2[i] = numba_fe_type(smem[i].x, smem[i].y)


    input = np.array([3.14+2.71j, -3.71-2.71j, 3.71-9.84j, -45.58-987j], dtype=numpy_type)
    output1 = np.zeros_like(input)
    output2 = np.zeros_like(input)

    f[1,1](input, output1, output2)
    cuda.synchronize()

    assert np.allclose(input, output1)
    assert np.allclose(input, output2)

# Test we can build types from numpy dtypes
@pytest.mark.parametrize("size,numpy_type,numba_type,numba_fe_type,numba_basic_type", [
    (2, np.dtype([('x', np.float16), ('y', np.float16)]),                                       float16x2_type, float16x2, np.float16),
    (4, np.dtype([('x', np.float16), ('y', np.float16), ('z', np.float16), ('w', np.float16)]), float16x4_type, float16x4, np.float16),
    (2, np.dtype([('x', np.float32), ('y', np.float32)]),                                       float32x2_type, float32x2, np.float32),
    (2, np.dtype([('x', np.float64), ('y', np.float64)]),                                       float64x2_type, float64x2, np.float64),
])
def test_dtypes_numpy_numba_interop(size, numpy_type, numba_type, numba_fe_type, numba_basic_type):

    print(f"Test for {numpy_type} <-> {numba_type}")

    @cuda.jit
    def f(input, output):
        for i in range(4):
            v = input[i]
            if size == 2:
                w = numba_fe_type(v.x, v.y)
                output[i].x = w.x
                output[i].y = w.y
            elif size == 4:
                w = numba_fe_type(v.x, v.y, v.z, v.w)
                output[i].x = w.x
                output[i].y = w.y
                output[i].z = w.z
                output[i].w = w.w


    if size == 2:
        input = np.array([ (3.14, 2.71           ), (2.71, 42          ), (1.0, -100              ), (-1.0, 1.0      ) ], dtype=numpy_type)
    elif size == 4:
        input = np.array([ (3.14, 2.71, -0.1, 0.1), (2.71, 42, 3.0, 4.0), (1.0, -100, 1000, -10000), (-1.0, 1.0, 0, 0) ], dtype=numpy_type)
    output = np.zeros_like(input)

    f[1,1](input, output)
    cuda.synchronize()

    assert np.allclose(input.view(dtype=numba_basic_type), output.view(dtype=numba_basic_type))

@pytest.mark.parametrize("numba_type,count,bitwidth,numba_basic_type", [
    (float16x2_type, 2, 2 * 16, types.float16),
    (float16x4_type, 4, 4 * 16, types.float16),
    (float32x2_type, 2, 2 * 32, types.float32),
    (float64x2_type, 2, 2 * 64, types.float64),
])
def test_vector_types(numba_type, count, bitwidth, numba_basic_type):
    print(f"Test for {numba_type}")
    assert numba_type.bitwidth == bitwidth
    assert numba_type.count == count
    assert numba_type.dtype == numba_basic_type

def test_views():

    @cuda.jit
    def f(input, output):

        v0 = input.view(np.float16)
        v1 = cuda.local.array(shape=(4 * 7,), dtype=np.float16)
        v2 = v1.view(float16x2_type)
        v3 = v2.view(float16x4_type)
        v4 = output.view(float16x4_type)

        for j in range(7):

            for i in range(4):
                v1[4 * j + i] = v0[4 * j + i]

            for i in range(2):
                v2[2 * j + i] = float16x2(v1[4 * j + 2 * i], v1[4 * j + 2 * i+1])

            v3[j] = float16x4(v2[2 * j].x, v2[2 * j].y, v2[2 * j + 1].x, v2[2 * j + 1].y)
            v4[j] = v3[j]

    input = np.linspace(0.0, 3.14, 4 * 7, dtype=np.float16)
    output = np.zeros_like(input)
    f[1,1](input, output)
    cuda.synchronize()

    assert np.allclose(input, output)

def test_views_vector_load_store():

    input = np.linspace(0.0, 3.14, 4, dtype=np.float16)
    output = np.zeros_like(input)

    # Ensure that we get the right vectorized version with float16x4 types
    @cuda.jit
    def f_vectorized(input, output):
        input4 = input.view(float16x4_type)
        output4 = output.view(float16x4_type)
        output4[0] = input4[0]

    f_vectorized[1,1](input, output)
    cuda.synchronize()
    assert np.allclose(input, output)

    ptx = [v for k,v in f_vectorized.inspect_asm().items()]
    assert len(ptx) == 1
    ptx = ptx[0]

    assert 'ld.global.u64' in ptx
    assert 'st.global.u64' in ptx

    assert 'ld.global.u16' not in ptx
    assert 'ld.global.u16' not in ptx

    # Ensure that we *don't* get the right vectorized version without
    @cuda.jit
    def f_non_vectorized(input, output):
        for i in range(4):
            output[i] = input[i]

    f_non_vectorized[1,1](input, output)
    cuda.synchronize()
    assert np.allclose(input, output)

    ptx = [v for k,v in f_non_vectorized.inspect_asm().items()]
    assert len(ptx) == 1
    ptx = ptx[0]

    assert 'ld.global.u64' not in ptx
    assert 'st.global.u64' not in ptx

    assert 'ld.global.u16' in ptx
    assert 'ld.global.u16' in ptx
