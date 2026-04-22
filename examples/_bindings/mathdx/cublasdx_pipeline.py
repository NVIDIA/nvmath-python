# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from nvmath.bindings import mathdx

m = 256
n = 128
k = 16
num_threads = 128
arch = 900
sm = [arch, mathdx.CommondxArchModifier.ARCH_SPECIFIC]

# Create the cuBLASDx descriptor
h = mathdx.cublasdx_create_descriptor()
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.FUNCTION, mathdx.CublasdxFunction.MM)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.EXECUTION, mathdx.CommondxExecution.BLOCK)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.API, mathdx.CublasdxApi.TENSORS)
mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.PRECISION,
    3,
    [
        mathdx.CommondxPrecision.I8,
        mathdx.CommondxPrecision.I8,
        mathdx.CommondxPrecision.I32,
    ],
)
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SM, len(sm), sm)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.TYPE, mathdx.CublasdxType.REAL)
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.BLOCK_DIM, 3, [num_threads, 1, 1])
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SIZE, 3, [m, n, k])

mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.ARRANGEMENT,
    3,
    [
        mathdx.CublasdxArrangement.ROW_MAJOR,
        mathdx.CublasdxArrangement.COL_MAJOR,
        mathdx.CublasdxArrangement.ROW_MAJOR,
    ],
)

mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.WITH_PIPELINE, 1)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.ENABLE_INPUT_STREAMING, 1)

mathdx.cublasdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "matmul")

shape_a = [m, k]
shape_b = [k, n]
shape_c = [m, n]
strides_a = [k, 1]
strides_b = [1, n]
strides_c = [n, 1]
big_gmem_a = mathdx.cublasdx_create_tensor_strided(
    mathdx.CublasdxMemorySpace.GMEM, mathdx.CommondxValueType.R_8I, 0, len(shape_a), shape_a, strides_a
)
big_gmem_b = mathdx.cublasdx_create_tensor_strided(
    mathdx.CublasdxMemorySpace.GMEM, mathdx.CommondxValueType.R_8I, 0, len(shape_b), shape_b, strides_b
)
big_gmem_c = mathdx.cublasdx_create_tensor_strided(
    mathdx.CublasdxMemorySpace.GMEM, mathdx.CommondxValueType.R_8I, 0, len(shape_c), shape_c, strides_c
)

acc_c = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.ACCUMULATOR_C)

device_pipeline = mathdx.cublasdx_create_device_pipeline(
    h, mathdx.CublasdxDevicePipelineType.SUGGESTED, 1, mathdx.CublasdxBlockSizeStrategy.FIXED, big_gmem_a, big_gmem_b
)
tile_pipeline = mathdx.cublasdx_create_tile_pipeline(h, mathdx.CublasdxTilePipelineType.PIPELINE_DEFAULT, device_pipeline)

tensors = [big_gmem_a, big_gmem_b, big_gmem_c, acc_c]
pipelines = [device_pipeline, tile_pipeline]

mathdx.cublasdx_finalize_pipelines(len(pipelines), pipelines)
mathdx.cublasdx_finalize_tensors(len(tensors), tensors)

acc_size = mathdx.cublasdx_get_tensor_trait_int64(acc_c, mathdx.CublasdxTensorTrait.STORAGE_BYTES)
acc_alignment = mathdx.cublasdx_get_tensor_trait_int64(acc_c, mathdx.CublasdxTensorTrait.ALIGNMENT_BYTES)
print(f"Accumulator tensor {acc_c}: storage size {acc_size}, alignment {acc_alignment}")

device_pipeline_storage_size = mathdx.cublasdx_get_pipeline_trait_int64(
    device_pipeline, mathdx.CublasdxPipelineTrait.STORAGE_BYTES
)
device_pipeline_storage_alignment = mathdx.cublasdx_get_pipeline_trait_int64(
    device_pipeline, mathdx.CublasdxPipelineTrait.STORAGE_ALIGNMENT_BYTES
)
print(
    f"Device pipeline {device_pipeline}: storage size {device_pipeline_storage_size}, "
    f"alignment {device_pipeline_storage_alignment}"
)

shared_memory_buffer_size = mathdx.cublasdx_get_pipeline_trait_int64(device_pipeline, mathdx.CublasdxPipelineTrait.BUFFER_SIZE)
shared_memory_buffer_alignment = mathdx.cublasdx_get_pipeline_trait_int64(
    device_pipeline, mathdx.CublasdxPipelineTrait.BUFFER_ALIGNMENT_BYTES
)
print(
    f"Tile pipeline {tile_pipeline}: shared memory buffer size {shared_memory_buffer_size}, "
    f"alignment {shared_memory_buffer_alignment}"
)

device_pipeline_block_dim = np.zeros(3, dtype=np.int64)
mathdx.cublasdx_get_pipeline_trait_int64s(
    device_pipeline, mathdx.CublasdxPipelineTrait.BLOCK_DIM, 3, device_pipeline_block_dim.ctypes.data
)
print(f"Device pipeline {device_pipeline}: block dim {device_pipeline_block_dim.tolist()}")

tile_pipeline_storage_size = mathdx.cublasdx_get_pipeline_trait_int64(tile_pipeline, mathdx.CublasdxPipelineTrait.STORAGE_BYTES)
tile_pipeline_storage_alignment = mathdx.cublasdx_get_pipeline_trait_int64(
    tile_pipeline, mathdx.CublasdxPipelineTrait.STORAGE_ALIGNMENT_BYTES
)
print(f"Tile pipeline {tile_pipeline}: storage size {tile_pipeline_storage_size}, alignment {tile_pipeline_storage_alignment}")

init_acc = mathdx.cublasdx_create_device_function(h, mathdx.CublasdxDeviceFunctionType.CREATE, 1, [acc_c])
init_device_pipeline = mathdx.cublasdx_create_device_function_with_pipelines(
    h, mathdx.CublasdxDeviceFunctionType.CREATE, 0, 0, 1, [device_pipeline]
)
init_tile_pipeline = mathdx.cublasdx_create_device_function_with_pipelines(
    h, mathdx.CublasdxDeviceFunctionType.CREATE, 0, 0, 1, [tile_pipeline]
)
destroy_acc = mathdx.cublasdx_create_device_function(h, mathdx.CublasdxDeviceFunctionType.DESTROY, 1, [acc_c])
destroy_device_pipeline = mathdx.cublasdx_create_device_function_with_pipelines(
    h, mathdx.CublasdxDeviceFunctionType.DESTROY, 0, 0, 1, [device_pipeline]
)
destroy_tile_pipeline = mathdx.cublasdx_create_device_function_with_pipelines(
    h, mathdx.CublasdxDeviceFunctionType.DESTROY, 0, 0, 1, [tile_pipeline]
)
execute_pipeline = mathdx.cublasdx_create_device_function_with_pipelines(
    h, mathdx.CublasdxDeviceFunctionType.EXECUTE, 1, [acc_c], 1, [tile_pipeline]
)
copy_acc_big_gmem_c = mathdx.cublasdx_create_device_function(h, mathdx.CublasdxDeviceFunctionType.COPY, 2, [acc_c, big_gmem_c])

functions = [
    execute_pipeline,
    init_acc,
    init_device_pipeline,
    init_tile_pipeline,
    destroy_acc,
    destroy_device_pipeline,
    destroy_tile_pipeline,
    copy_acc_big_gmem_c,
]

for f in functions:
    symbol_size = mathdx.cublasdx_get_device_function_trait_str_size(f, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)
    symbol = bytearray(symbol_size)
    symbol_size = mathdx.cublasdx_get_device_function_trait_str(
        f, mathdx.CublasdxDeviceFunctionTrait.SYMBOL, len(symbol), symbol
    )
    symbol = symbol[:-1].decode()

    print(f"Device function {f}: symbol {symbol}")

# Compile the device function to lto_90
code = mathdx.commondx_create_code()
mathdx.commondx_set_code_option_int64s(code, mathdx.CommondxOption.TARGET_SM, len(sm), sm)
mathdx.commondx_set_code_option_str(code, mathdx.CommondxOption.EXTRA_NVRTC_ARGS, "-gen-opt-lto")
mathdx.cublasdx_finalize_device_functions(code, len(functions), functions)

# Extract the LTOIR
lto_size = mathdx.commondx_get_code_ltoir_size(code)
lto = bytearray(lto_size)
mathdx.commondx_get_code_ltoir(code, lto_size, lto)


print(f"Generated LTOIR for gemm device function, {lto_size} bytes at ..")

for t in tensors:
    mathdx.cublasdx_destroy_tensor(t)

for p in pipelines:
    mathdx.cublasdx_destroy_pipeline(p)

for f in functions:
    mathdx.cublasdx_destroy_device_function(f)

mathdx.commondx_destroy_code(code)
mathdx.cublasdx_destroy_descriptor(h)
