# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from nvmath.bindings import mathdx

try:
    from cuda.core import (
        Linker,
        LinkerOptions,
        ObjectCode,
        Program,
        ProgramOptions,
    )
except ImportError:
    from cuda.core.experimental import (
        Linker,
        LinkerOptions,
        ObjectCode,
        Program,
        ProgramOptions,
    )

m = 128
n = 128
k = 32
num_threads = 128
arch = 1200
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
        mathdx.CommondxPrecision.F16,
        mathdx.CommondxPrecision.F16,
        mathdx.CommondxPrecision.F32,
    ],
)
# WAR for the TMA descriptor issue on SM 12.0
if mathdx.get_version_ex() < (0, 3, 2):
    mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.SM, 890)
else:
    mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SM, 1, sm)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.TYPE, mathdx.CublasdxType.REAL)
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.BLOCK_DIM, 3, [num_threads, 1, 1])
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SIZE, 3, [m, n, k])

mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.ALIGNMENT, 3, [16, 16, 16])
mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.ARRANGEMENT,
    3,
    [
        mathdx.CublasdxArrangement.COL_MAJOR,
        mathdx.CublasdxArrangement.ROW_MAJOR,
        mathdx.CublasdxArrangement.ROW_MAJOR,
    ],
)

mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.WITH_PIPELINE, 1)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.ENABLE_INPUT_STREAMING, 1)

mathdx.cublasdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "matmul")

shape_a = [m, k, 2]
shape_b = [k, n, 2]
strides_a = [1, m, m * k]
strides_b = [k, 1, n * k]
big_gmem_a = mathdx.cublasdx_create_tensor_strided(
    mathdx.CublasdxMemorySpace.GMEM, mathdx.CommondxValueType.R_16F, 0, len(shape_a), shape_a, strides_a
)
big_gmem_b = mathdx.cublasdx_create_tensor_strided(
    mathdx.CublasdxMemorySpace.GMEM, mathdx.CommondxValueType.R_16F, 0, len(shape_b), shape_b, strides_b
)

device_pipeline = mathdx.cublasdx_create_device_pipeline(
    h, mathdx.CublasdxDevicePipelineType.SUGGESTED, 2, mathdx.CublasdxBlockSizeStrategy.FIXED, big_gmem_a, big_gmem_b
)

tensors = [big_gmem_a, big_gmem_b]
pipelines = [device_pipeline]

mathdx.cublasdx_finalize_pipelines(len(pipelines), pipelines)
mathdx.cublasdx_finalize_tensors(len(tensors), tensors)

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

device_pipeline_block_dim = np.zeros(3, dtype=np.int64)
mathdx.cublasdx_get_pipeline_trait_int64s(
    device_pipeline, mathdx.CublasdxPipelineTrait.BLOCK_DIM, 3, device_pipeline_block_dim.ctypes.data
)
print(f"Device pipeline {device_pipeline}: block dim {device_pipeline_block_dim.tolist()}")

init_device_pipeline = mathdx.cublasdx_create_device_function_with_pipelines(
    h, mathdx.CublasdxDeviceFunctionType.CREATE, 0, 0, 1, [device_pipeline]
)

functions = [
    init_device_pipeline,
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

init_pipeline_func = ObjectCode.from_ltoir(bytes(lto))

# CUDA C source code for our kernel
init_pipeline_source = (
    f"#define create_dev_pipe {symbol}\n"
    + """
struct libmathdx_tensor_0s_0s { void* ptr; };
struct libmathdx_pipeline { void* ptr; };

extern "C" __device__ void create_dev_pipe(libmathdx_pipeline, libmathdx_tensor_0s_0s, libmathdx_tensor_0s_0s);

extern "C" __global__ void create_device_pipeline(void* device_pipeline_ptr, void* ga_storage, void* gb_storage) {
    auto ga = libmathdx_tensor_0s_0s { ga_storage };
    auto gb = libmathdx_tensor_0s_0s { gb_storage };
    auto device_pipeline = libmathdx_pipeline { device_pipeline_ptr };
    create_dev_pipe(device_pipeline, ga, gb);
}
"""
)


# Compiler arguments
cc_arch = f"sm_{arch // 10}{'a' if len(sm) > 1 and sm[1] == mathdx.CommondxArchModifier.ARCH_SPECIFIC else ''}"
print("Arch:", cc_arch)
program_options = ProgramOptions(link_time_optimization=True, arch=cc_arch)
linker_options = LinkerOptions(link_time_optimization=True, arch=cc_arch)

# Compile the CUDA code into a program
program = Program(init_pipeline_source, code_type="c++", options=program_options)
compiled_program = program.compile(target_type="ltoir")

with open(f"device_func.{init_pipeline_func.code_type}", "wb") as f:
    f.write(init_pipeline_func.code)

linker = Linker(compiled_program, init_pipeline_func, options=linker_options)
linked_program = linker.link("cubin")
