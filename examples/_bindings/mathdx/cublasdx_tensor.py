# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import mathdx

m = 256
n = 128
k = 16
num_threads = 128
arch = 900

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
        mathdx.CommondxPrecision.F32,
        mathdx.CommondxPrecision.F32,
        mathdx.CommondxPrecision.F32,
    ],
)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.SM, arch)
mathdx.cublasdx_set_operator_int64(h, mathdx.CublasdxOperatorType.TYPE, mathdx.CublasdxType.REAL)
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.BLOCK_DIM, 3, [num_threads, 1, 1])
mathdx.cublasdx_set_operator_int64s(h, mathdx.CublasdxOperatorType.SIZE, 3, [m, n, k])

mathdx.cublasdx_set_operator_int64s(
    h,
    mathdx.CublasdxOperatorType.ARRANGEMENT,
    3,
    [
        mathdx.CublasdxArrangement.COL_MAJOR,
        mathdx.CublasdxArrangement.COL_MAJOR,
        mathdx.CublasdxArrangement.COL_MAJOR,
    ],
)

mathdx.cublasdx_set_option_str(h, mathdx.CommondxOption.SYMBOL_NAME, "matmul")

# Define the input and output tensors
smem_a = mathdx.cublasdx_bind_tensor(h, mathdx.CublasdxTensorType.SUGGESTED_SMEM_A)
smem_b = mathdx.cublasdx_bind_tensor(h, mathdx.CublasdxTensorType.SUGGESTED_SMEM_B)
rmem_c = mathdx.cublasdx_bind_tensor(h, mathdx.CublasdxTensorType.SUGGESTED_RMEM_C)

tensors = [smem_a, smem_b, rmem_c]
mathdx.cublasdx_finalize_tensors(h, len(tensors), tensors)

for t in tensors:
    uid = mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.UID)
    alignment = mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.ALIGNMENT_BYTES)
    size = mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.STORAGE_BYTES)
    name_size = mathdx.cublasdx_get_tensor_trait_str_size(t, mathdx.CublasdxTensorTrait.OPAQUE_NAME)
    name = bytearray(name_size)
    mathdx.cublasdx_get_tensor_trait_str(t, mathdx.CublasdxTensorTrait.OPAQUE_NAME, len(name), name)
    name = name[:-1].decode()

    print(f"Tensor {t}: name {name}, storage size {size}B, alignment {alignment}B, uid {uid}")

# Define a function operating on those input and output tensors
gemm_sa_sb_rc = mathdx.cublasdx_bind_device_function(h, mathdx.CublasdxDeviceFunctionType.EXECUTE, len(tensors), tensors)
name_size = mathdx.cublasdx_get_device_function_trait_str_size(gemm_sa_sb_rc, mathdx.CublasdxDeviceFunctionTrait.NAME)
mangled_name_size = mathdx.cublasdx_get_device_function_trait_str_size(gemm_sa_sb_rc, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)
name = bytearray(name_size)
mangled_name = bytearray(mangled_name_size)
name_size = mathdx.cublasdx_get_device_function_trait_str(
    gemm_sa_sb_rc, mathdx.CublasdxDeviceFunctionTrait.NAME, len(name), name
)
mangled_name_size = mathdx.cublasdx_get_device_function_trait_str(
    gemm_sa_sb_rc, mathdx.CublasdxDeviceFunctionTrait.SYMBOL, len(mangled_name), mangled_name
)
name = name[:-1].decode()
mangled_name = mangled_name[:-1].decode()

print(f"Device function {gemm_sa_sb_rc}: name: {name}, mangled name {mangled_name}\n")

# Compile the device function to lto_90
code = mathdx.commondx_create_code()
mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, arch)
mathdx.cublasdx_finalize_device_functions(code, 1, [gemm_sa_sb_rc])

# Extract the LTOIR
lto_size = mathdx.commondx_get_code_ltoir_size(code)
lto = bytearray(lto_size)
mathdx.commondx_get_code_ltoir(code, lto_size, lto)


print(f"Generated LTOIR for gemm device function, {lto_size} bytes at ..")

mathdx.commondx_destroy_code(code)
# TODO: destroy update in original example (cpp)
mathdx.cublasdx_destroy_descriptor(h)
