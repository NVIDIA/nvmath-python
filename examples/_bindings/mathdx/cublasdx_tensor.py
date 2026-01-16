# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
smem_a = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.SUGGESTED_SMEM_A)
smem_b = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.SUGGESTED_SMEM_B)
rmem_c = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.SUGGESTED_RMEM_C)

gmem_a = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.GMEM_A)
gmem_b = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.GMEM_B)
gmem_c = mathdx.cublasdx_create_tensor(h, mathdx.CublasdxTensorType.GMEM_C)

gmem_tensors = [gmem_a, gmem_b, gmem_c]
gemm_tensors = [smem_a, smem_b, rmem_c]

rmem_c_fp64 = mathdx.cublasdx_make_tensor_like(rmem_c, mathdx.CommondxValueType.R_64F)
gmem_c_fp64 = mathdx.cublasdx_make_tensor_like(gmem_c, mathdx.CommondxValueType.R_64F)

tensors_fp64 = [rmem_c_fp64, gmem_c_fp64]
tensors = gmem_tensors + gemm_tensors + tensors_fp64
mathdx.cublasdx_finalize_tensors(h, len(tensors), tensors)


for t in tensors:
    uid = mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.UID)
    alignment = mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.ALIGNMENT_BYTES)
    size = mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.STORAGE_BYTES)
    logical_size = (
        mathdx.cublasdx_get_tensor_trait_int64(t, mathdx.CublasdxTensorTrait.LOGICAL_SIZE)
        if mathdx.get_version_ex() >= (0, 2, 4)
        else 0
    )
    name_size = mathdx.cublasdx_get_tensor_trait_str_size(t, mathdx.CublasdxTensorTrait.OPAQUE_NAME)
    name = bytearray(name_size)
    mathdx.cublasdx_get_tensor_trait_str(t, mathdx.CublasdxTensorTrait.OPAQUE_NAME, len(name), name)
    name = name[:-1].decode()

    print(f"Tensor {t}: name {name}, size {logical_size}, storage size {size}B, alignment {alignment}B, uid {uid}")

# Define a function operating on those input and output tensors
gemm_sa_sb_rc = mathdx.cublasdx_create_device_function(
    h, mathdx.CublasdxDeviceFunctionType.EXECUTE, len(gemm_tensors), gemm_tensors
)

copy_c = mathdx.cublasdx_create_device_function(h, mathdx.CublasdxDeviceFunctionType.COPY, 2, [gmem_c, rmem_c])

device_functions = [gemm_sa_sb_rc, copy_c]

copy_c_back_fp64 = mathdx.cublasdx_create_device_function(
    h, mathdx.CublasdxDeviceFunctionType.COPY, 2, [rmem_c_fp64, gmem_c_fp64]
)
device_functions += [copy_c_back_fp64]

for f in device_functions:
    mangled_name_size = mathdx.cublasdx_get_device_function_trait_str_size(f, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)
    mangled_name = bytearray(mangled_name_size)
    mangled_name_size = mathdx.cublasdx_get_device_function_trait_str(
        f, mathdx.CublasdxDeviceFunctionTrait.SYMBOL, len(mangled_name), mangled_name
    )
    mangled_name = mangled_name[:-1].decode()

    print(f"Device function {f}: mangled name {mangled_name}\n")

# Compile the device function to lto_90
code = mathdx.commondx_create_code()
mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, arch)
mathdx.cublasdx_finalize_device_functions(code, len(device_functions), device_functions)

# Extract the LTOIR
lto_size = mathdx.commondx_get_code_ltoir_size(code)
lto = bytearray(lto_size)
mathdx.commondx_get_code_ltoir(code, lto_size, lto)
mathdx.commondx_destroy_code(code)

print(f"Generated LTOIR for gemm device function, {lto_size} bytes at ..")

for i in range(len(gemm_tensors)):
    tensor = gemm_tensors[i]
    gmem_tensor = gmem_tensors[i]
    copy = mathdx.cublasdx_create_device_function(
        h,
        mathdx.CublasdxDeviceFunctionType.COPY,
        2,
        [gmem_tensor, tensor],
    )
    copy_name_size = mathdx.cublasdx_get_device_function_trait_str_size(copy, mathdx.CublasdxDeviceFunctionTrait.SYMBOL)
    copy_name = bytearray(copy_name_size)
    copy_name_size = mathdx.cublasdx_get_device_function_trait_str(
        copy, mathdx.CublasdxDeviceFunctionTrait.SYMBOL, len(copy_name), copy_name
    )
    copy_name = copy_name[:-1].decode()

    print(f"Device function {copy}: mangled name {copy_name}")

    # Compile the device function to lto_90
    code = mathdx.commondx_create_code()
    mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, arch)
    mathdx.cublasdx_finalize_device_functions(code, 1, [copy])

    # Extract the LTOIR
    lto_size = mathdx.commondx_get_code_ltoir_size(code)
    lto = bytearray(lto_size)
    mathdx.commondx_get_code_ltoir(code, lto_size, lto)

    mathdx.commondx_destroy_code(code)
    mathdx.cublasdx_destroy_device_function(copy)

    print(f"Generated LTOIR for copy device function, {lto_size} bytes at ..")

for t in tensors:
    mathdx.cublasdx_destroy_tensor(t)

for f in device_functions:
    mathdx.cublasdx_destroy_device_function(f)

# TODO: destroy update in original example (cpp)
mathdx.cublasdx_destroy_descriptor(h)
