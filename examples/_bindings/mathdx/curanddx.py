# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from nvmath.bindings import mathdx

dx_sm = 800
target_sm = 800

h = mathdx.curanddx_create_descriptor()

mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.GENERATOR, mathdx.CuranddxGenerator.PCG)
mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.EXECUTION, mathdx.CommondxExecution.THREAD)
mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.DISTRIBUTION, mathdx.CuranddxDistribution.UNIFORM)

if mathdx.get_version_ex() <= (0, 3, 1):
    # curanddxSetOperatorDoubles and CURANDDX_OPERATOR_DISTRIBUTION_PARAMETERS
    # removed in v0.3.2
    mathdx.curanddx_set_operator_doubles(h, mathdx.CuranddxOperatorType.DISTRIBUTION_PARAMETERS, 2, [-2.0, 3.0])

mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.OUTPUT_TYPE, mathdx.CommondxValueType.R_32F)
mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.GENERATE_METHOD, mathdx.CuranddxGenerateMethod.SINGLE)
mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.SM, dx_sm)

device_functions = (
    mathdx.CuranddxDeviceFunctionType.GENERATE
    | mathdx.CuranddxDeviceFunctionType.INIT_STATE
    | mathdx.CuranddxDeviceFunctionType.DESTROY_STATE
)
mathdx.curanddx_set_operator_int64(h, mathdx.CuranddxOperatorType.DEVICE_FUNCTIONS, device_functions)

code = mathdx.commondx_create_code()
mathdx.commondx_set_code_option_int64(code, mathdx.CommondxOption.TARGET_SM, target_sm)
mathdx.curanddx_finalize_code(code, h)

lto_size = mathdx.commondx_get_code_ltoir_size(code)
lto = bytearray(lto_size)
mathdx.commondx_get_code_ltoir(code, lto_size, lto)
mathdx.commondx_destroy_code(code)

generate_name_size = mathdx.curanddx_get_trait_str_size(h, mathdx.CuranddxTraitType.SYMBOL_GENERATE_NAME)
generate_name = bytearray(generate_name_size)
mathdx.curanddx_get_trait_str(h, mathdx.CuranddxTraitType.SYMBOL_GENERATE_NAME, generate_name_size, generate_name)
generate_name = generate_name.rstrip(b"\x00").decode("utf-8")

init_name_size = mathdx.curanddx_get_trait_str_size(h, mathdx.CuranddxTraitType.SYMBOL_INIT_STATE_NAME)
init_name = bytearray(init_name_size)
mathdx.curanddx_get_trait_str(h, mathdx.CuranddxTraitType.SYMBOL_INIT_STATE_NAME, init_name_size, init_name)
init_name = init_name.rstrip(b"\x00").decode("utf-8")

destroy_name_size = mathdx.curanddx_get_trait_str_size(h, mathdx.CuranddxTraitType.SYMBOL_DESTROY_STATE_NAME)
destroy_name = bytearray(destroy_name_size)
mathdx.curanddx_get_trait_str(h, mathdx.CuranddxTraitType.SYMBOL_DESTROY_STATE_NAME, destroy_name_size, destroy_name)
destroy_name = destroy_name.rstrip(b"\x00").decode("utf-8")

state_size = mathdx.curanddx_get_trait_int64(h, mathdx.CuranddxTraitType.STATE_SIZE)
state_align = mathdx.curanddx_get_trait_int64(h, mathdx.CuranddxTraitType.STATE_ALIGNMENT)

print(f"Successfully generated LTOIR, {lto_size} Bytes for curanddx device function {generate_name} and {init_name}")
print(f"  destroy_state function: {destroy_name}")
print(f"  state_size: {state_size}, state_alignment: {state_align}")

mathdx.curanddx_destroy_descriptor(h)
