# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


try:
    import torch
except ImportError:
    torch = None
import pytest
from .utils import sample_matrix, allow_cublas_unsupported, matmul_with_random_autotune
from .fp8_utils import assert_fp8_equal, fp8_matmul_reference, simple_scales, generate_inputs, choose_scales
from nvmath.linalg.advanced import Matmul, MatmulEpilog as Epilog
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.bindings import cublasLt as cublaslt
from .test_fp8 import SUPPORTED_TYPE_COMBINATIONS, expected_result_type
from contextlib import nullcontext

if torch is None:
    pytest.skip("Torch is required for FP8 tests", allow_module_level=True)

COMPUTE_CAPABILITY = (torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor)

if COMPUTE_CAPABILITY < (8, 9):
    pytest.skip("CC>=8.9 is required for FP8 tests", allow_module_level=True)

if cublaslt.get_version() < 120800:
    pytest.skip("cuBLAS 120800 is required for FP8 tests", allow_module_level=True)


def unpack_bitmask(bitmask, shape):
    if len(bitmask.shape) > 2:
        return torch.stack([unpack_bitmask(bitmask[i], shape) for i in range(bitmask.shape[0])])
    result = torch.zeros(shape)
    n, m = shape
    for i in range(n):
        for j in range(m):
            result[i][j] = bool(int(bitmask[i // 8][j].item()) & (1 << i % 8))
    return result


@pytest.mark.parametrize(
    "m,n,k",
    (
        (16, 16, 16),
        (80, 96, 16),
    ),
)
@pytest.mark.parametrize(
    "a_batch,b_batch,c_batch,d_batch",
    (
        ((), (), (), ()),
        ((2, 3), (2, 3), (2, 3), (2, 3)),
    ),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "epilog_name,order,epilog_aux_type,epilog_aux_amax",
    (
        ("BIAS", "col", None, False),
        ("RELU", "col", None, False),
        ("RELU", "row", None, False),
        ("RELU_AUX", "col", None, False),
        ("RELU_BIAS", "col", None, False),
        ("RELU_AUX_BIAS", "col", None, False),
        ("GELU", "col", None, False),
        ("GELU_AUX", "col", None, False),
        ("GELU_BIAS", "col", None, False),
        ("GELU_AUX_BIAS", "col", None, False),
        ("BGRADA", "col", None, False),
        ("BGRADB", "col", None, False),
        ("DRELU", "col", None, False),
        ("DRELU_BGRAD", "col", None, False),
        ("DGELU", "col", None, False),
        ("DGELU_BGRAD", "col", None, False),
        ("GELU_AUX", "col", "float8_e4m3fn", True),
        ("GELU_AUX", "col", "float8_e4m3fn", False),
    ),
)
def test_epilogs(
    m,
    n,
    k,
    atype,
    btype,
    ctype,
    dtype,
    epilog_name,
    order,
    a_batch,
    b_batch,
    c_batch,
    d_batch,
    epilog_aux_type,
    epilog_aux_amax,
    use_cuda,
):
    epilog = getattr(Epilog, epilog_name)

    result_type = expected_result_type(atype, btype, ctype, dtype)
    inferred_ctype = ctype or ("float16" if "float8" in result_type else result_type)

    # Handle gaps in cuBLAS support
    allow_not_supported = False
    allow_not_supported |= (inferred_ctype, result_type) == ("float32", "float32") and "GELU" in epilog_name
    allow_not_supported |= epilog_name in ("DGELU", "GELU_AUX") and inferred_ctype == "float16" and result_type == "float16"
    allow_not_supported |= epilog_name in ("DRELU", "DGELU") and "float16" in inferred_ctype and "float8" in result_type
    allow_not_supported |= "BGRAD" in epilog_name
    allow_not_supported |= epilog_name.startswith("RELU_AUX") and atype != btype
    allow_not_supported |= epilog_name == "RELU_AUX_BIAS" and inferred_ctype == "float32"
    allow_not_supported |= epilog_name in ("GELU", "GELU_BIAS") and "float8" in result_type
    allow_not_supported |= epilog_aux_type is not None and not (
        atype == "float8_e4m3fn" and btype == "float8_e4m3fn" and ctype == "float16" and epilog_name == "GELU_AUX"
    )
    if COMPUTE_CAPABILITY <= (8, 9):
        allow_not_supported |= "AUX" in epilog_name and "float8" in result_type and d_batch != ()
        allow_not_supported |= epilog_name.startswith("GELU_AUX") and atype != btype

    def sample_batch(batch_shape, matrix_shape, type, transposed=False):
        shape = (*batch_shape, *matrix_shape)
        if transposed:
            shape = (*shape[:-2], shape[-1], shape[-2])
        x = sample_matrix("torch", type, shape, use_cuda=use_cuda, min=-0.2, max=1)
        return x.swapaxes(-1, -2) if transposed else x

    a = sample_batch(a_batch, (m, k), atype, transposed=False)
    b = sample_batch(b_batch, (k, n), btype, transposed=True)
    alpha, beta = 0.8, None
    if ctype is not None:
        c = sample_batch(c_batch, (m, n), ctype, transposed=(order == "col"))
        beta = 0.12
    else:
        c = None
        beta = None

    quantization_scales = simple_scales(atype, btype, ctype, dtype)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    preferences = {
        "epilog": {
            "aux_type": NAME_TO_DATA_TYPE[epilog_aux_type] if epilog_aux_type else None,
            "aux_amax": epilog_aux_amax,
        }
    }

    # Prepare epilog inputs if needed
    inputs = {}
    if epilog_aux_type and "float8" in epilog_aux_type:
        inputs["aux_quantization_scale"] = 0.45
    if "BIAS" in epilog_name:
        bias_type = "float16" if inferred_ctype == "float16" else "bfloat16"
        bias = sample_matrix("torch", bias_type, (m,), use_cuda=use_cuda, min=0, max=1)
        inputs["bias"] = bias
    if "DRELU" in epilog_name:
        round_16 = lambda x: (x + 15) // 16 * 16
        inputs["relu_aux"] = torch.randint(low=0, high=256, size=(n, round_16(m // 16))).type(torch.uint8).T
    if "DGELU" in epilog_name:
        if order == "col":
            inputs["gelu_aux"] = sample_matrix("torch", result_type, (n, m), use_cuda=use_cuda, min=-5, max=5).T
        else:
            inputs["gelu_aux"] = sample_matrix("torch", result_type, (m, n), use_cuda=use_cuda, min=-5, max=5)

    # Run matmul. Allow cuBLAS NOT_SUPPORTED error for certain configurations (see above)
    def unpack_matmul(result):
        return result if isinstance(result, tuple) else (result, {})

    with (
        nullcontext()
        if not allow_not_supported
        else allow_cublas_unsupported(
            message=f"FP8 epilog not supported by cuBLAS: {epilog_name} for A:{atype} B:{btype} C:{ctype} D:{dtype}",
            allow_invalid_value=True,
        )
    ):
        result, aux = unpack_matmul(
            matmul_with_random_autotune(
                a,
                b,
                c,
                alpha=alpha,
                beta=beta,
                epilog=epilog,
                quantization_scales=quantization_scales,
                options=options,
                preferences=preferences,
                epilog_inputs=inputs,
            )
        )

    assert result.shape == (*d_batch, m, n)

    # Compute the reference and compare
    reference, reference_aux = unpack_matmul(
        fp8_matmul_reference(
            a,
            b,
            c,
            alpha=alpha,
            beta=beta,
            epilog=epilog,
            quantization_scales=quantization_scales,
            options=options,
            preferences=preferences,
            epilog_inputs=inputs,
        )
    )
    if "GELU" in epilog_name and result_type not in ("float16", "float32"):
        assert_fp8_equal(result, reference, atol=1e-1, rtol=1e-1)
    else:
        assert_fp8_equal(result, reference)

    # Compare auxiliary outputs
    assert set(aux.keys()) == set(reference_aux.keys())
    for key in aux:
        if key == "relu_aux":
            x = unpack_bitmask(aux[key], (m, n))
            y = unpack_bitmask(reference_aux[key], (m, n))
            assert torch.mean((x == y).type(torch.float32)) > 0.99
        elif key == "gelu_aux":
            assert_fp8_equal(aux[key], reference_aux[key])
            if epilog_aux_type is not None:
                assert str(aux[key].dtype).split(".")[-1] == epilog_aux_type
        elif key == "gelu_aux_amax":
            assert torch.allclose(
                aux["gelu_aux_amax"],
                (aux["gelu_aux"].type(torch.float32) / inputs.get("aux_quantization_scale", 1)).abs().max(),
                atol=1e-1,
                rtol=1e-1,
            )
        elif key == "drelu_bgrad" or key == "dgelu_bgrad":
            assert_fp8_equal(aux[key], reference.sum(axis=-1, keepdims=(d_batch != ())), atol=1e-1, rtol=1e-1)
            if epilog_aux_type is not None:
                assert str(aux[key].dtype).split(".")[-1] == epilog_aux_type
        else:
            raise RuntimeError(f"Test for {key} not implemented")


@pytest.mark.parametrize(
    "m,n,k",
    ((80, 96, 16),),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
@pytest.mark.parametrize("atype,btype,ctype,dtype", (("float8_e4m3fn", "float8_e4m3fn", None, None),))
@pytest.mark.parametrize(
    "epilog_name,epilog_aux_type",
    (("GELU_AUX", "float8_e4m3fn"),),
)
def test_epilog_aux_scale_reset(
    m,
    n,
    k,
    atype,
    btype,
    ctype,
    dtype,
    epilog_name,
    epilog_aux_type,
    use_cuda,
):
    """
    Test if reset_operands can reset epilog aux scales.
    """
    epilog = getattr(Epilog, epilog_name)

    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {
        "result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None,
    }
    preferences = {
        "epilog": {
            "aux_type": NAME_TO_DATA_TYPE[epilog_aux_type] if epilog_aux_type else None,
        }
    }
    inputs = {"aux_quantization_scale": 10}
    mm = Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    mm.plan(epilog=epilog, epilog_inputs=inputs, preferences=preferences)
    result, aux = mm.execute()
    reference, reference_aux = fp8_matmul_reference(
        a,
        b,
        c,
        alpha=alpha,
        beta=beta,
        quantization_scales=quantization_scales,
        options=options,
        preferences=preferences,
        epilog_inputs=inputs,
        epilog=epilog,
    )
    assert_fp8_equal(result, reference)
    assert_fp8_equal(aux["gelu_aux"], reference_aux["gelu_aux"])

    inputs2 = {"aux_quantization_scale": -0.1}
    mm.reset_operands(a=a, epilog_inputs=inputs2)
    result2, aux2 = mm.execute()
    reference2, reference_aux2 = fp8_matmul_reference(
        a,
        b,
        c,
        alpha=alpha,
        beta=beta,
        quantization_scales=quantization_scales,
        options=options,
        preferences=preferences,
        epilog_inputs=inputs2,
        epilog=epilog,
    )
    assert_fp8_equal(result2, reference2)
    assert_fp8_equal(aux2["gelu_aux"], reference_aux2["gelu_aux"])
