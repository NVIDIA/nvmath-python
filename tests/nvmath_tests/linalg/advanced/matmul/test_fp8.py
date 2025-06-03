# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


try:
    import torch
except ImportError:
    torch = None
import pytest
from .utils import sample_matrix, assert_tensors_equal, matmul_with_random_autotune
from .fp8_utils import choose_scales, generate_inputs, assert_fp8_equal, fp8_matmul_reference
from nvmath.linalg.advanced import Matmul, matmul, MatmulQuantizationScales
from nvmath.internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.bindings import cublasLt as cublaslt

if torch is None:
    pytest.skip("Torch is required for FP8 tests", allow_module_level=True)

if (torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor) < (8, 9):
    pytest.skip("CC>=8.9 is required for FP8 tests", allow_module_level=True)

if cublaslt.get_version() < 120800:
    pytest.skip("cuBLAS 120800 is required for FP8 tests", allow_module_level=True)

SUPPORTED_TYPE_COMBINATIONS = [
    # No type specification for D.
    ("float8_e4m3fn", "float8_e4m3fn", "float16", None),
    ("float8_e4m3fn", "float8_e4m3fn", "bfloat16", None),
    ("float8_e4m3fn", "float8_e4m3fn", "float32", None),
    ("float8_e4m3fn", "float8_e5m2", "float32", None),
    ("float8_e4m3fn", "float8_e5m2", "float16", None),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", None),
    # No type specification for D. No C.
    ("float8_e4m3fn", "float8_e4m3fn", None, None),
    ("float8_e4m3fn", "float8_e5m2", None, None),
    ("float8_e5m2", "float8_e4m3fn", None, None),
    # Explicit type specification for A, B, C, D.
    ("float8_e4m3fn", "float8_e4m3fn", "float16", "float16"),
    ("float8_e4m3fn", "float8_e4m3fn", "float16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e4m3fn", "bfloat16", "bfloat16"),
    ("float8_e4m3fn", "float8_e4m3fn", "bfloat16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e4m3fn", "float32", "float32"),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", "bfloat16"),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", "float8_e5m2"),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e5m2", "float16", "float16"),
    ("float8_e4m3fn", "float8_e5m2", "float16", "float8_e5m2"),
    ("float8_e4m3fn", "float8_e5m2", "float16", "float8_e4m3fn"),
    ("float8_e5m2", "float8_e4m3fn", "bfloat16", "bfloat16"),
    ("float8_e5m2", "float8_e4m3fn", "bfloat16", "float8_e5m2"),
    ("float8_e5m2", "float8_e4m3fn", "bfloat16", "float8_e4m3fn"),
    ("float8_e5m2", "float8_e4m3fn", "float16", "float16"),
    ("float8_e5m2", "float8_e4m3fn", "float16", "float8_e5m2"),
    ("float8_e5m2", "float8_e4m3fn", "float16", "float8_e4m3fn"),
    # Explicit type specification for A, B, D. No C.
    ("float8_e4m3fn", "float8_e4m3fn", None, "float16"),
    ("float8_e4m3fn", "float8_e4m3fn", None, "bfloat16"),
    ("float8_e4m3fn", "float8_e4m3fn", None, "float32"),
    ("float8_e4m3fn", "float8_e5m2", None, "bfloat16"),
    ("float8_e4m3fn", "float8_e5m2", None, "float8_e5m2"),
    ("float8_e4m3fn", "float8_e5m2", None, "float16"),
    ("float8_e5m2", "float8_e4m3fn", None, "bfloat16"),
    ("float8_e5m2", "float8_e4m3fn", None, "float8_e5m2"),
    ("float8_e5m2", "float8_e4m3fn", None, "float16"),
    ("float8_e5m2", "float8_e4m3fn", None, "float8_e4m3fn"),
]

if cublaslt.get_version() < 120600:
    SUPPORTED_TYPE_COMBINATIONS = [(a, b, c, d) for (a, b, c, d) in SUPPORTED_TYPE_COMBINATIONS if a == b]


def expected_result_type(atype, btype, ctype, dtype):
    return dtype or ctype or atype


SUPPORTED_TYPE_COMBINATIONS_WITH_FP8_D = [
    (a, b, c, d) for (a, b, c, d) in SUPPORTED_TYPE_COMBINATIONS if "float8" in expected_result_type(a, b, c, d)
]

SUPPORTED_TYPE_COMBINATIONS_WITH_NON_FP8_D = [
    t for t in SUPPORTED_TYPE_COMBINATIONS if t not in SUPPORTED_TYPE_COMBINATIONS_WITH_FP8_D
]


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (16, 16, 16),
        (32, 16, 16),
        (16, 32, 16),
        (16, 16, 32),
        (64, 32, 16),
        (16, 96, 32),
        (64, 16, 32),
        (64, 96, 16),
    ),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_stateful(m, n, k, atype, btype, ctype, dtype, use_cuda):
    """
    General test of FP8 multiplication.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

    mm = Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    mm.plan()
    result = mm.execute()

    assert str(result.dtype).split(".")[-1] == expected_result_type(atype, btype, ctype, dtype)

    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert_fp8_equal(result, reference)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    ((16, 16, 16),),
)
@pytest.mark.parametrize("amax", (True, False))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_autotuning(m, n, k, atype, btype, ctype, dtype, amax, use_cuda):
    """
    Tests if autotuning works with FP8 multiplication.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    result_type = expected_result_type(atype, btype, ctype, dtype)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "result_amax": amax and "float8" in result_type}
    mm = Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    mm.plan()
    mm.autotune()
    result = mm.execute()
    if isinstance(result, tuple):
        result = result[0]
    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert_fp8_equal(result, reference)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    ((96, 128, 16),),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_stateless(m, n, k, atype, btype, ctype, dtype, use_cuda):
    """
    Tests if stateless `matmul` supports quantization_scales and options.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    result = matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert str(result.dtype).split(".")[-1] == expected_result_type(atype, btype, ctype, dtype)

    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)

    assert_fp8_equal(result, reference)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "a_batch,b_batch,c_batch,d_batch",
    (
        ((), (), (), ()),
        ((3,), (3,), (3,), (3,)),
        ((8,), (8,), (8,), (8,)),
        ((5,), (), (5,), (5,)),
        ((), (2,), (2,), (2,)),
        ((2, 3), (2, 3), (2, 3), (2, 3)),
    ),
)
@pytest.mark.parametrize(
    "m,n,k",
    ((16, 16, 16),),
)
@pytest.mark.parametrize(("use_cuda"), (True,))
def test_batching(m, n, k, atype, btype, ctype, dtype, a_batch, b_batch, c_batch, d_batch, use_cuda):
    """
    Tests if batching works with FP8.
    """

    def sample_batch(batch_shape, matrix_shape, type, transposed=False):
        shape = (*batch_shape, *matrix_shape)
        if transposed:
            shape = (*shape[:-2], shape[-1], shape[-2])
        x = sample_matrix("torch", type, shape, use_cuda=use_cuda, min=0, max=2)
        return x.swapaxes(-1, -2) if transposed else x

    a = sample_batch(a_batch, (m, k), atype, transposed=False)
    b = sample_batch(b_batch, (k, n), btype, transposed=True)

    if ctype is not None:
        c = sample_batch(c_batch, (m, n), ctype, transposed=False)
        beta = 0.12
    else:
        c = None
        beta = None

    alpha = 0.32

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    result = matmul_with_random_autotune(
        a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options
    )

    expected_result_shape = (*d_batch, m, n)
    assert result.shape == expected_result_shape

    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert_fp8_equal(result, reference)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    ((64, 32, 16),),
)
@pytest.mark.parametrize("a_scale_kind", ("scalar", "gpu", "cpu"))
@pytest.mark.parametrize("b_scale_kind", ("scalar", "gpu", "cpu"))
@pytest.mark.parametrize("c_scale_kind", ("scalar", "gpu", "cpu"))
@pytest.mark.parametrize("d_scale_kind", ("scalar", "gpu", "cpu"))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_tensor_scales(m, n, k, atype, btype, ctype, dtype, a_scale_kind, b_scale_kind, c_scale_kind, d_scale_kind, use_cuda):
    """
    Test scales provided as scalars/GPU tensors/CPU tensors
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    scalar_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

    def prepare_scales(scales):
        """
        Change some of the scales into tensors
        """

        def wrap_scale(x, kind):
            if kind == "scalar" or x is None:
                return x
            tensor = torch.as_tensor(x, dtype=torch.float32)
            return tensor.cuda() if kind == "gpu" else tensor

        return {
            "a": wrap_scale(scales["a"], a_scale_kind),
            "b": wrap_scale(scales["b"], b_scale_kind),
            "c": wrap_scale(scales["c"], c_scale_kind),
            "d": wrap_scale(scales["d"], d_scale_kind),
        }

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

    scales = prepare_scales(scalar_scales)
    mm = Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)
    mm.plan()
    result = mm.execute()
    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=scalar_scales, options=options)
    assert_fp8_equal(result, reference)

    # In-place modification of GPU scales
    if a_scale_kind == "gpu":
        scalar_scales["a"] *= 0.5
        scales["a"].copy_(scalar_scales["a"])
    if b_scale_kind == "gpu":
        scalar_scales["b"] *= -1
        scales["b"].copy_(scalar_scales["b"])
    if c_scale_kind == "gpu" and scalar_scales["c"] is not None:
        scalar_scales["c"] *= -1
        scales["c"].copy_(scalar_scales["c"])
    if d_scale_kind == "gpu" and scalar_scales["d"] is not None:
        scalar_scales["d"] *= 0.5
        scales["d"].copy_(scalar_scales["d"])
    result = mm.execute()
    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=scalar_scales, options=options)
    assert_fp8_equal(result, reference)

    # Reset of the scales
    new_a, new_b, new_c, new_alpha, new_beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    new_scalar_scales = choose_scales(new_a, new_b, new_c, atype, btype, ctype, dtype, alpha=new_alpha, beta=new_beta)
    new_scales = prepare_scales(new_scalar_scales)
    mm.reset_operands(a=new_a, b=new_b, c=new_c, quantization_scales=new_scales, alpha=new_alpha, beta=new_beta)
    result = mm.execute()
    reference = fp8_matmul_reference(
        new_a, new_b, new_c, alpha=new_alpha, beta=new_beta, quantization_scales=new_scalar_scales, options=options
    )
    assert_fp8_equal(result, reference)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS_WITH_FP8_D)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (16, 16, 16),
        (96, 64, 16),
    ),
)
@pytest.mark.parametrize(("stateless"), (True, False))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_amax(m, n, k, atype, btype, ctype, dtype, stateless, use_cuda):
    """
    Test if amax is computed properly.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "result_amax": True}
    if not stateless:
        with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options) as mm:
            mm.plan()
            result, aux = mm.execute()
    else:
        result, aux = matmul_with_random_autotune(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)
    assert len(aux) == 1
    amax = aux["result_amax"]
    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, options=options, quantization_scales=scales)
    assert_fp8_equal(result, reference)
    not_scaled_reference = fp8_matmul_reference(
        a,
        b,
        c,
        alpha=alpha,
        beta=beta,
        options=options,
        quantization_scales={k: v if k != "d" else 1.0 for k, v in scales.items()},
    )
    assert_tensors_equal(amax, (not_scaled_reference.abs().max()), atol=0.01, rtol=1e-3)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "reset_a,reset_b,reset_c",
    ((a, b, c) for a in (True, False) for b in (True, False) for c in (True, False) if (a, b, c) != (False, False, False)),
)
@pytest.mark.parametrize("reset_alpha", (True, False))
@pytest.mark.parametrize("reset_beta", (True, False))
@pytest.mark.parametrize("m,n,k", ((16, 16, 16),))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_reset_operands(m, n, k, atype, btype, ctype, dtype, reset_a, reset_b, reset_c, reset_alpha, reset_beta, use_cuda):
    """
    Tests if reset_operands works with FP8 matmuls without resetting the scales.
    """
    if reset_c and ctype is None:
        pytest.skip("Can't reset C because C is not specified")
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    mm = Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    mm.plan()
    result1 = mm.execute()
    reference1 = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert_fp8_equal(result1, reference1)

    new_a, new_b, new_c, new_alpha, new_beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    reset_kwargs = {}
    if reset_a:
        reset_kwargs["a"] = a = new_a
    if reset_b:
        reset_kwargs["b"] = b = new_b
    if reset_c:
        reset_kwargs["c"] = c = new_c
    if reset_alpha:
        reset_kwargs["alpha"] = alpha = new_alpha
    if reset_beta:
        reset_kwargs["beta"] = beta = new_beta

    mm.reset_operands(**reset_kwargs)
    result2 = mm.execute()
    reference2 = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert_fp8_equal(result2, reference2)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize("reset_a_scale", (True, False))
@pytest.mark.parametrize("reset_b_scale", (True, False))
@pytest.mark.parametrize("reset_c_scale", (True, False))
@pytest.mark.parametrize("reset_d_scale", (True, False))
@pytest.mark.parametrize("m,n,k", ((16, 16, 16),))
@pytest.mark.parametrize("use_cuda", (True,))
def test_reset_quantization_scales(
    m, n, k, atype, btype, ctype, dtype, reset_a_scale, reset_b_scale, reset_c_scale, reset_d_scale, use_cuda
):
    """
    Tests if reset_operands allows resetting (some or all) quantization_scales.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    mm = Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    mm.plan()
    result1 = mm.execute()
    reference1 = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
    assert_fp8_equal(result1, reference1)

    new_a, new_b, new_c, new_alpha, new_beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    new_quantization_scales = choose_scales(new_a, new_b, new_c, atype, btype, ctype, dtype, alpha=new_alpha, beta=new_beta)

    reset_kwargs = {"a": new_a}

    reset_quantization_scales = {}
    if reset_a_scale:
        reset_quantization_scales["a"] = quantization_scales["a"] = new_quantization_scales["a"]
    if reset_b_scale:
        reset_quantization_scales["b"] = quantization_scales["b"] = new_quantization_scales["b"]
    if reset_c_scale:
        reset_quantization_scales["c"] = quantization_scales["c"] = new_quantization_scales["c"]
    if reset_d_scale:
        reset_quantization_scales["d"] = quantization_scales["d"] = new_quantization_scales["d"]
    if reset_quantization_scales:
        reset_kwargs["quantization_scales"] = reset_quantization_scales

    mm.reset_operands(**reset_kwargs)
    result2 = mm.execute()
    reference2 = fp8_matmul_reference(
        new_a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options
    )
    assert_fp8_equal(result2, reference2)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS[0:1])
@pytest.mark.parametrize("m,n,k", ((16, 16, 16),))
@pytest.mark.parametrize("use_cuda", (True,))
def test_quantization_scales_as_object(m, n, k, atype, btype, ctype, dtype, use_cuda):
    """
    Tests if passing the scales as an instance of MatmulQuantizationScales works.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    dict_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    quantization_scales = MatmulQuantizationScales(
        a=dict_scales["a"], b=dict_scales["b"], c=dict_scales["c"], d=dict_scales["d"]
    )
    result = matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales)
    assert str(result.dtype).split(".")[-1] == expected_result_type(atype, btype, ctype, dtype)

    reference = fp8_matmul_reference(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales)

    assert_fp8_equal(result, reference)


###############
@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize("a_scale", (True, False))
@pytest.mark.parametrize("b_scale", (True, False))
@pytest.mark.parametrize("c_scale", (True, False))
@pytest.mark.parametrize("d_scale", (True, False))
@pytest.mark.parametrize("use_cuda", (True,))
def test_validation_required_quantization_scales(atype, btype, ctype, dtype, a_scale, b_scale, c_scale, d_scale, use_cuda):
    """
    Tests if unspecified quantization_scales trigger an error.
    """
    a, b, c, alpha, beta = generate_inputs(16, 16, 16, atype, btype, ctype, use_cuda=use_cuda)
    all_quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)

    quantization_scales = {}
    if a_scale:
        quantization_scales["a"] = all_quantization_scales["a"]
    if b_scale:
        quantization_scales["b"] = all_quantization_scales["b"]
    if c_scale:
        quantization_scales["c"] = all_quantization_scales["c"]
    if d_scale:
        quantization_scales["d"] = all_quantization_scales["d"]

    quantization_scales_ok = all(x in quantization_scales or all_quantization_scales[x] is None for x in "abcd")

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

    with pytest.raises(ValueError, match=r"Scales are required for narrow-precision \(FP8 and lower\) operations"):
        matmul(a, b, c, alpha=alpha, beta=beta, options=options)

    if quantization_scales_ok:
        result = matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)
        reference = fp8_matmul_reference(
            a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options
        )
        assert_fp8_equal(result, reference)
    else:
        with pytest.raises(ValueError):
            matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS[1:2])
@pytest.mark.parametrize("m,n,k", ((16, 16, 16),))
@pytest.mark.parametrize("use_cuda", (True,))
def test_validation_invalid_quantization_scales_type(m, n, k, atype, btype, ctype, dtype, use_cuda):
    """
    Tests what happens when an invalid type is provided for `quantization_scales`.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    with pytest.raises(
        TypeError, match="Scale factors must be provided as an object of type MatmulQuantizationScales or as a dict"
    ):
        matmul(a, b, quantization_scales="oh no!")


@pytest.mark.parametrize(
    "atype,btype, ctype",
    (
        ("float8_e4m3fn", "float8_e5m2", "float16"),
        ("float8_e5m2", "float8_e4m3fn", None),
        ("float8_e5m2", "float8_e4m3fn", "float32"),
    ),
)
@pytest.mark.parametrize("m,n,k", ((16, 16, 16),))
def test_validation_unsupported_different_ab_types(m, n, k, atype, btype, ctype):
    version = cublaslt.get_version()
    if version >= 120600:
        pytest.skip(f"Different A and B types are supported for cuBLAS {version}")
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=True)
    with pytest.raises(
        ValueError, match=f"FP8 multiplication of {atype} and {btype} is not supported in cuBLASLt version {version}"
    ):
        matmul(a, b, quantization_scales={"a": 1, "b": 1, "c": 1, "d": 1})


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (16, 16, 8),
        (32, 8, 16),
        (8, 16, 16),
        (32, 32, 12),
        (32, 36, 16),
        (4, 48, 32),
        (32, 96, 17),
        (80, 11, 64),
        (19, 96, 128),
        (33, 44, 55),
    ),
)
def test_validation_invalid_sizes(m, n, k, atype, btype, ctype, dtype):
    """
    Tests if invalid size raises an error.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=True)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

    with pytest.raises(ValueError, match="must be divisible by 16"):
        matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    ((64, 64, 64),),
)
@pytest.mark.parametrize("misaligned", ("a", "b"))
@pytest.mark.parametrize("offset", (1, 2, 4, 8, 12))
def test_validation_misaligned(m, n, k, atype, btype, ctype, dtype, misaligned, offset):
    """
    Tests if invalid alignment raises an error or returns a correct result.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=True)

    def gen_misaligned(shape, type):
        assert len(shape) == 2
        aligned = torch.rand(shape[0] * shape[1] + offset).type(getattr(torch, type)).cuda()
        return aligned[offset:].reshape(shape)

    if misaligned == "a":
        a = gen_misaligned((m, k), atype)
    if misaligned == "b":
        b = gen_misaligned((n, k), btype).T

    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}

    with pytest.raises(ValueError, match="should be aligned to 16 bytes"):
        matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)


@pytest.mark.parametrize(
    "atype,btype,ctype,dtype",
    [
        (a, b, c, d)
        for (a, b, c, d) in SUPPORTED_TYPE_COMBINATIONS
        if "float8" not in expected_result_type(a, b, c, d) or (c and "float8" not in c)
    ],
)
@pytest.mark.parametrize(
    "m,n,k",
    ((16, 16, 16),),
)
def test_validation_non_fp8_scale(m, n, k, atype, btype, ctype, dtype):
    """
    Tests if scales are prohibited for non-FP8 tensors
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=True)
    quantization_scales = {"a": 1, "b": 1, "c": 1, "d": 1}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    with pytest.raises(
        ValueError,
        match=r"Quantization scaling is not supported for . when it is not a narrow-precision \(FP8 and lower\) type",
    ):
        matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options)


@pytest.mark.parametrize(
    "atype,btype,ctype,dtype",
    [
        (a, b, c, d)
        for (a, b, c, d) in SUPPORTED_TYPE_COMBINATIONS
        if "float8" not in expected_result_type(a, b, c, d) or (c and "float8" not in c)
    ],
)
@pytest.mark.parametrize(
    "m,n,k",
    ((16, 16, 16),),
)
def test_validation_non_fp8_scale_reset(m, n, k, atype, btype, ctype, dtype):
    """
    Tests if attempt to reset the scale for non-FP8 tensor raises an error.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=True)
    quantization_scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None}
    with Matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=quantization_scales, options=options) as mm:
        mm.plan()
        with pytest.raises(
            ValueError,
            match=r"Quantization scaling is not supported for . when it is not a narrow-precision \(FP8 and lower\) type",
        ):
            mm.reset_operands(a=a, quantization_scales={"a": 1, "b": 1, "c": 1, "d": 1})


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS_WITH_NON_FP8_D)
@pytest.mark.parametrize(
    "m,n,k",
    ((16, 16, 16),),
)
@pytest.mark.parametrize(("stateless"), (True, False))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_validation_non_fp8_amax(m, n, k, atype, btype, ctype, dtype, stateless, use_cuda):
    """
    Test if amax is not supported for non-FP8 D.
    """
    a, b, c, alpha, beta = generate_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)
    scales = choose_scales(a, b, c, atype, btype, ctype, dtype, alpha=alpha, beta=beta)
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "result_amax": True}
    with pytest.raises(ValueError, match=r"result_amax=True is allowed only for narrow-precision \(FP8 and lower\) results"):
        matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)
