# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0


try:
    import torch
except ImportError:
    torch = None
import pytest
from .utils import sample_matrix
from .fp8_utils import assert_fp8_equal
from nvmath.linalg.advanced import Matmul, matmul, MatmulEpilog as Epilog
from nvmath.linalg.advanced.helpers import matmul as matmul_helpers
from nvmath._internal.typemaps import NAME_TO_DATA_TYPE
from nvmath.bindings import cublasLt as cublaslt
from nvmath._internal.utils import check_or_create_options
from nvmath.linalg.advanced import _configuration
from contextlib import nullcontext
from .utils import allow_cublas_unsupported

if torch is None:
    pytest.skip("Torch is required for MXFP8 tests", allow_module_level=True)

if torch.cuda.get_device_properties(0).major < 10:
    pytest.skip("CC>=10.0 is required for MXFP8 tests", allow_module_level=True)

if cublaslt.get_version() < 120800:
    pytest.skip("cuBLAS 12.8 is required for MXFP8 tests", allow_module_level=True)


def unpack_matmul(result):
    """
    Helper function which unpacks the result of `matmul` into D, d_out (or None), epilog aux
    """
    if isinstance(result, tuple):
        d, aux = result
        d_out = aux.pop("d_out_scale", None)
        return d, d_out, aux
    else:
        return result, None, {}


def expected_result_type(atype, btype, ctype, dtype):
    """
    Result type of FP8 matmul. ctype=None means no C. dtype=None means no explicit
    specification of the result type.
    """
    return dtype or ctype or atype


def generate_mxfp8_scales(x, scale_range, *, use_cuda, validate_shape=True):
    """
    Generates UE8M0 scales for x, randomly chosen from [2^scale_range[0], 2^scale_range[1]]
    """
    low = scale_range[0] + 127
    high = scale_range[1] + 127
    assert low >= 0
    assert high <= 255
    if validate_shape:
        assert all(s % 128 == 0 for s in x.shape[-2:])
    num_scales = x.nelement() // 32
    s = torch.randint(low=low, high=high + 1, size=(num_scales,)).to(torch.uint8)
    if use_cuda:
        s = s.cuda()
    return s


def expand_mxfp8_scales(x, scales):
    """
    Expands block UE8M0 scales tensor for `x` into a float32 tensor with actual scale
    factors.
    """
    idx = matmul_helpers.get_mxfp8_scale_offset(x, torch.meshgrid(*(torch.arange(d) for d in x.shape), indexing="ij"))
    if scales.is_cuda:
        idx = idx.cuda()
    return 2 ** (scales.type(torch.float32)[idx] - 127)


def generate_simple_inputs(m, n, k, atype, btype, ctype, *, c_transposed=False, use_cuda):
    """
    Generates matmul inputs of given shapes and types.
    """

    def random_choice(choices, shape=()):
        choices = torch.as_tensor(choices)
        idx = torch.randint(low=0, high=len(choices), size=shape)
        return choices[idx]

    def random_sign():
        return random_choice([-1, 1]).item()

    # Use non-symmetric distributions to reduce the risk of catastrophic cancellations
    a = random_choice([-1, 0, 0.5, 1, 1.5], shape=(m, k)).type(getattr(torch, atype))
    b = random_choice([-1, 0, 0.5, 1, 1.5], shape=(n, k)).type(getattr(torch, btype)).T

    alpha = random_sign() * 2 ** torch.randint(low=-10, high=-8, size=()).item()

    if ctype is not None:
        if c_transposed:
            c = random_choice([-0.25, 0, 0.5, 1], shape=(n, m)).type(getattr(torch, ctype)).T
        else:
            c = random_choice([-0.25, 0, 0.5, 1], shape=(m, n)).type(getattr(torch, ctype))
        beta = random_sign() * alpha * torch.rand(size=()).item()
    else:
        c = None
        beta = None
    if use_cuda:
        a = a.cuda()
        b = b.cuda()
        if c is not None:
            c = c.cuda()
    return a, b, c, alpha, beta


def mxfp8_matmul_reference(
    a, b, c=None, *args, d_out=None, quantization_scales=None, epilog_inputs=None, options=None, **kwargs
):
    """
    Computes MXFP8-like matmul, but with higher precision.
    """
    scales = check_or_create_options(_configuration.MatmulQuantizationScales, quantization_scales, "Matmul scales")
    options = check_or_create_options(_configuration.MatmulOptions, options, "Matmul options")
    options.result_type = None

    a_scale = expand_mxfp8_scales(a, scales.a)
    b_scale = expand_mxfp8_scales(b, scales.b)
    ascaled = a.type(torch.float32) * a_scale
    bscaled = b.type(torch.float32) * b_scale

    for key in ("bias", "gelu_aux"):
        if epilog_inputs and key in epilog_inputs:
            epilog_inputs[key] = epilog_inputs[key].type(torch.float32)

    d = matmul(
        ascaled,
        bscaled,
        c.type(torch.float32) if c is not None else None,
        *args,
        quantization_scales=None,
        epilog_inputs=epilog_inputs,
        options=options,
        **kwargs,
    )
    if d_out is not None:
        d_scale = expand_mxfp8_scales(d, d_out)
        d /= d_scale
    return d


SUPPORTED_TYPE_COMBINATIONS = (
    ("float8_e4m3fn", "float8_e4m3fn", None, None),
    ("float8_e4m3fn", "float8_e5m2", None, None),
    ("float8_e5m2", "float8_e4m3fn", None, None),
    ("float8_e4m3fn", "float8_e4m3fn", "float32", "float32"),
    ("float8_e4m3fn", "float8_e5m2", "float32", "float32"),
    ("float8_e5m2", "float8_e4m3fn", "float32", "float32"),
    ("float8_e4m3fn", "float8_e4m3fn", "float32", None),
    ("float8_e4m3fn", "float8_e5m2", "float32", None),
    ("float8_e5m2", "float8_e4m3fn", "float32", None),
    ("float8_e4m3fn", "float8_e4m3fn", "float16", "float16"),
    ("float8_e4m3fn", "float8_e5m2", "float16", "float16"),
    ("float8_e5m2", "float8_e4m3fn", "float16", "float16"),
    ("float8_e4m3fn", "float8_e4m3fn", "float16", None),
    ("float8_e4m3fn", "float8_e5m2", "float16", None),
    ("float8_e5m2", "float8_e4m3fn", "float16", None),
    ("float8_e4m3fn", "float8_e4m3fn", "bfloat16", "bfloat16"),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", "bfloat16"),
    ("float8_e5m2", "float8_e4m3fn", "bfloat16", "bfloat16"),
    ("float8_e4m3fn", "float8_e4m3fn", "bfloat16", None),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", None),
    ("float8_e5m2", "float8_e4m3fn", "bfloat16", None),
    ("float8_e4m3fn", "float8_e4m3fn", "float16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e5m2", "float16", "float8_e4m3fn"),
    ("float8_e5m2", "float8_e4m3fn", "float16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e4m3fn", "bfloat16", "float8_e4m3fn"),
    ("float8_e4m3fn", "float8_e5m2", "bfloat16", "float8_e4m3fn"),
    ("float8_e5m2", "float8_e4m3fn", "bfloat16", "float8_e4m3fn"),
)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (128, 128, 128),
        (2 * 128, 4 * 128, 2 * 128),
        (7 * 128, 5 * 128, 3 * 128),
    ),
)
@pytest.mark.parametrize("a_scale_range", ((-5, 5),))
@pytest.mark.parametrize("b_scale_range", ((-5, 5),))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_mxfp8(m, n, k, atype, btype, ctype, dtype, a_scale_range, b_scale_range, use_cuda):
    """
    Basic MXFP8 test.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    result_type = expected_result_type(atype, btype, ctype, dtype)
    ascales = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda)

    scales = {"a": ascales, "b": bscales}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}
    result, d_out, _ = unpack_matmul(matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options))

    reference = mxfp8_matmul_reference(
        a, b, d_out=d_out, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options
    )
    assert_fp8_equal(result, reference)
    assert str(result.dtype).split(".")[-1] == result_type


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize("a_scale_range", ((-5, 5),))
@pytest.mark.parametrize("b_scale_range", ((-5, 5),))
@pytest.mark.parametrize(
    "a_batch,b_batch,c_batch,d_batch",
    (
        ((), (), (), ()),
        ((3,), (3,), (3,), (3,)),
        ((8,), (8,), (8,), (8,)),
        ((2, 3), (2, 3), (2, 3), (2, 3)),
    ),
)
@pytest.mark.parametrize(
    "m,n,k",
    ((128, 128, 128),),
)
@pytest.mark.parametrize(("use_cuda"), (True,))
def test_batching(
    m, n, k, atype, btype, ctype, dtype, a_scale_range, b_scale_range, a_batch, b_batch, c_batch, d_batch, use_cuda
):
    """
    Tests if batching works with MXFP8.
    """

    def sample_batch(batch_shape, matrix_shape, type, transposed=False):
        shape = (*batch_shape, *matrix_shape)
        if transposed:
            shape = (*shape[:-2], shape[-1], shape[-2])
        x = sample_matrix("torch", type, shape, use_cuda=use_cuda, min=0, max=2)
        return x.swapaxes(-1, -2) if transposed else x

    a = sample_batch(a_batch, (m, k), atype, transposed=False)
    b = sample_batch(b_batch, (k, n), btype, transposed=True)

    ascales = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda)

    if ctype is not None:
        c = sample_batch(c_batch, (m, n), ctype, transposed=False)
        beta = 0.12
    else:
        c = None
        beta = None

    alpha = 0.32

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}
    scales = {"a": ascales, "b": bscales}
    result, d_out, _ = unpack_matmul(matmul(a, b, c, alpha=alpha, beta=beta, quantization_scales=scales, options=options))
    reference = mxfp8_matmul_reference(
        a, b, c=c, alpha=alpha, d_out=d_out, beta=beta, quantization_scales=scales, options=options
    )
    expected_result_shape = (*d_batch, m, n)
    assert result.shape == expected_result_shape

    reference = mxfp8_matmul_reference(
        a, b, c, d_out=d_out, alpha=alpha, beta=beta, quantization_scales=scales, options=options
    )
    assert_fp8_equal(result, reference)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    ((2 * 128, 4 * 128, 3 * 128),),
)
@pytest.mark.parametrize("a_scale_range", ((-3, 3),))
@pytest.mark.parametrize("b_scale_range", ((-3, 3),))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_reset(m, n, k, atype, btype, ctype, dtype, a_scale_range, b_scale_range, use_cuda):
    """
    Tests if in-place change of A/B scales and reset_operands works.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    ascales = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda)

    scales = {"a": ascales, "b": bscales}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}

    with Matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options) as mm:
        mm.plan()

        # Check initial result
        result, d_out, _ = unpack_matmul(mm.execute())
        reference = mxfp8_matmul_reference(
            a, b, c=c, alpha=alpha, d_out=d_out, beta=beta, quantization_scales=scales, options=options
        )
        assert_fp8_equal(result, reference)

        # Change A and B scales in place
        if use_cuda:
            ascales[: len(ascales) // 2] *= -1
            bscales[len(bscales) // 2 :] = bscales[0]
        result = mm.execute()
        result, d_out, _ = unpack_matmul(mm.execute())
        reference = mxfp8_matmul_reference(
            a, b, c=c, alpha=alpha, d_out=d_out, beta=beta, quantization_scales=scales, options=options
        )
        assert_fp8_equal(result, reference)

        # Reset A scale, keep B scale
        ascales2 = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
        scales2 = scales.copy()
        scales2["a"] = ascales2
        mm.reset_operands(a=a, b=b, quantization_scales={"a": ascales2})
        result, d_out, _ = unpack_matmul(mm.execute())
        reference = mxfp8_matmul_reference(
            a, b, c=c, alpha=alpha, d_out=d_out, beta=beta, quantization_scales=scales2, options=options
        )
        assert_fp8_equal(result, reference)

        # Reset A scale and B scale
        ascales3 = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
        bscales3 = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda)
        scales3 = scales2.copy()
        scales3["a"] = ascales3
        scales3["b"] = bscales3
        mm.reset_operands(a=a, b=b, quantization_scales={"a": ascales3, "b": bscales3})
        result, d_out, _ = unpack_matmul(mm.execute())
        reference = mxfp8_matmul_reference(
            a, b, c=c, alpha=alpha, d_out=d_out, beta=beta, quantization_scales=scales3, options=options
        )
        assert_fp8_equal(result, reference)


def unpack_bitmask(bitmask, shape):
    """
    Utility function unpacking ReLU aux bitmask.
    """
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
        (128, 128, 128),
        (4 * 128, 3 * 128, 2 * 128),
    ),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "epilog_name,order",
    (
        ("RELU", "col"),
        ("RELU", "row"),
        ("GELU", "col"),
        ("BGRADA", "col"),
        ("BGRADB", "col"),
        ("DRELU", "col"),
        ("DRELU_BGRAD", "col"),
        ("DGELU", "col"),
        ("DGELU_BGRAD", "col"),
        ("RELU_BIAS", "col"),
        ("BIAS", "col"),
        ("GELU_BIAS", "col"),
        ("RELU_AUX", "col"),
        ("RELU_AUX_BIAS", "col"),
        ("GELU_AUX", "col"),
        ("GELU_AUX_BIAS", "col"),
    ),
)
@pytest.mark.parametrize(
    "a_batch,b_batch,c_batch,d_batch",
    (
        ((), (), (), ()),
        ((2,), (2,), (2,), (2,)),
        ((2, 3), (2, 3), (2, 3), (2, 3)),
    ),
)
@pytest.mark.parametrize("a_scale_range", ((-2, 2),))
@pytest.mark.parametrize("b_scale_range", ((-3, 3),))
def test_epilogs(
    m,
    n,
    k,
    atype,
    btype,
    ctype,
    dtype,
    a_scale_range,
    b_scale_range,
    epilog_name,
    a_batch,
    b_batch,
    c_batch,
    d_batch,
    order,
    use_cuda,
):
    """
    Tests epilogs with MXFP8.
    """
    epilog = getattr(Epilog, epilog_name)

    result_type = expected_result_type(atype, btype, ctype, dtype)
    inferred_ctype = ctype or ("float16" if "float8" in result_type else result_type)

    # Currently, those are not supported by cuBLAS, so we allow them to fail with
    # "NOT_SUPPORTED".
    allow_not_supported = False
    allow_not_supported |= "BGRAD" in epilog_name
    allow_not_supported |= "DRELU" in epilog_name
    allow_not_supported |= "DGELU" in epilog_name
    allow_not_supported |= "AUX" in epilog_name

    def sample_batch(batch_shape, matrix_shape, type, transposed=False):
        shape = (*batch_shape, *matrix_shape)
        if transposed:
            shape = (*shape[:-2], shape[-1], shape[-2])
        x = sample_matrix("torch", type, shape, use_cuda=use_cuda, min=-0.2, max=1)
        return x.swapaxes(-1, -2) if transposed else x

    a = sample_batch(a_batch, (m, k), atype, transposed=False)
    b = sample_batch(b_batch, (k, n), btype, transposed=True)
    alpha, beta = 0.12, None
    if ctype is not None:
        c = sample_batch(c_batch, (m, n), ctype, transposed=(order == "col"))
        beta = 0.34
    else:
        c = None
        beta = None

    ascales = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda)

    inputs = {}
    if "BIAS" in epilog_name:
        bias_type = "float16" if inferred_ctype == "float16" else "bfloat16"
        bias = sample_matrix("torch", bias_type, (m,), use_cuda=use_cuda, min=0, max=1)
        inputs["bias"] = bias
    if "DRELU" in epilog_name:
        round_16 = lambda x: (x + 15) // 16 * 16
        inputs["relu_aux"] = torch.randint(low=0, high=256, size=(n, round_16(m // 8))).type(torch.uint8).T
    if "DGELU" in epilog_name:
        if order == "col":
            inputs["gelu_aux"] = sample_matrix("torch", result_type, (n, m), use_cuda=use_cuda, min=-5, max=5).T
        else:
            inputs["gelu_aux"] = sample_matrix("torch", result_type, (m, n), use_cuda=use_cuda, min=-5, max=5)

    scales = {"a": ascales, "b": bscales}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}

    # Run matmul. Allow cuBLAS NOT_SUPPORTED error for certain configurations (see above)
    with (
        nullcontext()
        if not allow_not_supported
        else allow_cublas_unsupported(
            message=f"MXFP8 epilog not supported by cuBLAS: {epilog_name} for A:{atype} B:{btype} C:{ctype} D:{dtype}",
            allow_invalid_value=True,
        )
    ):
        result, d_out, aux = unpack_matmul(
            matmul(
                a,
                b,
                c,
                alpha=alpha,
                beta=beta,
                epilog=epilog,
                quantization_scales=scales,
                options=options,
                epilog_inputs=inputs,
            )
        )

    # Compute the reference and compare
    reference = mxfp8_matmul_reference(
        a,
        b,
        c,
        d_out=d_out,
        alpha=alpha,
        beta=beta,
        epilog=epilog,
        quantization_scales=scales,
        options=options,
        epilog_inputs=inputs,
    )

    if isinstance(reference, tuple):
        reference, reference_aux = reference
    else:
        reference_aux = {}

    if "GELU" in epilog_name and result_type not in ("float16", "float32"):
        assert_fp8_equal(result, reference, atol=1e-1, rtol=1e-1)
    else:
        assert_fp8_equal(result, reference)

    # Compare auxiliary outputs
    assert aux.keys() == reference_aux.keys()
    for key in aux:
        if key == "relu_aux":
            x = unpack_bitmask(aux[key], (m, n))
            y = unpack_bitmask(reference_aux[key], (m, n))
            assert torch.mean((x == y).type(torch.float32)) > 0.99
        elif key == "gelu_aux":
            assert_fp8_equal(aux[key], reference_aux[key])
        elif key == "drelu_bgrad" or key == "dgelu_bgrad":
            assert_fp8_equal(aux[key], reference.sum(axis=1), atol=1e-1, rtol=1e-1)
        else:
            raise RuntimeError(f"Test for {key} not implemented")


def test_helpers():
    """
    Tests MXFP8 helpers.
    """
    x = torch.ones((1024, 3 * 1024), dtype=torch.float8_e4m3fn)
    scales = matmul_helpers.create_mxfp8_scale(x, 3)
    y = matmul_helpers.apply_mxfp8_scale(x, scales)
    assert_fp8_equal(y, x.type(torch.float32) * 8)
    z = matmul_helpers.apply_mxfp8_scale(y, matmul_helpers.invert_mxfp8_scale(scales))
    assert_fp8_equal(z, x.type(torch.float32))


@pytest.mark.parametrize("M,N", ((1024, 3 * 1024), (128, 128), (5 * 1024, 256)))
@pytest.mark.parametrize("nsamples", (1, 7, 100))
@pytest.mark.parametrize("input_format", ("vectors", "ints"))
def test_indexing_helpers(M, N, nsamples, input_format):
    """
    Tests indexing helpers.
    """
    tensor = torch.zeros((M, N), dtype=torch.float8_e4m3fn)
    xs, ys = (torch.randint(size=(nsamples,), low=0, high=d, dtype=torch.int32) for d in (M, N))

    full = matmul_helpers.get_mxfp8_scale_offset(tensor, torch.meshgrid(*(torch.arange(d) for d in (M, N)), indexing="ij"))
    reference = full[xs, ys]

    if input_format == "vectors":
        result = matmul_helpers.get_mxfp8_scale_offset(tensor, (xs, ys))
        assert torch.all(result == reference)
    elif input_format == "ints":
        result = [matmul_helpers.get_mxfp8_scale_offset(tensor, (x, y)) for x, y in zip(xs, ys, strict=True)]
        assert all(res == ref for res, ref in zip(result, reference, strict=True))
    else:
        raise RuntimeError


@pytest.mark.parametrize("order", ("t", "b", "tb", "bt", "tbt", "btb"))
def test_mxfp8_and_fp8(order):
    """
    Test if MXFP8 and FP8 work together
    """
    m, n, k = 256, 256, 256

    a = torch.zeros(m, k, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn).T

    for kind in order:
        if kind == "t":
            # Tensor-wide scaling
            matmul(a, b, quantization_scales={"a": 1, "b": 1, "d": 1})
        else:
            # Block scaling
            matmul(
                a,
                b,
                options={"block_scaling": True},
                quantization_scales={
                    "a": matmul_helpers.create_mxfp8_scale(a, 0),
                    "b": matmul_helpers.create_mxfp8_scale(b, 0),
                },
            )


@pytest.mark.parametrize(
    "m,n,k",
    ((128, 128, 128),),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
def test_validation_d_scale(m, n, k, atype, btype, ctype, dtype, use_cuda):
    """
    Test if an error is raised if D scale is provided.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    ascales = generate_mxfp8_scales(a, (-10, 10), use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, (-10, 10), use_cuda=use_cuda)
    scales = {"a": ascales, "b": bscales, "d": torch.zeros(m, n).type(torch.uint8).cuda()}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}
    with pytest.raises(ValueError, match="Quantization scaling is not supported for D when `block_scaling` option is enabled."):
        matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)


@pytest.mark.parametrize(
    "m,n,k",
    ((128, 128, 128),),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize("scales_provided", ("", "a", "b"))
def test_validation_all_scales_required(m, n, k, atype, btype, ctype, dtype, scales_provided, use_cuda):
    """
    Test if an error is raised if not all scales are provided.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    result_type = expected_result_type(atype, btype, ctype, dtype)
    ascales = generate_mxfp8_scales(a, (-10, 10), use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, (-10, 10), use_cuda=use_cuda)
    scales = {}
    if "a" in scales_provided:
        scales["a"] = ascales
    if "b" in scales_provided:
        scales["b"] = bscales

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}

    ok = "a" in scales_provided and "b" in scales_provided and ("d" in scales_provided or "float8" not in result_type)

    if not ok:
        with pytest.raises(ValueError, match=r"Scale for . is not specified"):
            matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize("a_scale_range", ((-5, 5),))
@pytest.mark.parametrize("b_scale_range", ((-5, 5),))
@pytest.mark.parametrize(
    "a_batch,b_batch",
    (
        ((3,), ()),
        ((), (3,)),
        ((2,), (3,)),
        ((2, 2), (1, 4)),
    ),
)
@pytest.mark.parametrize(
    "m,n,k",
    ((128, 128, 128),),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_validation_ab_batches_different(
    m, n, k, atype, btype, ctype, dtype, a_scale_range, b_scale_range, a_batch, b_batch, use_cuda
):
    """
    Tests if MXFP8 raises an error when batch sizes are different.
    """

    def sample_batch(batch_shape, matrix_shape, type, transposed=False):
        shape = (*batch_shape, *matrix_shape)
        if transposed:
            shape = (*shape[:-2], shape[-1], shape[-2])
        x = sample_matrix("torch", type, shape, use_cuda=use_cuda, min=0, max=2)
        return x.swapaxes(-1, -2) if transposed else x

    a = sample_batch(a_batch, (m, k), atype, transposed=False)
    b = sample_batch(b_batch, (k, n), btype, transposed=True)

    ascales = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda)
    bscales = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda)

    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}
    scales = {"a": ascales, "b": bscales}
    with pytest.raises(
        ValueError,
        match=r"When block_scaling=True, the batch dimensions of A and B must match \(broadcasting is not supported\).",
    ):
        matmul(a, b, quantization_scales=scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (128 + 1, 128, 128),
        (128, 128 + 1, 128),
        (128, 128, 128 + 1),
        (128 + 64, 128, 128),
        (128, 128 + 64, 128),
        (128, 128, 128 + 64),
    ),
)
@pytest.mark.parametrize("a_scale_range", ((-5, 5),))
@pytest.mark.parametrize("b_scale_range", ((-5, 5),))
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_validation_shapes(m, n, k, atype, btype, ctype, dtype, a_scale_range, b_scale_range, use_cuda):
    """
    Tests if an error is raised when M, N, K are not divisible by 128.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    ascales = generate_mxfp8_scales(a, a_scale_range, use_cuda=use_cuda, validate_shape=False)
    bscales = generate_mxfp8_scales(b, b_scale_range, use_cuda=use_cuda, validate_shape=False)

    scales = {"a": ascales, "b": bscales}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}
    with pytest.raises(ValueError, match=f"M={m} N={n} K={k} must be divisible by 128 when block_scaling=True"):
        matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (128, 128, 128),
        (128, 256, 512),
        (512, 512, 128),
    ),
)
@pytest.mark.parametrize(("use_cuda"), (True, False))
@pytest.mark.parametrize(
    "a_err,b_err",
    (
        (
            (1, 0),
            (0, 1),
            (1, 1),
            (32, 32),
            (-1, -1),
            (-1, 0),
            (0, -1),
        )
    ),
)
def test_validation_scales_shapes(m, n, k, atype, btype, ctype, dtype, a_err, b_err, use_cuda):
    """
    Tests if an error is scale shapes don't match input shapes.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=use_cuda)

    ascales = torch.zeros(size=(a.nelement() // 32 + a_err,), dtype=torch.uint8)
    bscales = torch.zeros(size=(a.nelement() // 32 + b_err,), dtype=torch.uint8)
    scales = {"a": ascales, "b": bscales}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}

    with pytest.raises(ValueError, match=r"Scales for (A|B) should have shape .* Got .*"):
        matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS[:3])
@pytest.mark.parametrize(("use_cuda"), (True, False))
def test_validation_scalar_scales(atype, btype, ctype, dtype, use_cuda):
    """
    Tests if scalar scales are disallowed when block_scaling=True.
    """
    a, b, c, alpha, beta = generate_simple_inputs(128, 128, 128, atype, btype, ctype, use_cuda=use_cuda)
    scales = {"a": 1, "b": 1}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}
    with pytest.raises(ValueError, match="A scalar tensor-wide scale factor is not allowed when block_scaling=True."):
        matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)


@pytest.mark.parametrize("atype,btype,ctype,dtype", SUPPORTED_TYPE_COMBINATIONS[:3])
@pytest.mark.parametrize(
    "m,n,k",
    ((128, 128, 128),),
)
@pytest.mark.parametrize(("scale_dtype"), (torch.int8, torch.float32))
def test_validation_scales_dtype(m, n, k, atype, btype, ctype, dtype, scale_dtype):
    """
    Tests if scales of invalid type are rejected.
    """
    a, b, c, alpha, beta = generate_simple_inputs(m, n, k, atype, btype, ctype, use_cuda=False)

    ascales = torch.zeros(size=(a.nelement() // 32,), dtype=scale_dtype)
    bscales = torch.zeros(size=(a.nelement() // 32,), dtype=scale_dtype)
    scales = {"a": ascales, "b": bscales}
    options = {"result_type": NAME_TO_DATA_TYPE[dtype] if dtype else None, "block_scaling": True}

    with pytest.raises(ValueError, match="Block scales for (A|B) should be uint8 tensor"):
        matmul(a, b, c=c, alpha=alpha, beta=beta, quantization_scales=scales, options=options)
