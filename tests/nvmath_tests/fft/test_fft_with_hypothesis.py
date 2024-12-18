import itertools

import cupy as cp
import numpy as np
import scipy.fft

from hypothesis import given, reproduce_failure, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

import nvmath

from nvmath_tests.helpers import nvmath_seed

# FIMXE: Lower minimum side length to 1 after refactoring of array traits
shape_st = array_shapes(min_dims=1, max_dims=3, min_side=2, max_side=256)

element_properties = dict(
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=True,
    min_magnitude=0.0,
    max_magnitude=1.0,
    min_value=-0.5,
    max_value=+0.5,
)

c32_array_st = arrays(
    np.complex64,
    shape=shape_st,
    elements=element_properties,
)
c64_array_st = arrays(
    np.complex128,
    shape=shape_st,
    elements=element_properties,
)
f32_array_st = arrays(
    np.float32,
    shape=shape_st,
    elements=element_properties,
)
f64_array_st = arrays(
    np.float64,
    shape=shape_st,
    elements=element_properties,
)

options_st = st.fixed_dictionaries(
    {
        "result_layout": st.sampled_from(["natural", "optimized"]),
        "last_axis_parity": st.sampled_from(["odd", "even"]),
        # TODO more options
    }
)

execution_st = st.sampled_from(
    [
        "cuda",
        "cpu",
        nvmath.fft.ExecutionCUDA(),
        nvmath.fft.ExecutionCPU(),
    ]
)

dtype_dict = {
    ("fft", "complex64"): "complex64",
    ("fft", "complex128"): "complex128",
    ("ifft", "complex64"): "complex64",
    ("ifft", "complex128"): "complex128",
    ("rfft", "float64"): "complex128",
    ("rfft", "float32"): "complex64",
    ("irfft", "complex128"): "float64",
    ("irfft", "complex64"): "float32",
}

axes_strategy = st.sampled_from(
    list(
        itertools.chain(
            itertools.permutations(range(3)),
            itertools.permutations((0, 1)),
            # itertools.permutations((0,2)),  # axes must be contiguous
            itertools.permutations((1, 2)),
            itertools.combinations(range(3), r=1),
            [None],
        )
    )
)


def is_axes_valid(a: np.ndarray, axes: tuple[int] | None, is_r2c: bool) -> bool:
    if axes is None:
        return True
    return all(
        [
            # axes must be in the range [0...N) where N is the number of dimensions
            all((n >= 0 and n < a.ndim) for n in axes),
            # axes must contain either the first or last dimension
            a.ndim - 1 in axes or 0 in axes,
            # the least significant dimension must be listed last for R2C,C2R
            (not is_r2c) or max(axes) == axes[-1],
            # FIXME: R2C only supports stride of 1 for last dimension?
            (not is_r2c) or max(axes) == a.ndim - 1,
        ]
    )


def verify_result(result, ref, orig, fft_type):
    assert result.dtype.name == dtype_dict[(fft_type, orig.dtype.name)]
    tol = 1e2 * np.finfo(orig.dtype).eps
    if np.linalg.norm(ref) == 0.0:
        assert np.linalg.norm(result - ref) < tol, f"error greater than tolerance for input shape {orig.shape}"
    else:
        assert (
            np.linalg.norm(result - ref) / np.linalg.norm(ref) < tol
        ), f"error greater than tolerance for input shape {orig.shape}"


@nvmath_seed()
@given(a=st.one_of(c32_array_st, c64_array_st), axes=axes_strategy, options=options_st, execution=execution_st)
def test_fft(a, axes, options, execution):
    if not is_axes_valid(a, axes, is_r2c=False):
        return
    try:
        b = nvmath.fft.fft(a, axes=axes, options=options, execution=execution)
    except cp.cuda.memory.OutOfMemoryError:
        # requiring too much GPU memory (>1GB), do nothing
        assert a.nbytes > 2**30, "suspicious OOM when requesting not too much GPU memory!"
        return
    except RuntimeError as e:
        if "The FFT CPU execution is not available" in str(e):
            assert (
                execution == "cpu"
                or isinstance(execution, nvmath.fft.ExecutionCPU)
                or (execution is None and isinstance(a, np.ndarray))
            )
            return
        raise e
    if execution == "cuda" or isinstance(execution, nvmath.fft.ExecutionCUDA):
        c = cp.asnumpy(cp.fft.fftn(cp.asarray(a), axes=axes, norm="backward"))
    else:
        c = scipy.fft.fftn(a, axes=axes, norm="backward")
    verify_result(b, c, a, "fft")


@nvmath_seed()
@given(a=st.one_of(c32_array_st, c64_array_st), axes=axes_strategy, options=options_st, execution=execution_st)
def test_ifft(a, axes, options, execution):
    if not is_axes_valid(a, axes, is_r2c=False):
        return
    try:
        b = nvmath.fft.ifft(a, axes=axes, options=options, execution=execution)
    except cp.cuda.memory.OutOfMemoryError:
        # requiring too much GPU memory (>1GB), do nothing
        assert a.nbytes > 2**30, "suspicious OOM when requesting not too much GPU memory!"
        return
    except RuntimeError as e:
        if "The FFT CPU execution is not available" in str(e):
            assert (
                execution == "cpu"
                or isinstance(execution, nvmath.fft.ExecutionCPU)
                or (execution is None and isinstance(a, np.ndarray))
            )
            return
        raise e
    if execution == "cuda" or isinstance(execution, nvmath.fft.ExecutionCUDA):
        c = cp.asnumpy(cp.fft.ifftn(cp.asarray(a), axes=axes, norm="forward"))
    else:
        c = scipy.fft.ifftn(a, axes=axes, norm="forward")
    verify_result(b, c, a, "ifft")


@nvmath_seed()
@given(a=st.one_of(f32_array_st, f64_array_st), axes=axes_strategy, options=options_st, execution=execution_st)
def test_rfft(a, axes, options, execution):
    if not is_axes_valid(a, axes, is_r2c=True):
        return
    try:
        b = nvmath.fft.rfft(a, axes=axes, options=options, execution=execution)
    except cp.cuda.memory.OutOfMemoryError:
        # requiring too much GPU memory (>1GB), do nothing
        assert a.nbytes > 2**30, "suspicious OOM when requesting not too much GPU memory!"
        return
    except RuntimeError as e:
        if "The FFT CPU execution is not available" in str(e):
            assert (
                execution == "cpu"
                or isinstance(execution, nvmath.fft.ExecutionCPU)
                or (execution is None and isinstance(a, np.ndarray))
            )
            return
        raise e
    if execution == "cuda" or isinstance(execution, nvmath.fft.ExecutionCUDA):
        c = cp.asnumpy(cp.fft.rfftn(cp.asarray(a), axes=axes, norm="backward"))
    else:
        c = scipy.fft.rfftn(a, axes=axes, norm="backward")
    verify_result(b, c, a, "rfft")


@nvmath_seed()
@given(a=st.one_of(f32_array_st, f64_array_st), axes=axes_strategy, options=options_st, execution=execution_st)
def test_irfft(a, axes, options, execution):
    if not is_axes_valid(a, axes, is_r2c=True):
        return
    # NOTE: Specifying output shape is the equivalent of `last_axis_parity` for scipy/numpy
    fft_shape = tuple(a.shape[e] for e in (range(a.ndim) if axes is None else axes))
    options["last_axis_parity"] = "odd" if fft_shape[-1] % 2 else "even"
    try:
        b = nvmath.fft.rfft(a, axes=axes, options=options, execution=execution)  # C2R needs complex-Hermitian input
        c = nvmath.fft.irfft(b, axes=axes, options=options, execution=execution)
    except cp.cuda.memory.OutOfMemoryError:
        # requiring too much GPU memory (>1GB), do nothing
        assert a.nbytes > 2**30, "suspicious OOM when requesting not too much GPU memory!"
        return
    except RuntimeError as e:
        if "The FFT CPU execution is not available" in str(e):
            assert (
                execution == "cpu"
                or isinstance(execution, nvmath.fft.ExecutionCPU)
                or (execution is None and isinstance(a, np.ndarray))
            )
            return
        raise e
    assert a.shape == c.shape, f"{a.shape} vs {c.shape}"
    if execution == "cuda" or isinstance(execution, nvmath.fft.ExecutionCUDA):
        c_ref = cp.asnumpy(cp.fft.irfftn(cp.asarray(b), s=fft_shape, axes=axes, norm="forward"))
    else:
        c_ref = scipy.fft.irfftn(b, s=fft_shape, axes=axes, norm="forward")
    verify_result(c, c_ref, b, "irfft")
