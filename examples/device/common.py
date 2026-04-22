# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from cuda.bindings import runtime as cudart

from nvmath.device.common_cuda import ComputeCapability
from nvmath.device.cusolverdx import Solver, _OrthogonalFactorizerProperties, _SolverProperties


def random_complex(shape, real_dtype, order="C") -> np.ndarray:
    return random_real(shape, real_dtype, order) + 1.0j * random_real(shape, real_dtype, order)


def random_real(shape, real_dtype, order="C") -> np.ndarray:
    # NOTE: reshape does not guarantee layout if order is provided. So we have
    #   to use copy.
    return np.random.randn(np.prod(shape)).astype(real_dtype).reshape(shape).copy(order=order)


def random_int(shape, int_dtype, order="C"):
    """
    Generate random integers in the range [-2, 2) for signed integers and [0, 4)
    for unsigned integers.
    """
    min_val, max_val = 0, 4
    if issubclass(int_dtype, np.signedinteger):
        min_val, max_val = -2, 2
    # NOTE: reshape does not guarantee layout if order is provided. So we have
    #   to use copy.
    return np.random.randint(min_val, max_val, size=shape, dtype=int_dtype).copy(order=order)


def random(shape, dtype, order=None, arrangement=None):
    assert order is None or arrangement is None, "Specify only one of order or arrangement"
    if arrangement is not None:
        order = "C" if arrangement == "row_major" else "F"
    if order is None:
        order = "C"
    if np.issubdtype(dtype, np.floating):
        return random_real(shape, dtype, order)
    elif np.issubdtype(dtype, np.complexfloating):
        return random_complex(shape, dtype, order)
    elif np.issubdtype(dtype, np.integer):
        return random_int(shape, dtype, order)


def CHECK_CUDART(err):
    if err != cudart.cudaError_t.cudaSuccess:
        err2, str = cudart.cudaGetErrorString(cudart.cudaError_t.cudaSuccess)
        raise RuntimeError(f"CUDArt Error: {str}")


def fft_perf_GFlops(fft_size, batch, time_ms, coef=1.0):
    fft_flops_per_batch = coef * 5.0 * fft_size * math.log2(fft_size)
    return batch * fft_flops_per_batch / (1e-3 * time_ms) / 1e9


def mm_perf_GFlops(size, batch, time_ms, coef=1.0):
    return coef * 2.0 * batch * size[0] * size[1] * size[2] / (1e-3 * time_ms) / 1e9


def fp16x2_to_complex64(data):
    return data[..., ::2] + 1.0j * data[..., 1::2]


def complex64_to_fp16x2(data):
    shape = (*data.shape[:-1], data.shape[-1] * 2)
    output = np.zeros(shape=shape, dtype=np.float16)
    output[..., 0::2] = data.real
    output[..., 1::2] = data.imag
    return output


def device_shared_memory(cc: ComputeCapability) -> int:
    # Source for these chip memory numbers:
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
    match cc.integer:
        case 1200 | 1210:
            return 99 * 1024
        case 900 | 1000 | 1010 | 1030 | 1100:
            return 227 * 1024
        case 890 | 860:
            return 99 * 1024
        case 800 | 870:
            return 163 * 1024
        case 750:
            return 64 * 1024
        case 700 | 720:
            return 96 * 1024
        case _:
            return 48 * 1024


def format_solver(solver: Solver | _SolverProperties | _OrthogonalFactorizerProperties) -> str:
    if isinstance(solver, _OrthogonalFactorizerProperties):
        return (
            f"Solver(size=({solver.size}), lda={solver.lda}, "
            + f"arr_a={solver.a_arrangement}, data_type={solver.data_type}, "
            + f"value_type={solver.value_type}, precision={solver.precision}, block_dim={solver.block_dim})"
        )

    return (
        f"Solver(size=({solver.size}), lda={solver.lda}, ldb={solver.ldb}, "
        + f"arr_a={solver.a_arrangement}, arr_b={solver.b_arrangement}, data_type={solver.data_type}, "
        + f"value_type={solver.value_type}, precision={solver.precision}, block_dim={solver.block_dim})"
    )


def verify_relative_error(data_test, data_ref, rel_err: float, abs_error: float, solver: Solver | _SolverProperties) -> bool:
    diff_norm = np.linalg.norm(data_test - data_ref)
    ref_norm = np.linalg.norm(data_ref)

    if diff_norm > ref_norm * rel_err + abs_error:
        solver_msg = format_solver(solver)
        msg = (
            "Failed: | data_test - data_ref | > | data_ref | * rel_err + abs_error with: "
            + f"| data_test - data_ref | = {diff_norm}, | data_ref | = {ref_norm}, "
            + f"| data_ref | * rel_err + abs_error = {ref_norm * rel_err + abs_error}"
            + f" with solver: {solver_msg}"
        )
        raise RuntimeError(msg)

    return diff_norm / ref_norm if ref_norm != 0.0 else 0.0


def verify_triangular_relative_error(
    data_test, data_ref, rel_err: float, abs_error: float, solver: _SolverProperties, fill_mode: str = "upper"
) -> bool:
    assert fill_mode in {"upper", "lower"}
    assert data_test.shape == data_ref.shape
    assert data_test.ndim == 2 and data_ref.ndim == 2

    rows, cols = data_test.shape
    if fill_mode == "upper":
        mask = np.triu(np.ones((rows, cols), dtype=bool))
    else:
        mask = np.tril(np.ones((rows, cols), dtype=bool))

    data_test_tri = np.where(mask, data_test, 0)
    data_ref_tri = np.where(mask, data_ref, 0)

    return verify_relative_error(
        data_test_tri,
        data_ref_tri,
        rel_err,
        abs_error,
        solver,
    )


def prepare_random_matrix(
    shape: tuple[int, int, int],
    dtype=np.float64,
    is_complex=False,
    is_hermitian=False,
    is_diag_dom=False,
    is_positive_definite=False,
):
    assert len(shape) == 3

    data = random_complex(shape, dtype, "C") if is_complex else random_real(shape, dtype, "C")
    if is_positive_definite:
        is_hermitian = True

    _, m, n = shape

    if is_hermitian:
        hermitian_size = min(m, n)
        row_idx, col_idx = np.triu_indices(hermitian_size, k=1)
        data[:, col_idx, row_idx] = np.conj(data[:, row_idx, col_idx])

        if is_complex:
            diag_idx = np.arange(hermitian_size)
            data[:, diag_idx, diag_idx] = data[:, diag_idx, diag_idx].real

    if is_positive_definite:
        data = data @ np.ascontiguousarray(data.conj().transpose(0, 2, 1))

    if is_diag_dom:
        diag_size = min(m, n)
        diag_idx = np.arange(diag_size)
        row_abs_sum = np.sum(np.abs(data[:, :diag_size, :]), axis=-1)
        data[:, diag_idx, diag_idx] = 2.0 * row_abs_sum + 5.0

    return data
