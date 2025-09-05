# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
import math

from nvmath.internal.utils import get_or_create_stream
from nvmath.internal.tensor_wrapper import maybe_register_package

import nvmath.internal.ndbuffer.package_utils as package_utils
import nvmath.internal.ndbuffer.ndbuffer as ndb
from nvmath.internal.ndbuffer.jit import _invalidate_kernel_cache
from .helpers import arange, zeros, permuted, assert_equal

import pytest
import numpy as np
import cuda.bindings.driver as cudadrv

try:
    import cupy as cp
except ImportError:
    pytest.skip("cupy is not installed", allow_module_level=True)


# ndbuffer uses asynchronous memory pool, let's use it in cupy
# too to decrease the amount of variables that impact the performance
# comment out this line to see performance difference that takes into
# account different allocation strategies
cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)


def get_l2_size(device_id):
    status, ret = cudadrv.cuDeviceGetAttribute(cudadrv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_id)
    assert status == 0, f"{status}"
    return ret


def flush_l2(device_id=0):
    l2_size = get_l2_size(device_id)
    n_floats = (l2_size + 3) // 4
    a = cp.empty(n_floats, dtype=cp.float32)
    a[:] = 42


def bench(
    device_id,
    callee,
    stream_holder,
    num_iters,
    warmup=5,
    sync_every_iter=False,
    include_compile_time=False,
):
    with cp.cuda.Device(device_id), stream_holder.ctx:
        l2_size = get_l2_size(device_id)
        n_floats = (l2_size + 3) // 4
        dummy = cp.empty(n_floats, dtype=cp.float32)

        start = cp.cuda.Event(disable_timing=False)
        end = cp.cuda.Event(disable_timing=False)
        for _ in range(warmup):
            callee()
            dummy[:] = 44

        if sync_every_iter:
            elapsed = 0
            for _ in range(num_iters):
                if include_compile_time:
                    _invalidate_kernel_cache()
                start.record(stream_holder.external)
                callee()
                end.record(stream_holder.external)
                stream_holder.external.synchronize()
                elapsed += cp.cuda.get_elapsed_time(start, end)
        else:
            stream_holder.external.synchronize()
            start.record(stream_holder.external)
            if include_compile_time:
                for _ in range(num_iters):
                    _invalidate_kernel_cache()
                    callee()
            else:
                for _ in range(num_iters):
                    callee()
            end.record(stream_holder.external)
            stream_holder.external.synchronize()
            elapsed = cp.cuda.get_elapsed_time(start, end)

    return elapsed / num_iters


def benchmark_case(device_id, direction, dst, src, stream_holder, show_logs=False, num_iters=10):
    if show_logs:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M:%S",
            force=True,
        )
    else:
        logger = None

    _stream = stream_holder.external
    match direction:
        case "h2d":
            _nd_dst = package_utils.wrap_cupy_array(dst)
            _nd_src = package_utils.wrap_numpy_array(src)

            def copy(_logger=None):
                ndb.copy_into(_nd_dst, _nd_src, stream=stream_holder, logger=_logger)

            def cupy_baseline():
                dst.set(src, stream=_stream)
                _stream.synchronize()

        case "d2d":
            _nd_dst = package_utils.wrap_cupy_array(dst)
            _nd_src = package_utils.wrap_cupy_array(src)

            def copy(_logger=None):
                ndb.copy_into(_nd_dst, _nd_src, stream=stream_holder, logger=_logger)

            def cupy_baseline():
                dst[:] = src

        case "d2h":
            _nd_dst = package_utils.wrap_numpy_array(dst)
            _nd_src = package_utils.wrap_cupy_array(src)

            def copy(_logger=None):
                ndb.copy_into(_nd_dst, _nd_src, stream=stream_holder, logger=_logger)

            def cupy_baseline():
                src.get(out=dst, stream=_stream)
                _stream.synchronize()

    with cp.cuda.Device(device_id):
        if show_logs:
            copy(logger)
        else:
            copy()
    _stream.synchronize()
    # test that the copy works as expected
    assert_equal(dst, src)

    time_cupy = bench(device_id, cupy_baseline, stream_holder, num_iters=num_iters)
    time_copy = bench(device_id, copy, stream_holder, num_iters=num_iters)
    return time_copy, time_cupy


def _copy_perf_case(device_id, stream_holder, shape, direction, dtype, perm, results):
    dst_device_id = "cpu" if direction == "d2h" else device_id
    src_device_id = "cpu" if direction == "h2d" else device_id
    with cp.cuda.Device(device_id), stream_holder.ctx:
        # Here we always permute src, which should be the worst-case scenario
        # for ndbuffer in comparison to cupy + numpy implementation,
        # because copy between cupy and numpy is not directly possible
        # if dst is not F or C, which requires additional logic in Python
        # to explicitly handle temporary copy followed by setting/getting
        # into the dst array
        dst = zeros(dst_device_id, stream_holder, permuted(shape, perm), dtype)
        src = arange(src_device_id, stream_holder, math.prod(shape), dtype).reshape(shape).transpose(perm)
    size_in_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    if size_in_bytes < 2**20:
        num_iters = 100
    else:
        num_iters = 10
    time_copy, time_cupy = benchmark_case(device_id, direction, dst, src, stream_holder, num_iters=num_iters)
    nd_cp_ratio = time_copy / time_cupy
    id_perm = tuple(range(len(shape)))
    if perm == id_perm:
        time_cupy_id = time_cupy
    else:
        time_cupy_id = results[shape, direction, dtype, id_perm]["time_cupy_id"]
    nd_id_ratio = time_copy / time_cupy_id
    cp_id_ratio = time_cupy / time_cupy_id
    assert (shape, direction, dtype, perm) not in results
    results[shape, direction, dtype, perm] = {
        "time_copy": time_copy,
        "time_cupy": time_cupy,
        "time_cupy_id": time_cupy_id,
        "nd_cp_ratio": nd_cp_ratio,
        "nd_id_ratio": nd_id_ratio,
        "cupy_id_ratio": cp_id_ratio,
    }


shapes = [
    (1,),
    (255,),
    (1023, 1023),
    (3, 1023 * 1023),
    (1023 * 1023, 3),
    (2, 3, 1023),
    (1023, 2, 3),
    (7, 1023, 511),
    (1023, 511, 3),
    (128, 128, 128),
    (255, 255, 255),
    (55, 55, 3, 3),
    (3, 3, 55, 55),
    (55, 55, 55, 13),
    (101, 101, 101, 101),
    (2,) * 25,
]

directions = ["d2d", "h2d", "d2h"]
dtypes = ["int8", "int16", "float32", "float64", "complex128"]


def test_copy_perf():
    device_id = 0
    with cp.cuda.Device(device_id):
        maybe_register_package("cupy")
        stream_holder = get_or_create_stream(device_id, cp.cuda.Stream(non_blocking=True), "cupy")
    results = {}

    print(
        f"Running test with {len(shapes)} shapes, {len(directions)} directions, "
        f"{len(dtypes)} dtypes, and different permutations"
    )
    print("time ndbuffer, time cupy - time a single copy took with ndbuffer and cupy respectively")
    print(
        "time cupy (base) - speed of light for a given shape and dtype, "
        "i.e. time a single copy took with cupy for non-permuted data"
    )
    print()
    for shape in shapes:
        for direction in directions:
            for dtype in dtypes:
                if len(shape) <= 4:
                    permutations = list(itertools.permutations(range(len(shape))))
                else:
                    ndim = len(shape)
                    permutations = [tuple(range(ndim)), tuple(reversed(range(ndim)))]
                    mid = ndim // 2
                    permutations.append(tuple(range(mid, ndim)) + tuple(range(mid)))
                for perm in permutations:
                    _copy_perf_case(device_id, stream_holder, shape, direction, dtype, perm, results)
            _shape_direction_summary(shape, direction, dtypes, permutations, results)


def _format_nd_cp_ratio(x, spec):
    if x > 1.01:
        # Mark unexpected slowdowns with double exclamation mark
        # For d2d copy we don't expect slowdowns compared to cupy
        # For d2h and h2d copy we don't expect slowdowns for big enough sizes,
        # but for small ones: cupy's approach to make permute-copy on the
        # host (using numpy) can play out better for some permutations.
        if spec["direction"] == "d2d" or (x > 1.5 or math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize >= 2**20):
            return f"(!!){x:.3f}"
        else:
            return f"(!){x:.3f}"
    elif x < 0.1:
        return f"(:o){x:.3f}"
    else:
        return f"{x:.3f}"


def _format_spec_elements(spec_element_name, spec_element_value):
    if (
        spec_element_name == "shape"
        and len(spec_element_value) >= 5
        and all(x == spec_element_value[0] for x in spec_element_value)
    ):
        return f"{(spec_element_value[0],)}*{len(spec_element_value)}"
    elif spec_element_name == "perm" and len(spec_element_value) >= 5:
        if spec_element_value == tuple(range(len(spec_element_value))):
            return "id"
        elif spec_element_value == tuple(reversed(range(len(spec_element_value)))):
            return "rev"
        else:
            return "custom"
    else:
        return str(spec_element_value)


def _print_results(
    results,
    spec_elements=None,
    cols=None,
    print_headers=True,
    col_widths=None,
    elipis_at=None,
):
    if spec_elements is None:
        spec_elements = ["shape", "direction", "dtype", "perm"]
    if cols is None:
        cols = [
            "nd_cp_ratio",
            "nd_id_ratio",
            "cupy_id_ratio",
            "time_copy",
            "time_cupy",
            "time_cupy_id",
        ]
    col_names = {
        "nd_cp_ratio": "ndbuffer / cupy",
        "nd_id_ratio": "ndbuffer / base",
        "cupy_id_ratio": "cupy / base",
        "time_copy": "time ndbuffer",
        "time_cupy": "time cupy",
        "time_cupy_id": "time cupy (base)",
    }
    cols_formatting = {
        "nd_cp_ratio": _format_nd_cp_ratio,
        "nd_id_ratio": lambda x, _: f"{x:.3f}",
        "cupy_id_ratio": lambda x, _: f"{x:.3f}",
        "time_copy": lambda x, _: f"{x:.6f}",
        "time_cupy": lambda x, _: f"{x:.6f}",
        "time_cupy_id": lambda x, _: f"{x:.6f}",
    }
    rows = []
    if print_headers:
        header = [", ".join(spec_elements)]
        for col in cols:
            header.append(col_names[col])
        rows.append(header)
    for spec, result in results:
        row = []
        row.append(", ".join(_format_spec_elements(spec_element, spec[spec_element]) for spec_element in spec_elements))
        for col in cols:
            row.append(cols_formatting[col](result[col], spec))
        rows.append(row)
    num_cols = len(rows[0])
    if col_widths is None:
        col_widths = [max(len(row[col]) for row in rows) for col in range(num_cols)]
    for i, row in enumerate(rows):
        if elipis_at is not None and i == elipis_at + int(print_headers):
            print("...")
        print(" | ".join(f"{element:<{element_width}}" for element, element_width in zip(row, col_widths, strict=True)))
    return col_widths


def _case_spec_as_dict(shape, direction, dtype, perm):
    return {
        "shape": shape,
        "direction": direction,
        "dtype": dtype,
        "perm": perm,
    }


def _shape_direction_summary(shape, direction, dtypes, permutations, results):
    shape_direction_results = [
        (
            _case_spec_as_dict(shape, direction, dtype, perm),
            results.get((shape, direction, dtype, perm)),
        )
        for dtype in dtypes
        for perm in permutations
    ]
    shape_direction_results = [(spec, result) for spec, result in shape_direction_results if result is not None]
    result_items = sorted(shape_direction_results, key=lambda x: x[1]["nd_cp_ratio"])
    tail_length = 5
    if len(result_items) <= 2 * tail_length:
        print(f"shape: {_format_spec_elements('shape', shape)}, direction: {direction}:")
        _print_results(result_items, spec_elements=["dtype", "perm"])
    else:
        print(f"shape: {_format_spec_elements('shape', shape)}, direction: {direction}:")
        _print_results(
            result_items[:tail_length] + result_items[-tail_length:],
            spec_elements=["dtype", "perm"],
            elipis_at=tail_length,
        )
    print()
