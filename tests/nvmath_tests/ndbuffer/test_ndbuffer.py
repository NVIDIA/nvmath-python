# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import math
import itertools
import random
import logging

from nvmath.internal import tensor_wrapper
import pytest
import cuda.core.experimental as ccx
import nvmath.internal.tensor_ifc_ndbuffer as tndb
import nvmath.internal.ndbuffer.ndbuffer as ndb
from nvmath.internal.utils import device_ctx
from .helpers import (
    np,
    cp,
    Param,
    _SL,
    idfn,
    assert_equal,
    as_ndbuffer,
    sliced_or_broadcast_1d,
    stride_tricks,
    arange,
    zeros,
    random_non_empty_slice,
    random_negated_strides,
    inv,
    permuted,
    dense_c_strides,
    abs_strides,
    as_array,
    create_stream,
    free_memory,
    wrap_operand,
)


def _permutations(rng, ndim, sample_size=10):
    if ndim <= 4:
        return list(itertools.permutations(range(ndim)))
    elif ndim <= 7:
        return rng.sample(list(itertools.permutations(range(ndim))), sample_size)
    else:
        p_id = tuple(range(ndim))
        p_reverse = tuple(reversed(range(ndim)))
        return [p_id, p_reverse]


def _shuffled(rng, l):
    l = list(l)
    rng.shuffle(l)
    return l


def _shape(rng, ndim):
    if ndim <= 9:
        return tuple(range(2, 2 + ndim))
    else:
        non_ones = rng.sample(list(range(ndim)), min(20, ndim))
        shape = [1] * ndim
        for i in non_ones:
            shape[i] = 2
        return tuple(shape)


def _empty_shape(rng, ndim):
    shape = [0] * ndim
    num_non_zero = rng.randint(max(0, ndim - 2), ndim - 1)
    non_zero_indices = rng.sample(range(ndim), num_non_zero)
    for i in non_zero_indices:
        shape[i] = rng.randint(1, 2 ** (63 // ndim))
    assert math.prod(shape) == 0
    return tuple(shape)


py_rng = random.Random(42)

dtypes = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


@pytest.mark.parametrize(
    (
        "ndim",
        "device_id",
        "dtype",
        "shape",
    ),
    [
        (
            Param("ndim", ndim),
            Param("device_id", device_id),
            Param("dtype", dtype),
            Param("shape", shape),
        )
        for ndim in [1, 2, 3, 4, 5, 31]
        for device_id in ["cpu", 0]
        for dtype in [py_rng.choice(dtypes)]
        for shape in [_empty_shape(py_rng, ndim)]
    ],
    ids=idfn,
)
def test_empty_tensor(ndim, shape, device_id, dtype):
    ndim = ndim.value
    shape = shape.value
    device_id = device_id.value
    dtype = dtype.value
    if cp is None and device_id != "cpu":
        pytest.skip("Cupy is required to run this test")
    stream_holder = create_stream(device_id)
    nd_device_id = ndb.CPU_DEVICE_ID if device_id == "cpu" else device_id
    ndbuffer = ndb.empty(
        shape=shape,
        dtype_name=dtype,
        itemsize=1,
        device_id=nd_device_id,
        stream=stream_holder,
    )
    assert ndbuffer.shape == shape
    assert ndbuffer.strides == tuple(0 for _ in range(ndim))
    assert ndbuffer.strides_in_bytes == tuple(0 for _ in range(ndim))
    assert ndbuffer.size_in_bytes == 0
    assert ndbuffer.size == 0
    assert ndbuffer.device_id == device_id
    assert ndbuffer.data_ptr == 0

    a = zeros(device_id, stream_holder, shape, dtype)
    src = wrap_operand(a)
    dst_device_id = 0 if device_id == "cpu" else "cpu"
    dst = src.to(dst_device_id, stream_holder=stream_holder)
    assert dst.shape == shape
    assert dst.tensor.strides_in_bytes == a.strides
    assert dst.tensor.size_in_bytes == 0
    assert dst.tensor.size == 0
    assert dst.device_id == dst_device_id
    assert dst.tensor.data_ptr == 0


def test_size_overflow():
    with pytest.raises(OverflowError):
        ndb.empty(shape=(2**31, 2**29, 13), dtype_name="int8", itemsize=1, device_id=0)
    with pytest.raises(OverflowError):
        ndb.empty(shape=(2**31, 2**29, 3), dtype_name="float32", itemsize=4, device_id=0)


@pytest.mark.parametrize(
    (
        "volume",
        "stride",
        "dtype",
        "direction",
    ),
    [
        (
            Param("volume", volume),
            Param("stride", stride),
            Param("dtype", dtype),
            Param("direction", direction),
        )
        for volume in [1, 13, 1024]
        for stride in [0, 1, -1, 2, -2, 3, -3]
        for dtype in ["int8", "int16", "float32", "complex64", "complex128"]
        for direction in ["h2d", "d2d", "d2h"]
    ],
    ids=idfn,
)
def test_1d_copy_src_strides(volume, stride, dtype, direction):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    volume = volume.value
    stride = stride.value
    direction = direction.value
    dtype = dtype.value
    device_id = 0
    stream_holder = create_stream(device_id)
    src_device_id = "cpu" if direction == "h2d" else device_id
    out_device_id = "cpu" if direction == "d2h" else device_id
    a = sliced_or_broadcast_1d(src_device_id, stream_holder, volume, stride, dtype)
    src = wrap_operand(a)
    out = wrap_operand(zeros(out_device_id, stream_holder, a.shape, dtype))
    with device_ctx(device_id):
        ndb.copy_into(out.asndbuffer(), src.asndbuffer(), stream=stream_holder)
    print(
        f"\nshape={out.shape} = {src.shape}, strides={out.strides} <- {src.strides},"
        f"device_id={out.device_id} <- {src.device_id}"
    )
    if direction == "d2d":
        stream_holder.obj.sync()
    assert_equal(out.tensor, src.tensor)


@pytest.mark.parametrize(
    (
        "volume",
        "stride",
        "dtype",
        "direction",
    ),
    [
        (
            Param("volume", volume),
            Param("stride", stride),
            Param("dtype", dtype),
            Param("direction", direction),
        )
        for volume in [1, 2555]
        for stride in [0, 1, -1, 2, 49, -49]
        for dtype in ["uint8", "uint16", "int32", "float64", "complex128"]
        for direction in ["h2d", "d2d", "d2h"]
    ],
    ids=idfn,
)
def test_1d_copy_dst_strides(volume, stride, dtype, direction):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    volume = volume.value
    stride = stride.value
    direction = direction.value
    dtype = dtype.value
    device_id = 0
    stream_holder = create_stream(device_id)
    src_device_id = "cpu" if direction == "h2d" else device_id
    out_device_id = "cpu" if direction == "d2h" else device_id
    with stream_holder.ctx:
        out = wrap_operand(sliced_or_broadcast_1d(out_device_id, stream_holder, volume, stride, dtype))
        out.tensor[:] = 0
        src = wrap_operand(arange(src_device_id, stream_holder, math.prod(out.tensor.shape), dtype))
    print(
        f"\nshape={out.shape} = {src.shape}, strides={out.strides} <- {src.strides},"
        f"device_id={out.device_id} <- {src.device_id}"
    )
    if volume > 1 and stride == 0:
        with pytest.raises(ValueError, match="could overlap in memory"):  # noqa: SIM117
            with device_ctx(device_id):  # noqa: SIM117
                ndb.copy_into(out.asndbuffer(), src.asndbuffer(), stream=stream_holder)
        return
    with device_ctx(device_id):
        ndb.copy_into(out.asndbuffer(), src.asndbuffer(), stream=stream_holder)
    if direction == "d2d":
        stream_holder.obj.sync()
    assert_equal(out.tensor, src.tensor)


@pytest.mark.parametrize(
    (
        "ndim",
        "shape",
        "permutation",
        "slice",
        "negate",
        "direction",
        "device_id",
        "dtype",
    ),
    [
        (
            Param("ndim", ndim),
            Param("shape", shape),
            Param("permutation", permutation),
            Param("slice", slice),
            Param("negate", negate),
            Param("direction", direction),
            Param("device_id", device_id),
            Param(
                "dtype",
                py_rng.choice(dtypes),
            ),
        )
        for ndim in [2, 3, 4, 5, 7, 13, 21, 32]
        for shape in [_shape(py_rng, ndim)]
        for permutation in _permutations(py_rng, ndim)
        for slice, negate in [(False, False), (True, False), (True, True)]
        for direction in ["d2h", "h2d"]
        for device_id in _shuffled(py_rng, [0, 1])
    ],
    ids=idfn,
)
def test_layout_preservation(ndim, shape, permutation, slice, negate, direction, device_id, dtype):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    ndim = ndim.value
    shape = shape.value
    permutation = permutation.value
    direction = direction.value
    dtype = dtype.value
    device_id = device_id.value

    if device_id > 0 and ccx.system.num_devices < 2:
        pytest.skip("Test requires at least 2 gpus")

    src_device_id = "cpu" if direction == "h2d" else device_id
    stream_holder = create_stream(device_id)

    a_base = arange(src_device_id, stream_holder, math.prod(shape), dtype)
    a_base = a_base.reshape(shape)
    a_base_strides = a_base.strides
    a = a_base
    if slice:
        a = random_non_empty_slice(py_rng, a)
    if negate:
        a = random_negated_strides(py_rng, a)
    a = a.transpose(permutation)
    assert abs_strides(a.strides) == permuted(a_base_strides, permutation)
    assert math.prod(a.shape) > 0
    if slice:
        assert math.prod(a.shape) != math.prod(a_base.shape)
    else:
        assert math.prod(a.shape) == math.prod(a_base.shape)
    src = wrap_operand(a)
    if direction == "h2d":
        dst = src.to(device_id=device_id, stream_holder=stream_holder)
        assert src.device_id == "cpu"
        assert dst.device_id == device_id
    else:
        assert direction == "d2h"
        dst = src.to(device_id="cpu", stream_holder=stream_holder)
        assert src.device_id == device_id
        assert dst.device_id == "cpu"
    print(
        f"shape={dst.shape} = {src.shape}, strides={dst.strides} <- {src.strides}, device_id={dst.device_id} <- {src.device_id}"
    )
    b = as_array(dst.tensor)
    if not slice and not negate:
        assert b.strides == a.strides
    else:
        expected = permuted(
            dense_c_strides(permuted(a.shape, inv(permutation)), a.itemsize),
            permutation,
        )
        b_strides = abs_strides(b.strides) if negate else b.strides
        assert b_strides == expected, f"{b_strides} != {expected}"
    assert_equal(b, a)


@pytest.mark.parametrize(
    (
        "shape",
        "transformation",
        "direction",
        "device_id",
        "dtype",
        "num_threads",
        "use_barrier",
    ),
    [
        (
            Param("shape", shape),
            Param("transformation", transformation),
            Param("direction", direction),
            Param("device_id", device_id),
            Param(
                "dtype",
                py_rng.choice(dtypes),
            ),
            Param("num_threads", num_threads),
            Param("use_barrier", use_barrier),
        )
        for shape in [(51,), (1024, 1023), (101, 101, 101)]
        for transformation in ["id", "slice", "reverse"]
        for direction in ["d2h", "d2d", "h2d"]
        for device_id in [0, 1]
        for num_threads in [1, 2, 16]
        for use_barrier in [True, False]
    ],
    ids=idfn,
)
def test_multithreaded(shape, transformation, direction, device_id, dtype, num_threads, use_barrier):
    import threading
    from io import StringIO

    if cp is None:
        pytest.skip("Cupy is required to run this test")
    shape = shape.value
    transformation = transformation.value
    direction = direction.value
    dtype = dtype.value
    device_id = device_id.value
    num_threads = num_threads.value

    if device_id > 0 and ccx.system.num_devices < 2:
        pytest.skip("Test requires at least 2 gpus")

    if use_barrier:
        # artificially increase contention for the caches in the ndbuffer code
        barier = threading.Barrier(num_threads)
    else:
        barier = None

    def copy_(thread_id, thread_data):
        try:
            for i in range(3):
                logger_name = f"ndbuffer_test_multithreaded_{thread_id}"
                log_stream = StringIO()
                logger = logging.Logger(logger_name, level=logging.DEBUG)
                logger.addHandler(logging.StreamHandler(log_stream))
                logger.setLevel(logging.DEBUG)
                stream_holder = create_stream(device_id)
                src_device_id = "cpu" if direction == "h2d" else device_id
                a_base = arange(src_device_id, stream_holder, math.prod(shape), dtype)
                a_base = a_base.reshape(shape)
                if transformation == "id":
                    a = a_base
                elif transformation == "slice":
                    a = a_base[((slice(None),) * (len(shape) - 1)) + (slice(None, None, -1),)]
                elif transformation == "reverse":
                    a = a_base.transpose(tuple(reversed(range(len(shape)))))
                else:
                    raise ValueError(f"Invalid transformation: {transformation}")
                src_wrapper = wrap_operand(a)
                dst_device_id = ndb.CPU_DEVICE_ID if direction == "d2h" else device_id
                if use_barrier:
                    barier.wait()
                nd_dst = ndb.empty(a.shape, dst_device_id, dtype, np.dtype(dtype).itemsize, stream=stream_holder)
                if use_barrier:
                    barier.wait()
                with device_ctx(device_id):
                    ndb.copy_into(nd_dst, src_wrapper.asndbuffer(), stream=stream_holder, logger=logger)
                    if direction == "d2d":
                        stream_holder.obj.sync()
                logs = log_stream.getvalue()
                launched_kernel = "Launching elementwise copy kernel" in logs or "Launching transpose copy kernel" in logs
                if launched_kernel:
                    if i == 0:
                        assert "Registered copy kernel includes" in logs
                    else:
                        assert "Registered copy kernel includes" not in logs, logs
                if "Compiling kernel" in logs:
                    thread_data["compiled"] += 1
                assert_equal(as_array(nd_dst), a)

        except Exception as e:
            thread_data["exception"] = e
            raise

    threads = []
    thread_data = [{"exception": None, "compiled": 0} for _ in range(num_threads)]
    for i in range(num_threads):
        t = threading.Thread(target=copy_, args=(i, thread_data[i]))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(num_threads):
        if thread_data[i]["exception"] is not None:
            raise AssertionError(f"Thread {i} failed") from thread_data[i]["exception"]
    total_compilations = sum(thread_data["compiled"] for thread_data in thread_data)
    assert total_compilations <= 1, f"total_compilations={total_compilations}"

    if direction != "d2h":
        import nvmath.internal.memory

        pool = nvmath.internal.memory.get_device_current_memory_pool(device_id)
        reserved_memory = pool.get_reserved_memory_size()
        with device_ctx(device_id) as device:
            device.sync()
            nvmath.internal.memory.free_reserved_memory()
            reserved_memory_after = pool.get_reserved_memory_size()
            assert reserved_memory_after < reserved_memory, (
                f"reserved_memory_after={reserved_memory_after} >= reserved_memory={reserved_memory}"
            )


@pytest.mark.parametrize(
    (
        "shape",
        "slice",
        "dtype",
        "needs_wide_strides",
        "transpose",
    ),
    [
        (
            Param("shape", shape),
            Param("slice", slice),
            Param("dtype", dtype),
            Param("needs_wide_strides", needs_wide_strides),
            Param("transpose", transpose),
        )
        for shape, slice, dtype, needs_wide_strides in [
            # this is a nice edge case:
            # 1. depending on the dtype max offset does or doesn't exceed INT_MAX
            # 2. the dot(shape - 1, strides) is less than INT_MAX but
            # the dot(shape, strides) is bigger than INT_MAX
            ((3, 2**24 + 1, 33), _SL[:, ::999, :], "int8", False),
            ((3, 2**24 + 1, 33), _SL[::-1, ::-999, ::-1], "int8", False),
            ((3, 2**24 + 1, 33), _SL[:, ::999, :], "int16", False),
            ((3, 2**24 + 1, 33), _SL[::-1, ::-999, ::-1], "int16", False),
            # volume and dot(shape, strides) exceed INT_MAX
            # but the actual max offset not
            ((1, 3, 715827883), _SL[:, ::-1, -19:], "int8", False),
            ((1, 3, 715827883), _SL[:, :, -19:], "int8", False),
            ((1, 3, 715827883), _SL[::-1, :, -19:], "int8", False),
            ((1, 3, 715827883), _SL[::-1, ::-1, 18::-1], "int8", False),
            # offset really exceeds INT_MAX (while sliced volume is still small)
            ((1, 4, 715827883), _SL[:, ::-1, -19:], "int8", True),
            ((1, 4, 715827883), _SL[:, :, -19:], "int8", True),
            ((1, 4, 715827883), _SL[::-1, :, -19:], "int8", True),
            ((1, 4, 715827883), _SL[::-1, ::-1, 18::-1], "int8", True),
            # like above but split 4 into 2x2 and check if wide strides
            # are used iff the strides have the same sign
            ((2, 2, 715827883), _SL[:, :, -19:], "int8", True),
            ((2, 2, 715827883), _SL[::-1, :, -19:], "int8", False),
            ((2, 2, 715827883), _SL[:, ::-1, -19:], "int8", False),
            ((2, 2, 715827883), _SL[::-1, ::-1, -19:], "int8", True),
        ]
        for transpose in [False, True]
    ],
    ids=idfn,
)
def test_wide_strides_small_volume_copy(caplog, shape, slice, dtype, needs_wide_strides, transpose):
    # test that wide strides are used when needed due to big offsets of the elements
    # (even when the volume is small)
    logger_name = "ndbuffer_test_wide_strides_copy"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if cp is None:
        pytest.skip("Cupy is required to run this test")

    free_memory()
    stream_holder = create_stream(0)

    shape = shape.value
    dtype = dtype.value
    slice = slice.value
    device_id = 0
    a = arange(device_id, stream_holder, math.prod(shape), dtype).reshape(shape)[slice]
    if transpose:
        a = a.transpose(0, 2, 1)
    b = zeros(device_id, stream_holder, a.shape, dtype)
    aw = wrap_operand(a)
    bw = wrap_operand(b)
    print(f"copy shape={bw.shape}: strides={bw.strides} <- {aw.strides}")
    caplog.clear()
    with device_ctx(device_id):  # noqa: SIM117
        with caplog.at_level(logging.DEBUG, logger_name):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder, logger=logger)
    log_text = caplog.text
    if needs_wide_strides:
        assert "TRANSPOSE_KERNEL(int64_t" in log_text or "ELEMENTWISE_KERNEL(int64_t" in log_text
    else:
        assert "TRANSPOSE_KERNEL(int32_t" in log_text or "ELEMENTWISE_KERNEL(int32_t" in log_text
    stream_holder.obj.sync()
    assert_equal(b, a)


@pytest.mark.parametrize(
    (
        "shape",
        "slice",
        "permutation",
        "dtype",
    ),
    [
        (
            Param("shape", shape),
            Param("slice", slice),
            Param("permutation", permutation),
            Param("dtype", dtype),
        )
        for shape, slice, permutation in [
            # 2**31 - 127 factorized, respectively sliced or transposed
            # to enforce elementwise and transpose kernels usage
            ((53, 419, 96703), (_SL[:, :, ::-1]), (0, 1, 2)),
            ((53, 419, 96703), (_SL[:, :, :]), (2, 1, 0)),
            # 2**32 - 127 factorized
            ((3, 23, 347, 179383), (_SL[:, ::-1, :, :]), (0, 1, 2, 3)),
            ((3, 23, 347, 179383), (_SL[:, :, :, :]), (3, 2, 1, 0)),
            # 4/3 * (2**32 - 1) factorized
            ((5, 4, 17, 257, 65537), (_SL[:, :, :, ::-1, :]), (0, 1, 2, 3, 4)),
            ((5, 4, 17, 257, 65537), (_SL[:, :, :, :, :]), (4, 3, 2, 1, 0)),
        ]
        for dtype in ["int8"]
    ],
    ids=idfn,
)
def test_wide_strides_large_volume_copy(caplog, shape, slice, permutation, dtype):
    # test that kernels properly compute offsets when the 64-bit strides are needed
    # this test uses large volumes to make sure that computing flat index and unravelling
    # it to ndim-coordinates does not overflow
    # NOTE, this test is slow
    logger_name = "ndbuffer_test_wide_strides_large_volume_copy"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    shape = shape.value
    dtype = dtype.value
    slice = slice.value
    permutation = permutation.value
    device_id = 0

    if cp is None:
        pytest.skip("Cupy is required to run this test")
    # we need to allocate src, dst and take into account that
    # cp testing assertion for tensor equality may copy the tensors
    # as well (likely due to their sliced/permuted layouts)
    if cp.cuda.Device(device_id).mem_info[1] < 4.1 * math.prod(shape) * np.dtype(dtype).itemsize:
        pytest.skip("Not enough memory to run the test")

    with cp.cuda.Device(device_id):
        cp.cuda.Device(device_id).synchronize()
        free_memory()
        stream_holder = create_stream(device_id)

    a = arange(device_id, stream_holder, math.prod(shape), dtype).reshape(shape)[slice].transpose(permutation)
    b = zeros(device_id, stream_holder, a.shape, dtype)
    aw = wrap_operand(a)
    bw = wrap_operand(b)
    print(f"copy shape={bw.shape}: strides={bw.strides} <- {aw.strides}")
    caplog.clear()
    with device_ctx(device_id):  # noqa: SIM117
        with caplog.at_level(logging.DEBUG, logger_name):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder, logger=logger)
    log_text = caplog.text
    assert "TRANSPOSE_KERNEL(int64_t" in log_text or "ELEMENTWISE_KERNEL(int64_t" in log_text
    stream_holder.obj.sync()
    assert_equal(b, a)


def test_unsupported_ndim():
    with pytest.raises(ValueError, match="Max supported ndim is 32"):
        ndb.empty(shape=(1,) * 33, dtype_name="int8", itemsize=1, device_id=ndb.CPU_DEVICE_ID)
    with pytest.raises(ValueError, match="Max supported ndim is 32"):
        wrap_operand(np.zeros(shape=(1,) * 34, dtype="float32")).asndbuffer()


@pytest.mark.parametrize(
    (
        "shape_a",
        "shape_b",
        "dtype",
    ),
    [
        (
            Param("shape_a", shape_a),
            Param("shape_b", shape_b),
            Param("dtype", dtype),
        )
        for shape_a, shape_b in [
            ((1, 2, 3), (1, 3, 2)),
            ((4,), (1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1)),
        ]
        for dtype in [py_rng.choice(dtypes)]
    ],
    ids=idfn,
)
def test_mismatched_shape(shape_a, shape_b, dtype):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    device_id = 0
    stream_holder = create_stream(device_id)
    shape_a = shape_a.value
    shape_b = shape_b.value
    dtype = dtype.value
    a = zeros("cpu", None, shape_a, dtype)
    b = zeros(device_id, stream_holder, shape_b, dtype)
    aw = wrap_operand(a)
    bw = wrap_operand(b)
    msg = "The shapes of the source and destination buffers must match"
    with device_ctx(device_id):
        with pytest.raises(ValueError, match=msg):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder)
        with pytest.raises(ValueError, match=msg):
            ndb.copy_into(aw.asndbuffer(), bw.asndbuffer(), stream=stream_holder)


@pytest.mark.parametrize(
    (
        "shape",
        "itemsize",
        "dtype",
        "msg",
    ),
    [
        (
            Param("shape", shape),
            Param("itemsize", itemsize),
            Param("dtype", dtype),
            Param("msg", msg),
        )
        for shape, itemsize, msg in [
            ((1, -2, 3), 1, "extents must be non-negative"),
            ((4,), -2, "itemsize must be positive"),
            ((4,), 3, "itemsize must be a power of two"),
        ]
        for dtype in [py_rng.choice(dtypes)]
    ],
    ids=idfn,
)
def test_empty_ndbuffer_wrong_shape(shape, itemsize, dtype, msg):
    shape = shape.value
    itemsize = itemsize.value
    dtype = dtype.value
    msg = msg.value
    device_id = 0
    with pytest.raises(ValueError, match=msg):
        ndb.empty(shape=shape, dtype_name=dtype, itemsize=itemsize, device_id=device_id)


def test_mismatched_dtype():
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    device_id = 0
    stream_holder = create_stream(device_id)
    a = zeros("cpu", None, (1, 2, 3), np.int32)
    b = zeros(device_id, stream_holder, (1, 2, 3), np.int64)
    aw = wrap_operand(a)
    bw = wrap_operand(b)
    msg = "The data types of the source and destination buffers must match"
    with device_ctx(device_id):
        with pytest.raises(ValueError, match=msg):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder)
        with pytest.raises(ValueError, match=msg):
            ndb.copy_into(aw.asndbuffer(), bw.asndbuffer(), stream=stream_holder)


@pytest.mark.parametrize(
    (
        "shape",
        "dtype",
        "expected_itemsize",
        "transpose",
        "device_id",
    ),
    [
        (
            Param("shape", shape),
            Param("dtype", dtype),
            Param("expected_itemsize", expected_itemsize),
            Param("transpose", transpose),
            Param("device_id", device_id),
        )
        for shape, dtype, expected_itemsize in [
            ((2, 255, 4), "int8", 4),
            ((2, 255, 4), "int16", 8),
            ((2, 255, 4), "float32", 8),
            ((2, 255, 4), "float64", 8),
            ((2, 255, 4), "complex128", 16),
            ((2, 255, 6), "int8", 2),
            ((2, 255, 6), "int16", 4),
            ((2, 255, 6), "float32", 8),
            ((2, 255, 6), "float64", 8),
            ((2, 255, 6), "complex128", 16),
            ((2, 255, 3), "int8", 1),
            ((2, 255, 3), "int16", 2),
            ((2, 255, 3), "float32", 4),
            ((2, 255, 3), "float64", 8),
            ((2, 255, 3), "complex128", 16),
        ]
        for transpose in [False, True]
        for device_id in [1, 0]
    ],
    ids=idfn,
)
def test_vectorized_copy(caplog, shape, dtype, expected_itemsize, transpose, device_id):
    logger_name = "ndbuffer_test_wide_strides_copy"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if cp is None:
        pytest.skip("Cupy is required to run this test")

    shape = shape.value
    dtype = dtype.value
    expected_itemsize = expected_itemsize.value
    device_id = device_id.value
    if device_id > 0 and ccx.system.num_devices < 2:
        pytest.skip("Test requires at least 2 gpus")

    stream_holder = create_stream(device_id)
    a_base = arange(device_id, stream_holder, math.prod(shape), dtype).reshape(shape)
    a = a_base[:, :-1, :]  # take a slice so that plain memcopy is not used
    b = zeros(device_id, stream_holder, a.shape, dtype)
    if transpose:
        a = a.transpose(2, 1, 0)
        b = b.transpose(2, 1, 0)
    aw = wrap_operand(a)
    bw = wrap_operand(b)
    with device_ctx(device_id):
        with caplog.at_level(logging.DEBUG, logger_name):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder, logger=logger)
            if expected_itemsize == np.dtype(dtype).itemsize:
                assert "Could not vectorize the copy" in caplog.text
            else:
                assert f"itemsize={expected_itemsize}" in caplog.text
        stream_holder.obj.sync()
        assert_equal(b, a)


@pytest.mark.parametrize(
    (
        "ndim",
        "shape",
        "permutation",
        "dtype",
    ),
    [
        (
            Param("ndim", ndim),
            Param("shape", shape),
            Param("permutation", permutation),
            Param("dtype", dtype),
        )
        for ndim in [2, 3, 4]
        for shape in [_shape(py_rng, ndim)]
        for permutation in _permutations(py_rng, ndim)
        for dtype in [py_rng.choice(dtypes)]
    ],
    ids=idfn,
)
def test_permuted_dense_strides_are_memcopied(caplog, ndim, shape, permutation, dtype):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    ndim = ndim.value
    shape = shape.value
    permutation = permutation.value
    dtype = dtype.value
    device_id = 0
    stream_holder = create_stream(device_id)
    a = arange(device_id, stream_holder, math.prod(shape), dtype).reshape(shape)
    a = a.transpose(permutation)
    b = zeros(device_id, stream_holder, shape, dtype)
    b = b.transpose(permutation)
    aw = wrap_operand(a)
    bw = wrap_operand(b)
    logger_name = "ndbuffer_test_permuted_dense_strides"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    caplog.clear()
    with device_ctx(device_id):  # noqa: SIM117
        with caplog.at_level(logging.DEBUG, logger_name):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder, logger=logger)
    log_text = caplog.text
    assert "can memcpy" in log_text
    stream_holder.obj.sync()
    assert_equal(b, a)


@pytest.mark.parametrize(
    (
        "base_shape",
        "broadcast_shape",
        "broadcast_strides",
        "dtype",
        "direction",
    ),
    [
        (
            Param("base_shape", base_shape),
            Param("broadcast_shape", broadcast_shape),
            Param("broadcast_strides", broadcast_strides),
            Param("dtype", py_rng.choice(dtypes)),
            Param("direction", direction),
        )
        for base_shape, broadcast_shape, broadcast_strides in [
            # broadcast
            ((1,), (7, 255, 3), (0, 0, 0)),
            ((255, 1), (255, 3), (1, 0)),
            ((1, 3), (255, 3), (0, 1)),
            ((6, 1, 12), (6, 3, 12), (6, 0, 1)),
            ((1, 1, 12), (6, 3, 12), (0, 0, 1)),
            ((1, 1, 12), (3, 6, 12), (0, 0, 1)),
            # broadcast and permute
            ((6, 1, 12), (12, 3, 6), (1, 0, 6)),
            ((1024,), (1024, 1024), (0, 1)),
            ((1024,), (1024, 1024), (1, 0)),
            # sliding window
            ((10,), (2, 7), (3, 1)),
            ((10,), (7, 2), (1, 3)),
        ]
        for direction in ["h2d", "d2d", "d2h"]
    ],
    ids=idfn,
)
def test_broadcast_copy(base_shape, broadcast_shape, broadcast_strides, dtype, direction):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    base_shape = base_shape.value
    broadcast_shape = broadcast_shape.value
    broadcast_strides = broadcast_strides.value
    dtype = dtype.value
    direction = direction.value
    stream_holder = create_stream(0)
    device_id = 0
    src_device_id = "cpu" if direction == "h2d" else device_id
    dst_device_id = "cpu" if direction == "d2h" else device_id
    a_base = arange(src_device_id, stream_holder, math.prod(base_shape), dtype).reshape(base_shape)
    a = stride_tricks(a_base, broadcast_shape, broadcast_strides, a_base.itemsize)
    aw = wrap_operand(a)
    if direction == "h2d":
        assert aw.device_id == "cpu"
        bw = aw.to(device_id=0, stream_holder=stream_holder)
        b = as_array(bw.tensor)
        print(f"\nh2d copy, shape={bw.shape}, strides={bw.strides} <- {aw.strides}")
    elif direction == "d2h":
        assert aw.device_id == 0
        bw = aw.to(device_id="cpu", stream_holder=stream_holder)
        b = as_array(bw.tensor)
        print(f"\nd2h copy, shape={bw.shape}, strides={bw.strides} <- {aw.strides}")
    else:
        assert direction == "d2d"
        assert aw.device_id == 0
        b = zeros(dst_device_id, stream_holder, broadcast_shape, dtype)
        bw = wrap_operand(b)
        with device_ctx(device_id):
            ndb.copy_into(bw.asndbuffer(), aw.asndbuffer(), stream=stream_holder)
        print(f"\nd2d copy, shape={bw.shape}, strides={bw.strides} <- {aw.strides}")
        stream_holder.obj.sync()
    assert_equal(b, a)
    expected_strides = dense_c_strides(broadcast_shape, a.itemsize)
    assert b.strides == expected_strides, f"{b.strides} != {expected_strides}"


@pytest.mark.parametrize(
    (
        "base_size",
        "device_id",
        "dtype",
    ),
    [
        (
            Param("base_size", base_size),
            Param("device_id", device_id),
            Param("dtype", dtype),
        )
        for base_size in [0, 1, 513, 1537, 2**20 + 1]
        for device_id in [0, 1]
        for dtype in [py_rng.choice(dtypes)]
    ],
    ids=idfn,
)
def test_default_device_allocation_size(base_size, device_id, dtype):
    device_id = device_id.value
    dtype = dtype.value
    base_size = base_size.value
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    if device_id > 0 and ccx.system.num_devices < 2:
        pytest.skip("Test requires at least 2 gpus")
    stream_holder = create_stream(device_id)
    itemsize = np.dtype(dtype).itemsize
    additional_sizes = 512 // itemsize + 1
    for i in range(additional_sizes):
        size = base_size + i
        shape = (size,)
        ndbuffer = ndb.empty(
            shape=shape,
            dtype_name=dtype,
            itemsize=itemsize,
            device_id=device_id,
            stream=stream_holder,
        )
        size_in_bytes = size * itemsize
        assert ndbuffer.shape == shape
        assert ndbuffer.strides == (0,) if size == 0 else (1,)
        assert ndbuffer.strides_in_bytes == (0,) if size == 0 else (itemsize,)
        assert ndbuffer.size_in_bytes == size_in_bytes
        assert ndbuffer.size == size
        assert ndbuffer.device_id == device_id

        rounded_size_in_bytes = (size_in_bytes + 511) // 512 * 512
        assert rounded_size_in_bytes >= size_in_bytes, f"{rounded_size_in_bytes} < {size_in_bytes}"
        if size_in_bytes % 512 == 0:
            rounded_size_in_bytes = size_in_bytes

        if size == 0:
            assert ndbuffer.data is None
            assert ndbuffer.data_ptr == 0
        else:
            assert ndbuffer.data.size == rounded_size_in_bytes, f"{ndbuffer.data.size} != {rounded_size_in_bytes}"
            assert ndbuffer.data_ptr != 0

        b = tensor_wrapper.wrap_operand(np.arange(size, dtype=dtype))
        with device_ctx(device_id):
            ndb.copy_into(ndbuffer, b.asndbuffer(), stream=stream_holder)
        assert_equal(as_array(ndbuffer), b.tensor)


@pytest.mark.parametrize(
    (
        "shape",
        "slice",
        "new_shape",
        "permutation",
        "allowed",
        "device_id",
        "dtype",
    ),
    [
        (
            Param("shape", shape),
            Param("slice", slice),
            Param("new_shape", new_shape),
            Param("permutation", permutation),
            Param("allowed", allowed),
            Param("device_id", device_id),
            Param("dtype", py_rng.choice(dtypes)),
        )
        for shape, slice, new_shape, permutation, allowed in [
            ((12,), _SL[:], (12,), (0,), True),
            ((12,), _SL[:], (13,), (0,), False),
            ((0,), _SL[:], (0,), (0,), True),
            ((0,), _SL[:], (1, 3), (0,), False),
            ((3,), _SL[3:], (3,), (0,), False),
            ((3,), _SL[3:], (0,), (0,), True),
            ((3, 0, 3), _SL[:], (2, 3, 4, 5, 6, 7, 0, 12), (0, 1, 2), True),
            ((3, 0, 3), _SL[:], (0,), (0, 1, 2), True),
            ((18,), _SL[:], (0,), (0,), False),
            ((12,), _SL[:], (2, 3, 2), (0,), True),
            ((12,), _SL[:], (2, 6), (0,), True),
            ((12,), _SL[:], (4, 3), (0,), True),
            ((12,), _SL[:], (3, 4), (0,), True),
            ((7, 12), _SL[:, :], (7, 12), (0, 1), True),
            ((12, 11), _SL[:, :], (2, 3, 2, 11), (0, 1), True),
            ((5, 12), _SL[:, :], (5, 2, 6), (0, 1), True),
            ((12, 7), _SL[:, :], (4, 3, 7), (0, 1), True),
            ((7, 12), _SL[:, :], (7, 3, 4), (0, 1), True),
            ((7, 12), _SL[:, :], (3, 4, 7), (0, 1), True),
            ((2, 3, 2), _SL[:, :, :], (12,), (0, 1, 2), True),
            ((2, 3, 2), _SL[:, :, :], (6, 2), (0, 1, 2), True),
            ((2, 3, 2), _SL[:, :, :], (2, 3, 2), (1, 2, 0), True),
            ((2, 3, 2), _SL[:, :, :], (6, 2), (1, 2, 0), True),
            ((2, 3, 2), _SL[:, :, :], (2, 6), (1, 2, 0), False),
            ((2, 3, 2), _SL[:, :, :], (12,), (1, 2, 0), False),
            ((2, 3, 2), _SL[:, :, :], (3, 2, 2), (1, 0, 2), True),
            ((10, 10, 10), _SL[::-1, ::-1, :], (10, 10, 10), (0, 1, 2), True),
            ((10, 10, 10), _SL[::-1, ::-1, :], (100, 10), (0, 1, 2), True),
            ((10, 10, 10), _SL[::-1, ::-1, ::-1], (1000,), (0, 1, 2), True),
            ((10, 10, 10), _SL[:, :, ::-1], (100, 10), (0, 1, 2), True),
            ((10, 10, 10), _SL[:, :, ::-1], (10, 100), (0, 1, 2), False),
            ((10, 10, 10), _SL[::-1, :, ::-1], (1000,), (0, 1, 2), False),
            ((10, 10, 10), _SL[::-1, ::-1, :], (100, 10), (1, 0, 2), False),
            ((10, 10, 10), _SL[::-1, ::-1, :], (10, 100), (0, 1, 2), False),
            ((5, 3), _SL[:-1, :], (12,), (0, 1), True),
            ((13, 3), _SL[1:, :], (6, 6), (0, 1), True),
            ((12, 4), _SL[:, :-1], (6, 2, 3), (0, 1), True),
            ((12, 4), _SL[:, :-1], (6, 6), (0, 1), False),
        ]
        for device_id in ["cpu", 0]
    ],
    ids=idfn,
)
def test_reshape(shape, slice, new_shape, permutation, allowed, device_id, dtype):
    if cp is None:
        pytest.skip("Cupy is required to run this test")
    shape = shape.value
    slice = slice.value
    new_shape = new_shape.value
    permutation = permutation.value
    allowed = allowed.value
    device_id = device_id.value
    dtype = dtype.value
    stream_holder = create_stream(0)
    a = arange(device_id, stream_holder, math.prod(shape), dtype).reshape(shape)
    a = a[slice]
    a = a.transpose(permutation)
    aw = tndb.NDBufferTensor(as_ndbuffer(a))
    if not allowed:
        if math.prod(new_shape) != math.prod(a.shape):
            msg = "The source and destination have different volumes"
        else:
            msg = "Cannot reshape the tensor without performing a copy"
        with pytest.raises(ValueError, match=msg):
            aw.reshape(new_shape)
    else:
        reshaped = aw.reshape(new_shape)
        print(f"\nReshaped: {reshaped.shape} <- {aw.shape}, strides: {reshaped.strides} <- {aw.strides}")
        if device_id == "cpu":
            bw = reshaped.to(device_id=0, stream_holder=stream_holder)
        else:
            assert device_id == 0
            bw = reshaped.to(device_id="cpu", stream_holder=stream_holder)
        b = as_array(bw.tensor)
        assert b.shape == new_shape
        c = a.reshape(new_shape)
        assert_equal(b, c)
