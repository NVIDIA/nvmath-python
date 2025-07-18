# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of (internal use) helper functions.
"""

import contextlib
import functools
import time
import typing
from collections.abc import Callable, Sequence

import cuda.bindings.runtime as cbr
import cuda.core.experimental as ccx

from . import formatters
from . import mem_limit
from . import package_wrapper
from .package_ifc import StreamHolder, Stream, AnyStream
from .tensor_ifc import TensorHolder, Tensor, AnyTensor
from nvmath._internal.layout import is_contiguous_and_dense


def infer_object_package(obj: object) -> str:
    """
    Infer the package that defines this object.
    """
    module = obj.__class__.__module__
    return module.split(".")[0]


"""A generic type for Options classes"""
Options = typing.TypeVar("Options")


def check_or_create_options(
    cls: type[Options], options: None | dict[str, typing.Any] | Options, options_description: str, *, keep_none=False
) -> Options | None:
    """
    Create the specified options dataclass from a dictionary of options or None.
    """

    if options is None:
        if keep_none:
            return options
        options = cls()
    elif isinstance(options, dict):
        options = cls(**options)

    if not isinstance(options, cls):
        raise TypeError(
            f"The {options_description} must be provided as an object "
            f"of type {cls.__name__} or as a dict with valid {options_description}. "
            f"The provided object is '{options}'."
        )

    return options


def check_or_create_one_of_options(clss, options, options_description, *, cls_key="name", default_name=None):
    """
    Create one of the specified options dataclasses by name or from a dictionary of options.
    """

    assert isinstance(clss, tuple)
    assert all(hasattr(cls, "name") and isinstance(cls.name, str) for cls in clss)

    if options is None:
        assert default_name is not None, "Internal error in execution options"
        options = _create_one_of_options_from_name(clss, default_name, options_description, cls_key=cls_key)
    elif isinstance(options, str):
        options = _create_one_of_options_from_name(clss, options, options_description, cls_key=cls_key)
    elif isinstance(options, dict):
        options = _create_one_of_options_from_dict(
            clss,
            options,
            options_description,
            cls_key=cls_key,
            default_name=default_name,
        )
    else:
        if not isinstance(options, clss):
            _raise_invalid_one_of_options(clss, options, options_description, cls_key=cls_key)
    return options


def _create_one_of_options_from_name(clss, cls_name, options_description, *, cls_key="name"):
    try:
        _cls_name = cls_name.lower()
        return next(cls for cls in clss if cls.name == _cls_name)()
    except StopIteration:
        raise _raise_invalid_one_of_options(clss, cls_name, options_description, cls_key=cls_key)


def _create_one_of_options_from_dict(
    clss,
    options,
    options_description,
    *,
    cls_key="name",
    default_name=None,
):
    cls_name = options.get(cls_key)
    if cls_name is None:
        cls_name = default_name
    if not isinstance(cls_name, str):
        raise _raise_invalid_one_of_options(clss, cls_name, options_description, cls_key=cls_key)
    try:
        _cls_name = cls_name.lower()
        cls = next(cls for cls in clss if cls.name == _cls_name)
    except StopIteration:
        raise _raise_invalid_one_of_options(clss, cls_name, options_description, cls_key=cls_key)
    return cls(**{key: name for key, name in options.items() if key != cls_key})


def _raise_invalid_one_of_options(clss, options, options_description, *, cls_key="name"):
    accepted_names = ", ".join(f"'{cls.name}'" for cls in clss)
    accepted_types = ", ".join(str(cls) for cls in clss)
    raise ValueError(
        f"The {options_description} must be: \n"
        f"1. an object of one of the following types {accepted_types} or\n"
        f"2. a string with the type name ({accepted_names}) or\n"
        f"3. a dict with a '{cls_key}' key and one of the type names {accepted_names}' as a value, "
        f"and optional options valid for that type. \n"
        f"The provided object is '{options}'."
    )


@contextlib.contextmanager
def device_ctx(new_device_id: int) -> typing.Iterator[ccx.Device]:
    """
    Semantics:

    1. The device context manager makes the specified device current from the point of entry
       until the point of exit.

    2. When the context manager exits, the current device is reset to what it was when the
       context manager was entered.

    3. Any explicit setting of the device within the context manager (using
       cupy.cuda.Device().use(), torch.cuda.set_device(), etc) will overrule the device set
       by the context manager from that point onwards till the context manager exits. In
       other words, the context manager provides a local device scope and the current device
       can be explicitly reset for the remainder of that scope.

    Corollary: if any library function resets the device globally and this is an undesired
        side-effect, such functions must be called from within the device context manager.

    Device context managers can be arbitrarily nested.
    """
    assert isinstance(new_device_id, int), "Internal Error. Setting device context for 'cpu' is not allowed."
    old_device = ccx.Device()
    try:
        if old_device.device_id != new_device_id:
            device = ccx.Device(new_device_id)
            device.set_current()
            yield device
        else:
            yield old_device
    finally:
        # We should always restore the old device at exit.
        old_device.set_current()


def is_hashable(obj: object) -> bool:
    try:
        hash(obj)
    except TypeError:
        return False
    return True


@functools.lru_cache(maxsize=128)
def cached_get_or_create_stream(
    device_id: int,
    stream: Stream | int,
    op_package: str,
) -> StreamHolder[Stream]:
    op_package_ifc = package_wrapper.PACKAGE[op_package]
    if isinstance(stream, int):
        ptr = stream
        if op_package == "torch":
            message = "A stream object must be provided for PyTorch operands, not stream pointer."
            raise TypeError(message)
        external = op_package_ifc.create_external_stream(device_id, ptr)
        ctx = op_package_ifc.to_stream_context(external)
        obj = op_package_ifc.create_stream(external)
        return StreamHolder(ctx=ctx, device_id=device_id, external=external, obj=obj, ptr=ptr, package=op_package)

    stream_package = infer_object_package(stream)
    if stream_package != op_package:
        message = "The stream object must belong to the same package as the tensor network operands."
        raise TypeError(message)

    ctx = op_package_ifc.to_stream_context(stream)
    ptr = op_package_ifc.to_stream_pointer(stream)
    external = stream
    obj = op_package_ifc.create_stream(stream)
    return StreamHolder(ctx=ctx, device_id=device_id, external=external, obj=obj, ptr=ptr, package=op_package)


def get_or_create_stream(
    device_id: int,
    stream: Stream | int | None,
    op_package: str,
) -> StreamHolder[Stream]:
    """
    Create a stream object from a stream pointer or extract the stream pointer from a stream
    object, or use the current stream.

    Args:
        device_id: The device ID.

        stream: A stream object, stream pointer, or None.

        op_package: The package the tensor network operands belong to.

    Returns:
        StreamHolder: Hold a CuPy stream object, package stream context, stream pointer, ...
    """
    op_package_ifc = package_wrapper.PACKAGE[op_package]
    if stream is None:
        stream = op_package_ifc.get_current_stream(device_id)
    return cached_get_or_create_stream(device_id, stream, op_package)


@functools.lru_cache(maxsize=128)
def get_memory_limit_from_device_id(memory_limit: int | float | str, device_id: int) -> int:
    with device_ctx(device_id):
        status, _, total_memory = cbr.cudaMemGetInfo()
        return _get_memory_limit(memory_limit, total_memory)


def _get_memory_limit(memory_limit: int | float | str, total_memory: int) -> int:
    """
    Parse user provided memory limit and return the memory limit in bytes.
    """
    if isinstance(memory_limit, int):
        if memory_limit < 0:
            raise ValueError("The specified memory limit must be greater than or equal to 0.")
        return memory_limit

    if isinstance(memory_limit, float):
        if memory_limit < 0:
            raise ValueError("The specified memory limit must be greater than or equal to 0.")
        if memory_limit <= 1.0:
            memory_limit *= total_memory
        return int(memory_limit)

    m = mem_limit.MEM_LIMIT_RE_PCT.match(memory_limit)
    if m:
        factor = float(m.group(1))
        if factor <= 0 or factor > 100:
            raise ValueError("The memory limit percentage must be in the range [0, 100].")
        return int(factor * total_memory / 100.0)

    m = mem_limit.MEM_LIMIT_RE_VAL.match(memory_limit)
    if not m:
        raise ValueError(mem_limit.MEM_LIMIT_DOC.format(kind="memory limit", value=memory_limit))

    base = 1000
    if m.group("binary"):
        base = 1024

    powers = {"": 0, "k": 1, "m": 2, "g": 3}
    unit = m.group("units").lower() if m.group("units") else ""
    multiplier = base ** powers[unit]

    value = float(m.group("value"))
    memory_limit = int(value * multiplier)

    return memory_limit


def create_empty_tensor(
    cls: type[TensorHolder[Tensor]],
    extents: Sequence[int],
    dtype: str,
    device_id: int | typing.Literal["cpu"],
    stream_holder: StreamHolder | None,
    verify_strides: bool,
    strides: Sequence[int] | None = None,
    **context,
) -> TensorHolder[Tensor]:
    """
    Create a wrapped tensor of the same type as (the wrapped) cls on the specified device
    having the specified extents and dtype.

    The tensor is created within a stream context to allow for asynchronous memory
    allocators like CuPy's MemoryAsyncPool.

    Note, the function assumes the `strides` are dense (possibly permuted).
    Otherwise, the behaviour is framework specific and tensor creation may fail
    or created tensor may be corrupted. Set `verify_strides` to True to check
    the layout and drop the strides if the layout is not dense.
    """
    assert isinstance(device_id, int) or device_id == "cpu", f"Internal Error: device_id must be int or 'cpu'; not {device_id}"
    # if device id is none the stream holder must be too
    assert device_id != "cpu" or stream_holder is None
    if strides is not None and verify_strides and not is_contiguous_and_dense(extents, strides):
        strides = None
    tensor = cls.empty(extents, dtype=dtype, device_id=device_id, strides=strides, stream_holder=stream_holder, **context)
    return tensor


def get_operands_device_id(operands: Sequence[TensorHolder[AnyTensor]]):
    """
    Return the id (ordinal) of the device the operands are on, or None if it is on the CPU.
    """
    device_id = operands[0].device_id
    if not all(operand.device_id == device_id for operand in operands):
        devices = {operand.device_id for operand in operands}
        raise ValueError(f"All operands are not on the same device. Devices = {devices}.")

    return device_id


def get_operands_dtype(operands: Sequence[TensorHolder[AnyTensor]]):
    """
    Return the data type name of the tensors.
    """
    dtype = operands[0].dtype
    if not all(operand.dtype == dtype for operand in operands):
        dtypes = {operand.dtype for operand in operands}
        raise ValueError(f"All tensors in the network must have the same data type. Data types found = {dtypes}.")
    return dtype


def get_operands_package(operands: Sequence[TensorHolder[AnyTensor]]) -> str:
    """
    Return the package name of the tensors.
    """
    package = infer_object_package(operands[0].tensor)
    if not all(infer_object_package(operand.tensor) == package for operand in operands):
        packages = {infer_object_package(operand.tensor) for operand in operands}
        raise TypeError(f"All tensors in the network must be from the same library package. Packages found = {packages}.")
    return package


def check_operands_match(
    orig_operands: Sequence[object] | object, new_operands: Sequence[object] | object, attribute: str, description: str
):
    """
    Check if the specified attribute matches between the corresponding new and old operands,
    and raise an exception if it doesn't.
    """
    if isinstance(orig_operands, Sequence):
        assert isinstance(new_operands, Sequence), "Internal Error. Cannot compare operand sequence with non-sequence."
        checks: list[bool] = [
            getattr(o, attribute) == getattr(n, attribute) for o, n in zip(orig_operands, new_operands, strict=True)
        ]

        if not all(checks):
            mismatch = [
                f"{location}: {getattr(orig_operands[location], attribute)} => {getattr(new_operands[location], attribute)}"
                for location, predicate in enumerate(checks)
                if predicate is False
            ]
            mismatch = formatters.array2string(mismatch)
            message = f"""\
The {description} of each new operand must match the {description} of the corresponding original operand.

The mismatch in {description} as a sequence of "position: original {description} => new {description}" is: \n{mismatch}"""
            raise ValueError(message)
    else:
        assert isinstance(new_operands, object)
        check: bool = getattr(orig_operands, attribute) == getattr(new_operands, attribute)
        if not check:
            message = f"""The {description} of the new operand must match the {description} of the original operand."""
            raise ValueError(message)


def check_attribute_match(orig_attribute: object, new_attribute: object, description: str):
    """
    Check if the specified attribute matches between the corresponding new and old operands,
    and raise an exception if it doesn't.
    """
    check = orig_attribute == new_attribute
    if not check:
        message = f"""The {description} of the new operand must match the {description} of the original operand."""
        raise ValueError(message)


def get_ptr_from_memory_pointer(mem_ptr):
    """
    Access the value associated with one of the attributes 'device_ptr', 'device_pointer',
    'ptr'.

    This is a compatibility function so that external memory pointer implementations from
    numba and cupy can be used instead of MemoryPointer from nvmath.memory.
    """
    attributes = (
        "device_ptr",  # nvmath.memory
        "device_pointer",
        "ptr",  # cupy.cuda.MemoryPointer
        "pointer",  # numba.cuda.MemoryPointer
    )
    for attr in attributes:
        if hasattr(mem_ptr, attr):
            return getattr(mem_ptr, attr)

    message = f"Memory pointer objects should have one of the following attributes specifying the device pointer: {attributes}"
    raise AttributeError(message)


class Value:
    """
    A simple value wrapper holding a default value.
    """

    def __init__(self, default, *, validator: Callable[[object], bool]):
        """
        Args:
            default: The default value to use.
            validator: A callable that validates the provided value.
        """
        self.validator = validator
        self._data = default

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = self._validate(value)

    def _validate(self, value):
        if self.validator(value):
            return value
        raise ValueError(f"Internal Error: value '{value}' is not valid.")


@contextlib.contextmanager
def cuda_call_ctx(
    stream_holder: StreamHolder[AnyStream], blocking=True, timing=True
) -> typing.Iterator[tuple[ccx.Event, Value]]:
    """
    A simple context manager that provides (non-)blocking behavior depending on the
    `blocking` parameter for CUDA calls. The call is timed only for blocking behavior when
    timing is requested.

    An `end` event is recorded after the CUDA call for use in establishing stream ordering
    for non-blocking calls. This event is returned together with a `Value` object that
    stores the elapsed time if the call is blocking and timing is requested, or None
    otherwise.

    Inside the context, the current device is the device which owns the provided stream.
    """
    device_id = stream_holder.device_id
    stream = stream_holder.obj

    with device_ctx(device_id):
        if timing:
            start = stream.record(options=ccx.EventOptions(enable_timing=blocking))

        end = stream.device.create_event(options=ccx.EventOptions(enable_timing=(timing and blocking)))

        time = Value(None, validator=lambda v: True)
        yield end, time

        stream.record(end)

        if not blocking:
            return

        end.sync()

        if timing:
            time.data = end - start


@contextlib.contextmanager
def host_call_ctx(timing=False):
    elapsed = Value(None, validator=lambda v: True)

    if timing:
        start_time = time.perf_counter_ns()

    yield elapsed

    if timing:
        elapsed.data = (time.perf_counter_ns() - start_time) * 1e-6


# Decorator definitions


def atomic(
    handler: Callable[[typing.Any, Exception | None], bool] | Callable[[Exception | None], bool], method: bool = False
) -> Callable:
    """
    A decorator that provides "succeed or roll-back" semantics. A typical use for this is to
    release partial resources if an exception occurs.

    Args:
        handler: A function to call when an exception occurs. The handler takes a single
            argument, which is the exception object, and returns a boolean stating whether
            the same exception should be reraised. We assume that this function does not
            raise an exception.

        method: Specify if the wrapped function as well as the exception handler are methods
            bound to the same object (method = True) or they are free functions (method =
            False).

    Returns:
        Callable: A decorator that creates the wrapping.
    """

    def outer(wrapped_function):
        """
        A decorator that actually wraps the function for exception handling.
        """

        @functools.wraps(wrapped_function)
        def inner(*args, **kwargs):
            """
            Call the wrapped function and return the result. If an exception occurs, then
            call the exception handler and reraise the exception.
            """
            try:
                result = wrapped_function(*args, **kwargs)
            except BaseException as e:
                if method:
                    flag = handler(args[0], e)
                else:
                    flag = handler(e)

                if flag:
                    raise e

            return result

        return inner

    return outer


def precondition(checker: Callable[..., None], what: str = "") -> Callable:
    """
    A decorator that adds checks to ensure any preconditions are met.

    Args:
        checker: The function to call to check whether the preconditions are met. It has the
            same signature as the wrapped function with the addition of the keyword argument
            `what`.
        what: A string that is passed in to `checker` to provide context information.

    Returns:
        Callable: A decorator that creates the wrapping.
    """

    def outer(wrapped_function):
        """
        A decorator that actually wraps the function for checking preconditions.
        """

        @functools.wraps(wrapped_function)
        def inner(*args, **kwargs):
            """
            Check preconditions and if they are met, call the wrapped function.
            """
            checker(*args, **kwargs, what=what)
            result = wrapped_function(*args, **kwargs)

            return result

        return inner

    return outer


COMMON_SHARED_DOC_MAP = {
    "operand": """\
A tensor (ndarray-like object). The currently supported types are :class:`numpy.ndarray`,
:class:`cupy.ndarray`, and :class:`torch.Tensor`.""".replace("\n", " "),
    #
    "stream": """\
Provide the CUDA stream to use for executing the operation. Acceptable inputs include
``cudaStream_t`` (as Python :class:`int`), :class:`cupy.cuda.Stream`, and
:class:`torch.cuda.Stream`. If a stream is not provided, the current stream from the operand
package will be used.""".replace("\n", " "),
    #
    "release_workspace": """\
A value of `True` specifies that the stateful object should release workspace memory back to
the package memory pool on function return, while a value of `False` specifies that the
object should retain the memory. This option may be set to `True` if the application
performs other operations that consume a lot of memory between successive calls to the (same
or different) :meth:`execute` API, but incurs a small overhead due to obtaining and
releasing workspace memory from and to the package memory pool on every call. The default is
`False`.""".replace("\n", " "),
}


class DefaultDocstring(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def docstring_decorator(doc_map, skip_missing=False):
    assert isinstance(doc_map, dict)

    def _format_doc(doc):
        if doc is not None:
            if skip_missing:
                # Using a default class to handle missing keys in doc_map
                doc = doc.format_map(DefaultDocstring(**doc_map))
            else:
                doc = doc.format(**doc_map)
        return doc

    def decorator(func_or_class):
        if isinstance(func_or_class, type):  # class decorator
            # update the docstring of all public methods with docstrings
            static_methods = []  # staticmethods appear to require special handling
            for name, method in vars(func_or_class).items():
                if isinstance(method, staticmethod):
                    static_methods.append(name)
                    continue
                if callable(method) and (not name.startswith("_")) and method.__doc__:
                    method.__doc__ = _format_doc(method.__doc__)
            # update the docstring of the constructor
            func_or_class.__doc__ = _format_doc(func_or_class.__doc__)
            for name in static_methods:
                method = getattr(func_or_class, name)
                method.__doc__ = _format_doc(method.__doc__)
            return func_or_class
        else:  # function decorator
            func_or_class.__doc__ = _format_doc(func_or_class.__doc__)
            return func_or_class

    return decorator
