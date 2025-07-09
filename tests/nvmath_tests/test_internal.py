import importlib
import typing

import cuda.core.experimental as ccx
import nvmath.internal.utils
from nvmath.internal import package_wrapper, tensor_wrapper
from hypothesis import given, strategies as st


def test_device_ctx():
    id0 = 0
    id1 = ccx.system.num_devices - 1
    device0 = ccx.Device(id0)
    device0.set_current()
    assert ccx.Device().device_id == id0
    with nvmath.internal.utils.device_ctx(id1) as device1:
        assert isinstance(device1, ccx.Device)
        assert device1.device_id == id1
        assert ccx.Device().device_id == id1
    assert ccx.Device().device_id == id0


@given(package_name=st.sampled_from(["cupy", "torch", "numpy"]), id0=st.sampled_from(["cpu", 0]))
def test_tensor_empty_device_ctx(package_name: str, id0: int | typing.Literal["cpu"]) -> None:
    try:
        tensor_wrapper.maybe_register_package(package_name)
    except ModuleNotFoundError:
        return
    tensor_type = tensor_wrapper._TENSOR_TYPES[package_name]
    if package_name == "cupy" and isinstance(id0, str):
        return
    if package_name == "numpy" and isinstance(id0, int):
        return
    id1 = ccx.system.num_devices - 1
    stream_holder = (
        None
        if isinstance(id0, str)
        else nvmath.internal.utils.get_or_create_stream(device_id=id0, stream=None, op_package=package_name)
    )
    with nvmath.internal.utils.device_ctx(id1):
        _ = nvmath.internal.utils.create_empty_tensor(
            tensor_type,
            device_id=id0,
            extents=(64, 64, 64),
            dtype="float32",
            stream_holder=stream_holder,
            verify_strides=False,
        )


@given(package_name=st.sampled_from(["cupy", "torch"]))
def test_stream_ifc(package_name: str):
    # make sure the package is registered
    try:
        tensor_wrapper.maybe_register_package(package_name)
    except ModuleNotFoundError:
        return
    package = package_wrapper.PACKAGE[package_name]
    core = package_wrapper.PACKAGE["cuda"]

    # test pointers are the same for current stream
    stream = package.get_current_stream(device_id=0)
    ptr = package.to_stream_pointer(stream)
    obj = package.create_stream(stream)
    core_ptr = core.to_stream_pointer(obj)
    assert core_ptr == ptr, f"cuda.core stream pointer ({core_ptr}) not equal to package stream pointer ({ptr})!"

    # test pointers are the same for new stream
    module = importlib.import_module(package_name)
    stream = module.cuda.Stream()
    ptr = package.to_stream_pointer(stream)
    obj = package.create_stream(stream)
    core_ptr = core.to_stream_pointer(obj)
    assert core_ptr == ptr, f"cuda.core stream pointer ({core_ptr}) not equal to package stream pointer ({ptr})!"
