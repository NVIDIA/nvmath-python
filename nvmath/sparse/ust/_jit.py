# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module defines a JIT runtime for the universal sparse tensor (UST).
"""

__all__ = []

import re
from collections.abc import Sequence

from nvmath.internal import utils

# TODO: support cuda.core >=0.5.0
try:
    from cuda.core import (
        LaunchConfig,
        Linker,
        LinkerOptions,
        ObjectCode,
        Program,
        ProgramOptions,
        launch,
    )
except ImportError:
    from cuda.core.experimental import (
        LaunchConfig,
        Linker,
        LinkerOptions,
        ObjectCode,
        Program,
        ProgramOptions,
        launch,
    )


def _check_numba_available():
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError("Numba is required to compile user-defined Python functions provided to the UST APIs.") from e


def compile_python_function(
    function,
    function_name,
    signature,
    *,
    representation="ltoir",
    compute_capability=None,
):
    """
    Compile the Python function to an intermediate representation using Numba.

    Args:
        function: The Python function to compile.
        function_name: The symbol to use for the function name.
        signature: the function signature as a sequence (return_type, *parameter_types). The
            type names should correspond to the C type names recognized by Numba.
        representation: the intermediate representation ("ltoir", "ptx").
        compute_capability: the device compute capability. If not specified, the
            current devices CC will be used.

    Returns:
        The compiled IR.
    """

    _check_numba_available()

    import numba
    import numba.cuda

    user_return_type, *user_arg_types = signature

    return_type = getattr(numba.cuda.types.types, user_return_type)

    arg_types = []
    for arg_type in user_arg_types:
        m = re.match(r"(\w+)((\[:\])|( *(\*)))?", arg_type)
        assert m is not None, "Internal error."
        t = getattr(numba.cuda.types.types, m.group(1))
        if m.group(2) is not None:
            if m.group(3) == "[:]":
                t = t[:]
            elif m.group(5) == "*":
                t = numba.cuda.types.types.voidptr
            else:
                raise AssertionError("Internal error.")
        arg_types.append(t)

    signature = numba.core.typing.signature(return_type, *arg_types)

    return numba.cuda.compile(
        function,
        signature,
        debug=None,
        lineinfo=False,
        device=True,
        fastmath=False,
        cc=compute_capability,
        opt=True,
        abi="c",
        abi_info={"abi_name": function_name},
        output=representation,
        forceinline=False,
        launch_bounds=None,
    )[0]


def compile_cpp_and_link(
    *,
    src_code=None,
    object_code=None,
    function_name,
    compiler_options=None,
    linker_options=None,
    intermediate_type="ltoir",
    target_type="cubin",
):
    """
    Compile the source code (if provided) and link with the intermediate code (if provided).

    Args:
        src_code: A string or sequence of strings containing the C++ source code.
        object_code: A byte-string or sequence of byte-string objects to be linked
            along with the source code.
        function_name: symbol name to use for the compiled and linked function.
        compiler_options: the compiler options to use as a
            :class:`cuda.core.ProgramOptions` object. Alternatively, a dictionary containing
            the options can be provided.
        linker_options: the linker options to use as a
            :class:`cuda.core.LinkerOptions` object. Alternatively, a dictionary containing
            the options can be provided.

    Returns:
        The linked kernel.
    """

    assert not (src_code is None and object_code is None), "Internal error."

    compiler_options = utils.check_or_create_options(ProgramOptions, compiler_options, "Compiler Options")
    linker_options = utils.check_or_create_options(LinkerOptions, linker_options, "Linker Options")

    if isinstance(src_code, str):
        src_code = [src_code]
    else:
        if src_code is not None and not isinstance(src_code, Sequence):
            raise TypeError("The source code type 'type(src_code)' must be a string or a sequence of strings.")

    # Handle the case where no source code is provided.
    if src_code is None:
        src_code = []

    # Compile source code to LTO-IR.
    compiled_code = [Program(s, "c++", options=compiler_options).compile(intermediate_type) for s in src_code]

    if isinstance(object_code, bytes):
        object_code = [object_code]
    else:
        if object_code is not None and not isinstance(object_code, Sequence):
            raise TypeError(
                "The intermediate code type 'type(object_code)' must be a byte-string or a sequence of byte-strings."
            )

    assert intermediate_type == "ltoir", "Internal error."

    # Handle the case where no object code is provided.
    if object_code is None:
        object_code = []

    object_code = [ObjectCode.from_ltoir(o) for o in object_code]

    # Link all objects together.
    linker = Linker(*(compiled_code + object_code), options=linker_options)
    linked_code = linker.link(target_type=target_type)

    # Extract kernel entry by function name.
    return linked_code.get_kernel(function_name)


# TODO: take grid instead of problem size.
def launch_kernel(kernel, parameters, problem_size, *, device_id, stream_holder, blocking=True):
    """
    Launch the kernel on the specified device and stream.

    Args:
        kernel: the CUDA kernel to launch.
        parameters: the kernel arguments as a sequence or None.
        problem_size: the size of the problem (work to divide among threads).
        device_id: the device ID on which to launch the kernel.
        stream_holder: a `nvmath.internal.utils.StreamHolder` object encapsulating the
            stream to use.
        blocking: if True, the call blocks.
    """

    if problem_size == 0:
        return

    # Prepare grid.
    threads_per_block = min(1024, problem_size)
    batch_size = (problem_size + threads_per_block - 1) // threads_per_block
    grid = LaunchConfig(grid=batch_size, block=threads_per_block)

    # Launch the kernel.
    with utils.device_ctx(device_id):
        launch(stream_holder.obj, grid, kernel, *parameters)

    # TODO: make this non-blocking by default.
    if blocking:
        stream_holder.obj.sync()
