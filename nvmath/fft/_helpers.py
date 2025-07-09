# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["compile_prolog", "compile_epilog"]

import functools

from nvmath.bindings import cufft as _cufft  # type: ignore


# Currently supported element data types for FFT callback functions.
VALID_DTYPES = {"float32": "Real", "float64": "DoubleReal", "complex64": "Complex", "complex128": "DoubleComplex"}

# Python <> CUDA magled types.
TYPE_MAP = {
    "float32": "f",  # float
    "float64": "d",  # double
    "complex64": "6float2",  # float2
    "complex128": "7double2",  # double2
}


def _check_numba_available():
    try:
        import numba  # noqa: F401
    except ModuleNotFoundError as e:
        raise RuntimeError("Numba is required to compile FFT prolog and epilog functions.") from e


def _get_compiler():
    import numba
    import numba.cuda

    @numba.core.compiler_lock.global_compiler_lock
    def compile_to(function, sig, name, *, compute_capability=None, representation="ltoir"):
        if compute_capability is None:
            compute_capability = numba.cuda.get_current_device().compute_capability
        else:
            if not isinstance(compute_capability, str):
                raise ValueError(
                    f"The compute capability must be specified as a string ('80', '89', ...). "
                    f"The provided value {compute_capability} is invalid."
                )
            cc_number = int(compute_capability)
            compute_capability = (cc_number // 10, cc_number % 10)

        if hasattr(numba.cuda.cudadrv.nvrtc, "get_arch_option"):
            # numba-cuda >=0.16.0
            get_arch_option = numba.cuda.cudadrv.nvrtc.get_arch_option
        elif hasattr(numba.cuda.cudadrv.nvvm, "get_arch_option"):
            get_arch_option = numba.cuda.cudadrv.nvvm.get_arch_option
        nvvm_options = {"opt": 3, "arch": get_arch_option(*compute_capability)}

        if representation == "ltoir":
            nvvm_options["gen-lto"] = None

        debug = False
        lineinfo = False

        args, return_type = numba.core.sigutils.normalize_signature(sig)

        cres = numba.cuda.compiler.compile_cuda(
            function,
            return_type,
            args,
            debug=debug,
            lineinfo=lineinfo,
            nvvm_options=nvvm_options,
            cc=compute_capability,
        )

        lib = numba.cuda.compiler.cabi_wrap_function(cres.target_context, cres.library, cres.fndesc, name, nvvm_options)

        return numba.cuda.cudadrv.nvvm.compile_ir(lib.llvm_strs, **nvvm_options)

    return compile_to


def _compile(
    function, element_dtype, user_info_dtype, *, name=None, phase=None, representation="ltoir", compute_capability=None
):
    _check_numba_available()

    import numba
    import numba.cuda

    assert phase in ["Load", "Store"], "Internal error."

    if element_dtype not in VALID_DTYPES:
        raise ValueError(
            f"The specified operand data type '{element_dtype}' is not currently supported. "
            f"It must be one of {VALID_DTYPES.keys()}."
        )

    data_type = getattr(numba.cuda.types.types, element_dtype)

    if user_info_dtype in VALID_DTYPES:
        info_type = getattr(numba.cuda.types.types, user_info_dtype)
    elif isinstance(user_info_dtype, numba.types.Type):
        info_type = user_info_dtype
    else:
        raise ValueError(
            f"The specified user information data type '{user_info_dtype}' is not supported. "
            f"It must be a Numba custom type or one of {VALID_DTYPES.keys()}."
        )

    dataptr_type = numba.types.CPointer(data_type)
    infoptr_type = numba.types.CPointer(info_type)
    offset_type = numba.types.uint64
    smemptr_type = dataptr_type

    if phase == "Load":
        insert = ""
        return_type = data_type
        signature = numba.core.typing.signature(return_type, dataptr_type, offset_type, infoptr_type, smemptr_type)
    else:
        insert = TYPE_MAP[element_dtype]
        return_type = numba.types.void
        signature = numba.core.typing.signature(return_type, dataptr_type, offset_type, data_type, infoptr_type, smemptr_type)

    if name is None:
        snippet = VALID_DTYPES[element_dtype]
        length = 16 + len(phase) + len(snippet)
        # the signature of callbacks differs - the offset argument
        # changed type from size_t to unsigned long long int
        offset_dtype = "m" if _cufft.get_version() < 11300 else "y"
        name = f"_Z{length}cufftJITCallback{phase}{snippet}Pv{offset_dtype}{insert}S_S_"

    compile_to = _get_compiler()
    return compile_to(function, signature, name, representation=representation, compute_capability=compute_capability)


compile_prolog = functools.wraps(_compile)(functools.partial(_compile, phase="Load"))
compile_epilog = functools.wraps(_compile)(functools.partial(_compile, phase="Store"))

SHARED_FFT_HELPER_DOCUMENTATION = {
    "element_dtype": """The data type of the ``data_in`` argument, one of ``['float32', 'float64', 'complex64',
            'complex128']``. It must have the same data type as that of the FFT operand for prolog functions or the FFT
            result for epilog functions.""",
    "user_info_dtype": """The data type of the ``user_info`` argument. It must be one of ``['float32', 'float64',
            'complex64', 'complex128']`` or an object of type :class:`numba.types.Type`. The offset is computed based on
            the memory layout (shape and strides) of the operand (input for prolog, output for epilog). If the user
            would like to pass additional tensor as `user_info` and access it based on the offset, it is crucial to know
            memory layout of the operand. Please note, the actual layout of the input tensor may differ from the layout
            of the tensor passed to fft call. To learn the memory layout of the input or output, please use stateful FFT
            API and :meth:`nvmath.fft.FFT.get_input_layout` :meth:`nvmath.fft.FFT.get_output_layout` respectively.

            .. note::
                Currently, in the callback, the position of the element in the input and output operands are described
                with a single flat offset, even if the original operand is multi-dimensional tensor.

            """,
    "compute_capability": """The target compute capability, specified as a string (``'80'``, ``'89'``, ...). The
            default is the compute capability of the current device.""",
}

compile_prolog.__doc__ = """
    compile_prolog(prolog_fn, element_dtype, user_info_dtype, *, compute_capability=None)

    Compile a Python function to LTO-IR to provide as a prolog function for
    :func:`~nvmath.fft.fft` and :meth:`~nvmath.fft.FFT.plan`.

    Args:
        prolog_fn: The prolog function to be compiled to LTO-IR. It must have the signature:
            ``prolog_fn(data_in, offset, user_info, reserved_for_future_use)``, and it
            essentially returns transformed ``data_in`` at ``offset``.

        element_dtype: {element_dtype}

        user_info_dtype: {user_info_dtype}

        compute_capability: {compute_capability}

    Returns:
        The function compiled to LTO-IR as `bytes` object.

    See Also:
        :func:`~nvmath.fft.fft`, :meth:`~nvmath.fft.FFT.plan`,
        :meth:`~nvmath.fft.compile_epilog`.

    Notes:
        - The user must ensure that the specified argument types meet the requirements
          listed above.
""".format(**SHARED_FFT_HELPER_DOCUMENTATION)
compile_prolog.__name__ = "compile_prolog"

compile_epilog.__doc__ = """
    compile_epilog(epilog_fn, element_dtype, user_info_dtype, *, compute_capability=None)

    Compile a Python function to LTO-IR to provide as an epilog function for
    :func:`~nvmath.fft.fft` and :meth:`~nvmath.fft.FFT.plan`.

    Args:
        epilog_fn: The epilog function to be compiled to LTO-IR. It must have the signature:
            ``epilog_fn(data_out, offset, data, user_info, reserved_for_future_use)``, and
            it essentially stores transformed ``data`` into ``data_out`` at ``offset``.

        element_dtype: {element_dtype}

        user_info_dtype: {user_info_dtype}

        compute_capability: {compute_capability}

    Returns:
        The function compiled to LTO-IR as `bytes` object.

    See Also:
        :func:`~nvmath.fft.fft`, :meth:`~nvmath.fft.FFT.plan`,
        :meth:`~nvmath.fft.compile_prolog`.

    Examples:

        The cuFFT library expects the end user to manage scaling of the outputs, so in order
        to replicate the ``norm`` option found in `other Python FFT libraries
        <https://numpy.org/doc/stable/reference/routines.fft.html#normalization>`_ we can
        define an epilog which performs the scaling.

        >>> import cupy as cp
        >>> import nvmath
        >>> import math

        Create the data for a batched 1-D FFT.

        >>> B, N = 256, 1024
        >>> a = cp.random.rand(B, N, dtype=cp.float64) + 1j * cp.random.rand(B, N, dtype=cp.float64)

        Compute a normalization factor that will create unitary transforms.

        >>> norm_factor = 1.0 / math.sqrt(N)

        Define the epilog function for the FFT.

        >>> def rescale(data_out, offset, data, user_info, unused):
        ...     data_out[offset] = data * norm_factor

        Compile the epilog to LTO-IR. In a system with GPUs that have different compute
        capability, the `compute_capability` option must be specified to the
        `compile_prolog` or `compile_epilog` helpers. Alternatively, the epilog can be
        compiled in the context of the device where the FFT to which the epilog is provided
        is executed. In this case we use the current device context, where the operands have
        been created.

        >>> with cp.cuda.Device():
        ...     epilog = nvmath.fft.compile_epilog(rescale, "complex128", "complex128")

        Perform the forward FFT, applying the rescaling as a epilog.

        >>> r = nvmath.fft.fft(a, axes=[-1], epilog=dict(ltoir=epilog))

        Test that the fused FFT run result matches the result of other libraries.

        >>> s = cp.fft.fftn(a, axes=[-1], norm="ortho")
        >>> assert cp.allclose(r, s)

    Notes:
        - The user must ensure that the specified argument types meet the requirements
          listed above.
""".format(**SHARED_FFT_HELPER_DOCUMENTATION)
compile_prolog.__name__ = "compile_prolog"
