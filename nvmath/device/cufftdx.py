# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["fft", "FFT", "compile_fft_execute"]
from functools import cached_property
from typing import Any
import warnings

from .common import (
    parse_code_type,
    check_code_type,
    SHARED_DEVICE_DOCSTRINGS,
    parse_sm,
)
from .common_cuda import (
    Code,
    Dim3,
)
from .common_backend import MATHDX_TYPES_TO_NP, get_isa_version, get_lto
from .cufftdx_backend import (
    generate_code,
    get_int_trait,
    get_knobs,
    get_str_trait,
    get_int_traits,
    get_data_type_trait,
    validate,
    generate_FFT,
    validate_execute_api,
)
from nvmath.internal.utils import docstring_decorator

from nvmath.bindings import mathdx

from ._deprecated import deprecated

CUFFTDX_DATABASE = None

FFTDX_DOCSTRING = SHARED_DEVICE_DOCSTRINGS.copy()
FFTDX_DOCSTRING.update(
    {
        "size": """\
The size of the FFT to calculate.""".replace("\n", " "),
        #
        "fft_type": """\
A string specifying the type of FFT operation, can be ``'c2c'``, ``'c2r'`` or ``'r2c'``.""".replace("\n", " "),
        #
        "direction": """\
A string specifying the direction of FFT, can be ``'forward'`` or ``'inverse'``. If not provided, will be ``'forward'``
if complex-to-real FFT is specified and ``'inverse'`` if real-to-complex FFT is specified.""".replace("\n", " "),
        #
        "ffts_per_block": """\
The number of FFTs calculated per CUDA block, optional. The default is 1. Alternatively, if provided as ``'suggested'``
will be set to a suggested value""".replace("\n", " "),
        #
        "elements_per_thread": """\
The number of elements per thread, optional. The default is 1. Alternatively, if provided as ``'suggested'``, will be
set to a suggested value. """.replace("\n", " "),
        #
        "real_fft_options": """\
A dictionary specifying the options for real FFT operation, optional.""".replace("\n", " "),
        "execute_api": """\
A string specifying the signature of the function that handles problems with input in register or in shared memory buffers.
Could be ``'shared_memory'`` or ``'register_memory'``.""".replace("\n", " "),
    }
)

_workspace_deprecation_warning = lambda: warnings.warn(
    "Using workspaces is deprecated and will be removed in future release.", DeprecationWarning
)


@docstring_decorator(FFTDX_DOCSTRING, skip_missing=False)
class FFT:
    """
    A class that encapsulates a partial FFT device function. A partial device function can
    be queried for available or optimal values for some knobs (such as `ffts_per_block`
    or `elements_per_thread`). It does not contain a compiled, ready-to-use,
    device function until finalized using :meth:`create`.

    .. versionchanged:: 0.7.0
        `FFT` has replaced `FFTOptions` and `FFTOptionsComplete`.

    Args:
        size (int): {size}

        precision (str): {precision}

        fft_type (str): {fft_type}

        sm (ComputeCapability): {sm}

        execution (str): {execution}

        direction (str): {direction}

        ffts_per_block (int): {ffts_per_block}

        elements_per_thread (int): {elements_per_thread}

        real_fft_options (dict): {real_fft_options} User may specify the following options
            in the dictionary:

            - ``'complex_layout'``, currently supports ``'natural'``, ``'packed'``, and
              ``'full'``.
            - ``'real_mode'``, currently supports ``'normal'`` and ``'folded``.

    .. seealso::
        The attributes of this class provide a 1:1 mapping with the CUDA C++ cuFFTDx APIs.
        For further details, please refer to `cuFFTDx documentation
        <https://docs.nvidia.com/cuda/cufftdx/index.html>`_.
    """

    def __init__(
        self,
        size,
        precision,
        fft_type,
        execution,
        *,
        sm=None,
        direction=None,
        ffts_per_block=None,
        elements_per_thread=None,
        real_fft_options=None,
    ):
        sm = parse_sm(sm)

        #
        # Check that the knobs are, individually, valid
        #

        validate(
            size=size,
            precision=precision,
            fft_type=fft_type,
            sm=sm,
            execution=execution,
            direction=direction,
            ffts_per_block=ffts_per_block,
            elements_per_thread=elements_per_thread,
            real_fft_options=real_fft_options,
        )

        if direction is None and fft_type == "r2c":
            direction = "forward"
        elif direction is None and fft_type == "c2r":
            direction = "inverse"

        #
        # Traits set by input
        #

        self._precision = precision
        self._fft_type = fft_type
        self._direction = direction
        self._size = size
        self._execution = execution
        self._ffts_per_block = ffts_per_block
        self._elements_per_thread = elements_per_thread
        self._real_fft_options = real_fft_options
        self._sm = sm

        #
        # Update suggested traits
        #

        if elements_per_thread == "suggested":
            self._elements_per_thread = self._suggested_elements_per_thread

        if ffts_per_block == "suggested":
            self._ffts_per_block = self._suggested_ffts_per_block

    @cached_property
    def _traits(self):
        return _FFTTraits(self)

    @property
    def elements_per_thread(self):
        if self._elements_per_thread is None:
            return self._traits.elements_per_thread
        return self._elements_per_thread

    @property
    def precision(self):
        return self._precision

    @property
    def ffts_per_block(self):
        if self._ffts_per_block is None:
            return self._traits.ffts_per_block
        return self._ffts_per_block

    @property
    def fft_type(self):
        return self._fft_type

    @property
    def direction(self):
        return self._direction

    @property
    def size(self):
        return self._size

    @property
    def execution(self):
        return self._execution

    @property
    def real_fft_options(self):
        return self._real_fft_options

    @property
    def sm(self):
        return self._sm

    #
    # Extensions
    #

    def valid(self, *knobs):
        if not (set(knobs) <= {"ffts_per_block", "elements_per_thread"}):
            raise ValueError(f"Unsupported knob. Only valid knobs are ffts_per_block and elements_per_thread but got {knobs}")

        return self._get_knobs(*knobs)

    @deprecated("definition is deprecated and may be removed in future versions")
    def definition(self):
        """
        .. deprecated:: 0.7.0
        """
        dd = {
            "size": self.size,
            "precision": self.precision,
            "fft_type": self.fft_type,
            "sm": self.sm,
            "execution": self.execution,
            "direction": self.direction,
            "ffts_per_block": self.ffts_per_block,
            "elements_per_thread": self.elements_per_thread,
            "real_fft_options": self.real_fft_options,
        }
        return dd

    @deprecated("create is deprecated and may be removed in future versions. Use `functools.partial` instead")
    def create(self, **kwargs):
        """
        Creates a copy of the instance with provided arguments updated.

        .. deprecated:: 0.7.0
            Please use :py:func:`functools.partial` instead.
        """
        code_type = kwargs.pop("code_type", None)
        if code_type is not None:
            DeprecationWarning("code_type is deprecated and will be removed in future releases. It is no longer needed.")
        compiler = kwargs.pop("compiler", None)
        if compiler is not None:
            DeprecationWarning("compiler is deprecated and will be removed in future releases. It is no longer needed.")
        dd = self.definition()
        dd.update(**kwargs)
        return FFT(**dd)

    @property
    def value_type(self):
        return self._traits.value_type

    @property
    def input_type(self):
        return self._traits.input_type

    @property
    def output_type(self):
        return self._traits.output_type

    @property
    def storage_size(self):
        return self._traits.storage_size

    @property
    def shared_memory_size(self):
        return self._traits.shared_memory_size

    @property
    def stride(self):
        return self._traits.stride

    @property
    def block_dim(self):
        return self._traits.block_dim

    @property
    def requires_workspace(self):
        _workspace_deprecation_warning()
        return False

    @property
    def workspace_size(self):
        return self._traits.workspace_size

    @property
    def implicit_type_batching(self):
        return self._traits.implicit_type_batching

    @property
    def extensions(self):
        raise NotImplementedError("Extensions not supported yet")

    def execute(*args):
        raise RuntimeError("execute is a device function and can not be called on host.")

    @deprecated("Calling MM(...) directly is deprecated, please use MM.execute(...) method instead.")
    def __call__(self, *args):
        raise RuntimeError("__call__ is a device function and can not be called on host.")

    @property
    @deprecated("files is deprecated and is no longer required and will be removed in future releases.")
    def files(self) -> list:
        return []

    #
    # Private implementations
    #

    def _suggested(self, what):
        # Generate full PTX
        h = generate_FFT(
            size=self._size,
            precision=self._precision,
            fft_type=self._fft_type,
            direction=self._direction,
            ffts_per_block=(None if self._ffts_per_block == "suggested" else self._ffts_per_block),
            elements_per_thread=(None if self._elements_per_thread == "suggested" else self._elements_per_thread),
            real_fft_options=frozenset(self._real_fft_options.items()) if self._real_fft_options else None,
            sm=self._sm,
            execution=self._execution,
        )

        if what == "elements_per_thread":
            return get_int_trait(h.descriptor, mathdx.CufftdxTraitType.ELEMENTS_PER_THREAD)

        if what == "suggested_ffts_per_block":
            return get_int_trait(h.descriptor, mathdx.CufftdxTraitType.SUGGESTED_FFTS_PER_BLOCK)

        raise Exception(f"Unknown suggested option '{what}'")

    @cached_property
    def _suggested_ffts_per_block(self):
        return self._suggested("suggested_ffts_per_block")

    @cached_property
    def _suggested_elements_per_thread(self):
        return self._suggested("elements_per_thread")

    def _get_knobs(self, *knobs):
        if not (set(knobs) <= {"ffts_per_block", "elements_per_thread"}):
            raise ValueError(f"Unsupported knob. Only valid knobs are ffts_per_block and elements_per_thread but got {knobs}")

        h = generate_FFT(
            size=self._size,
            precision=self._precision,
            fft_type=self._fft_type,
            direction=self._direction,
            ffts_per_block=self._ffts_per_block,
            elements_per_thread=self._elements_per_thread,
            real_fft_options=frozenset(self._real_fft_options.items()) if self._real_fft_options else None,
            sm=self._sm,
            execution=self._execution,
        )

        return get_knobs(h.descriptor, knobs)


class _FFTTraits:
    def __init__(self, FFT: FFT):
        h = generate_FFT(
            size=FFT._size,
            precision=FFT._precision,
            fft_type=FFT._fft_type,
            direction=FFT._direction,
            sm=FFT._sm,
            execution=FFT._execution,
            ffts_per_block=FFT._ffts_per_block if FFT._execution == "Block" else None,
            elements_per_thread=FFT._elements_per_thread if FFT._execution == "Block" else None,
            real_fft_options=frozenset(FFT._real_fft_options.items()) if FFT._real_fft_options else None,
        ).descriptor

        self.value_type = MATHDX_TYPES_TO_NP[get_data_type_trait(h, mathdx.CufftdxTraitType.VALUE_TYPE)]
        self.input_type = MATHDX_TYPES_TO_NP[get_data_type_trait(h, mathdx.CufftdxTraitType.INPUT_TYPE)]
        self.output_type = MATHDX_TYPES_TO_NP[get_data_type_trait(h, mathdx.CufftdxTraitType.OUTPUT_TYPE)]

        self.storage_size = get_int_trait(h, mathdx.CufftdxTraitType.STORAGE_SIZE)
        self.stride = get_int_trait(h, mathdx.CufftdxTraitType.STRIDE)
        self.elements_per_thread = get_int_trait(h, mathdx.CufftdxTraitType.ELEMENTS_PER_THREAD)
        self.implicit_type_batching = get_int_trait(h, mathdx.CufftdxTraitType.IMPLICIT_TYPE_BATCHING)

        self.workspace_size = 0
        if FFT.execution == "Block":
            self.block_dim: Dim3 | None = Dim3(*get_int_traits(h, mathdx.CufftdxTraitType.BLOCK_DIM, 3))
            self.shared_memory_size: int | None = get_int_trait(h, mathdx.CufftdxTraitType.SHARED_MEMORY_SIZE)
            self.ffts_per_block: int | None = get_int_trait(h, mathdx.CufftdxTraitType.FFTS_PER_BLOCK)
        else:
            self.block_dim = None
            self.shared_memory_size = None
            self.ffts_per_block = None


def compile_fft_execute(
    fft: FFT,
    code_type: Any,
    execute_api: str | None = None,
) -> tuple[Code, str]:
    code_type = parse_code_type(code_type)
    check_code_type(code_type, "cuFFTDx")
    validate_execute_api(fft.execution, execute_api)

    h = generate_FFT(
        size=fft._size,
        precision=fft._precision,
        fft_type=fft._fft_type,
        direction=fft._direction,
        sm=fft._sm,
        execution=fft._execution,
        ffts_per_block=fft._ffts_per_block if fft._execution == "Block" else None,
        elements_per_thread=fft._elements_per_thread if fft._execution == "Block" else None,
        real_fft_options=frozenset(fft._real_fft_options.items()) if fft._real_fft_options else None,
        execute_api=execute_api,
    ).descriptor

    code = generate_code(h, code_type.cc)

    # Compile
    lto_fn = get_lto(code.descriptor)
    isa_version = get_isa_version(code.descriptor)

    symbol = get_str_trait(h, mathdx.CufftdxTraitType.SYMBOL_NAME)

    return Code(code_type, isa_version, lto_fn), symbol


@docstring_decorator(FFTDX_DOCSTRING, skip_missing=False)
def fft(*, compiler=None, code_type=None, execute_api=None, **kwargs):
    """
    Create an :class:`FFT` object that encapsulates a compiled and ready-to-use FFT
    device function.

    .. deprecated:: 0.7.0

    Args:
        size (int): {size}

        precision (str): {precision}

        fft_type (str): {fft_type}

        sm (ComputeCapability): {sm}

        execution (str): {execution}

        direction (str): {direction}

        ffts_per_block (int): {ffts_per_block}

        elements_per_thread (int): {elements_per_thread}

        real_fft_options (dict): {real_fft_options} User may specify the following options
            in the dictionary:

            - ``'complex_layout'``, currently supports ``'natural'``, ``'packed'``, and
              ``'full'``.
            - ``'real_mode'``, currently supports ``'normal'`` and ``'folded'``.

        compiler: {compiler}

            .. versionchanged:: 0.7.0
                compiler is no longer needed and does not take effect. Use
                :py:func:`nvmath.device.compile_fft_execute` to get device
                function code.

        code_type (CodeType): {code_type}

            .. versionchanged:: 0.7.0
                code_type should be used by
                :py:func:`nvmath.device.compile_fft_execute` and no longer
                needed for numba-cuda usage.

        execute_api (str): {execute_api}

            .. versionchanged:: 0.7.0
                execute_api should be used by
                :py:func:`nvmath.device.compile_fft_execute` and no longer
                needed for numba-cuda usage.

    .. seealso::
        The attributes of :class:`FFT` provide a 1:1 mapping with the CUDA C++
        cuFFTDx APIs. For further details, please refer to `cuFFTDx documentation
        <https://docs.nvidia.com/cuda/cufftdx/index.html>`_.

    Examples:
        Examples can be found in the `nvmath/examples/device
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/device>`_ directory.
    """
    DeprecationWarning("fft is deprecated and will be removed in future releases. Please use FFT class directly.")
    if code_type is not None:
        DeprecationWarning("code_type is deprecated and will be removed in future releases. It is no longer needed.")
    if compiler is not None:
        DeprecationWarning("compiler is deprecated and will be removed in future releases. It is no longer needed.")
    if execute_api is not None:
        DeprecationWarning("execute_api is deprecated and will be removed in future releases. It is no longer needed.")
    return FFT(**kwargs)
