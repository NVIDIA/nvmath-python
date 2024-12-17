# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["fft", "FFTOptions"]
from functools import cached_property
import os

from .common import (
    make_binary_tempfile,
    check_in,
    find_unsigned,
    find_dim3,
    find_mangled_name,
    SHARED_DEVICE_DOCSTRINGS,
)
from .common_cuda import get_default_code_type, Code, CodeType, Symbol, ComputeCapability, Dim3
from .common_cpp import enum_to_np
from .common_mathdx import MATHDX_HOME
from .common_numba import NP_TYPES_TO_NUMBA_FE_TYPES
from .cufftdx_backend import validate, generate_block, generate_thread
from .cufftdx_db import cuFFTDxDatabase
from .cufftdx_numba import codegen
from .cufftdx_workspace import Workspace
from .nvrtc import compile
from .._internal.utils import docstring_decorator

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
    }
)


@docstring_decorator(FFTDX_DOCSTRING, skip_missing=False)
class FFTOptions:
    """
    A class that encapsulates a partial FFT device function. A partial device function can
    be queried for available or optimal values for the some knobs (such as
    `leading_dimension` or `block_dim`). It does not contain a compiled, ready-to-use,
    device function until finalized using :meth:`create`.

    Args:
        size (int): {size}

        precision (str): {precision}

        fft_type (str): {fft_type}

        code_type (CodeType): {code_type}

        execution (str): {execution}

        direction (str): {direction}

        ffts_per_block (int): {ffts_per_block}

        elements_per_thread (int): {elements_per_thread}

        real_fft_options (dict): {real_fft_options} User may specify the following options
        in the dictionary:

            - ``'complex_layout'``, currently supports ``'natural'``, ``'packed'``, and
              ``'full'``.
            - ``'real_mode'``, currently supports ``'normal'`` and ``'folded``.

    Note:
        The class is not meant to used directly with its constructor. Users are instead
        advised to use :func:`fft` create the object.

    See Also:
        The attributes of this class provide a 1:1 mapping with the CUDA C++ cuFFTDx APIs.
        For further details, please refer to `cuFFTDx documentation
        <https://docs.nvidia.com/cuda/cufftdx/index.html>`_.
    """

    def __init__(
        self,
        size,
        precision,
        fft_type,
        code_type,
        execution,
        *,
        direction=None,
        ffts_per_block=None,
        elements_per_thread=None,
        real_fft_options=None,
    ):
        if len(code_type) != 2:
            raise ValueError(f"code_type should be an instance of CodeType or a 2-tuple ; got code_type = {code_type}")
        code_type = CodeType(code_type[0], ComputeCapability(*code_type[1]))
        if code_type.cc.major < 7:
            raise RuntimeError(
                f"Minimal compute capability 7.0 is required by cuFFTDx, got {code_type.cc.major}.{code_type.cc.minor}"
            )

        #
        # Check that the knobs are, individually, valid
        #

        validate(
            size=size,
            precision=precision,
            fft_type=fft_type,
            code_type=code_type,
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
        self._code_type = code_type

        #
        # Update suggested traits
        #

        if elements_per_thread == "suggested":
            self._elements_per_thread = self._suggested_elements_per_thread

        if ffts_per_block == "suggested":
            self._ffts_per_block = self._suggested_ffts_per_block

    @property
    def elements_per_thread(self):
        return self._elements_per_thread

    @property
    def precision(self):
        return self._precision

    @property
    def ffts_per_block(self):
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
    def code_type(self):
        return self._code_type

    #
    # Extensions
    #

    def valid(self, *knobs):
        if not (set(knobs) <= {"ffts_per_block", "elements_per_thread"}):
            raise ValueError(f"Unsupported knob. Only valid knobs are ffts_per_block and elements_per_thread but got {knobs}")

        constraints = {
            "fft_type": self.fft_type,
            "size": self.size,
            "execution": self.execution,
            "arch": self.code_type.cc,
            "precision": self.precision,
            "direction": self.direction,
        }
        if self.real_fft_options is not None:
            constraints["real_fft_options"] = self.real_fft_options
        if self.elements_per_thread is not None:
            constraints["elements_per_thread"] = self.elements_per_thread
        if self.ffts_per_block is not None:
            constraints["ffts_per_block"] = self.ffts_per_block

        global CUFFTDX_DATABASE
        if CUFFTDX_DATABASE is None:
            CUFFTDX_DATABASE = cuFFTDxDatabase.create(os.path.join(MATHDX_HOME, "include/cufftdx/include/database/records/"))

        return CUFFTDX_DATABASE.query(knobs, constraints)

    def create(self, **kwargs):
        dd = {
            "size": self.size,
            "precision": self.precision,
            "fft_type": self.fft_type,
            "code_type": self.code_type,
            "execution": self.execution,
            "direction": self.direction,
            "ffts_per_block": self.ffts_per_block,
            "elements_per_thread": self.elements_per_thread,
            "real_fft_options": self.real_fft_options,
        }
        dd.update(kwargs)
        return fft(**dd)

    #
    # Private implementations
    #

    def _valid(self, knob):
        if knob == "elements_per_thread":
            return [self._suggested_elements_per_thread]
        elif knob == "ffts_per_block":
            return [1, self._suggested_ffts_per_block]
        else:
            raise ValueError("Unsupported knob")

    def _suggested(self, what):
        # Generate full PTX
        cpp = generate_block(
            size=self.size,
            precision=self.precision,
            fft_type=self.fft_type,
            direction=self.direction,
            ffts_per_block=(None if self.ffts_per_block == "suggested" else self.ffts_per_block),
            elements_per_thread=(None if self.elements_per_thread == "suggested" else self.elements_per_thread),
            real_fft_options=self.real_fft_options,
            code_type=self.code_type,
        )
        _, ptx = compile(cpp=cpp["cpp"], cc=self.code_type.cc, rdc=True, code="ptx")
        return find_unsigned(what, ptx)

    @cached_property
    def _suggested_ffts_per_block(self):
        return self._suggested("suggested_ffts_per_block")

    @cached_property
    def _suggested_elements_per_thread(self):
        return self._suggested("elements_per_thread")


class FFTOptionsComplete(FFTOptions):
    def __init__(self, **kwargs):
        FFTOptions.__init__(self, **kwargs)

        if self.execution == "Block":
            self._cpp = generate_block(
                size=self.size,
                precision=self.precision,
                fft_type=self.fft_type,
                direction=self.direction,
                code_type=self.code_type,
                ffts_per_block=self.ffts_per_block,
                elements_per_thread=self.elements_per_thread,
                real_fft_options=self.real_fft_options,
            )
        else:
            self._cpp = generate_thread(
                size=self.size,
                precision=self.precision,
                fft_type=self.fft_type,
                direction=self.direction,
                code_type=self.code_type,
                real_fft_options=self.real_fft_options,
            )

        _, self._ptx = compile(cpp=self._cpp["cpp"], cc=self.code_type.cc, rdc=True, code="ptx")

        # Look into PTX for traits
        self._value_type = enum_to_np(find_unsigned("value_type", self._ptx))
        self._input_type = enum_to_np(find_unsigned("input_type", self._ptx))
        self._output_type = enum_to_np(find_unsigned("output_type", self._ptx))
        self._storage_size = find_unsigned("storage_size", self._ptx)
        self._stride = find_unsigned("stride", self._ptx)
        self._elements_per_thread = find_unsigned("elements_per_thread", self._ptx)
        self._implicit_type_batching = find_unsigned("implicit_type_batching", self._ptx)
        self._workspace_size = 0
        self._requires_workspace = False
        if self.execution == "Block":
            self._block_dim = Dim3(*find_dim3("block_dim", self._ptx))
            self._shared_memory_size = find_unsigned("shared_memory_size", self._ptx)
            self._ffts_per_block = find_unsigned("ffts_per_block", self._ptx)
        else:
            self._block_dim = None
            self._shared_memory_size = None
            self._ffts_per_block = None

    @property
    def value_type(self):
        return self._value_type

    @property
    def input_type(self):
        return self._input_type

    @property
    def output_type(self):
        return self._output_type

    @property
    def storage_size(self):
        return self._storage_size

    @property
    def shared_memory_size(self):
        return self._shared_memory_size

    @property
    def stride(self):
        return self._stride

    @property
    def block_dim(self):
        return self._block_dim

    @property
    def requires_workspace(self):
        return self._requires_workspace

    @property
    def workspace_size(self):
        return self._workspace_size

    @property
    def implicit_type_batching(self):
        return self._implicit_type_batching


class FFTCompiled(FFTOptionsComplete):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        version, lto_fn = compile(cpp=self._cpp["cpp"], cc=self.code_type.cc, rdc=True, code="lto")
        self._ltos = [Code(self.code_type, version, lto_fn)]

        if self.execution == "Block":
            apis = ["thread", "smem"]
        else:
            apis = ["thread"]
        self._symbols = {api: find_mangled_name(self._cpp["names"][api], self._ptx) for api in apis}
        self._tempfiles = [make_binary_tempfile(lto_fn, ".ltoir")]

    @property
    def files(self):
        return list(v.name for v in self._tempfiles)

    @property
    def symbols(self):
        return [Symbol(k, v) for k, v in self._symbols.items()]

    @property
    def codes(self):
        return self._ltos

    def workspace(self):
        raise NotImplementedError("Workspace not supported yet")


class FFTNumba(FFTCompiled):
    def __init__(self, **kwargs):
        if "code_type" not in kwargs:
            kwargs["code_type"] = get_default_code_type()

        FFTCompiled.__init__(self, **kwargs)

        self._codegened = codegen(
            {
                "value_type": self.value_type,
                "scale": None,
                "symbols": self._symbols,
                "requires_workspace": self.requires_workspace,
                "execution": self.execution,
            },
            self,
            Workspace,
        )

    def __call__(self, *args):
        raise Exception("__call__ should not be called directly outside of a numba.cuda.jit(...) kernel.")

    @property
    def value_type(self):
        return NP_TYPES_TO_NUMBA_FE_TYPES[super(FFTCompiled, self).value_type]

    @property
    def input_type(self):
        return NP_TYPES_TO_NUMBA_FE_TYPES[super(FFTCompiled, self).input_type]

    @property
    def output_type(self):
        return NP_TYPES_TO_NUMBA_FE_TYPES[super(FFTCompiled, self).output_type]

    @property
    def extensions(self):
        raise NotImplementedError("Extensions not supported yet")


@docstring_decorator(FFTDX_DOCSTRING, skip_missing=False)
def fft(*, compiler=None, **kwargs):
    """
    Create an :class:`FFTOptions` object that encapsulates a compiled and ready-to-use FFT
    device function.

    Args:
        size (int): {size}

        precision (str): {precision}

        fft_type (str): {fft_type}

        compiler (str): {compiler}

        code_type (CodeType): {code_type}. Optional if compiler is specified as ``'Numba'``.

        execution (str): {execution}

        direction (str): {direction}

        ffts_per_block (int): {ffts_per_block}

        elements_per_thread (int): {elements_per_thread}

        real_fft_options (dict): {real_fft_options} User may specify the following options
        in the dictionary:

            - ``'complex_layout'``, currently supports ``'natural'``, ``'packed'``, and
              ``'full'``.
            - ``'real_mode'``, currently supports ``'normal'`` and ``'folded'``.

    See Also:
        The attributes of :class:`FFTOptions` provide a 1:1 mapping with the CUDA C++
        cuFFTDx APIs. For further details, please refer to `cuFFTDx documentation
        <https://docs.nvidia.com/cuda/cufftdx/index.html>`_.

    Examples:
        Examples can be found in the `nvmath/examples/device
        <https://github.com/NVIDIA/nvmath-python/tree/main/examples/device>`_ directory.
    """
    check_in("compiler", compiler, [None, "numba"])
    if compiler is None:
        return FFTCompiled(**kwargs)
    elif compiler == "numba":
        return FFTNumba(**kwargs)
