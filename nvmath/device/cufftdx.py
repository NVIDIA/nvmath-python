# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

__all__ = ["fft", "FFTOptions"]
from functools import cached_property
import warnings

from .common import (
    make_binary_tempfile,
    check_in,
    SHARED_DEVICE_DOCSTRINGS,
)
from .common_cuda import MAX_SUPPORTED_CC, get_default_code_type, Code, CodeType, ComputeCapability, Dim3
from .common_backend import MATHDX_TYPES_TO_NP, get_isa_version, get_lto
from .common_numba import NP_TYPES_TO_NUMBA_FE_TYPES
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
from .cufftdx_numba import codegen
from nvmath.internal.utils import docstring_decorator

from nvmath.bindings import mathdx

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
class FFTOptions:
    """
    A class that encapsulates a partial FFT device function. A partial device function can
    be queried for available or optimal values for some knobs (such as `ffts_per_block`
    or `elements_per_thread`). It does not contain a compiled, ready-to-use,
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

        execute_api:
            .. versionchanged:: 0.5.0
                execute_api is not part of the FFT type. Pass this argument to
                :py:func:`nvmath.device.fft` instead.

    Note:
        The class is not meant to be used directly with its constructor. Users are instead
        advised to use :func:`fft` to create the object.

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
        if (code_type.cc.major, code_type.cc.minor) > MAX_SUPPORTED_CC:
            raise RuntimeError(
                "The maximum compute capability currently supported by device "
                f"APIs is {MAX_SUPPORTED_CC}, "
                f"got {code_type.cc.major}.{code_type.cc.minor}"
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

        return self._get_knobs(*knobs)

    def definition(self):
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
        return dd

    def create(self, **kwargs):
        dd = self.definition()
        dd.update(**kwargs)
        return fft(**dd)

    #
    # Private implementations
    #

    def _suggested(self, what):
        # Generate full PTX
        h = generate_FFT(
            size=self.size,
            precision=self.precision,
            fft_type=self.fft_type,
            direction=self.direction,
            ffts_per_block=(None if self.ffts_per_block == "suggested" else self.ffts_per_block),
            elements_per_thread=(None if self.elements_per_thread == "suggested" else self.elements_per_thread),
            real_fft_options=frozenset(self.real_fft_options.items()) if self.real_fft_options else None,
            code_type=self.code_type,
            execution=self.execution,
            # TODO: remove after migrating to libmathdx 0.2.2+
            execute_api="register_memory",
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
            size=self.size,
            precision=self.precision,
            fft_type=self.fft_type,
            direction=self.direction,
            ffts_per_block=self.ffts_per_block,
            elements_per_thread=self.elements_per_thread,
            real_fft_options=frozenset(self.real_fft_options.items()) if self.real_fft_options else None,
            code_type=self.code_type,
            execution=self.execution,
            # TODO: remove after migrating to libmathdx 0.2.2+
            execute_api="register_memory",
        )

        return get_knobs(h.descriptor, knobs)


class FFTOptionsComplete(FFTOptions):
    def __init__(self, **kwargs):
        FFTOptions.__init__(self, **kwargs)

        h = generate_FFT(
            size=self.size,
            precision=self.precision,
            fft_type=self.fft_type,
            direction=self.direction,
            code_type=self.code_type,
            execution=self.execution,
            ffts_per_block=self.ffts_per_block if self.execution == "Block" else None,
            elements_per_thread=self.elements_per_thread if self.execution == "Block" else None,
            real_fft_options=frozenset(self.real_fft_options.items()) if self.real_fft_options else None,
            # TODO: remove after migrating to libmathdx 0.2.2+
            execute_api="register_memory",
        ).descriptor

        self._value_type = MATHDX_TYPES_TO_NP[get_data_type_trait(h, mathdx.CufftdxTraitType.VALUE_TYPE)]
        self._input_type = MATHDX_TYPES_TO_NP[get_data_type_trait(h, mathdx.CufftdxTraitType.INPUT_TYPE)]
        self._output_type = MATHDX_TYPES_TO_NP[get_data_type_trait(h, mathdx.CufftdxTraitType.OUTPUT_TYPE)]

        self._storage_size = get_int_trait(h, mathdx.CufftdxTraitType.STORAGE_SIZE)
        self._stride = get_int_trait(h, mathdx.CufftdxTraitType.STRIDE)
        self._elements_per_thread = get_int_trait(h, mathdx.CufftdxTraitType.ELEMENTS_PER_THREAD)
        self._implicit_type_batching = get_int_trait(h, mathdx.CufftdxTraitType.IMPLICIT_TYPE_BATCHING)

        self._workspace_size = 0
        if self.execution == "Block":
            self._block_dim = Dim3(*get_int_traits(h, mathdx.CufftdxTraitType.BLOCK_DIM, 3))
            self._shared_memory_size = get_int_trait(h, mathdx.CufftdxTraitType.SHARED_MEMORY_SIZE)
            self._ffts_per_block = get_int_trait(h, mathdx.CufftdxTraitType.FFTS_PER_BLOCK)
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
        _workspace_deprecation_warning()
        return False

    @property
    def workspace_size(self):
        return self._workspace_size

    @property
    def implicit_type_batching(self):
        return self._implicit_type_batching


class FFTCompiled(FFTOptionsComplete):
    def __init__(self, **kwargs):
        execute_api = kwargs.pop("execute_api", None)
        super().__init__(**kwargs)

        # Fixup typo introduced in earlier versions.
        if execute_api == "registry_memory":
            warnings.warn(
                "The execute_api 'registry_memory' is deprecated and will be "
                "removed in future releases. "
                "Please use 'register_memory' instead.",
                DeprecationWarning,
            )
            execute_api = "register_memory"

        validate_execute_api(self.execution, execute_api)

        if execute_api is None:
            execute_api = "register_memory"

        self._execute_api = execute_api

        h = generate_FFT(
            size=self.size,
            precision=self.precision,
            fft_type=self.fft_type,
            direction=self.direction,
            code_type=self.code_type,
            execution=self.execution,
            ffts_per_block=self.ffts_per_block if self.execution == "Block" else None,
            elements_per_thread=self.elements_per_thread if self.execution == "Block" else None,
            real_fft_options=frozenset(self.real_fft_options.items()) if self.real_fft_options else None,
            execute_api=execute_api,
        ).descriptor

        code = generate_code(h, self.code_type.cc)

        # Compile
        lto_fn = get_lto(code.descriptor)
        isa_version = get_isa_version(code.descriptor)

        self._ltos = [Code(self.code_type, isa_version, lto_fn)]

        self._symbol = get_str_trait(h, mathdx.CufftdxTraitType.SYMBOL_NAME)

    @cached_property
    def _tempfiles(self):
        """
        Create temporary files for the LTO functions.
        """
        return [make_binary_tempfile(lto.data, ".ltoir") for lto in self._ltos]

    @property
    def files(self):
        return [v.name for v in self._tempfiles]

    @property
    def symbol(self):
        return self._symbol

    @property
    def codes(self):
        return self._ltos

    def workspace(self):
        _workspace_deprecation_warning()
        raise NotImplementedError("Workspace not supported yet")

    @property
    def execute_api(self):
        return self._execute_api

    def definition(self):
        dd = super().definition()
        dd.update(execute_api=self.execute_api)
        return dd


class FFTNumba(FFTCompiled):
    def __init__(self, **kwargs):
        if "code_type" not in kwargs:
            kwargs["code_type"] = get_default_code_type()

        FFTCompiled.__init__(self, **kwargs)

        codegen(
            {
                "value_type": self.value_type,
                "symbol": self._symbol,
                "execute_api": self._execute_api,
                "execution": self.execution,
            },
            self,
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

        code_type (CodeType): {code_type}. Optional if compiler is specified as ``'numba'``.

        execution (str): {execution}

        direction (str): {direction}

        ffts_per_block (int): {ffts_per_block}

        elements_per_thread (int): {elements_per_thread}

        real_fft_options (dict): {real_fft_options} User may specify the following options
            in the dictionary:

            - ``'complex_layout'``, currently supports ``'natural'``, ``'packed'``, and
              ``'full'``.
            - ``'real_mode'``, currently supports ``'normal'`` and ``'folded'``.

        execute_api (str): {execute_api}

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
