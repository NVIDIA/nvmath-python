# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device APIs for RNGs: bit generators and distribution generators.

The APIs follow the cuRAND device APIs: https://docs.nvidia.com/cuda/curand/group__DEVICE.html
"""

import functools
import re
import sys

import nvmath.device
from nvmath.device import curand_kernel, random_helpers
from nvmath.device import random_states as states

# Common APIs (initialization, bit generation).
_COMMON_APIS = ["init", "rand", "rand4"]
_COMMON_APIS_PREFIX = ["curand_", "cu", "cu"]
_INIT_DOC = """init(..., state)
    Initialize the RNG state.

    The arguments depend upon the selected bit generator (see the overloads of `curand_init`
    in `cuRAND docs <https://docs.nvidia.com/cuda/curand/group__DEVICE.html>`_).

    Example:

        >>> from numba import cuda
        >>> from nvmath.device import random
        >>> compiled_apis = random.Compile(cc=None)

        We will be working on a grid of 64 blocks, with 64 threads in each.

        >>> threads = 64
        >>> blocks = 64
        >>> nthreads = blocks * threads

        Let us show how to use `init` with :class:`nvmath.device.random.StatesPhilox4_32_10`
        states. The same applies to :class:`nvmath.device.random.StatesMRG32k3a` and
        :class:`nvmath.device.random.StatesXORWOW`.

        First, create an array of states (one per thread) using
        :class:`nvmath.device.random.StatesPhilox4_32_10` constructor.

        >>> states = random.StatesPhilox4_32_10(nthreads)

        Define a kernel to initialize the states. Each thread will initialize one element of
        `states`. For the `Philox4_32_10
        <https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE_1ga613d37dacbc50494f2f859ef0d378b8>`_
        generator, the `init` arguments are: `seed`, `subsequence`, `offset`.

        >>> @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
        ... def setup(states):
        ...     i = cuda.grid(1)
        ...     random.init(1234, i, 0, states[i])

        Run the kernel to initialize the states:

        >>> setup[blocks, threads](states)

        Now, you can use the `states` array to generate random numbers using the random
        samplers available.

        For Sobol' family of quasirandom number generators, initialization is a bit more
        complex as it requires preparing a set of *direction vectors* and *scramble
        constants*. In this example, we will setup
        :class:`nvmath.device.random.StatesScrambledSobol64` states.

        Direction vectors can be obtained with
        :func:`nvmath.device.random_helpers.get_direction_vectors64`:

        >>> from nvmath.device import random_helpers
        >>> hostVectors = random_helpers.get_direction_vectors64(
        ...     random.random_helpers.DirectionVectorSet.SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6, nthreads)
        >>> sobolDirectionVectors = cuda.to_device(hostVectors)

        To get scramble constants, use
        :func:`nvmath.device.random_helpers.get_scramble_constants64`:

        >>> hostScrambleConstants = random_helpers.get_scramble_constants64(nthreads)
        >>> sobolScrambleConstants = cuda.to_device(hostScrambleConstants)

        As `init` expects a pointer to direction vectors, we will use `cffi` to obtain it.

        >>> states = random.StatesScrambledSobol64(nthreads)
        >>>
        >>> import cffi
        >>> ffi = cffi.FFI()
        >>>
        >>> @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
        ... def setup(sobolDirectionVectors, sobolScrambleConstants, states):
        ...     id = cuda.grid(1)
        ...     dirptr = ffi.from_buffer(sobolDirectionVectors[id:])
        ...     random.init(dirptr, sobolScrambleConstants[id], 1234, states[id])
"""
_RAND_DOC = """rand(state)
    Generate 32 or 64 bits of randomness depending on the provided bit generator.

    Args:
        state: a state object corresponding to one of the bit generators.

    Returns:
        32 or 64 bits of randomness depending on the provided bit generator.
"""
_RAND4_DOC = """rand4(state)
    Generate a 4-tuple of 32-bits of randomness from a Philox4_32_10 generator.

    Args:
        state: a state object corresponding to the Philox4_32_10 generator.

    Returns:
        A 4-tuple of 32-bits of randomness from a Philox4_32_10 generator as an uint32x4 object.
"""
_COMMON_APIS_DOC = [_INIT_DOC, _RAND_DOC, _RAND4_DOC]

# Bit generator state types.
_STATES = [
    "StatesMRG32k3a",
    "StatesPhilox4_32_10",
    "StatesSobol32",
    "StatesSobol64",
    "StatesScrambledSobol32",
    "StatesScrambledSobol64",
    "StatesTest",
    "StatesXORWOW",
]

# Sampling from distributions.
_SAMPLERS_NORMAL = ["normal", "normal_double", "normal2", "normal2_double", "normal4"]
_SAMPLERS_LOG_NORMAL = ["log_normal", "log_normal_double", "log_normal2", "log_normal2_double", "log_normal4"]
_SAMPLERS_POISSON = ["poisson", "poisson4"]
_SAMPLERS_UNIFORM = ["uniform", "uniform_double", "uniform2_double", "uniform4"]
_SAMPLERS = _SAMPLERS_NORMAL + _SAMPLERS_LOG_NORMAL + _SAMPLERS_POISSON + _SAMPLERS_UNIFORM

# Skip ahead functions.
_SKIPPERS = ["skipahead", "skipahead_sequence", "skipahead_subsequence"]

# Templates.
_NUMBER_TO_WORD = {"": "a", "2": "two", "4": "four"}

_DTYPE_NAME_MAP = {"uint": "uint32", "float": "float32", "double": "float64"}

_SAMPLER_DESCRIPTION_TEMPLATE = r"""{name}(state{extra_arguments_str})
Sample {num_values_word} {dtype_word} from {distribution_word} distribution using the specified bit generator state.
{arg_docs}
{sampler_return}

Example:

    The `states` parameter of the kernel below should be an array of already initialized bit generator states.
    See the documentation of :func:`nvmath.device.random.init` for more details on how to create and initialize the
    bit generator states.

    >>> from numba import cuda
    >>> from nvmath.device import random
    >>> compiled_apis = random.Compile()
    >>> @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    ... def kernel(states{extra_arguments_str}):
    ...     i = cuda.grid(1)
    ...     value = random.{name}(states[i]{extra_arguments_str})
"""

_DISTRIBUTION_NAME_SPECIAL_CASES = {
    "poisson": "Poisson",
    "log_normal": "log-normal",
}

_SAMPLER_COMMON_ARGS_TEMPLATE = r"""

Args:
    state: a state object corresponding to one of the bit generators.
"""

_SAMPLER_RETURN_TEMPLATE = r"""

Returns:
    A value of type :class:`{vector_type_description}`.
"""

_STATES_DESCRIPTION_TEMPLATE = r"""
States of {generator_name} bit generator.

Once created, the states can be initialized by :func:`nvmath.device.random.init` function.
See the documentation of :func:`nvmath.device.random.init` for details and examples.
"""


def _wrap(module, orig_name, new_name):
    global functools

    function = getattr(module, orig_name)
    wrapped = functools.wraps(function)(function)
    wrapped.__name__ = new_name
    return wrapped


def _wrap_sampler(new_name, *, extra_arguments_list: list | None = None, extra_arguments_doc: list[str] | None = None):
    global re
    global curand_kernel

    if extra_arguments_list is None:
        extra_arguments_list = []
    if extra_arguments_doc is None:
        extra_arguments_doc = []
    # Additional arguments for sampler, if any, other than 'state'.
    num_extra_arguments = len(extra_arguments_list)
    extra_arguments_str = "" if num_extra_arguments == 0 else ", {}".format(", ".join(extra_arguments_list))

    # Infer data needed for the description template from the name.
    m = re.match(r"([a-z_]+)([24])?(_([\w]+))?", new_name)
    if m is None:
        raise AssertionError(
            f"Random sampler name '{new_name}' is invalid or unexpected. "
            "The regex in nvmath.device.random.py should be updated."
        )
    # Distribution type.
    distribution = m.group(1)
    # Number of return values.
    num_values = "" if m.group(2) is None else m.group(2)
    num_values_word = _NUMBER_TO_WORD[num_values]
    # The dtype of the return values.
    dtype = m.group(4)
    if match_dtype_in_name := re.match("(.*)_(float|double)", distribution):
        # If dtype was parsed as a part of distribution name, extract it.
        distribution = match_dtype_in_name.group(1)
        dtype = match_dtype_in_name.group(2)
    if not dtype:
        # If dtype is not qualified in the name, uint for Poisson, float otherwise.
        dtype = "uint" if distribution == "poisson" else "float"
    distribution_word = _DISTRIBUTION_NAME_SPECIAL_CASES.get(distribution, distribution)
    dtype_word = dtype + "s" if num_values else dtype

    # Wrap function.
    wrapped = _wrap(curand_kernel, "curand_" + new_name, new_name)

    # Argument docs
    arg_docs = _SAMPLER_COMMON_ARGS_TEMPLATE
    for extra_argument, extra_argument_doc in zip(extra_arguments_list, extra_arguments_doc, strict=False):
        arg_docs += f"    {extra_argument}: {extra_argument_doc}.\n"

    # Return docs
    vector_type_description = _DTYPE_NAME_MAP[dtype]
    if num_values:
        vector_type_description = "nvmath.device." + vector_type_description + "x" + num_values
    sampler_return = _SAMPLER_RETURN_TEMPLATE.format(vector_type_description=vector_type_description)

    # Update docstring.
    description = _SAMPLER_DESCRIPTION_TEMPLATE.format(
        name=new_name,
        extra_arguments_str=extra_arguments_str,
        num_values_word=num_values_word,
        distribution_word=distribution_word,
        dtype_word=dtype_word,
        arg_docs=arg_docs,
        sampler_return=sampler_return,
    )
    wrapped.__doc__ = description

    return wrapped


def _wrap_skipahead(name):
    global re
    global curand_kernel

    # Infer data needed for the description template from the name.
    m = re.match("skipahead(_([a-z]+))?", name)
    skiptype = "element" if m.group(2) is None else m.group(2)

    # Wrap function.
    wrapped = _wrap(curand_kernel, name, name)

    description = f"""{name}(n, state)
    Update the bit generator state to skip ahead ``n`` {skiptype}s.

    Args:
        n: The number of {skiptype}s to skip ahead.
        state: The bit generator state to update.
"""
    if skiptype != "element":
        description += """
    .. note::
        For the XORWOW and Philox4_32_10 bit generators, the term *sequence* used in the API name and the argument `n`
        essentially refers to the notion of *subsequence* in these algorithms.
        """

    # Update docstring.
    wrapped.__doc__ = description

    return wrapped


def _create_symbols():
    global sys
    global curand_kernel
    global c_ext_shim_source

    random_module = sys.modules[__name__]

    # Wrap shim function.
    c_ext_shim_source = _wrap(curand_kernel, "c_ext_shim_source", "c_ext_shim_source")

    # Wrap common APIs.
    for api, prefix, doc in zip(_COMMON_APIS, _COMMON_APIS_PREFIX, _COMMON_APIS_DOC, strict=True):
        function = _wrap(curand_kernel, prefix + api, api)
        function.__doc__ = doc
        setattr(random_module, api, function)
        # TODO: Update doc string for

    # Wrap skipahead APIs.
    for skipper in _SKIPPERS:
        function = _wrap_skipahead(skipper)
        setattr(random_module, skipper, function)

    # Samplers with no extra arguments.
    for sampler in _SAMPLERS_NORMAL + _SAMPLERS_UNIFORM:
        function = _wrap_sampler(sampler)
        setattr(random_module, sampler, function)

    # The log_normal distribution requires two extra arguments.
    extra_arguments_list = ["mean", "stddev"]
    extra_arguments_doc = ["The mean value", "The standard deviation"]
    for sampler in _SAMPLERS_LOG_NORMAL:
        function = _wrap_sampler(sampler, extra_arguments_list=extra_arguments_list, extra_arguments_doc=extra_arguments_doc)
        setattr(random_module, sampler, function)

    # The poisson distribution requires an extra argument.
    extra_arguments_list = ["Lambda"]
    extra_arguments_doc = ["The parameter characterizing the Poisson distribution"]
    for sampler in _SAMPLERS_POISSON:
        function = _wrap_sampler(sampler, extra_arguments_list=extra_arguments_list, extra_arguments_doc=extra_arguments_doc)
        setattr(random_module, sampler, function)


_create_symbols()


def _wrap_states():
    global sys
    global re
    global states
    global states_arg_handlers

    random_module = sys.modules[__name__]

    # Wrap arg handlers extension.
    states_arg_handlers = states.states_arg_handlers

    for s in _STATES:
        setattr(random_module, s, getattr(states, "curand" + s))
        state = getattr(random_module, s)
        generator_name = re.match("States(.*)", s).group(1)
        state.__doc__ = _STATES_DESCRIPTION_TEMPLATE.format(generator_name=generator_name)
        state.__name__ = s


_wrap_states()


class Compile:
    """
    Compile the random device APIs with the specified compute capability.

    The ``files`` and ``extension`` attributes should be used as the arguments for
    :py:func:`numba.cuda.jit` decorator in Numba kernels which use random device APIs.

    Args:
        cc: (optional) the compute capability specified as an object of type
            :py:class:`nvmath.device.ComputeCapability`. If not specified, the default
            compute capability will be used.

    Example:
        >>> from numba import cuda
        >>> from nvmath.device import random
        >>> compiled_apis = random.Compile()
        >>> @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
        ... def kernel():
        ...     pass  # use random device APIs here
    """

    def __init__(self, cc: nvmath.device.ComputeCapability | None = None):
        global c_ext_shim_source

        if cc is None:
            cc = nvmath.device.common_cuda.get_default_code_type().cc
        elif not isinstance(cc, nvmath.device.ComputeCapability):
            raise ValueError(
                f"The specified compute capability {cc} is not valid. "
                "It must be an object of type :py:class:`nvmath.device.ComputeCapability`."
            )
        self.cc = cc

        # Compile APIs to LTO-IR and materialize in 'files' for linking into Numba kernels.
        _, self._lto = nvmath.device.nvrtc.compile(cpp=c_ext_shim_source.data, cc=self.cc, rdc=True, code="lto")  # type: ignore
        self._files = [nvmath.device.common.make_binary_tempfile(self._lto, ".ltoir")]

    @property
    def files(self):
        """
        The data needed to link random device APIs with Numba kernels.
        """
        return [f.name for f in self._files]

    @property
    def extension(self):
        """
        The extension needed to use random device APIs with Numba kernels.
        """
        global states_arg_handlers
        return states_arg_handlers


__all__ = ["Compile", "random_helpers"] + _COMMON_APIS + _SAMPLERS + _SKIPPERS + _STATES

del functools, re, curand_kernel, states
