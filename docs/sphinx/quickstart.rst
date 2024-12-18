Getting Started
***************

nvmath-python brings the power of the NVIDIA math libraries to the Python ecosystem.
The package aims to provide intuitive pythonic APIs that provide users full access
to all the features offered by NVIDIA's libraries in a variety of execution spaces.
nvmath-python works seamlessly with existing Python array/tensor frameworks and focuses
on providing functionality that is missing from those frameworks.

To learn more about the design of nvmath-python, visit our :doc:`Overview<overview>`.

Installation
============

To quickly install nvmath-python just run the following command:

.. code-block:: bash

    pip install nvmath-python[cu12,dx]

For more details visit the :doc:`Installation Guide<installation>`.

Examples
=========

In the examples below, we quickly demonstrate the basic capabilities
of nvmath-python. You can find more examples in our
`GitHub repository <https://github.com/NVIDIA/nvmath-python/tree/main/examples>`_.

Matrix multiplication
---------------------

Using the nvmath-python API allows access to all parameters of the underlying
NVIDIA cuBLASLt library.
Some of these parameters are unavailable in other wrappings of NVIDIA's C-API libraries.

.. doctest::

    >>> import cupy as cp
    >>> import nvmath
    >>>
    >>> m, n, k = 123, 456, 789
    >>> a = cp.random.rand(m, k).astype(cp.float32)
    >>> b = cp.random.rand(k, n).astype(cp.float32)
    >>>
    >>> # Use the stateful nvmath.linalg.advanced.Matmul object in order to separate planning
    >>> # from actual execution of matrix multiplication. nvmath-python allows you to fine-tune
    >>> # your operations by, for example, selecting a mixed-precision compute type.
    >>> options = {
    ...     "compute_type": nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_16F
    ... }
    >>> with nvmath.linalg.advanced.Matmul(a, b, options=options) as mm:
    ...     algorithms = mm.plan()
    ...     result = mm.execute()

To learn more about matrix multiplication in nvmath-python, have a look at
:py:class:`~nvmath.linalg.advanced.Matmul`.

FFT with callback
-----------------

User-defined functions can be `compiled to the LTO-IR format
<https://docs.nvidia.com/cuda/cufft/index.html#lto-load-and-store-callback-routines>`_ and
provided as epilog or prolog to the FFT operation, allowing for Link-Time Optimization and
fusing.

This example shows how to perform a convolution by providing a Python callback function as
prolog to the IFFT operation.

.. doctest::

    >>> import cupy as cp
    >>> import nvmath
    >>>
    >>> # Create the data for the batched 1-D FFT.
    >>> B, N = 256, 1024
    >>> a = cp.random.rand(B, N, dtype=cp.float64) + 1j * cp.random.rand(B, N, dtype=cp.float64)
    >>>
    >>> # Create the data to use as filter.
    >>> filter_data = cp.sin(a)
    >>>
    >>> # Define the prolog function for the inverse FFT.
    >>> # A convolution corresponds to pointwise multiplication in the frequency domain.
    >>> def convolve(data_in, offset, filter_data, unused):
    ...     # Note we are accessing `data_out` and `filter_data` with a single `offset` integer,
    ...     # even though the input and `filter_data` are 2D tensors (batches of samples).
    ...     # Care must be taken to assure that both arrays accessed here have the same memory
    ...     # layout.
    ...     return data_in[offset] * filter_data[offset] / N
    >>>
    >>> # Compile the prolog to LTO-IR.
    >>> with cp.cuda.Device():
    ...     prolog = nvmath.fft.compile_prolog(convolve, "complex128", "complex128")
    >>>
    >>> # Perform the forward FFT, followed by the inverse FFT, applying the filter as a prolog.
    >>> r = nvmath.fft.fft(a, axes=[-1])
    >>> r = nvmath.fft.ifft(r, axes=[-1], prolog={
    ...         "ltoir": prolog,
    ...         "data": filter_data.data.ptr
    ...     })

For further details, see the :ref:`FFT callbacks documentation <fft-callback>`.

Device APIs
-----------

The device APIs of nvmath-python allow you to access the functionalities
of cuFFTDx, cuBLASDx, and cuRAND libraries in your kernels.

This example shows how to use the cuRAND to sample
a single-precision value from a normal distribution.

First, create the array of bit-generator states (one per thread).
In this example, we'll use
:py:class:`Philox4_32_10<nvmath.device.random.StatesPhilox4_32_10>` generator.

.. doctest::

    >>> from numba import cuda
    >>> from nvmath.device import random
    >>> compiled_apis = random.Compile()
    >>>
    >>> threads, blocks = 64, 64
    >>> nthreads = blocks * threads
    >>>
    >>> states = random.StatesPhilox4_32_10(nthreads)
    >>>
    >>> # Next, define and launch a setup kernel, which will initialize the states using
    >>> # nvmath.device.random.init function.
    >>> @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    ... def setup(states):
    ...     i = cuda.grid(1)
    ...     random.init(1234, i, 0, states[i])
    >>>
    >>> setup[blocks, threads](states)
    >>>
    >>> # With your states array ready, you can use samplers such as
    >>> # nvmath.device.random.normal2 to sample random values in your kernels.
    >>> @cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
    ... def kernel(states):
    ...     i = cuda.grid(1)
    ...     random_values = random.normal2(states[i])

To learn more about this and other Device APIs,
visit the documentation of :mod:`nvmath.device`.
