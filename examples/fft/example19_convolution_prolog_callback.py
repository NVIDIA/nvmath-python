# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to perform a convolution by providing
a Python callback function as prolog to the IFFT operation.

To run this example, CUDA Toolkit 12.6U2 and device API (dx) dependencies
are required. The quickest way for pip users to set them up is to install
nvmath as ``pip install nvmath-python[cu12,dx]``.

For further details, please see :ref:`FFT callbacks <fft-callback>`.
"""

import cupy as cp

import nvmath

# Create the data for the batched 1-D FFT.
B, N = 256, 1024
a = cp.random.rand(B, N, dtype=cp.float64) + 1j * cp.random.rand(B, N, dtype=cp.float64)

# Create the data to use as filter.
filter_data = cp.sin(a)


# Define the prolog function for the inverse FFT.
def convolve(data_in, offset, filter_data, unused):
    """
    A convolution corresponds to pointwise multiplication in the frequency domain. We also scale by the FFT size N here.
    """
    # Note we are accessing `data_out` and `filter_data` with a single `offset` integer,
    # even though the input and `filter_data` are 2D tensors (batches of samples).
    # Care must be taken to assure that both arrays accessed here have the same memory layout.
    # For a reference, see the `example19_convolution_callback_memory_layout` example.
    return data_in[offset] * filter_data[offset] / N


# Compile the prolog to LTO-IR.
# In a system with GPUs that have different compute capability, the `compute_capability` option must be specified to the
# `compile_prolog` or `compile_epilog` helpers. Alternatively, the prolog can be compiled in the context of the device
# where the FFT to which the prolog is provided is executed. In this case we use the current device context, where the
# operands have been created.
with cp.cuda.Device():
    prolog = nvmath.fft.compile_prolog(convolve, "complex128", "complex128")

# Perform the forward FFT...
r = nvmath.fft.fft(a, axes=[-1])

# ... followed by the inverse FFT, applying the filter as a prolog.
r = nvmath.fft.ifft(r, axes=[-1], prolog={"ltoir": prolog, "data": filter_data.data.ptr})

# Finally, we can test that the fused FFT run
# result matches the result of separate calls
s = cp.fft.fftn(a, axes=[-1])
s *= filter_data
s = cp.fft.ifftn(s, axes=[-1])

print(cp.allclose(r, s))
