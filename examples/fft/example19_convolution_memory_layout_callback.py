# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to use callbacks with user-provided filter data,
focusing on memory-layoyt pitfalls.

To run this example, CUDA Toolkit 12.6U2 and device API (dx) dependencies
are required. The quickest way for pip users to set them up is to install
nvmath as ``pip install nvmath-python[cu12,dx]``.

For further details, please see :ref:`FFT callbacks <fft-callback>`.
"""

import cupy as cp

import nvmath

# Create the data for the batched 2-D C2C FFT.
# Note, the batch extent comes last, making the samples interleaved.
N1, N2, B = 128, 128, 64
a = cp.random.rand(N1, N2, B, dtype=cp.float64) + 1j * cp.random.rand(N1, N2, B, dtype=cp.float64)

# Create the data to use as filter.
filter_data = cp.sin(a)


def get_tensor_strides(t):
    """
    Get the strides of the tensor as a number of elements.
    Cupy describes the strides in terms of bytes.
    """
    return tuple(s // t.itemsize for s in t.strides)


# Define the prolog function for the inverse FFT.
def convolve(data_out, offset, element, filter_data, unused):
    """
    A convolution corresponds to pointwise multiplication in the frequency domain.
    We also scale by the FFT size N1 * N2 here.
    """
    data_out[offset] = element * filter_data[offset] / (N1 * N2)


# Compile the epilog to LTO-IR.
# In a system with GPUs that have different compute capability, the `compute_capability` option must be specified to the
# `compile_prolog` or `compile_epilog` helpers. Alternatively, the epilog can be compiled in the context of the device
# where the FFT to which the epilog is provided is executed. In this case we use the current device context, where the
# operands have been created.
with cp.cuda.Device():
    epilog = nvmath.fft.compile_epilog(convolve, "complex128", "complex128")

# Perform the forward FFT...
with nvmath.fft.FFT(a, axes=(0, 1), options={"result_layout": "optimized"}) as fft:
    output_shape, output_strides = fft.get_output_layout()
    # C2C transform preserves shape
    assert output_shape == filter_data.shape
    # However, there comes a surprise for the output_strides
    assert output_strides != get_tensor_strides(filter_data)
    # Even though the input and filter_data strides match,
    assert a.strides == filter_data.strides
    # the output operand has different stride order.
    assert output_strides == (N1, 1, N1 * N2)
    # Here, the reason for the difference is setting the result_layout=optimized
    # (which is default). One way to address this, could be to pass
    # result_layout=natural, so that nvmath preserves the `a`'s C-like stride order
    assert get_tensor_strides(a) == (N2 * B, B, 1)
    # In general, the operand strides depend on many factors.
    # Even the input operand's layout can change if nvmath must
    # perform an internal copy.
    # Users are advised to always check that filter's layout
    # matches input's (for prolog) or output's (for epilog) layout
    # to avoid insidious memory access bugs.
    #
    # Here, we permute the filter_data to match the output layout
    filter_copy = filter_data.transpose((2, 0, 1)).copy().transpose((1, 2, 0))
    assert output_shape == filter_copy.shape
    # And make sure the strides match
    assert output_strides == get_tensor_strides(filter_copy)
    # We can safely pass the permuted filter to epilog
    fft.plan(epilog={"ltoir": epilog, "data": filter_copy.data.ptr})
    r = fft.execute()

# ... followed by the inverse FFT.
r = nvmath.fft.ifft(r, axes=(0, 1))

# Finally, we can test that the fused FFT run
# result matches the result of separate calls
s = cp.fft.fftn(a, axes=(0, 1))
s *= filter_data
s = cp.fft.ifftn(s, axes=(0, 1))

print(cp.allclose(r, s))
