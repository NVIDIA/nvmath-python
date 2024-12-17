<h1 align="center"><img
    src="docs/sphinx/_static/nvmath-python-green-r4.svg"
    alt="A cube with three sides visible. Dots, a sine wave, and a grid on the faces."
    width="200"/>
</h1>

# nvmath-python: NVIDIA Math Libraries for the Python Ecosystem

nvmath-python brings the power of the NVIDIA math libraries to the Python ecosystem. The
package aims to provide intuitive pythonic APIs that provide users full access to all the
features offered by NVIDIA's libraries in a variety of execution spaces. nvmath-python works
seamlessly with existing Python array/tensor frameworks and focuses on providing
functionality that is missing from those frameworks.

## Some Examples

Using the nvmath-python API allows access to all parameters of the underlying NVIDIA
cuBLASLt library. Some of these parameters are unavailable in other wrappings of NVIDIA's
C-API libraries.

```python
import cupy as cp
import nvmath

# Prepare sample input data. nvmath-python accepts input tensors from pytorch, cupy, and
# numpy.
m, n, k = 123, 456, 789
a = cp.random.rand(m, k).astype(cp.float32)
b = cp.random.rand(k, n).astype(cp.float32)
bias = cp.random.rand(m, 1).astype(cp.float32)

# Use the stateful Matmul object in order to perform multiple matrix multiplications
# without replanning. The nvmath API allows us to fine-tune our operations by, for
# example, selecting a mixed-precision compute type.
mm = nvmath.linalg.advanced.Matmul(
    a,
    b,
    options={
        "compute_type": nvmath.linalg.advanced.MatmulComputeType.COMPUTE_32F_FAST_16F
    },
)

# Plan the matrix multiplication. Planning returns a sequence of algorithms that can be
# configured. We can also select epilog operations which are applied to the result of
# the multiplication without a separate function call.
mm.plan(
    epilog=nvmath.linalg.advanced.MatmulEpilog.BIAS,
    epilog_inputs={"bias": bias},
)

# Execute the matrix multiplication.
result = mm.execute()

# Remember to free the Matmul object when finished or use it as a context manager
mm.free()

# Synchronize the default stream, since by default the execution is non-blocking for
# GPU operands.
cp.cuda.get_current_stream().synchronize()
print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
print(f"Result type = {type(result)}, device = {result.device}")
```

nvmath-python exposes NVIDIA's device-side (Dx) APIs. This allows developers to call NVIDIA
library functions inside their custom device kernels. For example, a numba jit function can
call cuFFT in order to implement FFT-based convolution.

```python
import numpy as np
from numba import cuda
from nvmath.device import fft

def random_complex(shape, real_dtype):
    return (
        np.random.randn(*shape).astype(real_dtype)
        + 1.j * np.random.randn(*shape).astype(real_dtype)
    )

def main():

    size = 128
    ffts_per_block = 1
    batch_size = 1

    # Instantiate device-side functions from cuFFTDx.
    FFT_fwd = fft(
        fft_type="c2c",
        size=size,
        precision=np.float32,
        direction="forward",
        ffts_per_block=ffts_per_block,
        elements_per_thread=2,
        execution="Block",
        compiler="numba",
    )
    FFT_inv = fft(
        fft_type="c2c",
        size=size,
        precision=np.float32,
        direction="inverse",
        ffts_per_block=ffts_per_block,
        elements_per_thread=2,
        execution="Block",
        compiler="numba",
    )

    value_type          = FFT_fwd.value_type
    storage_size        = FFT_fwd.storage_size
    shared_memory_size  = FFT_fwd.shared_memory_size
    fft_stride          = FFT_fwd.stride
    ept                 = FFT_fwd.elements_per_thread
    block_dim           = FFT_fwd.block_dim

    # Define a numba jit function targeting CUDA devices
    @cuda.jit(link=FFT_fwd.files + FFT_inv.files)
    def f(signal, filter):

        thread_data = cuda.local.array(shape=(storage_size,), dtype=value_type)
        shared_mem = cuda.shared.array(shape=(0,), dtype=value_type)

        fft_id = (cuda.blockIdx.x * ffts_per_block) + cuda.threadIdx.y
        if(fft_id >= batch_size):
            return
        offset = cuda.threadIdx.x

        for i in range(ept):
            thread_data[i] = signal[fft_id, offset + i * fft_stride]

        # Call the cuFFTDx FFT function from *inside* your custom function
        FFT_fwd(thread_data, shared_mem)

        for i in range(ept):
            thread_data[i] = thread_data[i] * filter[fft_id, offset + i * fft_stride]

        FFT_inv(thread_data, shared_mem)

        for i in range(ept):
            signal[fft_id, offset + i * fft_stride] = thread_data[i]


    data = random_complex((ffts_per_block, size), np.float32)
    filter = random_complex((ffts_per_block, size), np.float32)

    data_d = cuda.to_device(data)
    filter_d = cuda.to_device(filter)

    f[1, block_dim, 0, shared_memory_size](data_d, filter_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()
    data_ref = np.fft.ifft(np.fft.fft(data, axis=-1) * filter, axis=-1) * size

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"L2 error {error}")

    assert error < 1e-5

if __name__ == "__main__":
    main()
```

nvmath-python provides the ability to write custom prologs and epilogs for FFT functions as
a Python functions and compiled them LTO-IR. For example, to have unitary scaling for an
FFT, we can define an epilog which rescales the output by 1/sqrt(N).

```python
import cupy as cp
import nvmath
import math

# Create the data for the batched 1-D FFT.
B, N = 256, 1024
a = cp.random.rand(B, N, dtype=cp.float64) + 1j * cp.random.rand(B, N, dtype=cp.float64)

# Compute the normalization factor for unitary transforms
norm_factor = 1.0 / math.sqrt(N)

# Define the epilog function for the FFT.
def rescale(data_out, offset, data, user_info, unused):
    data_out[offset] = data * norm_factor

# Compile the epilog to LTO-IR.
with cp.cuda.Device():
    epilog = nvmath.fft.compile_epilog(rescale, "complex128", "complex128")

# Perform the forward FFT, applying the filter as a epilog...
r = nvmath.fft.fft(a, axes=[-1], epilog={"ltoir": epilog})

# Finally, we can test that the fused FFT run result matches the result of separate
# calls
s = cp.fft.fftn(a, axes=[-1], norm="ortho")

assert cp.allclose(r, s)
```

## License

All files hosted in this repository are subject to the [Apache 2.0](./LICENSE) license.

## Disclaimer

nvmath-python is in a Beta state. Beta products may not be fully functional, may contain
errors or design flaws, and may be changed at any time without notice. We appreciate your
feedback to improve and iterate on our Beta products.
