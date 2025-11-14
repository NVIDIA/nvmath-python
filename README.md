<h1 align="center"><img
    src="docs/sphinx/_static/nvmath-python-green-r4.svg"
    alt="A cube with three sides visible. Dots, a sine wave, and a grid on the faces."
    width="200"/>
</h1>

# nvmath-python: NVIDIA Math Libraries for the Python Ecosystem

nvmath-python brings the power of the NVIDIA math libraries to the Python ecosystem.
The package aims to provide intuitive pythonic APIs giving users full access to all
features offered by NVIDIA's libraries in a variety of execution spaces. nvmath-python works
seamlessly with existing Python array/tensor frameworks and focuses on providing
functionality that is missing from those frameworks.

## Some Examples

Below are a few representative examples showcasing the three main categories of
features nvmath-python offers: host, device, and distributed APIs.

### Host APIs

Host APIs are called from host code but can execute in any supported execution
space (CPU or GPU). The following example shows how to compute a matrix multiplication
on CuPy matrices. Using the nvmath-python API allows access to *all* parameters
of the underlying NVIDIA cuBLASLt library, a distinguishing feature of nvmath-python
from other wrappings of NVIDIA's C-API libraries.

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

nvmath-python provides the ability to write custom prologs and epilogs for FFT functions as
Python functions and compile them to LTO-IR. For example, to have unitary scaling for an
FFT, we can define an epilog which rescales the output by `1/sqrt(N)`.

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

### Device-side APIs

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
    )
    FFT_inv = fft(
        fft_type="c2c",
        size=size,
        precision=np.float32,
        direction="inverse",
        ffts_per_block=ffts_per_block,
        elements_per_thread=2,
        execution="Block",
    )

    # Define a numba jit function targeting CUDA devices
    @cuda.jit
    def f(signal, filter):

        thread_data = cuda.local.array(
            shape=(FFT_fwd.storage_size,), dtype=FFT_fwd.value_type,
        )
        shared_mem = cuda.shared.array(shape=(0,), dtype=FFT_fwd.value_type)

        fft_id = (cuda.blockIdx.x * ffts_per_block) + cuda.threadIdx.y
        if(fft_id >= batch_size):
            return
        offset = cuda.threadIdx.x

        for i in range(FFT_fwd.elements_per_thread):
            thread_data[i] = signal[fft_id, offset + i * FFT_fwd.stride]

        # Call the cuFFTDx FFT function from *inside* your custom function
        FFT_fwd(thread_data, shared_mem)

        for i in range(FFT_fwd.elements_per_thread):
            thread_data[i] *= filter[fft_id, offset + i * FFT_fwd.stride]

        FFT_inv(thread_data, shared_mem)

        for i in range(FFT_fwd.elements_per_thread):
            signal[fft_id, offset + i * FFT_fwd.stride] = thread_data[i]


    data = random_complex((ffts_per_block, size), np.float32)
    filter = random_complex((ffts_per_block, size), np.float32)

    data_d = cuda.to_device(data)
    filter_d = cuda.to_device(filter)

    f[1, FFT_fwd.block_dim, 0, FFT_fwd.shared_memory_size](data_d, filter_d)
    cuda.synchronize()

    data_test = data_d.copy_to_host()
    data_ref = np.fft.ifft(np.fft.fft(data, axis=-1) * filter, axis=-1) * size

    error = np.linalg.norm(data_test - data_ref) / np.linalg.norm(data_ref)
    print(f"L2 error {error}")

    assert error < 1e-5

if __name__ == "__main__":
    main()
```

### Distributed APIs

Distributed APIs are called from host code but execute on a distributed
(multi-node multi-GPU) system. The following example shows the use of the
function-form distributed FFT with CuPy ndarrays:

```python
import cupy as cp
from mpi4py import MPI

import nvmath.distributed
from nvmath.distributed.distribution import Slab

# Initialize nvmath.distributed.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
device_id = rank % cp.cuda.runtime.getDeviceCount()
nvmath.distributed.initialize(device_id, comm, backends=["nvshmem"])

# The global 3-D FFT size is (512, 256, 512).
# In this example, the input data is distributed across processes according to
# the cuFFTMp Slab distribution on the X axis.
shape = 512 // nranks, 256, 512

# cuFFTMp uses the NVSHMEM PGAS model for distributed computation, which requires GPU
# operands to be on the symmetric heap.
a = nvmath.distributed.allocate_symmetric_memory(shape, cp, dtype=cp.complex128)
# a is a cupy ndarray and can be operated on using in-place cupy operations.
with cp.cuda.Device(device_id):
    a[:] = cp.random.rand(*shape, dtype=cp.float64) + 1j *
        cp.random.rand(*shape, dtype=cp.float64)

# Forward FFT.
# In this example, the forward FFT operand is distributed according
# to Slab.X distribution. With reshape=False, the FFT result will be
# distributed according to Slab.Y distribution.
b = nvmath.distributed.fft.fft(a, distribution=Slab.X, options={"reshape": False})

# Distributed FFT performs computations in-place. The result is stored in the same
# buffer as operand a. Note, however, that operand b has a different shape (due
# to Slab.Y distribution).
if rank == 0:
    print(f"Shape of a on rank {rank} is {a.shape}")
    print(f"Shape of b on rank {rank} is {b.shape}")

# Inverse FFT.
# Recall from previous transform that the inverse FFT operand is distributed according
# to Slab.Y. With reshape=False, the inverse FFT result will be distributed according
# to Slab.X distribution.
c = nvmath.distributed.fft.ifft(b, distribution=Slab.Y, options={"reshape": False})

# The shape of c is the same as a (due to Slab.X distribution). Once again, note that
# a, b and c are sharing the same symmetric memory buffer (distributed FFT operations
# are in-place).
if rank == 0:
    print(f"Shape of c on rank {rank} is {c.shape}")

# Synchronize the default stream
with cp.cuda.Device(device_id):
    cp.cuda.get_current_stream().synchronize()

if rank == 0:
    print(f"Input type = {type(a)}, device = {a.device}")
    print(f"FFT output type = {type(b)}, device = {b.device}")
    print(f"IFFT output type = {type(c)}, device = {c.device}")

# GPU operands on the symmetric heap are not garbage-collected and the user is
# responsible for freeing any that they own (this deallocation is a collective
# operation that must be called by all processes at the same point in the execution).
# All cuFFTMp operations are inplace (a, b, and c share the same memory buffer), so
# we take care to only free the buffer once.
nvmath.distributed.free_symmetric_memory(a)
```

## License

All files hosted in this repository are subject to the [Apache 2.0](./LICENSE) license.

## Disclaimer

nvmath-python is in a Beta state. Beta products may not be fully functional, may contain
errors or design flaws, and may be changed at any time without notice. We appreciate your
feedback to improve and iterate on our Beta products.
