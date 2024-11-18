# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

from numba import cuda, uint32, uint64
import nvmath.device.random as R
from functools import cache
from .compiled_apis import compiled_apis
import cffi

ffi = cffi.FFI()


class Generator:
    def curand_setup(self, *, blocks, threads, seed, offset, states):
        """
        Launches setup kernel with specified seed and offset.
        """
        raise NotImplementedError

    def curand_states(self, *args, **kwargs):
        """
        Instantiates curand States.* object
        """
        raise NotImplementedError

    def name(self):
        """
        Readable generator name
        """
        raise NotImplementedError

    def supports_skipahead(self) -> bool:
        """
        If skipahead() is supported
        """
        return False

    def get_skipahead_subsequence_function(self):
        """
        Returns a function skipping ahead a subsequence.
        It's call either skipahead_sequence or skipahead_subsequence, depedning
        on the generator.
        """
        return None

    def supports_skipahead_subsequence(self):
        return self.get_skipahead_subsequence_function() is not None


@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def seeded_setup_kernel(seed, offset, states):
    i = cuda.grid(1)
    R.init(seed, uint64(i), uint64(offset), states[i])


class SimpleSeededGenerator(Generator):
    def curand_setup(self, *, blocks, threads, seed, offset, states):
        seeded_setup_kernel[blocks, threads](uint64(seed), uint64(offset), states)


class XorwowGenerator(SimpleSeededGenerator):
    def curand_states(self, *args, **kwargs):
        return R.StatesXORWOW(*args, **kwargs)

    def name(self):
        return "xorwow"

    def supports_skipahead(self):
        return True

    def get_skipahead_subsequence_function(self):
        return R.skipahead_sequence  # misnamed in curand


class MrgGenerator(SimpleSeededGenerator):
    def curand_states(self, *args, **kwargs):
        return R.StatesMRG32k3a(*args, **kwargs)

    def name(self):
        return "mrg"

    def supports_skipahead(self):
        return True

    def get_skipahead_subsequence_function(self):
        return R.skipahead_subsequence


class PhiloxGenerator(SimpleSeededGenerator):
    def curand_states(self, *args, **kwargs):
        return R.StatesPhilox4_32_10(*args, **kwargs)

    def name(self):
        return "philox"

    def supports_skipahead(self):
        return True

    def get_skipahead_subsequence_function(self):
        return R.skipahead_sequence  # misnamed in curand


@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def sobol32_setup_kernel(vectors, offset, states):
    i = cuda.grid(1)
    R.init(ffi.from_buffer(vectors[i]), uint32(offset), states[i])


class Sobol32Generator(Generator):
    def curand_setup(self, *, blocks, threads, seed, offset, states):
        vectors = R.random_helpers.get_direction_vectors32(
            R.random_helpers.DirectionVectorSet.DIRECTION_VECTORS_32_JOEKUO6,
            blocks * threads,
        )
        vectors = cuda.to_device(vectors)
        offset += seed * 1000000007  # good enough for our tests
        sobol32_setup_kernel[blocks, threads](vectors, uint64(offset), states)

    def curand_states(self, *args, **kwargs):
        return R.StatesSobol32(*args, **kwargs)

    def name(self):
        return "sobol32"


@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def sobol64_setup_kernel(vectors, offset, states):
    i = cuda.grid(1)
    R.init(ffi.from_buffer(vectors[i]), uint32(offset), states[i])


class Sobol64Generator(Generator):
    def curand_setup(self, *, blocks, threads, seed, offset, states):
        vectors = R.random_helpers.get_direction_vectors64(
            R.random_helpers.DirectionVectorSet.DIRECTION_VECTORS_64_JOEKUO6,
            blocks * threads,
        )
        vectors = cuda.to_device(vectors)
        offset += seed * 1000000007  # good enough for our tests
        sobol64_setup_kernel[blocks, threads](vectors, uint64(offset), states)

    def curand_states(self, *args, **kwargs):
        return R.StatesSobol64(*args, **kwargs)

    def name(self):
        return "sobol64"


@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def scrambled_sobol32_setup_kernel(vectors, scramble_constants, offset, states):
    i = cuda.grid(1)
    R.init(ffi.from_buffer(vectors[i]), scramble_constants[i], uint32(offset), states[i])


class ScrambledSobol32Generator(Generator):
    def curand_setup(self, *, blocks, threads, seed, offset, states):
        vectors = R.random_helpers.get_direction_vectors32(
            R.random_helpers.DirectionVectorSet.DIRECTION_VECTORS_32_JOEKUO6,
            blocks * threads,
        )
        vectors = cuda.to_device(vectors)
        scramble_constants = R.random_helpers.get_scramble_constants32(blocks * threads)
        scramble_constants = cuda.to_device(scramble_constants)

        offset += seed * 1000000007  # good enough for our tests
        scrambled_sobol32_setup_kernel[blocks, threads](vectors, scramble_constants, uint64(offset), states)

    def curand_states(self, *args, **kwargs):
        return R.StatesScrambledSobol32(*args, **kwargs)

    def name(self):
        return "scrambledsobol32"


@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def scrambled_sobol64_setup_kernel(vectors, scramble_constants, offset, states):
    i = cuda.grid(1)
    R.init(ffi.from_buffer(vectors[i]), scramble_constants[i], uint32(offset), states[i])


class ScrambledSobol64Generator(Generator):
    def curand_setup(self, *, blocks, threads, seed, offset, states):
        vectors = R.random_helpers.get_direction_vectors64(
            R.random_helpers.DirectionVectorSet.DIRECTION_VECTORS_64_JOEKUO6,
            blocks * threads,
        )
        vectors = cuda.to_device(vectors)
        scramble_constants = R.random_helpers.get_scramble_constants64(blocks * threads)
        scramble_constants = cuda.to_device(scramble_constants)

        offset += seed * 1000000007  # good enough for our tests
        scrambled_sobol64_setup_kernel[blocks, threads](vectors, scramble_constants, uint64(offset), states)

    def curand_states(self, *args, **kwargs):
        return R.StatesScrambledSobol64(*args, **kwargs)

    def name(self):
        return "scrambledsobol64"


GENERATORS = [
    XorwowGenerator,
    MrgGenerator,
    PhiloxGenerator,
    Sobol32Generator,
    Sobol64Generator,
    ScrambledSobol32Generator,
    ScrambledSobol64Generator,
]
